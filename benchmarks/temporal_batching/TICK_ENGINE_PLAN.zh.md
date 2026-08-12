# 节拍引擎施工方案(基于四条缝合线的源码勘察)

**日期**: 2026-08-12 · 前置:`TICK_ENGINE.zh.md`(设计)、`RESULTS.zh.md`(retrofit 实证)
勘察对象:vllm 0.26.0 v1(installed)+ vllm-omni fork。所有 file:line 均已核实。

## 零、总原则与一个修正性发现

**保留**(不动):worker/kernel、KV block 池、权重加载、两个 IO 线程
(zmq 输入/输出线程与步频已解耦,queue.Queue 缓冲,天然兼容 tick)、
connector 的 put/get 接口签名、omni 的 stage/colocation 架构。

**替换**只有三件事:
1. 循环起搏:自由跑 → tick 等待(WP1)
2. 调度时机:每步在线决策 → 每事件计划 + 每拍重放(WP2/3)
3. 传输时机:异步队列+轮询 → 定相信箱(WP4)

**修正性发现**(勘察实测):稳态下上游 schedule() + update_from_output 的
Python 成本约 5-15µs/请求/步,32 session ≈ 0.3-0.5ms/tick——**不到 80ms 的
1%**。所以"调度开销归零"不是主要收益;真正买到的是:
(a) **尾抖动**:分配重试环、驱逐扫描、调度 pass 与模型步的对齐随机性;
(b) **CUDA graph 恒定命中**:batch descriptor 在事件间不变,混合 prefill
不再破坏 FULL 重放——这是步时方差的最大来源;
(c) **传输相位**:9 跳异步链(见 WP4)的抖动结构性消失;
(d) **硬容量语义**:准入=装箱检查,抢占结构性不可能。

---

## WP1 tick 心跳(改一个函数 + colocation 时钟)

**核心改动点只有一处**:`vllm/v1/engine/core.py:1368-1397`
`_process_input_queue` 的契约从"有活立即返回"改成"带截止期地
`input_queue.get(timeout=距下一拍余量)`,拍边沿返回"。

- 输入仍**即时处理**(admission-to-scheduler 即时;admission-to-batch 在
  下一拍边沿由计划器接手)——中途到的新轮不丢响应性。
- 删 1ms GIL-yield sleep(core.py:1413,松弛期本身就是 yield);成员为空
  时退回今天的阻塞等待(idle/pause 回调路径 1372-1374 原样保留)。
- `step_with_batch_queue` 的异步重叠机制**保留**(submit 非阻塞 + 延迟
  harvest 正是"拍边沿提交、下拍边沿收割"要的形状,depth=2 语义不变),
  只是提交时刻从"队列不满就提"改成"拍边沿提"。
- **colocation 是硬点**:同进程 N 个 stage 各自自由循环在 N 个线程上
  (stage_engine_core_proc.py:215-275,零协调)→ 换成每进程一个
  monotonic 时钟源 + `threading.Condition` 定时唤醒 + 每 stage 相位偏移。
- **新增 watchdog**:tick 超期(deadline miss)计数——现在引擎没有任何
  周期抖动度量(勘察确认 steady-state 无任何 wall-clock 监测)。

验收(M1):挂上现有 v3 屏障跑长回答 u16,步进间隔 p99 从 81.2ms 收敛到
80±1ms,超期率可观测。

## WP2 事件计划器 + 每拍重放(调度器侧)

调度决策收缩到**成员事件**,四类事件与现有代码 1:1 对应,直接改造为
计划事件处理器:

| 事件 | 现有代码(复用为 handler) |
|---|---|
| join | waiting 准入段 scheduler.py:666-1053(含 prefix 查询+首次分配) |
| leave | _free_request / _free_blocks :2207-2239 |
| turn-start | _update_request_as_session :1286-1327 |
| turn-end | _handle_stopped_request :1988-2004(停止检查触发) |

事件时产出 **TickPlan**:cohort 名单、每 session 配额、K 拍的 KV 块预算、
CUDA graph descriptor。每拍重放只做:位置 += quantum(一次向量加)、
跨块边界时 append 块(每 block_size token 一次)、采样 token 落账 +
check_stop(不可约的数据依赖,stop 即转化为 turn-end 事件)。

- **KV 预分配**:allocate_slots 已支持超量(num_lookahead_tokens 先例);
  事件时一次分配 K 拍的块,拍内零分配器调用。失效仅由池耗尽/中止触发。
- **抢占结构性消灭**(已验证:上游抢占只由分配失败触发,scheduler.py:
  564-607;waiting 准入从不抢占):把现有三件套准入门(watermark
  :402-409、full_sequence_must_fit :411-427、reserved_blocks :927-947)
  升级为**充分条件**——按计划视界的最坏块数准入。
- 保留 deferred-free 写栅栏语义(:318-324, 2248-2287):leave 事件释放的
  块必须等 in-flight 拍收割后归池——信箱的双缓冲有同构的 hazard。
- **陷阱**:optimistic-advance 契约——_update_after_schedule(:1236-1262)
  在调度时提前推进 computed,runner 消费 pre-advance 值;计划必须按此
  顺序存每拍位置,不能混两种口径。
- fork 侧落点:替换 OmniARScheduler.schedule() 的 super() 调用(:509)与
  temporal_pacer 的列表手术(:491-534)——v3 的手术每步都在证明 cohort
  事件间稳定,计划/重放只是把假装的稳定变成真的。

## WP3 runner 的 cohort 重放快路(最重的工程)

勘察结论:InputBatch 的持久行(token_ids_cpu、block table、采样参数)
已经是事件驱动的;但**所有形状/索引张量每步从零重建**
(_prepare_inputs:numpy req_indices/cu_num_tokens/query_start_loc 重建
+ CPU token gather + ~8 次 H2D + Triton slot-mapping),attention
metadata 每 KV group 每步 fresh build。

- 事件期编译一次:req_indices / cu_num_tokens / query_start_loc /
  discard_mask 都是 cohort 常量;每拍只推进 positions/seq_lens(一次
  GPU add)+ slot_mapping 增量。
- **attention metadata 需要 advance(delta) 变体**——per-backend builder
  现在没有;照 build_for_cudagraph_capture 的持久缓冲方式做。
- **CUDA graph 有一个真坑**:FULL 重放要求 uniform decode
  (query_len == 1+num_spec_tokens)。**thinker 2 token/拍不满足**。
  两个出路:(a) 借 spec-token 式捕获(uniform query_len=2);
  (b) thinker 每拍跑 2 次 graph 步(简单,先用这个)。talker 1 帧/拍
  天然 uniform ✓。
- token 回写:采用 GPU 常驻回路(async 调度的 prev_sampled scatter)为
  唯一回路;-1 哨兵只是 CPU 侧最终一致性的机制——它的消费者
  (penalties/bad_words/logitsprocs、omni 自定义 sampler 的强制同步)
  逐个安排到事件期或向量化。
- _bookkeeping_sync 的每步 Python 循环(:3766-3791)→ 固定形状向量写。
- **prefill 进松弛槽的红利**:混合批是破坏 FULL graph 重放和 metadata
  重建的唯一大源;分离后周期路**永远 uniform、永远命中同一张图**。

## WP4 定相信箱(传输)

勘察确认现状一条 chunk 走 **9 跳**:update_from_output → save_async 入队
→ save 线程 → payload 构建(detach().cpu())→ connector.put(msgpack
全拷贝 + flock + **每 chunk 新建一个 POSIX shm 段**)→ 消费侧线程 1ms
轮询(100ms 空闲回退)→ 解码+改请求 → 下一个调度 pass 才观察到 →
postprocess 挂载。相位抖动进入点:每次线程唤醒、可变序列化(实测
~225MB/s)、轮询量子、以及**调度 pass 对齐**(到货只在下一 pass 顶部
被看见)。

- **保留** put/get(from_stage,to_stage,key) 签名与工厂注册——调度器和
  adapter 调用点零改动;key 的 chunk_id 重释义为 tick 序号。
- 同进程:ColocInProcConnector 的 take-once dict → 每 (edge,session)
  **两个固定槽位、tick 奇偶寻址**,去锁(单写者 T+0 / 单读者 T+φ)。
- 跨进程:常驻环形 shm 段 + 固定布局槽位(去 per-chunk shm 创建/解锁
  syscall、去 msgpack 全拷贝)。
- send 变 T+0 相位同步工作(**删 save 线程**及其记录在案的四类竞态);
  recv 变 T+φ 相位回调(**删 recv 线程**与 1ms/100ms 轮询)。
- **相位保证下可删除**:WAITING_FOR_CHUNK 状态补丁、两个停车 deque、
  origin_status 表、held_non_active、has_requests 死锁覆盖、大部分孤儿
  恢复——这一整个状态机存在的唯一理由是"到货不可预测"。
- **小心**:runner 的 omni_connector_model_runner_mixin 里还有一份复制的
  poll/send 状态机(自己的线程和间隔)——两处必须一起换,否则留一半
  抖动源。

## WP5 输入节拍化(v4c,模型侧的大活)

- entrypoint 加音频逐拍 append 通道(帧的 prefill-on-arrival 已有先例);
- 音频编码器流式化:带左上下文的分块编码,**必须做音质回归验证**;
- turn 起点大 prefill → 松弛分片,TTFA = ceil(prefill_tokens/每拍松弛
  token 数) 拍,变成可计算量。

## WP6 双车道 + 准入(产品化)

- 周期实时道(cohort)+ 松弛 best-effort 道(text-only/离线请求走原
  调度路径)——Linux RT+CFS 的形状。
- 准入 = 装箱检查:Σ(每拍工作) + 余量 ≤ tick;超载即拒,不排队。
- 分布式:DP 的 step_counter 习语(32 步 all-reduce、dummy batch
  lockstep,core.py:2015-2093)天然映射为 tick 计数——已是"按步数的
  周期性",现成的扩展先例。

## 里程碑(每步独立可测)

| | 内容 | 验收 |
|---|---|---|
| M1 | WP1 心跳 | 步进 p99 → 80±1ms,超期率可观测 |
| M2 | WP2+3 talker 先行 | 每拍 CPU、graph 命中 100%、碎步归零 |
| M3 | WP4 信箱 | 0→1 相位散布归零(sched_steps 直方图) |
| M4 | thinker 重放 + WP5 | 全链节拍;TTFA 可计算性验证 |
| M5 | WP6 | N=装箱上限连续 1h 零 miss(硬保证演示) |

粗估(单人):WP1 2-3 天;WP2 1-1.5 周;WP3 1.5-2 周(attention advance
+ graph 是硬骨头);WP4 1 周;WP5 1-2 周(编码器验证);WP6 3-4 天。
到 M4 约 6-8 周。

## 风险表

| 风险 | 缓解 |
|---|---|
| thinker 2 token/拍 ≠ uniform decode,graph 不重放 | 先每拍 2 次 graph 步,后 spec 式捕获 |
| MoE 步时方差吃掉拍余量 | 按 70% 利用率装箱;超期 watchdog 降级(丢 1 拍配额,lead 缓冲吸收) |
| 上游 vLLM 合并漂移 | 接缝全部选在 omni fork 已覆盖的类(StageEngineCoreProc / OmniARScheduler / omni runner / connector);上游文件只碰 core.py 一个函数 |
| 编码器流式化损音质 | WP5 独立音质回归门,不过不并 |
| KV roll/压缩与固定计划冲突 | roll 定义为计划内事件(leave+join 原子对) |
