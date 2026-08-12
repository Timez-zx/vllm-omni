# Temporal Batching 实验设计

**分支**: `temporal-batching`(基于 `live-agent-web`)
**日期**: 2026-08-12
**问题**: 对于 video+audio 输入、audio 输出的 streaming 多用户负载,把生成节奏对齐到音频的天然周期(1 codec 帧 = 1920 样本 @ 24kHz = **80ms,12.5Hz**)的 "temporal batching",相比现状的贪心 continuous batching,在**吞吐**与**延迟可预测性**上是否更好?

---

## 一、现状与动机

当前系统(见 workflow.zh.md 与调度器代码)的行为:

- talker(stage 1)对每轮回复**贪心解码**:音频帧以远快于播放的速度爆发式生成(单轮 RTF≈0.1–0.3),然后该 session 闲置直到下一轮。
- 多用户下,各 session 的"爆发"随机碰撞:continuous batching 虽然能合批,但**轮与轮在时间上未必重叠**——平均 batch size 低,而碰撞发生时后到的轮排队。
- 播放端实际只消费 12.5 帧/秒。超前生成的帧只是坐在缓冲区里;为它们付出的"抢跑"代价由同时在跑的其他 session 承担。

**Temporal batching 假设**:让所有 session 在全局 tick 格(80ms 或 160ms)上齐步走,每 tick 每 session 只生成播放进度所需的帧。批的组成变成周期性、可预测的;decode 是访存受限的,batch N 的 step 时间接近 batch 1,每帧 GPU 成本被摊薄。

**Pacing 覆盖整条流水线,不只 talker。** thinker 和 talker 遵守同一个全局
tick 时钟:thinker 每 tick 生成固定 token 配额,talker 每 tick 生成播放所
需帧数。理由:只 pace talker 的话,thinker 的解码依旧是爆发式的——它给
talker 的 hidden state 到达是突发的(talker 输入侧不可预测),thinker 卡上
多 session 的批组成也依旧随机。全流水线同钟后,两张卡的每 tick 工作量都有
界、批组成都周期化,整个系统变成一个按 12.5Hz 时钟运转的同步数据流。

**物理约束**:thinker 的 pacing 速率必须高于 talker 消费文本的速度。语音
约 4–6 token/s(中文 ≈ 字/秒),即 talker 每帧消费 ≈0.4 token;thinker
速率低于这个值会让 talker "没词可念",音频欠载——这是 pacing 引入的新失败
模式,必须单独测量(talker 中途因等 chunk 停车的次数)。默认给足余量
(25 tok/s = 2 token/80ms tick ≈ 5 倍语音速率),并且回复文本远短于其语音
时长,thinker 总会提前收尾,pacing 的作用是把它的每 tick 工作量削平,而
不是拖到和语音一样慢。

## 二、假设(可证伪)

- **H1(延迟可预测性)**: temporal pacing 显著降低 chunk 到达间隔的抖动(p99/p50 比值)与 deadline miss rate,尤其在 N≥8 时。
- **H2(吞吐/容量)**: 齐步进批使稳态 batch size ≈ 活跃 session 数,每帧摊薄的 GPU 时间下降;满足 "RTF<1 且 miss<1%" 的最大并发 N 不低于(预期高于)基线。
- **H3(公平性)**: 基线下先到的轮垄断 GPU、后到轮 TTFA 随负载排队恶化;temporal 模式下新轮首帧延迟有界(≤ 1 tick + 首帧计算时间),TTFA 的 p99 随 N 增长更平缓。

**预期代价(必须诚实测量,不许回避)**:

1. **TTFA 可能变差**:贪心冲刺能更早攒齐首个音频 chunk;pacing 后首 chunk(4 帧)最快也要 ~4 个 tick。缓解:**initial burst 豁免**——每轮前 K 帧不受 pace 限制。
2. **失去 buffer lead**:贪心模式很快积累数秒客户端缓冲,后续任何抖动都被缓冲吸收;paced 模式每个 tick 都是软 deadline。缓解:**lead 参数**——允许超前播放进度 L ms。
3. tick 量化本身平均引入 ~tick/2 的调度延迟。
4. 停车/放行的调度器开销(每步 O(running) 检查)。

## 三、对照条件

| 条件 | 说明 |
|---|---|
| **A 基线** | 现状贪心 continuous batching(代码路径完全不变,开关关闭) |
| **B** | temporal pacing(thinker+talker 同钟),tick=80ms,lead=240ms(3 帧),initial burst 豁免 |
| **C** | temporal pacing(同上),tick=160ms,同 lead / burst |
| **D(消融)** | 只限速不量化(pacing 但释放时刻不对齐全局格)——分离"限速"与"齐步合批"两个效应 |

B/C/D 中 thinker 与 talker 都受 pace:talker 速率 = 播放速率(1 帧/80ms),
thinker 速率 = 25 tok/s(按 tick 折算配额),两者共用同一全局 tick 格。

## 四、负载模型

- **N ∈ {1, 2, 4, 8, 16}** 并发 session,engine 直连 websocket(`/v1/video/chat/stream`,`session_scoped_request=true`,不经 web proxy、无浏览器 VAD 噪声)。
- **输入侧节奏(temporal grid 的输入半边)**:音频以 **80ms/块(12.5Hz)** 流式发送,不是一次性发大块;**视频不设自己的时钟**——帧间隔必须是音频颗粒度的整数倍,实验用 **480ms(= 6×80ms)**,帧落在音频格点上。摄像头帧在整个 session 期间持续发送(包括收听回复时),模拟真实通话摄像头常开。
- 每 session 循环:按上述节奏流 ~3s 合成语音 → `video.query` → 收完整轮音频 → think time ~ U(1s, 4s)。session 启动时间随机错开,避免人为同步偏向 temporal 条件。
- **范围界定(诚实说明)**:输入的周期性由负载生成器保证;服务端仍按现状缓冲、turn 触发时一次性 prefill。"每 tick 增量 prefill 输入音频"需要 entrypoint 与音频编码器增量化改造,留作后续扩展——本实验验证的是**生成侧**的 temporal pacing,输入侧节奏保证各条件看到相同的到达过程。
- 每 session ≥ 10 轮,**丢弃前 2 轮**(JIT/缓存暖机)。
- thinker `temperature=0.0`:同一输入音频 → 确定性回复文本 → A/B 间回复长度分布一致,差异只来自调度。固定 prompt 集合,各条件复用。
- 每个 (条件, N) 格子至少 2 次重复。

## 五、指标

客户端记录每个 audio delta 的到达时刻 t_i 与音频时长 d_i(由 WAV 字节解析)。

**延迟**
- TTFA:`video.query` 发出 → 首个 `response.audio.delta` 到达(p50/p95/p99)。

**可预测性(核心)**
- 播放模型:anchor = 首 chunk 到达即开播;chunk i 的 deadline = anchor + Σ_{j<i} d_j。
- **deadline miss rate**:P(t_i > deadline_i);**stall 总时长**(miss 后重锚定累计)。
- **inter-chunk jitter**:gap_i = t_i − t_{i−1};报告 (gap_i − d_{i−1}) 分布与 p99/p50。

**吞吐**
- 聚合生成音频秒数 / 墙钟秒(全部 session 合计);单 session RTF。
- **容量**:满足 RTF<1 且 miss<1% 的最大 N。

**引擎侧**(env-gated 日志,离线解析)
- thinker 和 talker 每 scheduler step:墙钟时刻、batch size、调度 token 数 → 两个 stage 的 batch size 时间线直接验证"齐步合批"是否发生。
- 每轮 [TIMING]:thinker 完成 → talker 首帧 → 首 chunk 出声的分解。
- **thinker 引发的 talker 饥饿**:talker 请求轮中因等 chunk 停车(WAITING_FOR_CHUNK)的次数/时长——pacing thinker 引入的新失败模式,必须为零或可忽略。

## 六、部署(两卡隔离)

为了把 talker 的批行为干净归因,消除跨 stage 的 GPU 争抢:

- **GPU0**:thinker 独占(bf16 权重 59.4 GiB)。
- **GPU1**:talker + code2wav(colocate)。
- 实验期间 SoulX-LiveAct avatar server 停用(它占着 GPU0)。
- `max_num_seqs ≥ 20`(N=16 + shadow/roll headroom);具体 yaml 见 `deploy_temporal_2gpu.yaml`。

## 七、实现草图(经代码侦察确认)

**thinker 和 talker 遵守同一个全局 tick 时钟**。两个 stage 都是 `OmniARScheduler`,
一份 pacing gate 实现,按 stage 配速率,**默认关闭**,env 开关:

```
VLLM_OMNI_TEMPORAL_TICK_MS          # 0=关(基线);80 / 160,全局时钟,两 stage 共用
VLLM_OMNI_TEMPORAL_LEAD_MS          # talker 允许超前播放进度的余量,默认 240
VLLM_OMNI_TEMPORAL_INITIAL_FRAMES   # talker 每轮前 K 帧豁免,默认=首 chunk 帧数
VLLM_OMNI_TEMPORAL_THINKER_TPS      # thinker 速率上限 token/s,默认 25(≈5 倍语音消费速率)
VLLM_OMNI_TEMPORAL_THINKER_BURST    # thinker 每轮前 B token 豁免,默认 16
VLLM_OMNI_TEMPORAL_NO_QUANT         # 1=只限速不对齐 tick(条件 D)
```

判定(每个 scheduler pass,对 RUNNING 的请求;talker 速率 = 1 帧/80ms,
thinker 速率 = THINKER_TPS):

```
produced = 本轮已生成单位数(talker:帧;thinker:token)
ahead    = produced / rate − (now − turn_start)
if produced > burst and ahead > lead:
    release_at = quantize_to_next_tick(now + ahead − lead)   # 全局格
    本步停车(挪出可调度集合),到 release_at 放回
```

侦察确认的关键事实(代码位置见 RESULTS 引用):

- **talker 1 个采样 token = 1 个 codec 帧 = 80ms**(codebook-0 由 LM 采样,
  其余 codebook 由 code predictor MTP 同步补齐);帧计数 = `len(output_token_ids)`,
  需过滤 codec 特殊 token(pad/bos/eos ≥ 4196,有效 id < 2048)。
- **每轮重置已存在**:非最终 segment stop 时 `_output_token_ids.clear()`
  (omni_ar_scheduler.py:908),所以 `len(output_token_ids)` 天然是"本轮产出";
  新轮开始 = status 从 WAITING_FOR_STREAMING_REQ 翻回 WAITING 且输出为 0,
  `arrival_time` 被该轮 StreamingUpdate 刷新。
- **停车机制照抄 `_held_non_active` 先例**(chunk_transfer_adapter.py:110):
  在 super().schedule() 前把 paced 请求 pop 出 running,finally 里放回——
  同一次 schedule() 调用内完成,**不改 status、不动计数器**,对
  has_requests()/孤儿恢复/parked 计数完全不可见。这是最小侵入方案。
- **prefill 天然豁免**:gate 只看输出 token 计数,prefill 期间为 0,不会停车
  ——新轮的输入处理(prefill)不受 pace,TTFA 不背 pacing 的锅。

关键点:放行时刻量化到**全局** tick 格(所有 session、两个 stage 共用同一格),
这样各 session 的下一单位在同一时刻变为可调度 → 调度器自然合成大 batch。

## 八、有效性威胁

- 合成语音 vs 真实语音:影响 thinker 输出长度分布 → 各条件用同一 prompt 集合抵消。
- code2wav chunk 粒度(web demo 4 帧)与 tick 的相位交互 → 记录 chunk 边界,分析时对齐讨论;必要时补 25 帧(官方默认)对照。
- 单次运行方差 → 每格 ≥2 次重复,报告区间。
- 观察者效应:step 日志频率过高会自伤延迟 → 日志走内存缓冲、退出时落盘,或采样。

## 九、产出

- 负载生成器 + 分析脚本:`benchmarks/temporal_batching/`
- pacing 原型:talker 调度器,env-gated
- 结果与客观结论(赢在哪、输在哪、什么条件下值得):写回本目录 `RESULTS.zh.md` + workflow.zh.md 摘要
