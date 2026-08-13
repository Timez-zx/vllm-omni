# mass-teardown 挂起/引擎死亡:尸检与根治

**日期**: 2026-08-12 · 证据:`/home/ubuntu/data/results/camp_*/engine_slice.log`(21 cell 逐 cell 切片)

## 一、症状盘点(21 cell 全查)

| 现象 | cell | 签名 |
|---|---|---|
| 引擎死亡(IndexError) | T_short_u24, A_long_u24 | `gpu_ar_model_runner.py:1429` hidden_states 维度 0 |
| 引擎死亡(CUDA assert) | M_long_u24, T_long_u24, A_mixed_u24 | `vectorized_gather_kernel: ind < ind_dim_size` |
| 集体停摆 ~170-180s(不死) | M/T_mixed_u24 | 全 24 用户 turn 6-8 同窗冻结,超时后自愈 |
| u32 墙 | A/M/T_short_u32 | 三臂齐刷刷 ~95/320 超时 |

**关键相关性:凡引擎死亡的 cell 必有 `SHORT: placeholder` + `mrope_positions covered` 告警;无死亡的 cell 两者皆无。** 死亡与停摆是两个独立疾病。

## 二、疾病 1:段边界状态刷新缺失(死亡 + 无声音质污染)

**机制链(实测,M_long 13:19:34 同秒同请求)**:

1. talker 请求在 RUNNING 队列中 park 等 chunk(`WAITING_FOR_CHUNK`,origin=RUNNING);
2. 新 turn 的开段载荷(带 `embed.prefill`,每段只随首 chunk 出现一次)在此刻加载完成;
3. 按 origin=RUNNING 恢复 → 调度器以 **CachedRequestData** 发给 runner;
4. cached 路径**不执行任何段级刷新**(NewRequestData 路径会做:`_update_streaming_request` 刷 prompt/mrope/output 吸收 + `_update_streaming_input_additional_info` 把 `num_processed_tokens` 清零);
5. 陈旧 `num_processed_tokens`(上一 turn 累计,如 30;think 窗内每个 boundary-only 视频 append 再 +1)对**新段的新鲜行**切片 `[30,58)` → 28 行张量切出 **0 行**;
6. worker 钳位 `seg_len=min(span,rows)` → 该请求的 28 个槽位留着 `torch.empty` 未初始化垃圾:
   - 独批 → 前向 0 行 → **IndexError**(引擎死);
   - 混批 → 垃圾 embeds → 采样垃圾 id → **CUDA gather assert**(引擎死,全 batch 陪葬);
   - 钳位恰好未越界 → **无声降级**(该 turn 音频条件在错误前缀上,无异常)。

Qwen3-Omni 走**追加式** prompt 扩展(从不携带 `replace_streaming_prompt` 标记,只有 MiniCPM-o 用),所以"新段"只能靠接收侧推断:segment_finished 之后的第一个带数据 chunk。

**修复(三层,任何交错顺序都成立)**:

1. **adapter 改道**(`chunk_transfer_adapter.py`):接收侧记录段边界(`_expect_segment_opener`),开段 chunk 加载完成的请求禁止按 RUNNING 恢复——摘出 running 队列、置 WAITING、经 `waiting_for_chunk_waiting_requests` → `restore_queues` 回 waiting 队列 → 上游必然以 NewRequestData 发出(上游 `status==WAITING → scheduled_new_reqs` 已核实)→ runner 全量刷新。触发时打 WARNING 计数。
2. **runner 兜底**(`gpu_model_runner.py` `_update_additional_information`):cached 路径若仍见到开段载荷(replace 标记或 `embed.prefill` 在场),merge 后强制 `num_processed_tokens=0`(与健康路径完全一致的重置),打 ERROR——此路应不可达,响一次就说明改道有洞。
3. **模型钳位保命**(`qwen3_omni.py` `talker_preprocess_prefill`):prefill 行不足跨度时补零行到整跨度 + ERROR——一个请求音频降级,batch 对齐保住,引擎和其他 23 个用户活着。

(mrope 的既有自愈 override 是第 0 层,继续保留。)

## 三、疾病 2:stage0 KV 池耗尽 → 全体冻结(集体停摆/u32 墙)

**首个假设(encoder cache 死锁)被自己的复现实验证伪**:预防性的 schedule() 时
encoder 释放扫描上线后,val_T_mixed_u24 停摆原样复现(22 超时,p99 173s)。扫描保留
(无害且让 encoder 引用更短命),但**它不是解药**——如实记录。

**真因(复现切片上闭合式验证)**:

- 停摆瞬间 stage0 的 24 个请求 `sum(computed) = 275,084 / 275,008 = 100.0% KV 池`
  (gross 304k 含 24 份共享 system prompt 前缀的重复计数 = 物理满池),需求 102.9%;
- 多轮 session 上下文**无界线性增长**(视频帧主导,~1.5k token/turn,与臂无关):
  mixed/long 到 turn 6-8、short 到 turn 8-10 时 24 × ~11.5k ≈ 275k 撞墙;
- 撞墙后的病态:全员 parked/waiting、无 running 可抢占(`preemptions=0`),waiting
  队头 `allocate_slots` 失败即 break → **调度器空转零推进**;每个用户的最后一个
  已录入 chunk = 1 帧视频 append(222 token),故全员整齐卡在 `computed = prompt - 222`;
- 出口只有 180s 客户端 watchdog abort 释放整 session → 实测"~170-180s 自愈";
- 容量核对:u16 = 176k < 275k 全净 ✓;u24 边缘(旧 A_short_u24 擦边幸存)✓;
  u32-short 三臂 ~95/320 超时 ✓;臂无关 ✓。

**定性**:这不是引擎实现 bug,是**容量超售**——无界上下文 × N≥24 > KV 池。服务栈
对这堵墙的既有设计答案就是上下文压缩(前一研究已建好,Gemini Live 同款
trigger/target 语义,影子 roll 免冷启动):`context_compression_trigger_tokens` +
`context_compression_target_tokens`,容量规则 **并发数 ≤ 池 / trigger**。campaign
没开它,等于故意用无界上下文去测有界池。

**处置**:高并发实验全部按 N 配置压缩(两臂同配,`trigger = 0.75 × 275008 / N`,
`target = trigger/2`),写进 run_pressure.sh / run_video_freq.sh。冻结行为本身
(满池时零推进而非优雅降级/准入拒绝)记为引擎侧遗留改进项:空闲 parked session
的 KV 逐出(按占空比付费的 KV 语义)是结构性正解,列入后续。

## 四、验证(修复后,T 臂,一次启动连跑三个最恶劣 cell)

| cell | 旧表现 | 修复后 |
|---|---|---|
| mixed_u24 | 25-28 超时 + 停摆(M/T) | 引擎零死亡零陈旧状态告警;KV 冻结仍在(22 超时)→ 归因疾病 2,见上 |
| short_u24 | T 引擎死亡(turn 9) | **零死亡**;KV 冻结出现(22 超时,10 turns 累积更深) |
| short_u32 | 三臂 ~95 超时 | (KV 冻结,预期内——无压缩配置的对照) |

**疾病 1 判定:根除**——三个 cell 合计零 IndexError/CUDA assert/fatal、零
SHORT/mrope 告警,引擎单次启动扛完全部三 cell(旧 campaign 在此负载下需多次重启);
且这是在 KV 冻结波(最大搅动环境)下取得的。
冒烟(4 用户×3 turn):12/12 ok,TTFA p50 168ms,文本正常。

**疾病 2 解药验证(闭环)**:同引擎同负载,仅加压缩配置(trigger 8594 / target 4297):
**192/192 ok、零超时、零致命**(未压缩:22 超时 + ~170s 全体冻结)。代价如实:
p95 7.5s、>1s 占 16.7%——24 个 session 在相近时刻越线,进程级 2 个 warm-up permit
排不开,部分退化为 blocking roll(冷 turn ~3.7× TTFA)。这是有界上下文服务的
真实锯齿形态;两臂同价,不影响 M/T 对比公平。counter_leak_clamped=5(既有
安全网正常工作)。

## 五、对后续实验的约束

- 一切 N≥24 的实验必须带按 N 缩放的压缩配置(run_pressure.sh / run_video_freq.sh 已内置);
- u48 时 trigger≈4.3k(≈17 帧窗口),压缩 churn 占比上升,解读时把"压缩税"
  与调度机制的贡献分开看(两臂同税);
- 遗留改进(未做):满池时的优雅降级(空闲 parked session 的 KV 逐出 =
  按占空比付费的 KV 语义)、warm-up permit 随池余量自适应。

## 六、压缩 burst 根治(2026-08-13,三迭代)

**症状**:高压下压缩波把 TTFA p99 打到几十秒(long u32 基线 64.5s,turn 4 中位 14.8s)。

**三次迭代,每次失败都有解剖**:

| 版本 | 改法 | 结果 | 解剖 |
|---|---|---|---|
| v1 | 阻塞 roll 挪到 turn 尾(think 空档) | **恶化** p99 99.9s | 波期 roll 本身要几十秒,think 只有 2-6s,下一轮继承剩余等待;且 32 个无准入重建同时开工互相踩踏 |
| v2 | waive(影子没好就不搬)+ 工位 2→6 | **恶化** p99 147s | 第一波影子 1-3s 顺畅出炉;但 0.75×份额的触发线下,波峰(人人骑 1-1.5×份额)顶穿 KV 池,第二批种子 computed=0 卡满 30s 全灭 → 无人换新 → 冻结。**旧阻塞 roll 其实是泄压阀** |
| v3 | v2 + 触发线 = 份额/2 | **根治** p99 3.0s | 波峰每人 = 硬线 1.5t + 种子 0.5t = 2t = 恰好一份额,池永不被顶穿 |

**v3 终局数据**(long u32,同种子):TTFA p99 64.5s→3.0s(21×),max 7.7s;逐轮中位
0.4-0.7s 无波;64 次压缩全部影子无感切换(0 放弃 0 阻塞);卡顿 283→36ms;
吞吐 15.3→18.7;质量正常(字/秒 15.0,0 空 0 超时);miss 0.53→0.85%(<1% 线内)。

**配方**:①`video_stream_base.py` waive 逻辑(~30 行,`_can_defer_roll` 应急天花板
2×trigger 兜底)②`_MAX_CONCURRENT_SHADOW_WARMUPS` 2→6 ③配置规则
**trigger = 0.5 × 池/N**(不是 0.75——波峰预算进池是无波的前提)。
代价:窗口 6.4k→4.3k(每人记忆 -1/3),miss +0.3pp。

**附带修正**:T 臂 wave 高于 M 的旧谜团 = 波峰 KV 顶穿时的分配排队,
与内联发送无关(消融证伪);v3 后该现象整体消失。

**滑窗对照(同日)**:kernel 级滑窗(双级 window=份额)burst 同样归零但
**越窗即哑**(turn 4 起 25-38/40 空回复,attention sink 滑出窗口,模型未经
SWA 训练)——重建式滑动 + v3 配方是当前唯一质量安全且无 burst 的组合。

**v4(尾巴再收敛)**:v3 剩余尾巴解剖 = 8/128 轮 TTFT 1.8-7.7s,全在压缩窗口,
全是 stage0 prefill 排队(swap turn + 种子挤 8192/步预算)。两改:①触发线
每 session ×[0.80,1.00) 向下抖动(集体越线摊开 ~2 轮,uuid 哈希,波峰预算不破)
②stage0 max_num_batched_tokens 8192→16384。结果:p95 1827→1157,尾巴轮
8→4,最差 7.7→5.2s,miss 0.85→0.62%,逐轮 p50 压缩窗隆起抹平(544/491/403/367),
质量无损。p99 指标在 n=128 下已是单样本噪声(3-4s 区间)。
剩余 4 个离群轮的下一层手段(未做,均为代码工程):种子 prefill 降优先级
(只吃 turn prefill 剩余预算)、影子 tailing(swap 零携带)。

**v5(种子不带帧,用户提议的对照)**:carry_frames=false,种子 2.1k→~200 token。
p95 1157→974(首次进 1s),p90 778→741;**但尾巴轮仍是 4/128**——解剖显示
4 个离群全挤在同一个 7s 窗口(t=76-83s),全是 thinker 首字、零放弃零阻塞:
是 32 用户同步开局导致的 turn-3 prefill 对撞(压测人为产物,真实错峰流量
天然不存在),与种子大小无关。**结论:v4(带帧)为默认——牺牲视觉记忆
买不到离群消除;v5 记为"极致延迟优先"可选档。**
梯子终态:基线 64.5s → v3 3.0s → v4 p95 1.16s → v5 p95 0.97s。
