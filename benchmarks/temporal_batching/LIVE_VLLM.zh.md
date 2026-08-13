# live-vllm:周期性负载的原生引擎重设计

分支 `live-vllm`(基于 temporal-batching,2026-08-13 起)。目标:把六条设计
原则从"补丁"变成"架构"——引擎以拍为执行单位,请求驱动退化为事件处理器。

## 0. 方案选型:重设计什么、复用什么

**结论:翻转控制面,复用数据面。** 从零重写 vLLM 是低效路径:模型 runner、
KV 管理、多模态编码、连接器这些数据面组件与"请求驱动还是拍驱动"无关,全部
保留。要翻转的是控制面——谁决定"这一拍做什么":

```
现状(temporal-batching):请求驱动为脊柱,tick 是七个 env 补丁
  (TICK/BARRIER/ENGINE/REPLAY/MAILBOX/INLINE_SEND/FRAME_TICK)
live-vllm:拍执行器为脊柱,补丁转正为器官;决策只在事件发生
```

依据(实验背书,见 PRESSURE/BUGFIX/TICK_ENGINE 各文档):
- talker 内核活跃仅 29%,每步重复决策/发射在周期负载下是纯浪费;
- WP7 证明收益主要来自"非周期关键路径净空",不是合批本身;
- 声码器 25 帧左上下文重算 = 每有用帧 7 倍冗余,GPU1 实测瓶颈;
- 压缩波(p99 64s)与 KV 超卖冻死的机理都是"非周期工作无配额、容量无结构"。

## 1. 架构:六原则 → 六个组件

| 原则 | 组件 | 位置 |
|---|---|---|
| ① 拍是执行单位 | TickExecutive:每拍重放 cohort 计划,决策仅在成员事件 | stage_engine_core_proc + omni_ar_scheduler(REPLAY 转正) |
| ② 周期/非周期分离 | SlackLedger:每拍先扣解码心跳需求,余量为松弛槽,非周期预填按配额切片 | omni_ar_scheduler.schedule |
| ③ 结构化容量 | AdmissionLedger:引擎发布每拍占用,准入按"再加一人还装得下吗"收/拒,不排队 | 引擎心跳 → video_stream_base 准入 |
| ④ 相位是资源 | PhaseAllocator:admission 发相位;声码 1/K 轮值组;触发抖动纳入相位框架 | ChunkTickGate 升级 + 会话准入 |
| ⑤ 算子带状态 | StatefulVocoder:常驻卷积边界状态,增量解码,消灭左上下文重算 | qwen3_omni_code2wav |
| ⑥ 两个 SLO 两套机制 | 松弛槽内 turn-open 专用道(最新开口优先);节拍保播放 | SlackLedger 的排序策略 |

本分支默认全开(env 仍可显式关闭以做对照):周期性不是可选优化,是架构假设。

## 2. 施工阶段与验收

- **P1 拍执行器默认化**:七个开关在本分支默认 on;REPLAY 语义核查(替代路径
  覆盖率日志)。验收:冒烟 + u32 长答与 T 臂持平。
- **P2 松弛槽+专用道**:每拍预填配额 = 拍预算 − 解码需求;跨拍切片;最新
  turn-open 优先。验收:压缩/开口大预填不再挤掉解码拍(心跳 p99 ≤ 基线),
  开口 p50 不升。
- **P3 准入台账**:tick occupancy EWMA + KV 台账 → 服务端收/拒。验收:超载
  压入优雅拒绝,已收 session 指标不塌。
- **P4 相位分配器**:声码 1/K 轮值。验收:GPU1 削峰,播放不劣化。
- **P5 有状态声码器**:增量解码。验收:波形与重算路径一致(容差内),
  GPU1 每拍耗时下降。
- **P6 整体对照**:live-vllm vs temporal-batching T 臂,long u32/u48 + v160,
  同 FP8 同真实到达;容量探顶 u56/u64。

## 3. 测试基线

对照臂 = temporal-batching 的 T(出厂认证配置):FP8 全局、不带图压缩、
触发 0.75×份额、真实到达。所有已建立的证据线沿用:SCHED-STEP、turnprobe、
ChunkTickGate 日志、analyze.py 指标全套。

(以下各阶段的实测结果随施工追加。)
