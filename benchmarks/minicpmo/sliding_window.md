# MiniCPM P/D：滑动 KV 窗口

**2026-09-09 最新状态：18K × 300 s，8/9 用户通过、10 用户不通过。** 内容退化与 serving 错误分开，见 [生成与服务核查](sliding_quality.md)。

新增门槛：滑窗后，每个用户长期 RTF 必须严格 **>1**，且每轮 `max(0, 上一轮 D 完成 − 本轮完整输入到齐)` 必须 **≤500 ms**。首次滑窗输入保留其窗前前驱，不能漏掉进入滑窗时已有的积压。该指标不含本轮处理，不是 TTFA；任何一次超限都失败，追回不能抵消。`--max-backlog-ms 500` 只用于离线判定，不限流、不丢输入。137 项相关回归通过。

10 用户离线重判：**403 轮超限、10/10 用户超限、最大积压 7,098.226 ms**。旧 `analysis.json` 和 RTF-only 汇总不改写；新结论保存在同目录 `analysis-backlog500.json`。下文 RTF 通过不再等于容量通过。

## 最新测试：18K × 8/9 用户 × 300 s

同下方 10 用户的四卡、MPS、pipeline、HD4 AV、FP8/BF16 Q、Triton 自动选路与 pin128 配置。serving 和已安装 vLLM 源码未改；benchmark 仅增加离线门槛及记录字段。各档正式输入均为 300 s；8/9 用户之间源码 manifest 和完整部署配置完全一致。

| 用户数 | 最低逐用户长期 RTF | 最大前轮积压 | 超过 500 ms 的轮数 | 判定 |
|---:|---:|---:|---:|---|
| 8 | 1.001756 | 127.641 ms | 0 | 通过 |
| 9 | 1.001732 | 301.425 ms | 0 | 通过 |
| 10 | 1.001471 | 7,098.226 ms | 403 | 不通过 |

8 用户 **2,400/2,400 输入全部完成**，每人第 84–85 轮进入滑窗、滑窗后 216–217 轮，最终逻辑历史 65.0k–65.9k。满窗后完整 AV 到齐→D 完成 p50/p95/p99 为 **413/792/894 ms**，最大 1,072 ms；这与不含本轮执行的积压指标不同。P/D 驻留最多 1,134 blocks，D 每轮输入 prefill 为 1 token，KV delta 中位数/p99 为 227/245 tokens。零 AV fallback、运行错误或 KV 抢占，接收事件/导出审计 0 FAIL/UNKNOWN；最大发送偏差 6.25 ms，源码运行前后不变。仅确认本次负载和时长下的两条输入容量标准，未认证模型质量或完整音频 SLO。

复现下方命令时使用 `--users 8 --duration-s 300 --max-backlog-ms 500`，其余参数相同。归档：`/home/ubuntu/data/experiments/minicpm-pd-capacity-review-20260909/users-8x300-w18000-backlog500-r1/`，新判据记录在 `analysis.json` 的 `bounded_backlog_audit` 与 `sliding_window_audit.long_horizon_rtf` 中。

9 用户有效补测 **2,700/2,700 完成**：每人第 84–85 轮进入滑窗，滑窗后 216–217 轮，最终逻辑历史 65.3k–66.1k。满窗后完整 AV 到齐→D 完成 p50/p95/p99 为 **488/975/1,065 ms**，最大 1,227 ms。P/D 驻留最多 1,134 blocks，D 每轮输入 prefill 为 1 token，KV delta 中位数/p99 为 228/245 tokens；零 AV fallback、运行错误或 KV 抢占，接收事件/导出审计 0 FAIL/UNKNOWN。最大发送偏差 **5.86 ms**，源码运行前后未变。

9 用户首测出现一次几乎同时影响两条输入的客户端唤醒迟到，最大 **25.37 ms**，违反预设 10 ms 发送质量门槛；原始数据保留但不计有效容量点。没有更改代码或放宽阈值，同配置重跑后时序通过。有效归档为同根目录 `users-9x300-w18000-backlog500-r2/`；`r1/TIMING_INVALID.md` 解释无效原因。复现只将上述 `--users` 改为 9。9 用户是本次 workload/时长下的最高通过档，不是其他输入或无限会话的容量保证。

## 10 用户对照：18K × 300 s

按用户要求将本次窗口改为 **18,000 tokens + pin128**、输入改为 **300 s**，其余四卡布局、MPS、FP8、BF16 Q、Triton 原生自动选路、HD4 AV workload 不变。预热 12 s 使用独立 session；输入结束后观察 30 s。默认部署文件仍为 36K，本次通过命令行覆盖。

- **3,000/3,000 完成，10/10 用户长期 RTF≥1，最低 1.0014706348。** 每人第 85 轮进入滑窗，滑窗后 216 轮，逻辑历史最终 64.9k–66.3k，均覆盖完整窗口替换。
- 滑窗后完整 AV 到齐→D 完成 p50/p95/p99：**585/4,578/6,167 ms**，最大 **8,087 ms**。中途发生积压、最终追回；长期 RTF 通过不代表低尾延迟，也不证明 10 用户是最大容量。
- P/D 驻留最多 **1,134 blocks**；D 每轮输入 prefill 为 1 token，传输 delta 中位数/p99 **229/245 tokens**。零 AV fallback、运行错误或 KV 抢占；接收事件/导出审计 **0 FAIL/UNKNOWN**，最大发送偏差 6.77 ms。
- 源码运行前后未变；这是未提交开发快照的测量，不是 clean-build、模型语义或完整语音播放认证。18K GPU attention/缓存/配置回归 **65 项通过**。与下方 36K 对照同时改变了窗口、用户数和时长，不能将差异全归因于单一因素。

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 10 --duration-s 300 \
  --kv-window-tokens 18000 --pinned-prefix-tokens 128 \
  --kv-cache-dtype fp8 --triton-disable-q-quantization \
  --no-triton-force-2d-attention --quality-capture --out-dir /path/to/new-run
```

归档：`/home/ubuntu/data/experiments/minicpm-pd-capacity-review-20260909/users-10x300-w18000-native-attention-r1/`；同级 `capacity-summary-10-w18000.json` 为汇总。原 420 s 任务在服务启动期间取消，未产生测量结果。

## 36K 对照：原生 attention 路径

沿用下文四卡布局与 workload，显式使用 **36k+pin128、BF16 Q、FP8 KV**；正常编译/Graph、私有 MPS、无同步探针。去掉此前数值诊断的 `triton_force_2d_attention`：它禁用了 decode split-K。单请求、36k 历史的单层 attention 实测 1.258→0.137 ms；这只证明该诊断配置有额外成本，不代表全部端到端积压已解释。

| 用户数 | 完成输入 | 最低跨窗长期 RTF | 满窗后延迟 p50 / p95 / p99 | 判定 |
|---:|---:|---:|---|---|
| 6 | 2,520/2,520 | 0.985506 | 745 / 9,661 / 11,006 ms | 未通过 |
| 7 | 2,940/2,940 | 0.983531 | 1,621 / 20,998 / 24,309 ms | 未通过 |
| 8 | 3,360/3,360 | 0.951623 | 19,735 / 37,194 / 39,317 ms | 未通过 |

延迟为完整 1 秒 AV 到齐→D 完成，**含此前轮次积压，不是本轮纯 GPU 时间，也不是 TTFA**。RTF 是每用户满窗后输入秒数/对应 wall time，按未舍入值≥1判断，不按单轮迟到次数判断。6/7/8 分别有 5/5/6 位用户 RTF<1。

- 每人满窗后 254–257 轮、最终逻辑历史 91.8k–94.0k；P/D 最多 2,259 个有效 block（含 8 个 pinned blocks），没有 prompt 重建。
- 全部 8,820 轮的 D 输入 prefill 均为 1 token；满窗后传输 delta 中位数/p99 为 230/245 tokens。零音视频 fallback、运行错误或 KV 抢占，功能事件/导出审计均为 0 FAIL/UNKNOWN。
- 增量 P 的直接执行证据另来自此前 32 个正式 forward 抽样：其中 18 个跨窗样本只计算 211–230 个连续新位置，最后已到 91k 历史。源码/回归与本轮 D 账目互补，不能把 D 账目冒充本轮所有 P forward 的逐位置探针。
- 6/7/8 用户的 D 完成→客户端接收 p99 范围为 15–19/20–25/29–39 ms；原多秒输出阻塞未复现。最大发送偏差分别为 6.43/6.67/6.95 ms，低于 10 ms。
- 三档 Omni/vLLM 源码 manifest 一致，运行前后 hash 未变。代码未提交，报告的是开发快照结果；不宣称模型语义质量或完整音频 SLO 通过。4 用户档在启动期按用户要求中止，不算失败点。

结果根目录：`/home/ubuntu/data/experiments/minicpm-pd-capacity-review-20260909/`。各 `users-{6,7,8}x420-native-attention-r1/` 保留配置、源码、MPS、发送/完成记录、文本/PCM 与审计；`capacity-summary.json` 为汇总，脚本 `summarize_capacity.py` 可离线复算。

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 6 --duration-s 420 \
  --kv-window-tokens 36000 --pinned-prefix-tokens 128 \
  --kv-cache-dtype fp8 --triton-disable-q-quantization \
  --no-triton-force-2d-attention --quality-capture --out-dir /path/to/new-run
```

仅将 `--users` 改为 7 或 8 即复现其他档；下文旧容量表不用于当前版本结论。

## 当前实现

- P/D 使用原生 token 级 sliding-window attention，窗口为 **36,000 tokens**。到顶后逐块回收窗口外 KV，不重开短 prompt，不重新 prefill 整个窗口。
- 逻辑 token 历史、位置和完整历史 hash 继续增长。D 用 vLLM `SlidingWindowManager` 匹配仍驻留的窗口内块；不是把截断后的 token 当成一个新的、相同 prefix。
- 原生 AV 使用调度占位 token，不能据此跨用户复用 KV。每条新 lineage 生成独立随机 cache namespace，P、提前注册的 D 和正式 D 都保留同一 salt；同 session 后续 unit 不换 salt。
- NIXL registration 携带缺失块的**绝对逻辑位置**，P 只传对应的有效块。已回收位置是 null block，禁止传输；不能将剩余块简单按列表开头对齐。
- P-only/D-only 模式保留，D→下一轮 P 的依赖不变。旧的短上下文 rollover 只在未开启滑窗的部署中保留。
- Talker 的 20 层使用独立 **4,096-token** 滑窗，逻辑位置上限 65,536，防止长时间说话超过原来的 4k 输入数组。其 binary 调度 token 不代表真实 conditioning/code，所以关闭跨请求 prefix 查找；同一活动请求的 KV 保留和滑窗不受影响。

这是 attention 策略变更，不宣称模型效果无损，也不等同于官方的 whole-unit 裁剪和 RoPE 重定位。默认 pin=0 时最早的 system/reference KV 会移出窗口；最新核查显式使用 pin128，保留开头 8 个块。逻辑位置上限为 **262,144**，不是无限会话；36k 是驻留 attention/KV 窗口，不是逻辑 prompt 长度。

## Setup 与判据

GPU0 Thinker-P；GPU1 Thinker-D + Talker；GPU2 Vision + Audio Encoder；GPU3 Code2Wav。私有 MPS 开启，保持 pipeline。P/D/Talker 权重 FP8，P/D KV FP8；P/D 各固定 64 GiB KV，Talker 8 GiB（KV 保持 BF16）。

真实 960×540 AV、`max_slice_nums=4`；200 ms 音频 chunk、1 FPS 视频、1 s model unit。随机 session 相位、±50 ms 到达 jitter、seed `20260908`。预处理在计时前完成，输入不等待回复，非预期发送偏差不得超过 10 ms。

每档独立重启服务，先用独立单用户会话预热 12 s；随后正式用户从空历史输入 **420 s**，再观察输出 30 s。预热不提供正式用户的历史。所有代码及已安装 vLLM 源码在测试前归档，测试后检查未变化。

容量判据（2026-09-09 更新）：逐用户同时满足**滑窗后长期 RTF > 1** 和**最大前轮积压 ≤ 500 ms**，均按未舍入值比较。RTF 为：

```text
RTF_sw = 滑窗后完成的 N 个 1 秒输入 unit /
         (最后一个 D 完成时间 − 首个滑窗 unit 的完整输入到齐时间)
```

分子为 N 秒，包含首个 unit 的 1 秒处理预算；分母是一段连续墙钟时间，包含排队、P/D、handoff，不累加单轮延迟，也不平均单轮 RTF。起点明确为**完整输入到齐**，不是旧 `stream_rtf` 的第一块媒体到达。旧值仍保留作诊断，不能混用。

每个用户填满窗口后至少 120 个 unit，逻辑 prompt 达到至少窗口的两倍，证明窗口完整替换一次。滑窗后的前轮积压超过 500 ms 即失败，不能靠末尾追回通过；本轮执行耗时和 p99 单列。输入按实时发送，因此 RTF 接近 1 不代表 GPU 接近饱和；有限时长通过也不保证无限会话稳定。

同时核验：每个用户的 P/D 实际驻留块数受窗口约束、无 prompt 重置、输入/帧全部完成、无 fallback/抢占/运行错误、发送时序有效。此结论只覆盖 Thinker 输入消费；音频播放连续性另列，不能默认通过。

## 复现

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 6 --duration-s 420 --kv-window-tokens 36000 \
  --out-dir /home/ubuntu/data/experiments/minicpm-pd-sliding-repeat
```

输出目录必须不存在。保留 `deploy.yaml`、`commands.json`、`warmup.json`、`run.json`、`analysis.json`、`server.log`、`mps.json`、源码快照和 `source-stability.json`。

## 归档：旧 RTF-only 结果（2026-09-09）

同一批原始长测按新标准离线重算：**6 用户通过，8 用户失败；7 用户未测**。没有重新运行服务或改写原始记录。仅适用于上述配置、输入和 420 s 测量时长，不是无限会话或所有 workload 的容量保证。

| 用户数 | 最慢用户满窗后输入预算 | 对应墙钟耗时 | 最小长期 RTF | 容量判定 |
|---:|---:|---:|---:|---|
| 6 | 253 s | 252.845578 s | 1.000611 | 通过 |
| 8 | 252 s | 253.211870 s | 0.995214 | 失败 |

判定使用未舍入值。8 用户失败是最慢用户在约 253 秒窗口中未维持输入速度，不是下面的 123 次单轮超时触发了容量闸门。

下表延迟为**完整 1 秒 AV 到齐 → physical D 完成**，只统计各用户进入满窗口后的 unit；不是 TTFA。

| 用户数 | 全程完成 unit | 每人满窗后 unit | 满窗后延迟 p50 / p95 / p99 | 积压次数（诊断） | 最大单轮超期 |
|---:|---:|---:|---|---:|---:|
| 6 | 2,520 / 2,520 | 252–254 | 284 / 706 / 814 ms | 0 | 0 ms |
| 8 | 3,360 / 3,360 | 252–254 | 322 / 1,064 / 2,002 ms | 123 | 1,400 ms |

- 全程延迟 p99：6 用户 **785 ms**，8 用户 **1,780 ms**。两档首轮均无积压；8 用户首次超期在第 394 个输入，123 次超期全部发生在满窗之后。
- 每人逻辑历史最终约 **90k–92k**，没有短 prompt 重建；P/D 各最多 **2,251 个有效 16-token block**。两档滑窗后每轮传输 KV 中位数均为 **225 tokens**，p99 为 244/245 tokens，不是 36k 整窗重传。
- 视频与音频 unit 全部由 arrival sidecar 处理，零 fallback；零运行错误、零观测到的 KV 抢占。每个用户首次 D 的本地 prefix 命中均为 0。最大发送偏差为 **6.38 / 6.94 ms**，均低于 10 ms。
- 8 用户最慢 1% 回合的均值分解：上一轮 D 尚未完成而等待 **1,029 ms**，本轮 D 提交前 **138 ms**，D 提交至完成 **908 ms**，合计 **2,074 ms**。可以定位到 D 路径与前轮积压；D 段仍包含 ingress、KV 就绪、调度和 decode，**不能直接归因为纯 GPU decode**。

语音输出持续运行，但播放观察仍记录到缓冲耗尽（6/8 用户合计 193/315 次），不据此宣布完整语音播放无卡顿。这里只给出输入消费容量。

结果目录：`/home/ubuntu/data/experiments/minicpm-pd-sliding-20260908/`：

- 通过档：`users-6x420-w36000-v7b`；失败档：`users-8x420-w36000-v7`。
- 两档代码/config manifest 一致，运行前后源码 hash 未变，MPS 四阶段接入通过。283 项相关回归测试通过。
- 新判据结果保存为各目录的 `analysis-long-window-rtf.json`；原 `analysis.json` 保留旧零积压判据。`input_capacity_pass` 是长期 RTF 结果，`input_deadline_pass` 只报告旧单轮判据。此次只修改分析逻辑，serving 代码和测量配置未变。
- 当前工作树未提交，分析器的 `capacity_pass` 仍受 clean-tree 检查限制；上表是已归档源码的**开发快照长期输入 RTF 结果**。不能将其写成已提交版本的正式认证。

早期 v1–v5 因配置、缓存隔离或 Talker 边界问题无效；v6 的 4/8 用户只用于搜索，早于 Talker 跨请求缓存防护。原始文件及原因保留在结果根目录，不混入最终表。额外修正了 D 以非原生 terminator 结束时丢失最后一个采样 token 的问题，保留真实 token，不 padding。

旧版“一轮一个 KV block”和旧容量不可直接沿用：旧测试既会重建短 context，也缺少 AV 占位 token 的跨用户缓存隔离。
