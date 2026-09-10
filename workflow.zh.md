# MiniCPM-o 原生全双工 P/D 工作流

英文版本：[workflow.md](workflow.md)。

**当前容量（2026-09-10）：21 用户，当前配置下的最高通过档；22 用户轻微超出积压门槛。** 判据与300 s跨滑窗测试范围见下文，不代表无限时长稳定保证或硬件理论上限。此前12用户失败已被工程修复结果更新；P的GPU排队/执行和D多步执行是当前主要延迟项，仍有CPU准备与控制开销。

## 当前设置与判据

GPU0=Thinker-P，GPU1=Thinker-D+Talker，GPU2=Vision/Audio Encoder，GPU3=Code2Wav；私有 MPS、跨阶段 pipeline。P/D/Talker FP8 权重、P/D FP8 KV，Encoder/Code2Wav 原精度，Talker BF16 KV。BF16 Q、原生 Triton attention；最新实验显式设置 D split-K 阈值为 32。

真实 HD4/1 FPS 视频、200 ms 音频分块组成原生 1 s 单元；随机相位、±50 ms jitter、seed=20260908。独立预热后从零历史开始，18K+pin128 滑窗，300 s 输入+30 s 输出观察。每用户满窗后 **RTF>1 且继承积压≤500 ms**；发送误差>10 ms 则容量点无效。滑窗后仍增量 prefill、传 KV delta。

## 无诊断长测

| 当前版本 / 用户数 | 完成输入 | 最低满窗后 RTF | 最大继承积压 | >500 ms 轮数 | 判定 |
|---|---:|---:|---:|---:|---|
| O，20 用户 | 6000/6000 | 1.000659 | 217.8 ms | 0 | 本次通过 |
| 38157bae，21 用户 | 6300/6300 | 1.000787 | 339.3 ms | 0 | 本次通过，当前容量 |
| 38157bae，22 用户 | 6600/6600 | 1.000383 | 523.0 ms | 5 | 失败 |
| O，24 用户 | 7200/7200 | 0.991656 | 7645.8 ms | 1889 | 失败 |

以上四档服务源码和完整部署配置一致；测量内源码稳定、发送有效、AV完整消费、零fallback/preemption，协议审计0 FAIL/UNKNOWN。21用户每人至少216轮在满窗后，最大积压339.3 ms，全部通过。22用户长期RTF全部通过，但2人共5轮积压超过500 ms；24用户中22人最终RTF>1，但所有人都曾超过积压门槛。最终追回不算通过。

21/22用户使用已提交的`38157bae`、干净工作树；21用户最大发送误差6.628 ms，P/D最多驻留1134块。原始记录与命令：[21用户结果](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/batch-policy-clean-21x300-r1/RESULTS.md)、[22用户结果](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/batch-policy-clean-22x300-r1/RESULTS.md)。每档一次300 s，仅确认本配置的最高通过档为21，不证明无限时长稳定。输入周期固定，自主生成的decode数量并不固定。

## 已修复与剩余问题

保留此前所有修复：独立推进 KV 完成通知；去掉完整历史重复解码/hash/转换和大 tensor 的 Python 列表运输；减少输出复制与无效 GPU 控制元数据；补齐相同采样规则的多用户 batching。

新确认的问题：P 每轮回传完整历史，现改为 suffix+offset 并严格核验已有前缀；D 在有效 batch 16→17 时切到较慢 attention 路径。相同 Q/K/V 的 18K 单层对照为 0.894→0.457 ms（17 请求），复用原生 split-K、阈值设为32；48/64 请求不再获益。没有改变窗口或输入，attention 不同归约顺序不保证逐位相同；42项滑窗 GPU 数值检查通过。

**最新 O 修复已长测：**将禁用 token 掩码、逐用户重复惩罚索引和温度处理也汇入既有采样 batch；共享/未指定 RNG 仍保留原行序。完整策略24行微测 4.53→2.03 ms/步，4400 token及RNG状态匹配；169项相关回归通过、1项不适用跳过。没有新 kernel 或减少生成步骤。实跑中 D 的16/17请求 forward均值为22.7/23.5ms，旧的约39ms断崖不再出现；不是相同输入的严格算子对照。

同版24用户的独立诊断，最慢72个当轮路径（不含继承积压）平均 **1640 ms = D提交前833 + 提交后807**。P接纳等待309、P执行至Core446；D接纳/调度41、执行及迭代控制749，其余约95ms。P自身forward event377ms，接纳等待中279ms与前批GPU完成等待阶段重叠；D约19.8步，forward event累计421ms。D另有准备124、采样/状态76、迭代间控制100ms，不能全部算成GPU计算或重复加GPU等待。

音频在该慢组关键路径仅3.6ms，KV post→通知8.6ms，P runner完成→Core仅2.0ms；P输出平均约123KiB、编码线程CPU2.3ms。此前编码未准备好、KV通知等下一批、整历史大消息等问题不再是大头。滑窗后P新增token中位数220、p99 230；D本地prefill始终1token，KV delta中位数229、p99 245tokens，P/D驻留最多1134块，没有整窗重算。

**诊断与容量分开：**无诊断24用户的最慢72轮当轮路径为1449ms（D前788、D后661）；诊断为1640ms，不能将诊断额外耗时归给生产路径，也不能拿不同生成决策的两轮差值直接校正分项。结论是GPU执行/排队成为主要项，不是“全部非GPU开销已消失”。

同轮60s硬件采样：P / D+Talker平均GPU busy为81.5% / 74.1%，SM active为76.9% / 50.9%，DRAM read active为4.6% / 49.0%；不是持续满载或理论峰值证明。P attention每块128线程、28,928B共享内存，每SM102,400B最多容纳3块，对应25%warp容量；低warp比例不等于75%资源可自由使用，也不证明kernel最优。旧版10s CUDA trace确认forward主要在执行kernel；Nsight stop/flush和native栈采样的扰动及后续积压均排除容量结论。

原始数据、每项修复、失败尝试、测试和命令见[本轮诊断](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/WRITE_PROGRESS.md)，当前复现命令见[benchmark README](benchmarks/minicpmo/README.md#reproduce-the-current-result)。这是输入消费/协议证据，不认证正常对话语义、完整音频播放SLO或无限时长。20/24用户测量时源码未提交，归档中的clean-build认证仍为false；后续提交不改写旧记录。21/22用户补测为干净源码、无诊断的有效容量测量。

## 提交前功能复核（2026-09-10）

- 当前服务源码与上述20/24用户长测快照逐文件一致；已安装vLLM 0.26.0的Python源码与发行包RECORD一致，没有遗漏的本地源码补丁。
- 重跑全部53个新增/修改测试文件：**1531项通过、1项不适用跳过**，包含显式GPU滑窗数值、跨用户cache/RNG隔离、P/D交接、采样、输出协议和取消处理。仅修正CPU测试对“前序测试不能初始化CUDA”的错误假定，未改服务逻辑。
- 重新审计两组长测：输入、session/response归属、文字/PCM导出和KV账目均为**0 FAIL、0 UNKNOWN**；未发现新的服务功能错误。滑窗后仍增量prefill/传KV delta，不是整窗重算。
- 边界：不保证所有输入的对话语义或连续播放质量；已有官方独立长历史对照也出现内容退化，见[生成核查](benchmarks/minicpmo/sliding_quality.md)。当前版本可作为已验证服务契约的研究基线，不等于模型质量全面认证。

本次回归日志和重审计结果保存在本轮目录的`precommit-functional-regression-r2.log`、`precommit-functional-audit-{20,24}.json`。

## 本轮早期修复记录（以下版本已被当前结果更新）

两项修复：NIXL 原有 writer 独立推进发送状态，不再等下一批 P 才发完成通知；duplex 文本输出复用 vLLM 自带的增量 detokenizer，只解码 prompt 尾部边界。保留既有优化、完整逻辑历史、18K 滑窗增量 prefill/KV delta、输入与采样策略。具体请求中，约 0.47 ms 的 KV copy 曾被延迟到 338 ms 才完成通知；修复后 16 用户慢组通知约 4.1 ms。API 原来 62% 的持锁栈样本落在完整历史解码，修复后该热点消失。

| 本轮版本/负载 | 完成输入 | 最低满窗后 RTF | 最大继承积压 | >500 ms 轮数 | 满窗后 Ready→D p50/p95/p99 |
|---|---:|---:|---:|---:|---|
| 两项修复，12 用户，无诊断 | 3600/3600 | 1.001887 | 4.6 ms | 0 | 435/759/848 ms |
| 同版，16 用户，带诊断 | 4800/4800 | 1.000238 | 392.7 ms | 0 | 546/1116/1231 ms |
| 同版，20 用户，无诊断 | 6000/6000 | 0.945461 | 26,143 ms | 2538 | 2720/17384/21409 ms |

均为下述相同四卡/FP8/MPS/pipeline、HD4、18K+pin128、300 s 设置；发送时序有效，视频/音频完整消费且零 fallback，协议审计 0 FAIL/UNKNOWN，测量内源码不变。16 用户带采样，不替代无诊断容量点。20 用户最慢 60 个满窗后当轮处理平均 1759 ms，其中 D 提交前 585、提交后 1174 ms；不含继承积压，D 段也不等于纯 forward。

继续验证：GPU 采样复用 vLLM 的 random_sample；固定输入的 540 次 token/RNG 对照一致，12 行完整采样由 7.81 降到 6.54 ms。该改动的端到端验证尚未完成，不混入上表。PyTorch 2.11 的 multinomial 校验已是异步，收益是减少重复校验/小 kernel，不能称为消除了逐用户强制同步。原始证据与复现见 [本轮诊断](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/WRITE_PROGRESS.md)。以上是归档开发快照的输入容量，不认证对话语义或完整音频播放 SLO。

## 上一轮修复记录（已由上表更新）

此前的 KV delta、完成块立即发布、Encoder pipeline、输出后台消费、禁用 token 索引缓存和 repetition penalty 向量化全部保留。本轮只补三处：P 已完成结果先交付再执行下一批；独立用户的采样反馈按阶段一次取回，减少逐用户 GPU/CPU 同步；D 等待远端 KV 时原有 NIXL 线程每 1 ms 检查，空闲仍休眠。不改模型、采样规则/RNG、输入和生成预算，不新增线程或传输协议。209 项回归通过；固定 logits 的 12 行采样由 8.975 降到 8.093 ms，token/RNG 一致，不代表采样开销清零。

配置不变：GPU0=P、GPU1=D+Talker、GPU2=AV Encoder、GPU3=Code2Wav，私有 MPS、pipeline、FP8 权重/KV、P/D 各 64 GiB KV、BF16 Q、原生 Triton 自动选路。真实 HD4/1 FPS 视频和 200 ms 音频输入，随机相位与 ±50 ms jitter，seed=20260908；从零历史开始，显式 18K+pin128 滑窗，300 s 输入+30 s 观察。每用户满窗后 RTF >1 且最大继承积压 ≤500 ms。

| 12 用户版本 | 完成输入 | 最低满窗后 RTF | 最大积压 | >500 ms 轮数 | 满窗后 Ready→D p50/p95/p99 | 结果 |
|---|---:|---:|---:|---:|---|---|
| 修复前：索引缓存+repetition 向量化 | 3600/3600 | 1.001755 | 1322 ms | 82 | 696/1368/1751 ms | 失败 |
| 再加 P 及时交付+采样反馈合并 | 3600/3600 | 1.001828 | 1108 ms | 25 | 516/1077/1489 ms | 失败 |
| 当前：再加 D 通知自行推进 | 3600/3600 | 1.001286 | 503.6 ms | 1 | 489/1002/1251 ms | 失败 |

三组无诊断测量的部署配置哈希相同，发送有效，视频/音频均完整消费，零 AV fallback、运行错误或 KV 抢占，可观察功能审计 0 FAIL/UNKNOWN；源码与配置在各次测量内不变。滑窗后仍增量 prefill/传 KV，P/D 驻留最多 1134 块，没有整窗重算。当前唯一超限：用户 1 第 186 轮处理 1344 ms（D 提交前 602、提交后 742），叠加已有等待，使第 187 轮继承 503.6 ms 积压；第 187 轮本身已在追回。

单独 12×300 非阻塞诊断确认：**P runner 完成→Core 交付 p99 从 190 降到 2.24 ms**，旧的“已算完却先执行下一批”延迟已消除。新版最慢 36 个当轮路径（均满窗后）平均 1222 ms：P 接纳前 289、P 调度/runner/交付 175、到 D 提交 27、D 提交到接纳 141、D 调度及多步执行 570、输出到 API 19 ms。这是同一组请求的可加分解，不含继承积压；D 执行内 GPU-event forward 约 298 ms、采样 113 ms、准备 81 ms，CPU/GPU 细分可能重叠，不能再机械相加。

**仍未排除的开销：**上述慢请求的 KV write→D ready 约 151 ms，其中通知入队前 142、入队后 9 ms；新轮询保证不依赖下一次 engine 唤醒，但没有证明这段 tail 显著下降，也不能称为纯 PCIe 拷贝。P 输入解析前另有 238 ms，包含 AV cache-ready、应用/control-plane 和 IPC，尚无可靠逐请求细分，未猜测式修改。不能把剩余全部归给 prefill 或宣称工程优化完成。诊断仍有 1 轮积压超限（540 ms），不替代无诊断结果；一次误开强制 GPU 同步的诊断已中止并排除。完整证据与复现：[本次结果](/home/ubuntu/data/experiments/minicpm-pd-delivery-sampler-20260910/RESULTS.md)。这是归档开发快照的输入容量，不认证正常对话质量或完整音频播放 SLO。

**以下为修复前记录（2026-09-09）：18K、300 s，8/9 用户通过，10 用户不通过。** 积压定义为 `max(0, 上一轮 D 完成时间 − 本轮完整 AV 到齐时间)`，不含本轮执行。

9 用户有效补测：**2,700/2,700 完成，最低 RTF 1.001732，最大积压 301.425 ms，超限 0 轮**；满窗后延迟 p50/p95/p99 **488/975/1,065 ms**，每人满窗后 216–217 轮。最大发送偏差 5.86 ms，低于 10 ms；零 AV fallback、运行错误或 KV 抢占，接收审计 0 FAIL/UNKNOWN。8/9 用户的源码与部署配置完全相同。9 用户首测出现 25.37 ms 发送迟到，原始数据保留但不作为有效容量点；没有放宽发送阈值。当前只确认该 workload、300 s 下的 9 用户通过点，不保证其他输入或无限时长。

8 用户：**2,400/2,400 完成，最低 RTF 1.001756，最大积压 127.641 ms，超限 0 轮**。满窗后完整 AV 到齐→D 完成 p50/p95/p99 **413/792/894 ms**；每人满窗后 216–217 轮。配置与 10 用户相同，serving/已安装 vLLM 源码未改；只更新分析判据和记录字段。零 AV fallback、运行错误或 KV 抢占，接收审计 0 FAIL/UNKNOWN。10 用户虽全部 RTF > 1，但 10/10 用户超出积压门槛，合计 403 轮，最大 7,098 ms；10 用户为原始记录离线重判。

18,000+pin128 测试沿用相同四卡、MPS、FP8 和原生 Triton 自动选路，未改 serving 算法。3,000/3,000 输入完成，每人第 85 轮进入滑窗、覆盖 216 个满窗后输入，最终逻辑历史 64.9k–66.3k。最低长期 RTF **1.001471**，满窗后完整 AV 到齐→D 完成 p50/p95/p99 为 **585/4,578/6,167 ms**：中途积压后追回，不是全程低延迟。零 AV fallback、运行错误或 KV 抢占，接收事件/导出审计 0 FAIL/UNKNOWN；P/D 驻留最多 1,134 块，KV delta 中位数/p99 为 229/245 tokens。详细复现见下方滑窗文档。

此前 36K 对照：

- 四卡布局：GPU0=P，GPU1=D+Talker，GPU2=AV Encoder，GPU3=Code2Wav；MPS、pipeline、FP8 权重/KV，显式使用 36k+pin128、BF16 Q。
- 排除了一个测量配置混杂：此前数值诊断强制 2D attention，关闭了 decode 的历史 KV 分块并行。单请求、36k 历史的单层 attention 实测为 **1.258 ms vs 原生 split-K 0.137 ms**；不是整模型加速比。容量复测恢复原生自动选路，没有修改采样、输入、窗口或 serving 算法。
- 同一代码快照、每档 **420 s 输入 + 30 s 收尾**：6/7/8 用户分别完成 **2,520/2,940/3,360** 个输入，最低跨窗长期 RTF 为 **0.985506/0.983531/0.951623**；按每用户未舍入 RTF≥1 判定，三档均失败。4 用户档按用户要求在启动期中止，不计结果。
- 每人满窗后覆盖 254–257 轮，逻辑历史达 91.8k–94.0k，P/D 每请求驻留最多 2,259 块。零编码 fallback、运行错误或 KV 抢占；三档接收协议/导出审计均为 0 FAIL/UNKNOWN。D 每轮输入 prefill 仅计算 1 token，满窗后 KV delta 中位数 230、p99 245 tokens，没有整窗重传。
- 原输出消费阻塞未复现：6/7/8 用户的逐用户 D 完成→客户端接收 p99 分别为 **15–19/20–25/29–39 ms**。但轮次处理仍积压，不能用“功能完成”替代实时达标。CPU 回归 1,166 项通过；独立 GPU/缓存检查 53 项通过，包括实际 36k 窗口和 72k 逻辑位置。

详细 setup、判据与复现见 [滑窗测量](benchmarks/minicpmo/sliding_window.md)。这是已归档的未提交开发快照，不是 clean-build 认证。官方路径也会出现的长历史内容退化仍单列；本轮测当前自主生成负载的输入消费能力，不宣称正常对话质量或完整语音播放 SLO 通过。此前修复与证据边界见 [生成与服务核查](benchmarks/minicpmo/sliding_quality.md)。

下文旧容量与探索记录仅作归档，不代表当前修复版的功能或可用容量。

## 阶段一：目标

测量单机四 GPU 在持续音视频输入下可长期维持的 session 容量。先核查输入、状态、KV 和输出传递的工程正确性，再讨论 serving 延迟与容量；官方路径也会出现的内容退化单列，不以修改模型能力作为前置任务。应用层和 connector 的工程混杂仍须排除。

## 阶段二：当前 serving 设计

```text
每 200 ms 音频 + 1 FPS 视频
  -> 一个原生 1 秒 model unit
  -> Vision + Audio Encoder sidecar
  -> Thinker-P 增量 prefill
  -> 传输 block-aligned KV delta
  -> Thinker-D 有限 decode
  -> D 输出进入下一轮 P lineage
  -> 可选 Talker + Code2Wav
```

| Stage | GPU | 作用 |
|---|---:|---|
| Thinker-P | 0 | 多模态增量 prefill |
| Thinker-D + Talker | 1 | 有限自回归 decode；语音 code 生成，MPS 共卡 |
| Vision + Audio Encoder | 2 | 无状态 HD4 视频和流式音频编码 |
| Code2Wav | 3 | 语音 code 转波形 |

- 应用层维护 session 历史和媒体 buffer；engine KV 是可丢弃的执行状态。
- 各 session 的 Thinker 请求独立进入 engine，没有全局 admission gate。Encoder 有独立跨用户 batch：空闲转忙时等待 50 ms，积压时连续处理。
- MiniCPM-o 会把 Thinker 输出反馈给下一 unit，因此真实依赖为 `D(i-1) -> P(i) -> D(i)`。下一 Thinker unit 不等待 Talker 或 Code2Wav。
- 本轮 P/D 显式使用 18,000-token 滑动 attention/KV 窗口和 128-token 保留前缀（启动脚本默认窗口仍为 36,000，复现需传参）；逻辑历史和完整 hash 继续增长，D 复用窗口内 KV，NIXL 按绝对逻辑块位置传 delta。逻辑位置上限为 262,144；提交前预留当前完整生成预算，超限明确关闭该 session，不影响其他用户，不静默重建历史。这不是无限时长支持，也不等同于官方 whole-unit/sink 裁剪，不宣称质量无损。
- Talker 使用独立 4,096-token 滑窗、65,536 逻辑位置上限。关闭基于 binary 占位 token 的跨请求 prefix 查找，但保留活动请求的 KV；避免长时间说话越过原 4k 输入数组。
- P/D 使用 FP8 E4M3 KV 和 `NixlDeltaPushConnector`。D 保留 prefix KV，只导入新增的 block-aligned suffix。

每个到达的视频帧和音频 unit 都在 GPU 2 预编码。完成的 embedding 保存在 CPU，直到对应 P 请求消费。不静默丢输入；编码失败时保留正确性 fallback，但出现 fallback 的测量不能认证容量。

当前配置为 `benchmarks/minicpmo/deploy_capacity_pd_d_talker_fp8.yaml`：P/D/Talker 权重 FP8，P/D KV FP8；Encoder/Code2Wav 保持原生精度。P、D 各固定 64 GiB KV，Talker 固定 8 GiB BF16 KV。Encoder 使用后台线程、每模态独立 CUDA stream 和完成事件。`run_pd_placement.py` 默认启用私有 MPS，归档完整配置并核验四个 stage 的实际接入。

主要计算均支持跨用户 batching：Vision 按切片形状、Audio 按输入与缓存兼容性组批，P/D/Talker 主干使用 engine batching，Code2Wav 按 codec/cache 兼容性组批。部分采样仍逐用户执行；“支持 batching”不等于每次都组成大 batch，也不证明已达最佳效率。下一 unit 的编码可与前一 unit 的 Thinker 重叠，Talker/Code2Wav 不阻塞下一 Thinker unit；同一 unit 的 P→D 仍有真实依赖。

保留的其他配置：`isolated` 为 P、D、Encoder 各一 GPU，Talker+Code2Wav 共 GPU3；`encoders-on-p` 将 Encoder 放 GPU0、Talker 放 GPU2。它们不是本轮拓扑，不能直接沿用旧容量数字。

## 阶段三：旧版 240 秒 workload 与判据（已归档）

本轮按 8→16→12→10 用户搜索，每档重新启动服务，输入 240 秒，再观察输出 30 秒：

- 循环真实 960x540 MP4、对齐的 16 kHz mono 音频和 reference audio；
- 音频每 200 ms 到达，视频为 1 FPS，模型每秒处理一个 unit；
- 使用官方 HD slicing，`max_slice_nums=4`；
- 每个 session 的相位在 `[0, 1 s)` 随机分布，并加入 +/-50 ms arrival jitter；
- seed 为 `20260908`，各 session 使用打散的音视频起始位置；
- context 从 0 开始，不剔除首轮。每个 session 在第 157 个输入附近实际 rollover，峰值 prompt 约 33.4k–33.8k；
- 模型保持自主 listen/speak，客户端按绝对时间发送，不等待模型回复。媒体解码/JPEG 预处理在计时前完成。

主要容量指标为：

```text
stream RTF = 已完成的 1 秒输入总量 /
             从首个媒体到达到最后一个 physical-D 完成的 wall time
```

输入容量通过要求：每个 session 的 `stream RTF >= 1`、全部预期 physical-D 请求完成、没有用户失败、全部视频帧被消费；下一秒完整输入到达时，上一秒必须完成 D，最后一个 unit 在 ready 后 1 秒内完成。中途积压不能用最后追上来抵消。此判据仅证明 Thinker 消费输入，不证明完整语音链路通过；缺少下游证据时，`end_to_end_capacity_pass` 为未知。

额外记录发送计划、实际发送、客户端唤醒偏差和 WebSocket 发送耗时。非预期发送偏差超过 10 ms（200 ms chunk 周期的 5%）则容量证据无效；主动设置的 ±50 ms jitter 不计入该偏差。旧结果缺少这些审计字段时，不按新标准判通过。

16 用户初测出现 177.7 ms 的客户端停顿，该档作废。后续将客户端 cyclic GC 延后到计时窗口外，保留正常引用计数回收，服务端不变；补充逐次迟到记录。10/12/16 用户重测最大发送偏差为 7.05/6.89/6.62 ms。RTF 认证使用未舍入值，不再用显示为 `1.000` 的三位小数判通过。

D 完成后不立即断开，继续观察整个 `post-stream-s` 窗口的音频输出。按每个回复的实际 PCM 时长累计 200 ms 播放缓冲；不同回复间的停顿不算卡顿，未结束的语音单独标记。该窗口可能包含自动续说，不能代替 Talker/Code2Wav 内部队列证据。GPU 采样移出发送事件循环，避免干扰输入节奏。

正式结果还要求：代码树干净；记录 server/client provenance；关闭诊断开关；没有 truncation、fallback、preemption；D prefix 和物理 KV 传输证据完整。dirty-tree 测量只能作为开发证据。

## 阶段四：已排除的工程混杂

| 混杂因素 | 当前处理 |
|---|---|
| 原始 AV 被复制到 D | D 只接收 prompt metadata 和导入的 KV |
| 每轮传输完整历史 KV | P 只发送 block-aligned delta |
| D 错误重算末尾媒体位置 | 缓存导入保留全部已计算位置；当前原生 P/D 每轮首次执行仅计算 P 生成的 1 个 token。旧的 2-token 情况是正确性错误，不是正常开销 |
| P/D 丢失采样和回合状态 | 有界状态随请求传递并回传 P；seed/offset 保持跨轮 RNG 连续，按 session/unit 身份恢复一次，不传媒体或 encoder 状态 |
| 回答结束与 unit 结束混淆 | TURN_EOS 后继续到原生 unit 边界；使用最终回合状态，不把中途 TURN_EOS 当作最终结束；P 已结束 unit 时 D 只确认 |
| 普通聊天 EOS/stop 提前截断 unit | Native Thinker 清除继承的聊天停止规则，只按三个原生 unit token 结束；不屏蔽普通 EOS 的生成 |
| 额外的 28 字符截断 | 删除非官方字符上限，保留原生每 unit 最多 20 token 的规则 |
| D→Talker 的 token/hidden 错位 | 每个文字 token 配对其自身被 forward 后的 hidden，而不是预测它的上一位置；只跳过 P 的首个决策 token |
| LISTEN 提前消耗 Talker 新回复标记 | 只有真正发往 Talker 时才提交其生命周期；新回复重置 KV，同回复各 unit 增量续接 |
| 旧回复的待发音频配上新文字 | 按 epoch/model-turn 隔离输出；拒绝旧回复迟到结果，不跨回复保留无归属音频 |
| 纯音频回复被文字门控丢弃 | 有明确 native 回复身份的非 LISTEN 音频正常创建/继续回复；仅真实 turn-end 结束回答 |
| 局部音频/文字坐标冒充累计坐标 | 使用播放游标维护累计边界；不伪造逐词对齐 |
| worker 丢掉片段身份，重复文字被误去重 | 保留 cache_epoch/chunk_seq/delta 标记；按真实片段身份去重，缺必需身份明确报错 |
| 异步 P→D 转交误读其他 stage 的 token | 使用 P 输出自带的边界 token，不读取各 stage 共用的临时值；错误身份和空边界仍拒绝 |
| 空文本 unit 错误中断续音 | 非 LISTEN 的空文本 unit 仍按官方规则推进 TTS；使用最终回合状态控制结束 |
| 未完成的 chunked prefill 污染采样状态 | P/D/Talker 均通过外层模型接收实际采样资格；未完成行不采样、不消耗 RNG、不更新回合状态；bookkeeping 不错误回退 RNG |
| Vision fallback 重编码 | 每个正式帧都消费 arrival-preencoded embedding |
| 音频、视频并发首次初始化覆盖缓存 | 加锁创建唯一 runtime，保留首帧缓存 |
| Encoder 队列丢帧 | 端到端审计 frame identity，不允许静默丢帧 |
| GPU2 -> GPU0 -> CPU 绕路 | Sidecar 输出直接从 GPU2 搬到 CPU |
| 持有全局 cache lock 做设备拷贝 | 拷贝在锁外执行，并使用 pending reservation |
| 迟到 encoder 结果污染已结束 session | session tombstone 拒绝迟到写入 |
| 重复解析图片/音频 metadata | 每次 planning transaction 只解析一次 |
| 重复复制完整 prompt list | 删除多余副本，D submit 后释放 bridge payload |
| 逐 row 读取 sampling metadata 并重复 clone logits | 每 batch 只搬一次 sampling 参数；每 row 只保留一份可写副本，RNG 顺序不变 |
| FlashInfer cache miss 启动依赖已激活的 shell | clean launcher 自动发现当前 Python 环境的 CUDA toolkit 和 `ninja`，并记录两者路径 |
| D 完成与 KV 复用证据不明确 | 每个 physical-D 完成都携带 request、prefix、suffix、block、token 和 byte 证据 |

没有新增 prepared-request 协议：复制并序列化 17k-token list 约为 0.1 ms、73 KiB，不足以解释 0.5-1 秒 tail。也没有引入无上限 pinned-memory cache，避免用新的内存风险处理非主要开销。

2026-09-08 修复验证：177 项回归测试通过；共卡和分卡各跑 `2 用户 × 10 秒输入 + 30 秒输出观察`，FP8、HD4、context 从 0 开始，seed `20260908`。两组均完成 20/20 个 D 输入和 20/20 帧，音视频 fallback、用户失败、输入积压均为 0；发送偏差最大分别为 3.75/4.11 ms，四个 stage 均接入私有 MPS。P、D 实际 KV 容量均为 932,064 tokens。

这是功能短测，不认证容量：输出观察仍记录到共卡 5 次、分卡 4 次缓冲耗尽，分卡另有 1 个回复在窗口结束时未观察到终止。这些现象尚未归因，不能宣称完整语音链路通过。完整配置、日志、输入和输出审计在 `/home/ubuntu/data/experiments/minicpm-pd-fairness-20260908/{colocated-2x10-v3,isolated-2x10-v1}/`。

## 阶段五：旧版上下文重建长测（2026-09-08）

均为上述 MPS、D+Talker 共卡、FP8、HD4、240 秒输入配置。输入完成、帧守恒、音视频 arrival cache 命中、context rollover 和 MPS 接入均核验；无运行错误、fallback 或抢占。源码未提交，已保存源码快照，只作为开发证据，不认证正式容量。

| 用户数 | 完成输入 | Ready→D p50/p99 | 输入积压次数 | 最大超期 | 最小 stream RTF（未舍入值保留六位展示） |
|---:|---:|---:|---:|---:|---:|
| 8 | 1920/1920 | 206/604 ms | 0 | 0 ms | 0.999648 |
| 10 | 2400/2400 | 200/343 ms | 1 | 6 ms | 0.999743 |
| 12 | 2880/2880 | 223/386 ms | 3 | 176 ms | 0.999475 |
| 16 | 3840/3840 | 234/577 ms | 9 | 442 ms | 0.999574 |

所有积压均在首个输入，后续 239 轮及压缩过程无输入积压。16 用户最慢首轮 Ready→D 为 1444 ms，其中 D 提交前 1227 ms、D service 217 ms；目前只定位到前段，未进一步证明是 P forward、encoder 首次初始化还是其他启动工作。

结论：仅按逐秒零积压，已验证 8 用户，10 用户也有一次 6 ms 超期，不能忽略。**这不是“GPU 最多支持 8 用户”的证据**；当前搜索首先撞到启动抖动，尚未找到持续处理容量上限。各档最小有限窗口 RTF 都略低于 1，严格 `RTF>=1` 仍未通过；不能通过四舍五入抹掉差异，也不能仅凭这个微小差异推断长期 backlog 增长。

完整语音链路仍不认证：8/10/12/16 用户分别观察到 225/229/559/199 次 200 ms 播放缓冲耗尽，以及 5/10/10/1 个未观察到终止的回复；PCM 格式错误均为 0。观察窗口含自动续说，不能把这些计数直接归因于 Talker/Code2Wav 饱和，也不能宣称播放流畅。

结果根目录：`/home/ubuntu/data/experiments/minicpm-pd-d-talker-capacity-20260908/`。有效时序档为 `users-{8,10,12}-fresh-v1` 和 `users-16-fresh-v2`；`users-16-fresh-v1` 因发送迟到无效。`search-v1` 仅预热：下游请求计数未归零，放弃复用服务，不能记成容量失败。保留原始 `analysis.json`；`analysis-exact-rtf.json` 为修正 RTF 舍入判定后的离线重分析。计时后未修改输入记录。

## 阶段六：已有测量（部署公平性修正前）

以下数字来自修正前的配置和判据，不代表固定 KV、统一编码执行路径后的新容量。修正后先做功能短测，再进行同配置容量对照。

两组均使用相同的 24x180 workload 和 seed。当前测试完成 4,320/4,320 个 physical-D 请求，消费 4,320/4,320 帧，fallback 和用户失败均为 0。

| 指标 | 清理前 | 当前 |
|---|---:|---:|
| Ready -> D p50/p95/p99 | `2069/5075/6034 ms` | `533/970/1244 ms` |
| 继承上一轮 D 等待 p50/p95/p99 | `1061/4081/5051 ms` | `0/0/236 ms` |
| 当前轮 fresh pre-D p50/p95/p99 | `574/772/856 ms` | `273/413/481 ms` |
| 当前轮 D service p50/p95/p99 | `388/701/914 ms` | `251/629/819 ms` |
| Fresh serial cycle p50/p95/p99 | `983/1300/1555 ms` | `532/929/1158 ms` |
| 最终 backlog p50/p95/p99 | `2476/3257/3286 ms` | `136/558/620 ms` |
| Stream RTF mean/min | `0.988/0.982` | `0.999/0.997` |

当前每个 unit 传输的 KV 中位数为 9 tokens、1 个 1.125 MiB block；p99 为 16 tokens、1 个 block。4,320 条传输记录全部有效。

旧的 5-6 秒 tail 主要是工程开销逐轮递推造成的。清理后，继承等待均值从 1,467 ms 降至 7 ms，且没有一次超过 1 秒。当前最慢 1% 中，继承等待占 15.6%，当前轮 pre-D 占 26.1%，D service 占 58.3%。

当前开发测试以极小差距未满足严格的有限窗口 RTF 判据（`min=0.997`），并且代码树为 dirty，因此不能认证正式容量。它能确定的是：多秒 backlog 已被消除。

### 28 用户下 Thinker-P 的实际 GPU 利用率

一次 28x180 长测在正式窗口内采集了 1,004 组 GPU0 硬件计数。这里不使用显存占用判断负载；`GPU kernel active` 也只表示有 kernel 驻留，不等于 GPU 算力已被充分利用。

| 指标 | 平均 | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| GPU kernel active | `59.6%` | `60.4%` | `96.1%` | `99.9%` |
| SM active | `39.6%` | `39.0%` | `73.5%` | `79.7%` |
| SM occupancy | `5.4%` | `5.3%` | `10.2%` | `11.2%` |
| Tensor Core active | `29.2%` | `27.8%` | `60.9%` | `66.1%` |
| DRAM bandwidth active | `9.0%` | `9.2%` | `14.7%` | `15.7%` |
| Power（上限约 600 W） | `311 W` | `322 W` | `349 W` | `361 W` |

GPU0 并未持续达到算力、带宽或功耗上限：平均 SM active 约 40%，Tensor Core active 约 29%，DRAM active 仅 9%。p95 的短时升高说明 prefill burst 到来时 GPU 会变忙，但这种压力不连续。SM occupancy 不能直接解释为“只用了 5.4% 峰值算力”，但它和 runner 的 batch p50 为 1 个请求、约 219 tokens 一致，说明多数 prefill batch 提供的并行度很低。

因此 28 用户下的 P 侧问题不是 GPU 物理能力已经耗尽，而是零碎的增量 prefill 不能持续形成高效 batch：平时硬件利用不足，短时 burst 又会形成排队并放大 tail。硬件计数证明“没有持续饱和”；结合 batch 形状和 runner 时间，才将低效率归因于碎片化 prefill。

完成 sampler 最终清理后，又用最终代码跑了非诊断 24x30 回归：720/720 个 physical-D 完成，720/720 帧被消费，fallback 和用户失败均为 0。Ready-to-D p50/p95/p99 为 `256/457/557 ms`，当前轮 pre-D 为 `121/215/267 ms`，D service 为 `126/278/354 ms`，继承等待 p99 为 0。结果与清理后的短测基线一致；由于只有 30 秒且代码树为 dirty，不能作为正式容量结果。

另用相同 production workload 和短测 seed 跑了独立的 24x30 诊断。诊断日志会扰动绝对延迟，因此下表只用于归因：

| 剩余路径 | p99 | 含义 |
|---|---:|---|
| 应用 ready -> 提交 P | `2.5 ms` | 应用 admission 不是 tail 来源 |
| P scheduler queue | `1.7 ms` | Core 接纳后很快被选中 |
| P runner 全部工作 | `222 ms` | 增量准备、forward 和 sampling/snapshot |
| P 结果暴露 | `33 ms` | 次要控制路径开销 |
| D 消息完成 IPC 解码后的 ingress | `134 ms` | Core 要等当前同步 runner step 结束后才读取输入队列 |
| D scheduler queue | `3.6 ms` | 接纳后调度很快 |
| D runner 全部 decode steps | `315 ms` | 顺序自回归计算；输出 token p99 为 8 |
| D 结果暴露 | `18 ms` | 次要控制路径开销 |

原始 StagePool send、Core receive、消息解码和 preprocessing 通常都低于 2 ms。P 到 D 的 `write()` 调用 p99 为 `3.0 ms`；write 到 D 完成 p99 为 `79 ms`，且与计算流水线重叠。因此 serialization、IPC、KV 带宽、scheduler queue 和应用 gate 都不足以解释剩余 tail。

剩余主要成本已经明确：P 的真实增量准备/forward、D 的顺序 decode，以及同步 Core 只能在 runner step 之间接纳新请求。前两项是模型计算。第三项是 engine 调度抽象：请求已经到达 Core 并完成解码，但不能加入正在执行的 batch。要消除它，需要 event-driven/thread-safe admission 或新的增量 batching scheduler，而不是再加应用层 gate。随机采样为保持各 session 的 RNG 顺序仍需逐 row 做 host 判断；该次要开销不能解释当前 tail。

## 阶段七：复现

每档独立启动并自动关闭服务及私有 MPS；例如复测 16 用户：

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 16 --duration-s 420 --kv-window-tokens 36000 --seed 20260908 \
  --out-dir /home/ubuntu/data/experiments/minicpm-pd-sliding-repeat
```

输出目录必须是新目录。当前脚本默认用独立会话预热 12 秒，正式用户从 0 开始输入 420 秒、输出观察 30 秒。保持真实 MP4/reference audio、HD4、随机相位和 jitter。满窗口后的逐用户长期 RTF 是容量判据，单轮超时与积压仅诊断；上述命令不会复现旧版短上下文重建。归档配置、命令、provenance、MPS、预热、输入/输出审计以及源码稳定性检查。

本轮源码快照在 `users-8-fresh-v1/{source,vllm-source}` 及 `users-16-fresh-v2/{source,vllm-source}`，后者与 10/12 用户计时期间的源码一致；RTF 精度修正仅影响事后分析。诊断开关只用于单独的定位实验，不能混入容量计时。
