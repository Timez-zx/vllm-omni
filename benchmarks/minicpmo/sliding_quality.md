# MiniCPM P/D：生成与滑窗核验

英文版：[sliding_quality.en.md](sliding_quality.en.md)。

## 当前结论（2026-09-09）

**最新 36k 长测的可观测服务契约通过；模型内容质量与 serving 正确性分开处理，不发布新容量。** 官方原始媒体、BF16、原生 encoder/生成器/basic 窗口的 420 轮对照，也在 unit150、context32,504、尚未滑窗时出现长纯控制输出。该实验不使用 vLLM、P/D 或我们的 attention mask，说明此类退化不依赖我们的 KV 搬运实现。它不能区分权重与官方生成策略的责任，也不证明所有 serving 代码无错。

后续只修输入、状态、KV 与输出传递问题，不通过改采样规则纠正内容。默认仍为 36k；8k 仅为归档对照，不是训练上限或正式修复。下文旧的“生成不通过”不再等同于“已定位 serving 错误”。

## Setup 与 workload

- GPU0：Thinker-P；GPU1：Thinker-D + Talker；GPU2：Vision/Audio Encoder；GPU3：Code2Wav。私有 MPS，各模块保持 pipeline。
- P/D/Talker 权重 FP8；P/D KV FP8。以下最新对照显式使用 BF16 Q，并让 P/D 都走 Triton 2D attention；这两个开关默认关闭。
- 960×540 原始视频、HD4、1 FPS；16 kHz 音频每 200 ms 到达，组成 1 s model unit。音视频对齐循环同一个 35 s 片段。
- seed 20260908；用户使用不同相位和媒体起始偏移；从空历史开始。独立预热 12 s，正式输入每人 420 s，之后观察 30 s。
- 显式保护开头 128 tokens，窗口为 36,000 或 8,000。只增量 prefill，P→D 仍传 block-aligned KV delta，不重建整窗。
- 滑窗仅限制物理 KV；绝对逻辑位置继续增长，受 262,144 上限保护。这与官方 whole-unit 裁剪并重定位 K 的策略不同，不宣称数值等价或质量无损。

## 最新服务核查与输出阻塞修复

发现一个与内容退化独立的应用错误：`_maybe_continue_native_response` 在输出消费循环里等待静音续接 scheduler，后者会等一秒定时器或前一静音 append。D 已完成的结果和语音因此排队。现在每 session 最多一个后台调度任务；输出继续消费，原有真实输入顺序、过期身份检查和静音续接策略保留。取消后台等待时使用 shield，不取消已经提交的 append；关闭 session 时清理定时任务。

真实 runtime bridge 的无 GPU 复现确认：修复前输出 consumer 被 pending scheduler 卡住，修复后可立即继续。对应 CPU 回归 1,166 项通过、18 项跳过。

三次均为上述 2×420 s、36k/pin128、FP8 权重/KV、BF16 Q、Triton 2D、MPS 和正常 CUDA Graph。没有同步 KV/语音探针，只有事后客户端导出；输入/config 相同，不假定自由生成的 token 轨迹完全相同。

| 运行 | 收尾观察 | 实际 D 完成 | 两位用户 D 完成→客户端收到 p99 |
|---|---:|---:|---|
| r1，修复前 | 30 s | 839/840，用户 0 缺最后一轮 | 34.34 s / 962 ms |
| r2，修复前 | 90 s | 840/840 | 17.15 s / 12.30 s |
| r3，修复后 | 30 s | 840/840 | **12.95 ms / 12.13 ms** |

这里量的是完成事件的传递延迟，不是 TTFA、GPU forward 或纯网络耗时。由逐事件接收单调时钟与对应输入发送时的 wall/monotonic 锚点离线对齐，保留时钟配对及导出舍入误差。r1 仍为不完整记录；延长 r2 收尾不是性能修复。r3 恢复原来的 30 s 收尾后仍完整完成。

r3 的最终核查：

- 840 帧和 840 个音频 unit 都由 arrival encoder 处理，零 fallback、零输入截断/请求运行错误；发送偏差最大 5.64 ms，源码运行前后未变。
- 每人有 254 个实际淘汰后的 unit，最终逻辑 prompt 为 91,773 / 92,369；P/D 最多 2,259 个驻留块，其中开头 8 块受保护。窗口完整替换，不重建短 prompt。
- 840 轮的 D 输入预填充均只计算 P 新采样的一个 token。`cached = local + external`，`prompt = cached + 1`，`transferred = external` 全部成立；滑窗后 KV delta 中位数 226 / 230 tokens，最大 245，不是整窗重传。
- 接收协议、session/response 身份、文字/音频坐标及 PCM→WAV 字节核验均通过：80 个正常回复，2 个取消回复单列；累计收到 1,569 字、424.24 s PCM，包含取消回复已送达的前缀。审计 0 FAIL、0 UNKNOWN。
- **最低滑窗后长期 RTF 为 0.943，仍有输入处理积压。** 修好了输出消费阻塞，不代表实时容量已通过；本次也没有对剩余 P/D 等待重新定责。

事件审计不证明每一个历史 KV 的数值相等或源端从未漏生成 chunk；本次未重新采集源 codec，也未重做 ASR/语义评分。先前的采样 KV 位级对照仍单独归档。模型内容退化可能改变 decode/语音负载，因此本轮使用 `--functional-only`，分析器不会将完整执行误报为正常对话容量认证。

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 2 --duration-s 420 \
  --kv-window-tokens 36000 --pinned-prefix-tokens 128 --kv-cache-dtype fp8 \
  --triton-disable-q-quantization --triton-force-2d-attention \
  --quality-capture --functional-only --out-dir /path/to/new-run
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/functional_audit.py /path/to/new-run
```

证据根目录：`/home/ubuntu/data/experiments/minicpm-pd-serving-contracts-20260909/`。三档依次为 `fp8-pin128-w36000-2x420-r1`、`fp8-pin128-w36000-2x420-drain90-r2`、`fp8-pin128-w36000-2x420-output-drain-fix-r3`。保留配置、命令、源码快照、完整事件/WAV、`functional-audit.json`、`analysis.json` 和 source-stability。汇总为 `final-summary.json`，离线脚本为 `summarize_contract_runs.py`；阻塞复现为 `check_output_drain_scheduler_wait.py` 及 before/after 日志。官方原始媒体对照在相邻 `minicpm-pd-native-reference-20260909/bf16-native-420-topk100-basic36000/`。

另修复旧分析规则：当前 D 不重算远端末 token，不能再要求 `transferred = external + 1` 或允许两 token replay；证据缺失仍记 UNKNOWN。`--post-stream-s` 默认仍是 30，仅控制收尾观察，不改变输入节奏或 RTF 公式。

## 归档定责：相同输入，脱离 serving 独立生成

2026-09-09 补充两步对照，仍不认证 36k 功能或容量：

1. **固定历史数值对照。** 36k、BF16 权重/KV 的两用户采集保存真实 P/D 输入 embedding 与输出 hidden；只记录第 0 层 KV，降低探针体积。6,503 次 forward 用官方 HF 重放，按下一轮 P 的真实历史排除 174 次废弃 D suffix，不把额外执行当有效输出。绝对位置超过 36k 的 3,002 次有效 D 采样位置，原生/当前 raw top1 有 2,878 次相同，原生 top1 全为控制 token。这是条件于当前历史的证据，不能单独说明坏历史最初从何而来。
2. **原生独立生成。** 只复用同一用户的 system/reference 和每轮 AV embedding，不输入 serving 生成的历史。官方 `streaming_generate`、`StreamDecoder.feed/decode` 自行维护输出与采样状态；BF16/SDPA、top-k100、每轮最多 20 tokens、240 轮。保留全部物理 KV，以显式 mask 实现 pin128 + 最近窗口，不做 P/D、KV 压缩、搬运或重定位，也不运行 TTS。

| 原生独立对照 | 长纯控制输出轮数 | 最长连续段 | mask 首次排除旧 KV / 最后普通文字 |
|---|---:|---|---|
| 36k，seed42 | 81/240 | unit157–223，共 67 轮 | unit168 / unit156 |
| 36k，seed7 | 127/240 | unit149–204，共 56 轮 | unit165 / unit148 |
| 8k mask，seed42，仅作对照 | 0/240 | 无 | unit39 / unit237 |

“长纯控制”在此指一个原生 unit 的生成决定至少含 10 个控制 token、没有普通文字；不计最后的 `</unit>`，不将正常单个 LISTEN 判坏。此口径是 unit，不是此前的 TTS handoff。8k/36k 的 seed42 对照前 43 轮生成 token 完全一致，第 44 轮才分歧；8k 对照绝对位置继续增长至 50,925。结果支持**可见长历史下的生成稳定性问题不依赖 vLLM/P-D 或错误的 KV 搬运**，不能简单归因于绝对位置超限。并不证明模型只支持 8k，也没有修好 36k；8k 不替代正式配置。

采集请求为 2×300 s，客户端观察期 D 仅完成 254/300、214/300；探针另记录了收尾期的执行，不能把这些记录冒充完成的容量测量。独立原生对照均完整完成 240 轮，但不验证音频输出。实验脚本、输入 ledger、逐位置比较、逐轮原始 token 和复现命令保存在 `/home/ubuntu/data/experiments/minicpm-pd-generation-fix-20260909/teacher-ledger-findings.md`。本轮选定 CPU 回归 379 项通过；新加入的是显式诊断选项，不是生成修复。

## 已确认的修复

| 边界 | 修复与证据 |
|---|---|
| 输入 | 保留完整参考 WAV，不再把 6.016 s 截成 6.000 s；按真实 processor 规则计算 token 预算。末尾参考 embedding 与官方对照的 cosine 从 0.750 升至 0.999 |
| 多用户音频 | 每个 session 独立持有 CPU audio processor；不再共享可变的 fixed/dynamic 归一化设置。真实并发 Mel 最大误差从 0.589 降为 0，reference 不受 live session 模式切换影响 |
| 自主回应 | 删除隐式 RMS/静音→LISTEN 规则及其状态清空副作用，只保留显式 force_listen。正式媒体未触发静音阈值，因此不把它称为已定位的长测根因 |
| P/D 状态 | 完整传递有界采样状态、重复惩罚历史、RNG 和 unit 身份；D 只计算 P 采样的首 token，不重算媒体 |
| 异步交接 | P→D 使用 P 输出自带的 token，不读取被其他 stage 覆盖的共享临时值；保留严格身份和边界校验 |
| 原生生成规则 | 区分回答结束与 unit 结束，清除普通 chat 的 EOS/stop 规则，删除额外字符截断；partial prefill 不推进采样状态 |
| Talker 生命周期 | token 配对自身 hidden；新回复真正提交 Talker 时才重置 KV，LISTEN 不消耗新回复标记，同回复仍增量执行 |
| 客户端输出 | 按 session/epoch/model-turn/chunk 隔离；不按文字内容误去重，不丢有合法身份的纯音频；使用真实累计文字/音频坐标 |
| 取消 | ACK_ONLY 仅提交客户端确认的播放历史。真实 writer/cancel 回归验证排队但未播放的尾部不会进入历史 |
| 诊断 | opaque NIXL handle 不强转整数；缺少 KV 样本记 UNKNOWN 而非错误的 PASS/FAIL；不完整测量明确返回非零状态 |

多用户处理器隔离和原生回应策略修复后的长测已完成：8k 的输入、状态和输出传递核验通过，36k 仍退化。联合 CPU 回归 997 项通过、3 项跳过；不能用测试数量代替生成核验。

## 最新结果

| 对照 | 结果 |
|---|---|
| 36k，统一 2D attention 的短测 | 同输入、同历史的 P/D 最终 hidden 差异从 32.07% 降至 0；所检完整 KV 和实际 WRITE 链路一致。短测没有覆盖滑窗及正常语音 |
| 36k，修复参考输入前，2×420 s | 840/840 输入完成；32 组实际传输、469,368 个层/位置 raw KV 与 scale 比较全部一致。但控制标记/乱码循环仍存在，52 个 WAV 完成 ASR，生成不通过 |
| 36k，保留完整参考输入，2×420 s | D 仅完成 407/420、402/420；观察期结束仍有积压，不能算完整长测。两用户首次异常约 141/59 s，均早于首次滑窗；约 157 s 起分别连续 56/59 次纯控制输出，此后未恢复正常对话 |
| 同组输出核查 | 109 个正常回复的文字、codec、PCM 守恒；111 个客户端 WAV 全部完成 ASR。2 个取消回复只收到连续前缀，另有 2 个后台回复未送达，明确单列。86/109 正常回复有 PCM 却无普通文字，说明传输完整不等于生成可用 |
| 36k，完整参考 + 音频隔离 + 原生策略，2×420 s | D 完成 391/420、401/420；源端约 53–55 s、11k context 已出现异常，早于 unit166 首次滑窗。跨窗后持续退化；74 个正常回复传递守恒，76 个 WAV 完成 ASR，不认证可用 |
| 8k，同源码/精度/输入，2×420 s | 840/840 输入完成，各跨窗 383 轮，KV 最多 509 块，无输入积压、fallback 或运行错误；89/183 次源 handoff 均无长纯控制循环 |
| 8k 完整语音核查 | 44 个回复全部正常结束，无取消尾部。771 字、5,911 codec、241.72 s PCM 完整对应；44 个 Talker turn 均从 position0 开始。44 个 WAV 全部完成 ASR，整体对应源文字；仍有跑题和计数错误，不代表模型能力完美 |
| 36k，同拓扑 BF16 Thinker 权重/KV，2×420 s | D 完成 418/420、420/420；纯控制输出首次约 86/93 s，早于 unit165/167 首次滑窗。最长连续控制 handoff 57/42 次，之后未恢复正常对话，仍不通过 |
| BF16 完整语音核查 | 53 个 Talker turn 均从 position0 开始；50 个正常回复的 417 字、425.56 s PCM 传递守恒。51 个 WAV 全部完成 ASR，退化语音仍存在。1 个取消回复的 180 字、10 s 未送达尾部单列，不当作正常回复丢失 |
| 8k，关闭内部探针、正常 CUDA Graph，2×420 s | P/D 编译与 Graph capture 成功；840/840 输入、各 383 轮跨窗，无输入积压、fallback 或运行错误，协议审计 0 FAIL/UNKNOWN。42 个回复全部正常结束，1,067 字、279.12 s PCM，42 个 WAV 全部完成 ASR；末尾仍能回应 |

上述首个“纯控制”信号定义为一次非 LISTEN 的 TTS handoff 没有普通文字、含至少 10 个控制 token；单次信号不直接判失败，持续循环及之后能否恢复需另查。后两组使用同步探针，出现几十秒积压，不能据此推导生产容量或唯一性能根因。

此前 8k 的 BF16-KV 与 FP8-KV/BF16-Q 对照均完成 840/840 输入、每人跨窗 383 轮，未复现旧的 182 s 重复短句。它们早于完整参考输入修复，仅支持继续验证 8k，不认证最新版。

最新 8k/36k FP8 两组源码快照相同，实质配置差别仅窗口。但自由采样在首次滑窗前就出现不同输出，不能把单次对比当作窗口的唯一数值因果证明。8k 是已取得正向证据的运行配置，不是已修好 36k 的证明；默认仍未改变。

有数值探针的上述长测使用 `enforce_eager=true`。因此另补了无数值/语音探针的 8k 正常执行对照，保留事后客户端导出；它通过基本输入、滑窗和传递核验，但不重新采集源 codec/hidden。个别长数词回复的 ASR 与文字不一致，不能单凭小型 ASR 判定是 TTS 还是识别错误；所有录音保留，不宣称逐字语音或 MOS 通过。原始容量报告还受 dirty tree 和旧版“D 重算远端末 token”审计规则限制，未改写为容量通过。

BF16 对照只把 P/D 权重与 KV 改为 BF16，并显式各预留 48 GiB KV，以容纳 D 同卡的 Talker；仍足够存下两用户窗口。Talker 仍 FP8，拓扑、媒体、pin128、2D attention 和模型/服务源码不变。它排除了“FP8 是必要原因”和“首次 eviction 才触发故障”，不证明本实现已完全正确。共享静态 KV scale 另作 CPU 短程评估，尚未加载到服务，也未当作质量修复。

## 目前排除了什么，尚未证明什么

- 已检查的 KV 传输正确，不支持“传坏 KV 导致乱码”。长测的位级比较只覆盖明列的采样位置，不代表所有位置。
- 统一 2D attention 消除了短测重算差异，但没有消除长程退化。滑窗后，16-token block 移动会改变 32-token tile 的归约分组；可见历史相同仍可能出现数值差异，尚未证明它导致生成退化。
- 官方 BF16、无 vLLM/P-D/TTS 的 36k 对照也退化，8k 对照仍能回应。这不等于排除了本实现的问题。官方长测 top-k=100，服务为 20；已有官方 150 轮对照两者逐 token 相同，但不能冒充全部采样条件完全匹配。
- `sent_ms` 是入队口径，不是网络送达证据。取消尾部必须逐回复核查，不按正常结束的回复计算丢失。
- ASR 只辅助核对文字与声音，不是 MOS、事实准确率或“所有场景均正常”的证明。
- 当前不发布正常对话容量认证。后续 serving 性能研究需要明确生成退化所改变的输出负载，不以内容错误直接判定 KV/协议有错。

## 复现与证据

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 2 --duration-s 420 \
  --kv-window-tokens 36000 --pinned-prefix-tokens 128 \
  --kv-cache-dtype fp8 --triton-disable-q-quantization \
  --triton-force-2d-attention --quality-capture \
  --speech-probe-dir /path/to/new-speech-probe \
  --numerical-probe-dir /path/to/new-numerical-probe \
  --numerical-probe-seqs 1-3 --out-dir /path/to/new-run
```

这是正确性诊断命令，不是容量命令。8k 对照显式改窗口参数；不能将主动中断的预热算成正式结果。

BF16 对照使用 `--thinker-quantization none --kv-cache-dtype bfloat16 --thinker-kv-cache-memory-gib 48`，其余参数保持不变。该预算选项只覆盖 P/D，不修改其他 stage 或部署默认值。

无内部探针对照将窗口设为 `8000`，去掉 `--speech-probe-dir`、`--numerical-probe-dir` 和 `--numerical-probe-seqs`，保留 `--quality-capture` 及精度/attention 开关。

- 根目录：`/home/ubuntu/data/experiments/minicpm-pd-generation-fix-20260909/`。
- 最新完整参考输入失败档：`pin128-w36000-fullref-qbf16-kvfp8-2d-2x420-r1/`，保存配置、源码、MPS、日志、输入完成记录、完整 WAV、ASR、`source-generation-findings.json`、`speech-delivery-owner-audit.json` 和 `functional-audit-v4.json`。
- 最新隔离修复对照：`pin128-w{36000,8000}-fullref-isolated-qbf16-kvfp8-2d-2x420-r2/`。8k 的协议审计为 0 FAIL、0 UNKNOWN；全部语音与源生成报告位于相应 run。主动中止的 8k r1 仅有预热，不是本组结果。
- BF16 对照：`pin128-w36000-fullref-isolated-bf16all-2d-2x420-r1/`，同样保留源生成、逐回复传递、WAV/ASR 和协议审计；838/840 完成，不能认证容量。
- 正常 Graph 对照：`pin128-w8000-fullref-isolated-qbf16-kvfp8-2d-noprobe-2x420-r1/`；结论与证据边界见该目录的 `final-functional-findings.md`。
- 数值证据：根目录 `attention-2d-short-result.md`、`attention-2d-long-numerical-result.md`；参考输入证据：该 run 的 `reference-input-comparison.json`。
- 官方窗口策略与对照：根目录 `native-context-policy-audit.md` 及相邻 `minicpm-pd-native-reference-20260909/`。
- 协议审计：`python benchmarks/minicpmo/functional_audit.py RUN --out RUN/functional-audit-v4.json`。其通过只表示可观测协议通过，不认证生成质量。
