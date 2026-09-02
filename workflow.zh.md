# MiniCPM-o 原生全双工 P/D 工作流

英文版本：[workflow.md](workflow.md)。

## 阶段一：目标

测量单机四 GPU 在持续音视频输入下可长期维持的 session 容量。只研究 serving 延迟和容量，不评价模型质量。在讨论 engine research 前，先排除应用层和 connector 的工程混杂。

## 阶段二：当前 serving 设计

```text
每 200 ms 音频 + 1 FPS 视频
  -> 一个原生 1 秒 model unit
  -> Vision Encoder sidecar
  -> Thinker-P 增量 prefill
  -> 传输 block-aligned KV delta
  -> Thinker-D 有限 decode
  -> D 输出进入下一轮 P lineage
  -> 可选 Talker + Code2Wav
```

| Stage | GPU | 作用 |
|---|---:|---|
| Thinker-P | 0 | 多模态增量 prefill |
| Thinker-D | 1 | 有限自回归 decode |
| Vision Encoder | 2 | 无状态 HD4 视频编码 |
| Talker + Code2Wav | 3 | 生成语音 code 和波形 |

- 应用层维护 session 历史和媒体 buffer；engine KV 是可丢弃的执行状态。
- 各 session 独立进入 engine；应用层没有全局 admission gate，也不做跨用户 batch。
- MiniCPM-o 会把 Thinker 输出反馈给下一 unit，因此真实依赖为 `D(i-1) -> P(i) -> D(i)`。下一 Thinker unit 不等待 Talker 或 Code2Wav。
- 估计 context 达到 36,000 tokens 时重开 lineage，只保留 system/reference input、上一完整 AV unit 及其确认后的 Thinker 输出、当前 unit。模型上限为 40,960 tokens。
- P/D 使用 FP8 E4M3 KV 和 `NixlDeltaPushConnector`。D 保留 prefix KV，只导入新增的 block-aligned suffix。

每个到达的视频帧都在 GPU 2 预编码。完成的 embedding 保存在 CPU，直到对应 P 请求消费。不静默丢帧；编码失败会显式报告，不用正式路径重编码掩盖失败。

## 阶段三：workload 与有效性

容量候选测试为 24 用户、180 秒：

- 循环真实 960x540 MP4、对齐的 16 kHz mono 音频和 reference audio；
- 音频每 200 ms 到达，视频为 1 FPS，模型每秒处理一个 unit；
- 使用官方 HD slicing，`max_slice_nums=4`；
- 每个 session 的相位在 `[0, 1 s)` 随机分布，并加入 +/-50 ms arrival jitter；
- seed 为 `20260915`；
- context age 在 0 到 154 个 unit 间随机预热，覆盖长 context 和 rollover；
- 正式测量 4,320 个 unit，另有 1,848 个不计入结果的预热 unit。

主要容量指标为：

```text
stream RTF = 已完成的 1 秒输入总量 /
             从首个媒体到达到最后一个 physical-D 完成的 wall time
```

容量通过要求：每个 session 的 `stream RTF >= 1`、全部预期 physical-D 请求完成、没有用户失败、全部视频帧被消费。单轮延迟和 stage RTF 只用于定位 jitter，不单独定义容量。

正式结果还要求：代码树干净；记录 server/client provenance；关闭诊断开关；没有 truncation、fallback、preemption；D prefix 和物理 KV 传输证据完整。dirty-tree 测量只能作为开发证据。

## 阶段四：已排除的工程混杂

| 混杂因素 | 当前处理 |
|---|---|
| 原始 AV 被复制到 D | D 只接收 prompt metadata 和导入的 KV |
| 每轮传输完整历史 KV | P 只发送 block-aligned delta |
| D 重算历史 | D 复用 prefix，只计算 1-2 个 suffix token |
| Vision fallback 重编码 | 每个正式帧都消费 arrival-preencoded embedding |
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

## 阶段五：最新测量

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

## 阶段六：复现

启动新的非诊断服务：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/minicpmo/clean_server.py \
  --provenance-out /tmp/minicpm-pd-server-provenance.json -- \
  python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113 \
  2>&1 | tee /tmp/minicpm-pd-server.log
```

运行 24x180 production profile 并分析：

```bash
python benchmarks/minicpmo/continuous_av.py \
  --url ws://127.0.0.1:8113/v1/realtime \
  --users 24 --duration-s 180 --workload-profile production --seed 20260915 \
  --connect-stagger-s 0 --admission-timeout-s 90 \
  --post-stream-s 120 --close-timeout-s 60 --gpus 0 1 2 3 \
  --media /path/to/omni_duplex1.mp4 --loop-media \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --out /tmp/minicpm-pd-24x180.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-server.log \
  --server-provenance-json /tmp/minicpm-pd-server-provenance.json \
  --run-json /tmp/minicpm-pd-24x180.json \
  --out /tmp/minicpm-pd-24x180-analysis.json
```

每个容量点都必须重启服务。诊断开关只能用于通过 `clean_server.py --allow-diagnostics` 启动的独立测试。
