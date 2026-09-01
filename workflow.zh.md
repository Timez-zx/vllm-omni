# MiniCPM-o 原生全双工 P/D 工作流

## 目标

测量单机四 GPU 在持续音视频输入下可长期维持的 session 容量。模型质量不在本实验范围内。

## 阶段一：serving 设计

```text
每个用户建立一个 WebSocket
  -> 每 200 ms 上传 PCM16 音频，每秒上传一帧视频
  -> 聚合成一个原生 1 秒 model unit
  -> Thinker-P 将新增 AV unit 追加到 session KV lineage
  -> 只把新增的 block-aligned KV suffix 传给 Thinker-D
  -> Thinker-D 完成有限 decode，并决定 listen/speak
  -> D 输出回写到下一轮 P lineage
  -> speak 时运行 Talker -> Code2Wav
```

- 应用层维护 session 和媒体 buffer；engine KV 是可丢弃的执行状态。
- 不同 session 独立进入 engine；应用层没有全局 gate，也不做跨用户 batch。
- `D(i-1)` 必须先于 `P(i)` 完成，因为 D 输出属于下一轮 Thinker lineage。下一轮 Thinker 不等待 Talker 或 Code2Wav。
- 估计 context 达到 36,000 tokens 时重开 lineage，只保留 system/reference input、上一完整 AV unit 及其确认后的 Thinker 输出、当前 unit。模型上限为 40,960 tokens。

部署：

| Stage | GPU | 作用 |
|---|---:|---|
| Thinker-P | 0 | 增量多模态 prefill |
| Thinker-D | 1 | 有限自回归 decode |
| Vision Encoder | 2 | 无状态 HD4 视频编码 |
| Talker + Code2Wav | 3 | 生成语音 code 和波形 |

每个到达的视频帧都会提交给 GPU 2。Encoder 同时只执行一个 RPC；执行期间的新帧进入下一 microbatch，不丢帧，也不限制每个 session 的 lookahead。完成的 embedding 保存在 CPU，正式 P 请求消费时才搬回 GPU 0，避免 backlog 占满 Thinker 显存。正式 P 等待对应 embedding ready；只有编码失败时才使用 request-local fallback。

这是容量诊断配置。生产部署仍需根据内存预算增加显式 backpressure 或丢帧策略，但不能用静默 fallback 重编码掩盖真实下游吞吐。

P/D 使用 `NixlDeltaPushConnector`。D 保留 prefix KV，每个 unit 只接收新增 KV suffix。
P/D 都使用 FP8 E4M3 存储 KV。MiniCPM-o 4.5 没有提供 KV scale，因此 P/D 使用相同的确定性默认 scale，不独立校准出两种不兼容的表示。实测 P/D cache 容量从 `497,504/490,416` 提高到 `995,024/980,848` tokens。

## 阶段二：workload 与容量判据

- 循环真实 MP4：960×540 视频、对齐的 16 kHz mono 音频和 reference audio。
- 音频每 200 ms 到达，视频为 1 FPS；模型以 1 Hz 消费一个 1 秒 unit。
- `max_slice_nums=4`，每个视频 unit 使用 HD4 路径。
- 运行 360 秒，session 相位在 `[0, 1 s)` 随机分布，seed `20260839`。
- 每个容量点都冷启动服务，再做一次单用户 JIT warm-up。不能在多个容量点间复用同一服务，因为残留的 session close 状态会污染下一次测量。
- 19 用户包含 6,840 个 unit，每个 session 发生两次 context rollover；20 用户应完成 7,200 个 unit。
- 最新根因实验使用 24 用户、360 秒、seed `20260904`，应完成 8,640 个正式 unit；另用随机历史预热覆盖长 context 与 rollover。
- 使用 NVIDIA DCGM profiler 以 1 Hz 采集从输入开始到最后一个 D 完成的硬件计数器。`SM active`、`SM occupancy`、`Tensor active` 和 `DRAM active` 表示实际硬件活动；NVML GPU busy 只用于对照，不作为饱和判据。

单轮 RTF 只用于诊断 jitter：

```text
unit RTF = 1000 ms / stage service time
```

容量使用长期处理速度：

```text
stream RTF = 已完成的 1 秒输入总量 /
             从首个 input-ready 到最后一个 D 完成的 wall time
```

只有全部 session 的 `stream RTF >= 1`、所有 D unit 完成且没有用户失败时，容量才通过。单轮超时后如果能追回 backlog，只属于 jitter。

## 阶段三：最终测量

### 容量边界参考

以下 19/20 用户边界来自前一版有界 arrival cache。最新无界预编码路径尚未重新扫描 19–23 用户，因此它是参考边界，不是当前代码的重新认证结果。

| 用户数 | 结果 | 完成的 D unit | 失败用户 | Stream RTF min/p50/p95 | Cycle RTF min/p50/p95 |
|---:|---|---:|---:|---|---|
| 19 | 通过 | 6,840/6,840 | 0 | `1.002/1.002/1.003` | `0.995/1.003/1.018` |
| 20 | 失败 | 6,162/7,200 | 17 | `0.777/0.790/0.810` | `0.772/0.772/0.781` |

20 用户时，每个 session 平均只完成 308.1 个 unit，最终积压平均为 80.6 秒。D 的有效完成速度约为 15.75 个 1 秒 unit/s，低于所需的 20 unit/s。

| 测试 | P service p50/p95/p99 | D service p50/p95/p99 | P/D 端到端 p50/p95/p99 | 等待上一轮 D p50/p95/p99 |
|---|---|---|---|---|
| 19 用户 | `192/736/1,029 ms` | `237/848/1,175 ms` | `473/2,190/2,770 ms` | `0.177/1,117/1,471 ms` |
| 20 用户 | `656/1,341/1,482 ms` | `440/946/1,269 ms` | `2,302/3,693/4,123 ms` | `1,152/1,903/2,132 ms` |

### 20 用户失败点的 P/D 真实硬件活动

统计覆盖从输入开始到最后一个 D 完成的 392 个一秒采样点。百分比是 profiler 活动比例，不是显存占用或进程驻留时间。

| 计数器 | Thinker-P mean/p95/max | Thinker-D mean/p95/max |
|---|---:|---:|
| SM active | `27.0/38.1/43.2%` | `15.5/36.1/46.9%` |
| SM occupancy | `3.9/5.5/6.4%` | `1.9/4.5/6.0%` |
| Tensor active | `18.7/27.7/32.1%` | `1.3/2.4/2.8%` |
| DRAM active | `8.8/15.7/17.2%` | `13.9/32.6/43.5%` |
| PCIe TX | `44.3/98.7/230.4 MiB/s` | `3.1/5.9/6.9 MiB/s` |
| PCIe RX | `69.3/89.2/101.9 MiB/s` | `51.7/108.1/241.4 MiB/s` |
| 功耗 | `238/295/326 W` | `138/200/231 W` |
| NVML GPU busy | `36.8/100/100%` | `19.2/51.5/73%` |

P 的 NVML busy p95 可以达到 100%，但同期 SM active p95 只有 38.1%，occupancy p95 只有 5.5%。因此 NVML busy 会造成“GPU 已饱和”的假象。P/D 的 SM、Tensor Core、DRAM、PCIe 和 600 W 功耗上限都没有饱和。

失败测试中没有 OOM、preemption 或 recomputation，D 仍命中几乎完整的 prefix。因此失败原因不是 KV 容量，也不是链路带宽，而是流水线效率：循环依赖 `D(i-1) -> P(i) -> D(i)`、有限 request/control 固定开销以及不规则的小 P/D batch，使两个 GPU 在 burst 之间存在空闲。20 用户时，等待上一轮 D 从可恢复 jitter 变成持续等待，backlog 开始增长，但 GPU 原始算力仍未被充分利用。

### 最新 24 用户根因实验

先消除 Encoder 混杂：所有到达帧都预编码，ready embedding 存 CPU，正式请求不再因为 cache 淘汰而回到 P runner 重编码。短测完成 720/720 帧，全部命中 arrival cache；长测无 OOM、无 formal fallback，Encoder ready p99 为 `608 ms`，正式 P 等待 embedding p99 仅 `1 ms`。

24 用户长测发送 8,640 个正式 unit，D 完成 7,300 个；所有 session 的长期 RTF 都低于 1。

| 指标 | 结果 |
|---|---:|
| Stream RTF min/p50/p95 | `0.776/0.784/0.794` |
| 最终 backlog p50/p95 | `83.6/86.6 s` |
| P/D 端到端 p50/p95/p99 | `2,479/3,219/3,604 ms` |
| 等待上一轮 D p50/p95/p99 | `1,252/1,694/1,921 ms` |
| P service p50/p95/p99 | `778/1,197/1,368 ms` |
| P 完成到 D 完成 p50/p95/p99 | `442/794/1,029 ms` |
| D service p50/p95/p99 | `479/828/1,064 ms` |

P 的吞吐证据是决定性的：正式窗口为 `390.7 s`，P runner 在其中执行了 `383.3 s`，duty 为 `98.1%`。P 共形成 980 个 batch，平均每批 `7.52` 个请求、`1,599` tokens、运行 `391 ms`，实际只能调度约 `18.8 unit/s`，低于输入的 `24 unit/s`。D runner duty 只有 `56.1%`，因此 D 和 handoff 会增加单轮延迟，但不是 24 用户的容量上限。

| DCGM 计数器 | Thinker-P mean/p95/max | Thinker-D mean/p95/max |
|---|---:|---:|
| SM active | `30.7/38.4/41.6%` | `15.8/24.9/30.4%` |
| SM occupancy | `4.3/5.3/5.8%` | `1.9/3.1/4.1%` |
| Tensor active | `21.9/28.2/30.2%` | `1.3/1.9/2.3%` |
| DRAM active | `7.8/9.5/10.8%` | `14.5/22.7/28.5%` |
| 功耗 | `256/302/313 W` | `143/173/194 W` |

`P runner duty=98.1%` 与 `SM active mean=30.7%` 不矛盾：前者表示 engine 几乎一直有 P batch 在执行，后者表示这些小增量 batch 只利用了约三成 SM 时间。当前 P 执行路径已无空闲容量，但每个 batch 的硬件效率仍低。

## 阶段四：结论

Tail 来自模型依赖：

```text
D(i-1) feedback -> P(i) -> KV handoff -> D(i) -> feedback -> P(i+1)
```

Context 变长会增加 P/D service time。P 低于输入速率后，`D(i-1)` 反馈变晚，下一 unit 的等待从 jitter 变成持续 backlog。这个等待是 P 容量不足的结果，不是第三个独立执行阶段。

最新实现已经排除 Encoder cache 淘汰、P 侧视觉重编码、GPU embedding 泄漏和 KV/OOM。24 用户的首要瓶颈是 Thinker-P runner 吞吐：它在时间上已饱和，但小增量 batch 的 SM、Tensor Core 和显存带宽利用率都偏低。研究重点应是提高长 context、小增量 P batch 的执行效率；优化 D/handoff 只能降低单轮延迟，不能单独补足 `18.8 -> 24 unit/s` 的吞吐缺口。

前一版实测边界为 19 用户通过、20 用户失败；当前代码确认 24 用户失败，但精确边界仍需重新扫描 19–23。FP8 KV 只解决驻留容量，不减少模型计算。默认 scale 的 FP8 KV 用于生产前仍需独立评估质量。

## 阶段五：复现

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
MINICPMO45_LOG_PREP_DIAG=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113

python benchmarks/minicpmo/continuous_av.py \
  --users 24 --duration-s 360 --phase-window-s 1 --seed 20260904 \
  --connect-stagger-s 0.5 --post-stream-s 30 --close-timeout-s 180 \
  --loop-media --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --gpus 0 1 2 3 \
  --out /tmp/minicpm-pd-cpu-cache-u24x360.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-cpu-cache-u24-server.log \
  --run-json /tmp/minicpm-pd-cpu-cache-u24x360.json \
  --out /tmp/minicpm-pd-cpu-cache-u24x360-analysis.json
```

重新扫描容量时依次使用 19–23 用户，并在切换用户数前重启服务。客户端运行期间，用下面的命令采集 profiler 计数器：

```bash
sudo dcgmi dmon \
  -e 1001,1002,1003,1004,1005,1009,1010,155,203,204 \
  -i 0,1,2,3 -d 1000
```
