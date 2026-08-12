# 节拍引擎(Tick Engine):按周期性重新设计推理引擎

**日期**: 2026-08-12 · **分支**: `temporal-batching` · 前置:`RESULTS.zh.md`

## 一、诊断:vLLM 引擎是"时间盲"的

源码证据(vllm 0.26.0, v1):

- **主循环**(`v1/engine/core.py: run_busy_loop`):`while True: 收输入 → step()`。
  没有时钟。引擎的世界里只有"有活儿/没活儿",没有"现在几点"。
- **调度器**(`v1/core/sched/scheduler.py`):唯吞吐——每步在 token/seq 预算内
  塞满;FCFS/优先级只排"谁先来"。全文搜 deadline / pacing / periodic /
  tick / rate:**零命中**。
- **Request**(`v1/request.py`):唯一时间字段 `arrival_time`,用于排序和统计。
  没有速率、没有截止期、没有"我的输出有个播放时钟"这个概念。

这是为**非周期负载**(聊天、批 API)做的正确设计:请求何时来不可知,来了
就尽快做完。但 realtime 多模态负载的本质完全不同,而本实验的每一个补丁
(拍边界屏障、拍内子格、消费点时刻表、防空转微睡)都是在无时钟循环上
"外挂时间"——它们的必要性本身就是诊断书。

## 二、关键洞察:这类负载是同步数据流(SDF)

realtime 音视频通话负载的速率是**先验已知且恒定**的:

```
每 80ms tick,每个活跃 session:
  1 块音频输入(1280 样本@16k)  → thinker 消化
  ~2 个文本 token               → thinker 产出(25 tok/s 上限)
  1 个 codec 帧                 → talker 产出(播放速率,物理规定)
  1/4 次 vocoder 调用           → code2wav(4 帧一块)
```

推论:**每 tick 的工作集是确定的**——不需要每步做调度决策,batch 组成可以
**预先计划**;每 tick 的 GPU 时间可测且稳定——容量与延迟可以**保证**而不是
祈祷。这正是信号处理(SDF)、实时音频(JACK/CoreAudio 的 period callback)、
时间触发架构(TDMA/TTA)和 RTOS cyclic executive 的世界观。Clockwork
(OSDI'20)证明过 DNN 推理的确定性足以支撑截止期调度;这里更进一步:
负载本身还是周期的。

## 三、节拍引擎:五个组件

### 1. 全局时钟 + 每 stage 相位(脉动流水)

一个 80ms 时钟源;各 stage 锁相,相位错开形成流水:

```
T+0     thinker 步(全体 session 一批:各 2 token)
T+δ     0→1 传输(双缓冲信箱翻页)
T+φ     talker 步(全体一批:各 1 帧)
T+ψ     code2wav 步(本 tick 轮到的 1/4 session,合批 vocode)
T+80    下一拍
```

code2wav 按 session 错相(每 tick 各 1/4)——连 vocoder 负载也被抹平。

### 2. Cohort 计划:调度从"每步"降到"每事件"

调度决策只在**成员变更**时发生(session 加入/离开、轮开始/结束):产出一份
cohort plan(谁在批里、各自的 KV 槽、每 tick 配额)。之后每 tick 直接
**重放计划**,只做增量(computed+=2 等)。vLLM 的每步 schedule() 是纯
Python、每秒跑几百次;对 talker 这种小模型,调度+launch 开销是延迟大头
(本 repo 早前测过 eager talker 内核活跃占比仅 29%)。计划重放让这块
开销结构性消失。

### 3. 静态批 → CUDA Graph 逐拍重放

Cohort 不变 ⇒ batch shape 恒定 ⇒ 一张图捕获、每拍重放。终局形态:整个
tick 的三段流水编成一个 stream program(thinker step → D2D 搬运 →
talker step → vocoder),一次提交。

### 4. 双缓冲信箱替代"队列 + 轮询"

现状 0→1/1→2 传输:save 线程逐个序列化 → 共享内存 → 消费侧轮询——每跳
毫秒级可变延迟,是实测打散相位的元凶(RESULTS 第三节)。节拍化后传输
是定相的:producer 在 T+0 写 buffer k,consumer 在 T+φ 读 buffer k,
**没有队列、没有轮询、没有抖动累积**。同卡 stage 直接翻 device 指针
(colocation 已有 D2D 先例),跨卡走预排程的 async copy。

### 5. 非周期工作进"松弛槽"(slack stealing)

每 tick 的周期工作用不满 80ms,余量就是松弛槽。turn 开始的 prefill
(300~2800 token)切片塞进各 tick 的松弛里跑——**非周期工作再也不撞节拍**,
且 TTFA 变成可计算量:`ceil(prefill_tokens / slack_per_tick_tokens)` 个
tick。这就是 RTOS 的 aperiodic server,也顺手完成"输入侧每 tick 增量
prefill"(音频编码器流式化后,输入消化本身成为周期工作的一部分)。

## 四、能买到什么(用实测数据外推)

| 维度 | 现状(贪心/我们的屏障 retrofit) | 节拍引擎 |
|---|---|---|
| batch 组成 | 靠运气 / 靠屏障捞(p50 7→11) | **by construction = N_active** |
| 步频(talker) | 40.9/s(贪心 u16)~60.7/s(v1) | **恒定 12.5/s**,与 N 无关 |
| 调度开销 | 每步 Python schedule() | 每事件一次,每拍重放 |
| 抖动来源 | 调度碰撞 + 传输相位 | 仅时钟精度(µs~ms) |
| 容量语义 | 压测出的经验值 | **每拍装得下 ⇒ 每拍都达标**(硬保证) |
| 尾延迟 | 统计性(p99 靠祈祷) | 结构性消失(超载即拒绝,不排队) |
| 功耗 | 爆发-空转 | 平稳(副产品) |

我们的屏障 retrofit(E2/E3)就是这个设计的**便宜近似 + 可行性证明**:
p99 步进间隔 81.2ms 的心跳、batch p50 11、全指标不劣于贪心——都是在
"外挂时间"的劣势下取得的。原生设计把剩余的散(追赶碎步、传输相位、
调度开销)从"补丁压制"变成"结构不存在"。

## 五、代价与开放问题(诚实)

1. **异构负载**:text-only 请求、离线批任务不合拍。方案:双车道——周期
   实时道(cohort)+ 松弛槽里的 best-effort 道(Linux RT+CFS 的形状)。
2. **每 tick 时间方差**:MoE 路由、prefill 分片大小都有方差 → 按余量规划
   (如目标利用率 70%)。买保证要付的标准价格,实时系统皆然。
3. **侵入性**:async scheduling、抢占、stats、engine loop 全假设自由循环。
   这是 fork 级改动,不是补丁。因此路径必须渐进(见下)。
4. **KV 增长与 roll**:session KV 无限长与固定 cohort 计划的冲突——roll/
   压缩要变成"计划内事件"。

## 六、落地路径(渐进,每步可测)

1. **v4a Cohort 重放(可在 vllm-omni 内做)**:talker stage 成员不变时跳过
   schedule(),直接重放上拍的调度输出(增量改 computed/slots);验证
   调度开销归零、步时缩短。
2. **v4b 双缓冲信箱**:0→1 边换成定相双缓冲(去 save 线程+轮询),测
   相位散布是否归零(sched_steps.py 直方图)。
3. **v4c 输入侧节拍化**:音频每 tick 增量 prefill(编码器流式化,大活)。
4. **v5 独立 tick-engine 核心循环**:以上验证后,把自由循环替换为
   `wait_until(next_tick) → replay plan → execute`,成为真正的节拍引擎。

## 七、假设边界:依赖负载周期性,不依赖模型架构

设计的依赖是两条,都与具体架构无关:**负载周期性**(输出被固定速率消费、
输入固定速率到达——realtime 媒体 I/O 的性质)+ **速率可静态化**(每 stage
的每媒体秒工作量静态或有上界、固定形状下步时可预测——SDF 可调度条件)。

- WP1 心跳:零架构依赖。WP2/3/4:只需"速率因子已知/形状随 cohort 固定/
  stage 间速率静态"——1 token=80ms 是**配置参数**而非前提;换 diffusion
  (SoulX 32 帧/block)即"每 N 拍一 block",同一设计的粗粒度情形;MoE 只
  带来步时方差,影响余量百分比不影响结构。
- 速率天然可变的节点(thinker:每秒语音的 token 数随内容变)只需**上界**
  (25 tok/s cap + lead 缓冲)即可入网。
- **唯一打开模型的地方**:WP5 输入编码器流式化(取决于感受野结构)。
  其余全部把模型当黑盒。
- 破坏假设的模型:每输出计算量不可预测(自适应计算、流中插工具调用、
  可变长思考)——进 best-effort 车道,这正是 WP6 双车道的理由。

正确抽象:引擎接受一张**速率标注的 stage 图**(每 stage 声明每 tick 的
消费/产出/步时上界)。周期性是负载与接口的契约,不是模型假设。同一引擎
无改动可服务 Moshi 类单体全双工(12.5Hz,连传输都省)、diffusion 数字人、
视频生成(每拍 1/N 帧)。

## 八、一句话

vLLM 把时间当排序键;realtime 负载需要把时间当**调度基底**。负载的周期性
意味着调度问题可以从"每步在线决策"退化为"每事件离线计划 + 每拍重放"——
这换来的不是几个百分点,是**从统计性能到结构保证的范式差**。
