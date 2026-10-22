# 第二时域模块计时与性能排查

本轮仅增加诊断计时、调整输出和断点兼容规则。model.py、数据、优化器、单步执行方式、数值精度、确定性配置和依赖版本均未改动。没有启用 CuPy、多步改写、kernel 融合、torch.compile 或 AMP。

## 22.114 s 前向 / 3.339 s 反向：v9 深入排查

**判断：这一比例可以由现有执行路径解释，不能据此认定梯度或网络算错；但也不能把它当作 4090 上此网络的正常效率。首要嫌疑是 PyTorch 2.5.1 的确定性膨胀卷积回退及其偏置填充。** 以下区分仓库已确认事实、底层源码推断和待服务器验证的部分。

### 补充论文效率表后的核对

用户补充并确认表头：**12 类数据集、0.5 s、LOSO**；SpikeTFNet 为 3.016 M 参数、32.016 M FLOPs、836.362 K SOPs、2.7350 ms/trial、365.63 trials/s。FLOPs 表征传统稠密计算，SOPs 表征脉冲时域分支的脉冲触发累加，后者在 1,800 个未见 trial 上取平均。

已复查本地 `SpikeTFNet_8.pdf`（原文第 8 页）：论文定义参数量为可训练参数之和；推理采用 batch=1、GPU warm-up，每次 forward 前后 CUDA 同步，排除 EEG 预处理、数据加载及磁盘 I/O；吞吐量按 1000/平均毫秒时延计算。原实验工作站为 RTX 3090 Ti、CUDA 11.6。故 1000/2.7350≈365.63，两列是同一测量的换算，不能作为两个独立效率证据，也不能与训练 batch=128 的均摊耗时直接相比。完整效率表不在当前第 1–8 页内，数据集、窗口及 1,800 trial 范围来自用户补充的表头与注释。

**当前 40 类工程实现的可训练参数实算为 4,584,088（4.584088 M）；不能以它高于 12 类表中的 3.016 M 判定复现偏差。** 这是两个数据集的不同输入/输出配置。当前仍有已获同意、尚未由作者配置确认的结构默认值，参数规模对齐需要对应的 40 类配置或效率表。

| 当前实现部分 | 可训练参数 |
| --- | ---: |
| 频域分支 | 4,376,648 |
| 其中：频域分类头（隐藏宽 40×6=240） | 2,429,560 |
| 其中：频域特征投影（10080→128） | 1,290,624 |
| 时域分支 | 85,544 |
| 其中：第二模块 | 50,304 |
| 融合模块 | 121,896 |
| 完整模型 | 4,584,088 |

分支参数行可相加得总量，“其中”行不重复相加。参数多的频域并非当前耗时最多的分支；参数数量和运行时间不是正比。原文 12 类使用 8 通道、0.25 Hz 频谱分辨率，当前 40 类使用 9 通道、0.2 Hz；按相同端点约定，实虚拼接长度为 448 与 560，展开特征长度及分类头输出规模均不同，不能只对比总参数量。图 2 和公式 (9)/(10) 确有并列分类头、特征投影；不为凑 12 类表的总量删除 40 类模型结构。Ts 和时间窗不改变当前模型参数量。

500 ms、Ts=4 时，当前第二模块稠密卷积有 24.576 M MAC/trial；按一乘一加算两次 FLOP 为 49.152 M，尚未计其他层。这是在 40 类输入下将时域卷积视为实际稠密计算的分析值；论文效率表区分传统稠密 FLOPs 与脉冲 SOPs，且对应 12 类输入，故不能拿 49.152 M 直接与 32.016 M 比较，更不能据此认定实现错误。后续对齐还要明确一 MAC 算几 FLOP、Ts 在等价稠密计数中的累计规则。

论文按 `SOP_l = fr_(l-1) × FLOP_l` 估计脉冲突触操作并单独报告；836.362 K 是已训练 12 类模型在 1,800 个未见 trial 上平均得到的统计量，不等于普通 GPU 的实际执行指令或 CUDA kernel 数。当前采用稠密 Conv1d，不会因输入含大量零脉冲就自动省去稠密卷积。不能把该 SOP 值代入当前 40 类模型；需使用本模型的已训练权重、层输入发放率、实际形状和对应 LOSO 测试 trial 统计。

此外，若按完整 LOSO 的 8160 trial 均摊，22.114 s/8160≈2.710 ms/trial，虽然数值接近论文 2.735 ms，但前者仅为 T2 卷积训练前向、batch 并行均摊且带诊断；后者是完整模型单 trial 推理，数值接近不构成效率一致的证明。本轮没有用未训练模型在 CPU 上产生所谓可对比的推理速度或 SOP 指标。

新日志增加 `MODEL PARAMETERS`，在每个任务开始时打印实际总量、分支和子模块参数，并保存到现有 provenance.json 的 model_parameters，便于核对服务器是否与当前配置一致。模型结构保持不变。

### 实际形状、调用次数和配置

| 配置/实现 | 当前值及影响 |
| --- | --- |
| 依赖 | requirements.txt 固定 torch 2.5.1+cu121；服务器实际版本由新日志确认 |
| 确定性 | `deterministic=True`；同时启用全局确定性算法和 cuDNN deterministic |
| cuDNN benchmark | False；没有自动搜索最快算法，但这不是下面回退的直接开关 |
| 输入和精度 | FP32，9 电极；卷积输入为 `[B,64,L]`；L=125/250 分别对应 500/1000 ms |
| 三个残差块 | 每块 `Conv1d(k=3,d=1/2/4)→BN→LIF→Conv1d(k=1,d=1)→BN→LIF→Dropout→残差相加` |
| 卷积细节 | 输入/输出通道均 64，groups=1，stride=1，默认 bias=True；padding 保持长度不变 |
| 仿真步 | Ts=4；三块在每一步重复调用，未将 EEG 采样点数误作 Ts |
| 每 batch 调用 | 6 Conv×4=24 次卷积前向，6 LIF×4=24 次 LIF 前向 |
| 正式 LOSO 每 epoch | 34×6×40=8160 个训练 trial；batch 128，无 drop_last，63 个完整 batch + 1 个 96 trial 尾 batch |
| 每 epoch 卷积调用 | 64×24=1536 次；22.114 s 对应全层混合平均约 14.40 ms/次，不能据此给每层分配相同时间 |

检查未发现通道/时间轴交换、意外扩大 padding、额外 Ts 循环、把全部目标被试加入训练或在 T2 中显式调用 `.cpu()`/`.item()`。N 和 M 决定独立任务数量，不直接扩大单任务 batch。两张卡各训练一个独立任务，并非同一 batch 分成两半。

膨胀率 1/2/4 的有效感受野为 3/5/9，但各自仍只有 3 个卷积核权重点；不是 dilation=4 就有四倍理想乘加数。第二模块每 trial 的卷积乘加数为 `3×Ts×L×64²×(3+1)`，500 ms 的整个 epoch 约 200.54 G MAC，1000 ms 约 401.08 G MAC。硬件峰值不能用来直接推断这个小算子密集工作负载的时间。

### 首要原因：确定性膨胀卷积绕过 cuDNN

PyTorch **v2.5.1** 的 `use_cudnn` 对 contiguous 格式检查 `deterministic && is_dilated()`，满足即返回 false。Conv1d 在该选择路径中扩展成 2D 表达；当前普通布局下，第二、第三块的 d=2/4 卷积会进入 `SlowDilated2d`。第一块 d=1 和三个逐点卷积不受这一条件阻挡。[版本固定的后端选择源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Convolution.cpp#L422-L430)

因此，在规定的版本/配置下，后端回退条件已经成立；**服务器是否确实使用这一版本、是否由这两层占据主要时间，需要实际 profiler 确认**。新日志应重点看 `1.dilated.0`、`2.dilated.0` 的 `T2 BACKEND` 是否出现 `aten::slow_conv_dilated2d`，并与 `0.dilated.0` 及全部 `pointwise.0` 对比。

不能只打开 benchmark 就假定解决：这两层已在选择 cuDNN 之前被拒绝。只关闭 `cudnn.deterministic`、却保留全局 deterministic algorithms，也不能保证退出该路径，因为后端选择对这两个标志取 OR。上游后来通过 PR #154292 移除了相关限制，说明必须核对具体 PyTorch 版本；本次未升级依赖。[上游修复](https://github.com/pytorch/pytorch/pull/154292)

### 为什么前向会比反向慢：偏置填充在前向独有

v2.5.1 的 CUDA 慢速膨胀实现逐 batch 元素处理；存在 bias 时，前向还逐输出通道填充，再执行 im2col 和 GEMM。这里的偏置默认存在，每次完整 batch 的单次膨胀卷积前向就有 `128×64=8192` 次通道填充。两层 d>1、Ts=4、8160 trial，对应 **4,177,920 次偏置通道填充/epoch**；这是根据源码循环计算的操作次数，不是实测 kernel 数。

反向走输入梯度、权重梯度的 GEMM/im2col/col2im 及偏置梯度归约，没有对应的逐输出通道初始化循环。因而数学上“反向通常有更多乘加”不能推出这条实现“反向必定更慢”。这是一条能直接解释前向异常突出的证据链。[CUDA 慢速膨胀实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/NaiveDilatedConvolution.cu#L247-L375)

还核查了偏置填充的同步语义：同设备 Tensor 版本的 fill_ 转入 copy_，并非每次通过 value.item() 取回 CPU。因此，本次没有把这四百多万次填充错误解释为四百多万次显式标量同步。应检查新采样中的 fill_/copy_/select 次数、相关 kernel 数量与 CPU self 时间；同时列出 item/_local_scalar_dense（如果出现）以验证真正的标量回传。[同版本 Fill.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Fill.cpp#L50-L62)

### 其他影响及现有数字的边界

- 单步 Torch LIF 和 Ts 循环确实增加调度，但这条日志 LIF 两项合计 0.284 s；在当前归属口径下，应优先核查卷积回退，不能将主要原因直接归为 LIF。
- 脉冲张量虽然可能很多位置为 0，当前仍使用普通稠密 Conv1d，没有按非零脉冲跳过乘加的稀疏算子。
- 学习率、Adam 权重衰减和辅助损失系数影响学习过程，但优化器/loss 不在这些卷积计时边界内；不能用它们解释前向中的填充循环。BN/Dropout、注意力头数、频域编码深度也不直接改变这里的六个 Conv 调用次数。
- 卷积宽度 64、Ts=4、batch 128 放大上述循环；缩短窗口未必按比例降低时间，因为偏置填充次数主要由 trial 数、通道数、仿真步数决定。
- TF32 的 matmul/cuDNN 开关分别控制不同路径。项目没有主动修改这两个默认值，新日志记录实际值。不能将 FP32 tensor 简化为所有底层运算都采用同一乘法精度。
- 双卡独立任务会同时向各 GPU 提交大量小操作，CPU/驱动提交压力可能更高；24 GB 显存大不代表这类工作负载能充分利用算力。没有服务器 profiler/硬件数据，暂不判断降频、共享 CPU 或其他进程争用。
- 旧四项使用 CUDA Event，同流开始事件排在上游工作之后，不是 CPU 秒表漏测异步操作。层内部 Event 区间仍可能含 CPU 提交间隙；反向按节点包围会排除节点之间的空隙，与整个层前向区间的粒度不同。22.114/3.339≈6.62 不能当作纯 kernel 计算量比值。[PyTorch 2.5.1 CUDA 语义](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst)

### v9 新增证据与读取方法

每个 epoch 继续输出四项总计，并增加 12 行 `T2 LAYER`：

```text
T2 LAYER 1.dilated.0 conv k=3 d=2 bias=True shape=128x64x125 F=...s B=...s FHost=...s BHost=...s Fn=256 Bn=... Fmax=...s Fstep=[...,...,...,...]s Bstep=[...,...,...,...]s Bshared=...s
```

这是字段示意，不是服务器实测数值。`0/1/2` 对应第一/二/三块；`dilated/pointwise` 对应膨胀/逐点子层；末尾 `.0` 是卷积，`.2` 是 LIF。`shape` 是首个输入形状，尾 batch 形状由详情中的 batch_size 区分。

F/B 为跨 batch 累计秒数。Fstep/Bstep 按调用顺序对应 Ts1..Ts4；参数 AccumulateGrad 跨步共享，单列 Bshared，不硬归给第一次调用。Fn 是层调用数，Bn 是反向图节点执行数，二者不是同一单位。Fmax 是单次层调用的最大区间，不是独立 trial 时延。Host 是主机提交区间，包含调用/调度/可能等待，不能与 CUDA 秒数相加。

细节文件 `details/t2_diagnostics.json` 按 epoch 保存层、仿真步、反向节点名称、batch size、首 batch/后续 batch、是否启用 profiler、累计/最大区间；无逐 batch 海量文件。该信息也随原断点历史提交，恢复时按成功提交 epoch 重建。

`t2_profile_batch=2` 默认在**每个任务每次启动/续训后的首个新 epoch**对第 2 个 batch 启用一次 PyTorch profiler；不足两个 batch 时采样末 batch。内部算子在 T2 层/节点 record_function 标记下归属，只导出压缩汇总。之后的 epoch 不重复采样；设为 0 关闭采样，time_branches=False 则全部诊断关闭。

| 日志 | 用途 |
| --- | --- |
| T2 RUNTIME | worker 实际版本、GPU、确定性、TF32、CPU 线程数及相关环境变量 |
| T2 LAYER | 每 epoch 定位慢块、慢层、仿真步差异；与四项总计可核对 |
| T2 BACKEND | 实际调用的 cudnn/slow_conv 等算子，避免只猜配置 |
| T2 SAMPLE | 一个 batch 的底层操作次数、CPU self 和直接关联的 kernel 累计秒数；明确保留 fill_/copy_/select/item 等线索 |
| T2 KERNEL | 每层每方向耗时最高的 kernel 名称、次数、累计秒数；完整列表在详情文件 |

Profiler 中只累加 CPU self 和直接关联 kernel，避免把 conv1d→convolution→slow_conv 等父子 inclusive 时间重复相加。LIF 反向进一步拆到 sigmoid、mul、sub/add 等实际算子；卷积数据/权重梯度若由同一原生调用合并，不人为编造它们各自的独立秒数。

采样会增加 CPU/内存/事件开销，profiler 收尾也可能同步；**采样 epoch 的总时间不是未插桩基准**。详情用 profiled 标记分离该 batch。Kernel 数据不可用时标记 NA，不把 CPU 时间冒充 GPU 时间。建议提供采样 epoch 和再后面 1–2 个无重型采样 epoch 的日志，用于排除采样干扰；事件 hooks 本身仍存在。

### 保留变量和下一步判据

本次没有关闭确定性、修改 bias、改变 dilation/Ts/宽度/batch、更换 Torch/SpikingJelly、重写 LIF 或启用混合精度。改变这些设置会影响后端、数值轨迹或模型，不能混入这次诊断。

若 d=2/4 明显慢，且 BACKEND=slow_conv、fill_/copy_ 次数与循环估计一致，优先验证后端回退及偏置填充；若全部卷积同样慢，或实际都使用 cuDNN，则检查 Host/设备区间差、kernel 分布、实际环境与双任务资源竞争；若主要只有首个 batch/step 异常，检查初始化和采样开销。后续优化应在独立对照实验中验证速度、输出/梯度与准确率，而不是直接改变当前训练轨迹。

## 新计时范围

`config.py` 中 `time_branches=True` 开启诊断，普通模型和包含第二时域模块的消融组共用同一实现。

每个 epoch 日志先显示以下四项累计秒数，v9 另附上面的细粒度诊断，不显示平均 trial 时间：

```text
T2 Conv Forward=...s | T2 Conv Backward=...s | T2 LIF Forward=...s | T2 LIF Backward=...s
```

| 项目 | 范围 |
| --- | --- |
| Conv Forward | `temporal.blocks` 内全部 `Conv1d`：膨胀卷积与逐点卷积前向 |
| Conv Backward | 对应卷积的输入/参数梯度与参数梯度累积节点 |
| LIF Forward | 同一范围内全部 `LIFNode` 的充电、发放及膜电位重置 |
| LIF Backward | 对应 LIF 的替代梯度、膜电位跨步依赖与实际参与 loss 的重置梯度 |

所有堆叠残差块、所有仿真步、所有训练 batch 分别累计到四项。默认 3 个残差块，每块 2 个卷积、2 个 LIF；4 个仿真步对应每个 batch 各 24 次前向调用。**调用次数不是 CUDA kernel 数量**。第一模块、注意力中的卷积/LIF 不计入；BN、Dropout、残差相加、融合、损失、优化器、监测和保存也不计入。四项之和不代表整个第二模块或整个 epoch 的总时间。

前向用层的 pre/post hook 包围；反向用该层所创建、且实际参与本次 loss 的 autograd 节点 pre/post hook 包围。输入上游图被排除；LIF 除输出脉冲外还记录重置后的 `v` 图，防止漏掉后续仿真步才使用的状态梯度。没有修改计算图或使用 full-backward module hook 改写输入视图。

CUDA 上将开始/结束 Event 记在当前执行流，epoch 末等待后解析 `elapsed_time`；CPU 使用 `perf_counter`。不在每个卷积或 LIF 内手工插入同步、`.item()`、CPU 拷贝；v9 的 profiler 采样收尾可能同步。诊断事件和 Python hooks 本身有开销；Event 区间仍可能包含主机提交间隙、设备争用或调度空闲，不能等同于 profiler 的 kernel 累加时间。

沿用 `training.log`、`details/history.csv`、任务 `report.md` 和已有组级/批次汇总。CSV/报告继续保留对应样本次数上的均摊值，日志不显示。`spectral_only` 四项为 0；关闭计时时，存在的第二模块显示 NA。

## 排查结果

### 逐时间步调用与 kernel 调度

当前 [模型](../model.py) 的 LIF 使用 `step_mode="s"`、`backend="torch"`，`TemporalBranch.forward` 显式循环仿真步。第二模块的普通 Conv1d、BN、LIF 在每步重复执行，没有跨时间步的卷积批处理，也没有 CuPy 多步神经元后端。

因此，当前实现具备产生大量小 kernel 提交的条件。SpikingJelly 论文将这种调用开销作为融合加速的动机，并区分无状态层的时间维并行与有状态神经元的 CUDA 融合。[SpikingJelly 论文的加速章节](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)

本项目安装的 0.0.0.0.14 仍对部分单步充电/重置函数使用 TorchScript，不能称为“完全没有任何局部融合”；但它没有启用文档介绍的 CuPy 多步加速路径。[对应版本官方说明](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/activation_based_en/11_cext_neuron_with_lbl.html)

这是一项已确认的执行方式和潜在瓶颈。最新四项统计显示卷积前向突出，因此 v9 优先追查上面的确定性膨胀卷积实现；本次不更改这些配置。

### CUDA 异步及同步归属

| 检查位置 | 发现 | 归属判断 |
| --- | --- | --- |
| 第二模块模型代码 | 无显式 `synchronize()`、`.item()` 或 `.cpu()` | 未发现第二模块内集中等待之前 GPU 工作的显式同步点 |
| 旧模块计时 | 已用 CUDA Event，并在 epoch 末同步后读数 | 不是只测 Python 提交时间的普通 CPU 秒表 |
| 新四项计时 | 同流 Event 配对，反向节点自身 pre/post 配对 | 不把相邻 BN/其他节点和节点间间隔直接归入卷积或 LIF |
| 每 batch 的 loss 有限值判断 | `if not torch.isfinite(loss)` 隐含 CUDA 标量转布尔等待 | 在整体前向完成后，不在第二模块计时区间内部；保留未改 |
| 梯度有限值检查 | `check_gradients=True` 时存在同类等待；当前默认 False | 在 backward 后，保留原开关 |
| epoch 边界和计时汇总 | 显式 CUDA 同步；epoch 指标 `.item()` 和参数有限值判断 | 位于算子区间外；原同步点保留未优化 |
| 监测/测试/保存 | 标量读取及 CPU 数据拷贝可能同步 | 不属于这四项训练算子时间 |

PyTorch 说明 CUDA 操作通常异步执行，准确测量需要同步或 CUDA Event；同一流操作按提交顺序执行。开始事件也排在之前工作之后，因此之前工作的排队等待不会直接成为这一对事件之间的时间。[PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)

仍需区分“事件时间正确覆盖了 GPU 时间线区间”和“区间全部用于计算”：如果 Python/hook 调度使下一项工作迟迟未提交，GPU 会在事件之间空闲。旧版反向连续区间尤其可能包含相邻节点之间的提交间隙，新版用节点 pre/post 缩小了这一归属范围；节点内部的调度间隙仍可能存在。不能用 CPU 冒烟耗时判定双 4090 上的实际 kernel 占比。

## 断点与实验可比性

清单升级为 v9。修改前 v8 的训练、评估和计时辅助文件需通过原归档及固定摘要认证，数据、模型、科学参数仍严格校验；更早版本的已认证升级规则继续保留。模型/优化器状态、RNG 和 DataLoader 顺序沿用原恢复协议。

保持原结果目录、训练配置和 `resume=True`，停止旧进程、同步代码后，继续执行 `python train.py` 或 `python -m experiments.ablation`。已有 v8 四项计时保留；旧 epoch 没有细粒度数据，不从总计拆分补造。旧日志原文保留；新日志附细节，新增一个详情 JSON。已完成任务不为补测而重训。

本轮计时边界、hook 数量与旧版不同，不宜将两个版本的绝对耗时直接相减解释为模型速度提升。本轮没有进行性能优化。所有本地检查均为 CPU；CUDA Event 路径用可控事件模拟验证，真实服务器耗时仍待测量。
