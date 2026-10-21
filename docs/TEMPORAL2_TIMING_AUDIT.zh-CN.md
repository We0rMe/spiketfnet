# 第二时域模块计时与性能排查

本轮仅增加诊断计时、调整输出和断点兼容规则。model.py、数据、优化器、单步执行方式、数值精度、确定性配置和依赖版本均未改动。没有启用 CuPy、多步改写、kernel 融合、torch.compile 或 AMP。

## 新计时范围

`config.py` 中 `time_branches=True` 开启诊断，普通模型和包含第二时域模块的消融组共用同一实现。

每个 epoch 日志只显示以下四项累计秒数，不显示平均 trial 时间：

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

CUDA 上将开始/结束 Event 记在当前执行流，epoch 末等待后解析 `elapsed_time`；CPU 使用 `perf_counter`。不在每个卷积或 LIF 内插入同步、`.item()`、CPU 拷贝。诊断事件和 Python hooks 本身有开销；Event 区间仍可能包含主机提交间隙、设备争用或调度空闲，不能等同于 profiler 的纯 kernel 累加时间。

沿用 `training.log`、`details/history.csv`、任务 `report.md` 和已有组级/批次汇总。CSV/报告继续保留对应样本次数上的均摊值，日志不显示。`spectral_only` 四项为 0；关闭计时时，存在的第二模块显示 NA。

## 排查结果

### 逐时间步调用与 kernel 调度

当前 [模型](../model.py) 的 LIF 使用 `step_mode="s"`、`backend="torch"`，`TemporalBranch.forward` 显式循环仿真步。第二模块的普通 Conv1d、BN、LIF 在每步重复执行，没有跨时间步的卷积批处理，也没有 CuPy 多步神经元后端。

因此，当前实现具备产生大量小 kernel 提交的条件。SpikingJelly 论文将这种调用开销作为融合加速的动机，并区分无状态层的时间维并行与有状态神经元的 CUDA 融合。[SpikingJelly 论文的加速章节](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)

本项目安装的 0.0.0.0.14 仍对部分单步充电/重置函数使用 TorchScript，不能称为“完全没有任何局部融合”；但它没有启用文档介绍的 CuPy 多步加速路径。[对应版本官方说明](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/activation_based_en/11_cext_neuron_with_lbl.html)

这是一项已确认的执行方式和潜在瓶颈，尚不能证明它造成服务器观察到的 90% 占比。卷积前向、卷积反向、LIF 前向、LIF 反向的新统计需由服务器提供。确定性卷积算法、形状和膨胀率也可能影响卷积速度，本轮不更改这些配置。

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

清单升级为 v8。修改前 v7 的训练和计时辅助文件需通过原归档及固定摘要认证，数据、模型、科学参数仍严格校验；更早版本的已认证升级规则继续保留。模型/优化器状态、RNG 和 DataLoader 顺序沿用原恢复协议。

保持原结果目录、训练配置和 `resume=True`，停止旧进程、同步代码后，继续执行 `python train.py` 或 `python -m experiments.ablation`。旧 epoch 没有四项算子时间，不能由旧模块总计拆分，显示缺失及覆盖率。旧日志原文保留；新日志、更新后的 CSV/报告只显示当前四项。已完成任务不为补测而重训。

本轮计时边界、hook 数量与旧版不同，不宜将两个版本的绝对耗时直接相减解释为模型速度提升。本轮没有进行性能优化。所有本地检查均为 CPU；CUDA Event 路径用可控事件模拟验证，真实服务器耗时仍待测量。
