**实现规格，2026-10-19**

本说明更新前期计划中的待定项。用户已明确同意未公开参数按经验设置；代码里的 P/R/A 注释区分原文、相关作者实现和工程假设。未重新调参寻找有利的测试结果。

**原文已经明确并采用**

- Dataset 2：35 人、40 类、6 blocks，250 Hz；九通道、0.64 s 起点、0.5–1.0 s 六个窗口；四阶 Butterworth 8–64 Hz。
- 图 2：SSVEPformer 编码器 2 层；多尺度脉冲残差块 3 层。
- 第 8 页：PyTorch、SpikingJelly、Adam、lr=0.001、weight_decay=0.001、100 epochs、batch=128、dropout=0.5、seed=42、LIF tau=2、默认 Ts=4。
- 最终 epoch 的网络直接测试，不设目标被试验证，不根据目标成绩选择 checkpoint。旧计划中的“源验证集方案”不再需要，所有源 trial 均用于训练。
- ITR 分母为 Tl+0.5 s，低于机会水平置零。其他效率指标暂缓，保留后续所需输入和模型资料。

**经过同意的工程补全**

| 项目 | 本实现 | 说明 |
| --- | --- | --- |
| 频谱 | `[8,64)`；F=560；FFT×2/L | 参考 [作者相关 SSVEPformer 源码](https://github.com/YuDongPan/DL_Classifier/blob/main/Model/SSVEPformer.py)，不是 SpikeTFNet 官方参数证明 |
| 频域通道组合 | 1×1 Conv，9→18，LN 沿频谱轴 | 与相关实现一致 |
| 频域块 | 核宽 31，stride=1，same padding；F→F Linear/GELU/Dropout | 核宽为经验默认，FFN 变换轴参考相关实现 |
| 频域辅助头 | Dropout→Linear(18F,240)→LN→GELU→Dropout→Linear(240,40) | 参考相关实现；Conv/Linear 权重 std=0.01 |
| 时域 stem | 9→64，核宽 7，stride=1，same padding，BN | 无下采样，L0=L |
| 时域残差块 | 宽度均 64；核宽 3；dilation 1、2、4；pointwise=1 | 采用正文示例的膨胀率，3 块由配置 tuple 长度控制 |
| 注意力 | 4 heads，每头宽 16；gamma=0.125 | 无 softmax，不擅自加位置编码或额外 FFN |
| Q/K/V 投影 | Conv1d(kernel=1)→BN→LIF | 等价于每个 token 的线性投影；头拆分后做注意力 |
| 时域与共享投影 | Linear→LN→GELU→Dropout；维度均 128 | 门控隐藏层也为 128，保留分支投影和共享投影两个阶段 |
| 辅助权重 | lambda_f=lambda_t=0.3 | 经验值，集中在 config.py |
| LIF | threshold=1、reset=0、Sigmoid alpha=4、detach_reset=False | SpikingJelly Torch 单步后端，逐仿真步计算 |
| LIF 电流 | decay_input=False | 在 reset=0 时实现式 (11)：Upre=(1−1/tau)U+I；库默认 True 则注入 I/tau，不能默默混用 |
| 状态与 BN | 每次模型 forward 前后 reset；stem 电流和 BN 仅计算一次 | 深层 BN 每个仿真步更新；测试使用源训练得到的 running statistics |
| Dropout | PyTorch 独立随机掩码，各仿真步重新采样 | 图 3 编码后也保留 dropout；不额外二值化残差流 |
| 滤波 | 每个完整 6 s trial 独立 SOS 双向滤波，再取窗；不重复 notch | 录制时已 notch；双向滤波是离线假设，窗后样本参与滤波边界响应，不宣称实时因果性能 |
| 归一化 | 不做额外去均值、加窗、幅度标准化或跨被试统计拟合 | 仅按 FFT 缩放；BN 统计只来自源训练数据 |
| 软件 | PyTorch 2.5.1、SpikingJelly 0.0.0.0.14；本地 Python 3.12 CPU | 与论文 Python 3.9/CUDA 11.6 工作站环境有区别；服务器设备信息另存 |

SpikingJelly 的输入衰减和硬重置语义依据其 [LIFNode 文档](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/sub_module/spikingjelly.activation_based.neuron.html)。若修改 reset、decay_input 或 tau，应重新核对式 (11)；当前默认 reset=0、decay_input=False 已做手算序列测试。

频域分支原文称 channel MLP，相关 SSVEPformer 实现实际为每个 feature channel 在频谱长度轴上的线性变换。本实现遵循该变换轴。共享门控严格采用式 (19)–(23)，不采用引言中“logits fusion”一词的直接加权解释。

残差加法输出可能大于 1，Dropout 也可能缩放发放值。本实现遵循显式残差公式，只有 LIF 的直接输出为二值；不因正文集合符号矛盾而添加额外 LIF。注意力 Q/K/V 均重新经 LIF。

**用户新增要求与论文区别**

- N 控制测试折 S1..SN；每折训练池始终为另外 34 人。
- M 控制独立种子次数；当前用户先设 M=1、seed=42、窗口 1000 ms，后续可扩为多种子。每个新任务重新初始化模型和优化器；恢复中的同一任务加载已保存状态。
- 原文跨被试报告均值±标准差；按用户要求输出均值±方差，并单列跨重复与跨被试的统计口径。
- 只计算 ACC/ITR 作为最终测试指标。训练曲线保留联合 loss 和训练 ACC；不画每 epoch 测试曲线。

**防泄漏边界与可追溯性**

预处理只做每个 trial 独立的固定变换；没有全数据标准化、目标自适应、目标 BN 更新或重叠滑窗。缓存包含所有 trial 但不进行跨 trial 拟合。训练 DataLoader 只接收源被试索引，测试在最终 epoch 后 eval/inference_mode 执行。每次 forward 清空 LIF 内存，跨 batch 不继承状态。完整训练和测试身份数组保存在 details/evaluation.npz，可直接审计。

本实现没有复制不完整参考文献中的未知设置，也未根据本地冒烟 ACC 调整模型。参数量、FLOPs/SOPs、配对检验和硬件效率分析需要正式实验及相应基线或服务器测量，不能从冒烟结果推断。

**2026-10-20：服务器双 GPU 执行**

`config.py` 增加训练设备池 `device/gpu_ids` 与独立重评估设备 `evaluation_device`。双 GPU 各执行独立的窗口/被试/种子任务，采用 PyTorch `spawn`，每个进程只在自己的设备创建模型。任务之间不共享模型、梯度、BN 或 LIF 内存；不引入 DDP/SyncBatchNorm，不改变每个模型的 batch size 和训练轮数。每个任务重新设定随机种子，完成后在同一卡测试，父进程等待所有任务成功后汇总。预处理仍在 CPU 完成，LOSO 身份边界保持不变。

服务器依赖固定为 CUDA 12.1 版 PyTorch 2.5.1 / torchvision 0.20.1；本地冒烟使用 `smoke_tests/requirements.txt` 中相同版本的 CPU 构建。双 CPU 进程测试用于验证调度和隔离，不能代替真实 GPU 内核、显存容量及性能验证。

**逐 epoch 计时、性能开销与恢复**

逐 batch 全梯度有限性扫描改由 `check_gradients` 控制，默认关闭；保留 loss 的即时有限性检查，epoch 结束时检查参数。训练 loss/ACC 与测试脉冲计数在设备端累计，减少逐 batch/逐神经元的标量同步；传输使用 non_blocking。未改变网络结构、注意力运算顺序、精度、batch size、Ts 或确定性设置。本机无 CUDA，不能据 CPU 冒烟给出 4090 加速倍数。

每个 epoch 的训练用时使用 CUDA 同步后的墙钟时间，包含数据读取、传输和参数更新；保存耗时另列。每个任务滚动保存一份 `details/resume.pt`，包含模型、Adam、随机数、数据打乱状态和历史。中断后回退到最后完整 checkpoint，完成后删除恢复文件。恢复不会按测试 ACC 选 epoch，源被试与目标被试隔离保持不变。

兼容旧版已完成任务；旧版中途未存权重的训练只能从头重训。M 缩减只改变待办任务与汇总范围，不按成绩选择种子。CPU 连续/中断恢复的一致性测试不能保证跨 GPU 型号或软件版本逐位一致。
