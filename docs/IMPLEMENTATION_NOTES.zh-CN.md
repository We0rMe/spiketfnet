**2026-10-26 当前变更（下文保留历史规格）**

日常配置精简至config.py；完整只读实验快照、固定协议和运行策略定义在utils/configuration.py，供归档和旧配置恢复，非第二个调参入口。新实验默认split_bias；历史快照缺失该字段仍恢复native。清单升至v13并归档、验证内部配置模块。

新增独立的temporal_dropout，只作用于时间编码后的dropout及三块残差块的dropout，全部投影头和频谱分支仍使用dropout。旧快照缺字段时继承它原来的dropout值，保持旧模型计算。全局平均读出、BN、LIF公式和LOSO流程未改。

本轮候选为temporal_dropout=0.1、stem_kernel=31、residual_kernel=7，其他主要超参数保留；7折×2种子、1000 ms，默认仅时间分支。该组合尚未完成服务器验证。更大核宽及独立dropout是基于机制探索的工程候选，并非作者配置；运行和对照建议见README。

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
- 按原文及用户更正后的要求报告均值±标准差，并单列跨重复与跨被试的统计口径。旧版方差记录见下文版本历史。
- 只计算 ACC/ITR 作为最终测试指标。训练曲线保留联合 loss 和训练 ACC；不画每 epoch 测试曲线。

**防泄漏边界与可追溯性**

预处理只做每个 trial 独立的固定变换；没有全数据标准化、目标自适应、目标 BN 更新或重叠滑窗。缓存包含所有 trial 但不进行跨 trial 拟合。训练 DataLoader 只接收源被试索引，测试在最终 epoch 后 eval/inference_mode 执行。每次 forward 清空 LIF 内存，跨 batch 不继承状态。完整训练和测试身份数组保存在 details/evaluation.npz，可直接审计。

本实现没有复制不完整参考文献中的未知设置，也未根据本地冒烟 ACC 调整模型。参数量可直接由模型结构计数；FLOPs 需明确输入及计数口径，SOPs/配对检验/硬件效率需相应已训练模型、正式测试或服务器测量，不能从冒烟 ACC 推断。

**2026-10-20：服务器双 GPU 执行**

`config.py` 增加训练设备池 `device/gpu_ids` 与独立重评估设备 `evaluation_device`。双 GPU 各执行独立的窗口/被试/种子任务，采用 PyTorch `spawn`，每个进程只在自己的设备创建模型。任务之间不共享模型、梯度、BN 或 LIF 内存；不引入 DDP/SyncBatchNorm，不改变每个模型的 batch size 和训练轮数。每个任务重新设定随机种子，完成后在同一卡测试，父进程等待所有任务成功后汇总。预处理仍在 CPU 完成，LOSO 身份边界保持不变。

服务器依赖固定为 CUDA 12.1 版 PyTorch 2.5.1 / torchvision 0.20.1；本地冒烟使用 `smoke_tests/requirements.txt` 中相同版本的 CPU 构建。双 CPU 进程测试用于验证调度和隔离，不能代替真实 GPU 内核、显存容量及性能验证。

**逐 epoch 计时、性能开销与恢复**

逐 batch 全梯度有限性扫描改由 `check_gradients` 控制，默认关闭；保留 loss 的即时有限性检查，epoch 结束时检查参数。训练 loss/ACC 与测试脉冲计数在设备端累计，减少逐 batch/逐神经元的标量同步；传输使用 non_blocking。未改变网络结构、注意力运算顺序、精度、batch size、Ts 或确定性设置。本机无 CUDA，不能据 CPU 冒烟给出 4090 加速倍数。

每个 epoch 的训练用时使用 CUDA 同步后的墙钟时间，包含数据读取、传输和参数更新；保存耗时另列。每个任务滚动保存一份 `details/resume.pt`，包含模型、Adam、随机数、数据打乱状态和历史。中断后回退到最后完整 checkpoint，完成后删除恢复文件。恢复不会按测试 ACC 选 epoch，源被试与目标被试隔离保持不变。

兼容旧版已完成任务；旧版中途未存权重的训练只能从头重训。M 缩减只改变待办任务与汇总范围，不按成绩选择种子。CPU 连续/中断恢复的一致性测试不能保证跨 GPU 型号或软件版本逐位一致。

**输出升级与训练集监测**

参考本地 SSVEPformer 的组织，增加任务 `training.log`、实时覆盖的 `training.png`/`report.md`、最终 `confusion.png` 和实验总进度报告。原结果层级和权重位置不移动。预测仍只存原 evaluation.npz；报告是指标的可读展示，不增设第二套机器可读预测或指标表。

源训练被试的第一个 block 用于 eval 模式的训练集监测（正式折为 1360 trial），每 5 epoch 加首/末轮计算一次，曲线每 epoch 更新。它不是独立验证集，不影响源训练池，不做选模。监测只读模型参数及 BN 统计，前后恢复 Python、NumPy、Torch、CUDA 及打乱生成器随机状态；测试被试仅在固定最后一轮之后进入模型。

对已审查 v2 的 train.py/evaluate.py，通过归档原文件哈希与规范换行后的固定摘要验证，允许仅输出功能升级到 v3。未知源码变化仍拒绝恢复。恢复文件和训练计算本身不改格式；旧 epoch 的监测值保留缺失。新增输出与监测在 CPU 上已与冻结旧训练代码比较，权重和最终预测逐位一致；实际 CUDA 恢复仍需服务器核验。

**2026-10-21：消融实现约定**

新增 full / spectral_only / temporal_only / equal_fusion 四组，共用原编码器和训练协议；普通训练的窗口及路径保持原配置，消融窗口独立可选，默认 500 ms。单分支保留该分支的融合投影和最终分类头，主损失加保留分支的辅助 CE（权重不变），这是用户确认的工程约定。等权组保留双分支及投影/分类头，只删除门控，使用不可学习的 0.5/0.5。

所有组先按完整模型相同顺序初始化，再删除指定模块，使相同种子的公共参数初值一致；不使用预训练权重。原 model.py/data.py 未改动。新断点显式保存组别，新 v4 实验检查 ablation_model.py 哈希；旧 v3 完整模型仅通过已归档源文件及固定摘要认证后迁移，仍检查科学参数和数据。详见 [消融运行说明](ABLATION.zh-CN.md)。

**配置入口与分支计时升级（v5）**

运行组仅由 ABLATION_GROUPS 选择，窗口由 ABLATION_WINDOWS_MS 设置，ABLATION_ROOT 控制批次根目录；组名/窗口/被试/种子子目录自动生成。model_variant 留作内部快照字段，不作为用户选择入口。

完整模型在同一 batch 内联合训练，一次 backward/Adam.step 更新所有保留参数。utils/timing.py 用前向模块边界及反向图节点归属记录分支专属区间；融合模块、损失、优化器不分摊给分支。CUDA 事件在 epoch 末统一解析，CPU 用 perf_counter；记录执行区间而非纯 kernel 时间，会有插桩开销。未修改模型/梯度/优化顺序，关闭开关只停计时。

逐 epoch 分支秒数写入原 history.csv 和日志，各级现有汇总增加三项训练累计和覆盖率；跨任务求和不等于双卡墙钟时间。只累计已提交历史，不补造旧 epoch 耗时。v5 增加计时代码哈希校验，允许已审查 v4 四组断点迁移；v2/v3 原迁移边界保留。

## 目录整理（v6）

根目录保留配置、数据、模型、训练、评估五个核心脚本。消融入口迁至 experiments/ablation.py，以 `python -m experiments.ablation` 运行；模型变体迁至 experiments/variants.py。计时迁至 utils/timing.py，断点源码校验收拢到 utils/provenance.py；复现计划归入 docs/。配置值、结果目录及科学计算不变。

源码归档保留包目录，恢复校验支持旧平铺文件名到新路径的映射。仅允许固定摘要认证的 v5 训练/评估代码升级，仍逐项检查原模型、消融模型、预处理、计时及科学参数；已支持的更早版本迁移继续保留。v6 后续恢复也校验新提取的辅助模块，不能以整理目录为由放行未知代码改动。

## 时域三模块计时（v7）

保持 model.py、数据、损失、优化器和参数键名不变，在 utils/timing.py 对现有模块注册临时钩子。Times1 包含 current/encoder/dropout；Times2 包含整个 blocks；Times3 包含 attention、token 平均、embedding/classifier 及跨步平均。所有仿真步累计，辅助分类头归入 Times3；融合投影仍只计入整体训练时间。

前向使用连续区间以覆盖函数式读出操作；收集反向归属时在已标注图节点处停止，避免重复归属上游，图遍历开销留在整体训练时间。反向按 Node 执行切换区间；CUDA 事件在 epoch 结束统一解析。启用/关闭计时不改变前向结果、梯度、随机状态或优化过程。

新输出只保留 Spectral Times、Temporal Times1/2/3、Total Time 五项。平均每 trial 按实际样本处理次数均摊；旧 trial 数仅在已验证 LOSO 划分及无 drop_last 的既有训练流程下恢复。旧时域总计不用于反推三模块数据，历史缺项保持未知。v6 的 train/evaluate/timing/provenance 及 v5 的旧 timing 需通过归档文件和固定摘要认证，才允许升级；数据与模型校验继续执行。原 checkpoint 结构及随机状态恢复协议不变。

## 第二模块卷积/LIF 前后向诊断（v8）

当前仅统计 temporal.blocks 内所有 Conv1d/LIFNode 的前向与反向四项。前向用层 pre/post hook；反向只包围实际参与 loss 的归属节点。输入上游节点排除，重置后的 LIF 膜电位图纳入，以覆盖跨步梯度；非目标 BN/残差等不计入。没有改变 model.py 或任何训练科学参数，原 epoch 边界同步和有限值检查保留；没有执行多步/CuPy/融合/AMP 等优化。

源码认证支持修改前 v7 的真实断点续训；旧四项时间未知，记录覆盖率。新的四项区间之和不等于原第二模块整体区间。说明、官方依据与审计结果见 [第二模块计时排查](TEMPORAL2_TIMING_AUDIT.zh-CN.md)。

## 第二模块逐层/逐步与底层算子诊断（v9）

保留 v8 四项总计及训练计算。T2 Conv/LIF 增加层名、仿真步、反向节点名称、实际 batch size、首次/后续 batch、累计/最大区间和 Host 区间；跨步共享 AccumulateGrad 单列 step=0。逐 epoch 日志不显示 trial 均摊值，详细数据合并在一个 details/t2_diagnostics.json，并随 checkpoint 历史保存。

新增 utils/operator_profile.py，使用内置 PyTorch profiler，仅对每次启动/续训后每个任务的首个新 epoch 中一个 batch 采样。只保留 T2 标记内的 CPU self、直接关联 kernel 及算子次数，避免父子 inclusive 时间重复累计；不可用的 CUDA 数据标记 NA。配置 t2_profile_batch=2，0 可关闭重型采样；不改变 batch size、数据顺序或梯度。模型、优化器、依赖、确定性等未改动。

固定版本源码排查发现 torch 2.5.1 的 contiguous 确定性膨胀卷积会绕开 cuDNN；慢速 CUDA 路径前向逐样本/通道填充 bias，而反向没有对应初始化循环。这是前向耗时突出的首要嫌疑；需服务器实际 BACKEND、copy_/fill_ 及 kernel 分布验证，不以 CPU 冒烟时间替代 GPU 测量。

支持固定摘要认证的 v8 train/evaluate/timing/provenance 升级到 v9，更早版本规则保留；v8 四项时间保留，新细节不补造历史。新采样工具纳入源码归档与后续恢复校验。真实 Benchmark 小样本上验证四组 v8 中断后恢复与原版连续训练权重、预测、训练/监测指标一致；本机没有 CUDA，GPU 性能结论待服务器验证。

用户补充的效率表属于 12 类、0.5 s：3.016 M 参数、32.016 M FLOPs、836.362 K SOPs、2.735 ms/trial；SOPs 为 1,800 个未见 trial 均值。当前 40 类模型实算 4,584,088 个可训练参数，数据集输入/输出尺寸不同，不能仅因总量不同判定复现偏差。启动日志及 provenance 增加参数分解记录；当前只诊断，不修改结构以拟合这张 12 类表。

## 膨胀卷积偏置外置（v10）

服务器证据确认 d=2/4 调用 slow_conv_dilated2d，单层单 batch 的 copy_ 次数为 128×64×4=32768。新增可选 t2_conv_execution=split_bias，仅在 T2 的 d>1 卷积中保留原 bias 参数并在卷积后广播相加；native 默认保留原计算。参数键名、尺寸与数学定义不变，但浮点求和及梯度归约顺序可能变化，不能保证 SNN 完整训练轨迹不变。

当前已完成 31 折的实验建议 native 完成；模式切换需独立 experiment_name / ABLATION_ROOT。严格认证的 v9 源码可升级 v10 native，未知源码仍拒绝，旧配置即使没有执行字段也始终恢复为 native。新增 CPU 双精度卷积输出/梯度检查、默认模型旧新逐位检查、四组 v9 断点升级与新 split_bias 模式中断恢复检查。GPU 收益由 smoke_tests/run_conv_benchmark.py 在空闲卡上验证；CPU 速度不作 GPU 结论。未修改依赖或关闭确定性。

## 三项耗时与测试报告（v11）

本轮不改变模型、数据、损失、优化器或执行模式。完整频域/时域的前反向分别计时，总时间覆盖训练循环；t2_profile_batch=0默认同时关闭细粒度hooks及profiler。旧细节保留，但缺失的完整分支时长不反推或补零。源码认证迁移支持经固定摘要审核的v10，新增分支计时工具加入归档与校验；已认证源码若只有LF/CRLF差异可兼容，任意内容改动不放行。

修正结束日志的统计轴：旧版打印M个种子的LOSO均值间方差，M=1必然得到0，不能反映35名测试被试差异。现在主结果采用先对每名被试平均种子，再跨被试汇总；种子间统计另存。仍遵循用户要求的均值±方差及variance_ddof，单观测方差显示NA（ddof=0也如此），两个以上相同观测的零方差不改。ITR仍逐折计算再平均，不用总体平均ACC推导。

根report.md首先显示各种子下的跨被试测试ACC/ITR，多种子完成后再显示跨种子及种子平均后的跨被试结果；分折进度折叠。单轮报告测试指标置顶，保留训练/监测表现、曲线和三项时间，移除大量诊断说明。已完成结果可调用evaluate.refresh_saved_reports，只重建报告及汇总，不训练、不推理、不改原始指标/预测/权重/日志。

上传native实验35折单种子1000ms的跨被试总体统计：ACC 78.0119047619±445.7349773243；ITR 143.2633182448±2999.6181859100。±后是方差而非参考项目的标准差，单位为对应指标单位平方。split_bias仅7个epoch，无最终测试结果，不得据其训练ACC断言最终精度。

## 标准差展示更正（v12）

用户更正统计展示要求：从均值±方差改为均值±标准差（SD）。计算调用np.std，汇总CSV字段改为std和mean_plus_std，普通实验和消融的报告/结束日志统一。保留原ddof=0，仅更正离散指标，不同时改变分母或聚合轴；variance_ddof字段名仅为兼容旧配置，0为总体SD，1为样本SD。单个观测的SD继续为NA。

35折单种子1000ms旧native实验现为ACC 78.011905±21.112437、ITR 143.263318±54.768770（ddof=0）；前文v11中方差数字仅是历史记录，不作为当前报告。原始指标、日志、预测和权重保持不变。v11训练/评估/恢复与消融源码以固定摘要认证后可升级v12，未知源码修改仍拒绝。
