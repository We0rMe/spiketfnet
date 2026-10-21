# SpikeTFNet / Benchmark

PyTorch + SpikingJelly 实现，使用 40 类 Benchmark 数据做跨被试 LOSO。仅实现 SpikeTFNet，不包含论文的其他基线。原文未公开的参数已按用户同意的工程默认值补齐，详见 [实现决策](docs/IMPLEMENTATION_NOTES.zh-CN.md)。这属于带明确假设的复现，不宣称与作者未公开代码逐项相同。

## 文件分工

| 文件 | 功能 |
| --- | --- |
| `config.py` | 唯一常规修改入口：数据路径、N/M/seed、六种窗口、训练和网络超参数 |
| `data.py` | 数据读取、完整 trial 独立滤波、裁窗、FFT、缓存和 LOSO 划分 |
| `model.py` | 频域分支、脉冲时域分支、门控及联合损失 |
| `train.py` | N 折 × M 种子训练；只测试固定最后一轮权重 |
| `evaluate.py` | ACC/ITR、分层统计及已保存模型的重新评估 |
| `experiments/ablation.py` | 消融批次入口、组间汇总；复用根目录的训练与评估流程 |
| `experiments/variants.py` | 原始模型与消融组的构建、各组损失；复用 `model.py` 中的分支 |
| `utils/timing.py` | 分支计时与耗时统计 |
| `utils/provenance.py` | 源码归档清单、历史版本认证与断点兼容校验 |
| `docs/` | 复现计划、实现约定及消融运行说明 |
| `smoke_tests/` | 全部冒烟脚本与结果；与正式实验隔离 |

前期结构梳理归档见 [复现计划](docs/REPRODUCTION_PLAN.zh-CN.md)，当前运行参数以 `config.py` 和实现说明为准。

在项目根目录执行所有运行命令。普通训练用 `python train.py`，消融实验用 `python -m experiments.ablation`；不要直接运行包内文件。所有可修改参数仍集中在 `config.py`。后续分支改进可在 `experiments/` 扩展实验，继续共享数据、训练和评估流程。

## 运行

先修改 `config.py`。没有命令行参数解析，也不需要额外 YAML/JSON 配置。

```python
data_root = Path("/your/server/benchmark")
N = 35                    # 测试被试依次 S1,...,SN
M = 1                     # 当前先做单种子；改为 3 时使用 42、43、44
seed = 42                 # 起始种子
windows_ms = (1000,)       # 当前只训练 1000 ms
simulation_steps = 4       # Ts，可以修改
device = "cuda"           # 训练设备：cuda 使用下列 GPU 池；auto 允许 CPU 回落
gpu_ids = (0, 1)           # 两张 GPU 各运行一个独立训练任务
evaluation_device = "cuda:0"  # 单独运行 evaluate.py 时使用的设备
resume = True             # 同名实验恢复；完成的任务自动跳过
check_gradients = False   # 正式训练关闭逐 batch 全梯度扫描
monitor_every = 5         # 源训练子集监测，非验证集；首轮、末轮也测，0 禁用
plot_every = 1            # 每个 epoch 覆盖更新训练曲线和单次报告
experiment_name = "spiketfnet"
```

N 只限制测试折数。即使 N=1，正式训练仍用除 S1 外的全部 34 名被试，不能把源被试池缩小为 N 个人。当前单窗口、35 折、单种子共 35 个独立模型；扩展到 6 个窗口 × 35 折 × 3 种子时共 630 个模型。调试服务器配置时可先设 N=1、M=1、`windows_ms=(500,)`，另起实验名称；不要把这种子集成绩当成完整 LOSO 结果。

服务器建议 Python 3.9–3.12（本地已测试 3.12）。`requirements.txt` 固定 PyTorch 2.5.1 / torchvision 0.20.1 的 CUDA 12.1 构建，需要兼容的 NVIDIA 驱动。构建对应关系见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。Linux 服务器安装和启动：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

Windows 服务器将激活命令换为 `.\.venv\Scripts\Activate.ps1`，其余相同。项目当前 Windows `.venv` 是冒烟专用 CPU 环境，不上传、不拷贝到服务器。库版本与论文所述 Python 3.9/CUDA 11.6 环境不完全相同，运行时版本会归档。

设备选项统一在 `config.py`，无需修改各脚本中的硬编码设备：

| 配置 | 运行方式 |
| --- | --- |
| `device="auto", gpu_ids=(0, 1)` | 使用池中可见的 GPU；一张卡时单卡，没有 CUDA 时 CPU |
| `device="cuda", gpu_ids=(0, 1)`（当前配置） | 要求两张卡均可用，否则报错 |
| `device="cuda:1"` | 仅使用编号 1 的卡；此时忽略 GPU 池 |
| `device="cpu"` | CPU 串行训练 |
| `evaluation_device="auto"/"cpu"/"cuda:0"/"cuda:1"` | 控制单独启动的 `evaluate.py`，一次用一张卡 |

双卡采用独立实验并行：按“窗口 × 留出被试 × 种子”分配任务，每张卡一个 `spawn` 进程，分别维护模型、优化器、BatchNorm 和 LIF 状态。每个模型的 batch size 仍为 128、学习率不变；每次训练完成的测试跟随该模型所在卡。不同任务写入各自目录，全部完成后才统一汇总。此模式不使用 DDP 拆分同一模型的 batch；只配置一个训练任务时只启用一张卡。GPU 编号以进程可见编号为准，设置 `CUDA_VISIBLE_DEVICES` 后会重新编号。

`data.py` 的 SciPy 预处理固定在 CPU，`model.py` 的参数和输入由训练/评估脚本统一搬到所选设备。预处理在父进程完成一次，各训练进程只接收数据。设备名称、GPU 型号及进程号保存在 `details/`，不增加根目录结果文件。独立模型可以用不同 GPU 型号执行，但实际耗时与数值可重复性需在服务器验证。

`python data.py` 可单独准备缓存，`python train.py` 也会自动调用它。缓存只有 `cache/benchmark_prepared.npz` 一个文件，保存选通道后的最大窗片段与数据依据；每次核验原文件 SHA-256、预处理配置和数据脚本哈希，不把旧配置缓存误当新数据。原始数据从不被改写。

使用 `python evaluate.py` 可重新评估 `output_root / experiment_name` 下所有最终 checkpoint。模型结构和预处理设置读取训练时快照；当前配置仅提供实验位置、数据根路径与运行设备。重新评估覆盖同一份评估文件，不再创建一套重复指标。代码/原数据变化会触发一致性检查。

`resume=True` 时，同名实验自动恢复；`resume=False` 时拒绝使用已有目录。更改 `experiment_name` 可启动全新实验。新版本通过文件锁阻止两个训练主进程同时写同一实验；旧版本没有此锁，升级前必须先停止旧训练并确认子进程退出。

## 消融实验

新增 `python -m experiments.ablation`，默认 Benchmark、500 ms、35 折、单种子，运行 full / spectral_only / temporal_only / equal_fusion 四组。
在 `config.py` 末尾只修改 `ABLATION_GROUPS` 选择组别，修改 `ABLATION_WINDOWS_MS` 选择窗口；N/M/seed、设备和其余训练设置继承 CONFIG。
例如 `ABLATION_GROUPS = ("spectral_only",)` 只运行频域组，默认四项列表则依次运行全部四组。
结果子目录按组名自动生成，不需要逐组修改 experiment_name。Config.model_variant 是内部记录字段，ABLATION_CONFIG 是派生配置，都不作为运行选择入口。
结果位于 `results/ablation/<组别>/`，各组独立恢复；总表为 `results/ablation/comparison.csv`。
单分支保留对应融合投影和最终分类头，等权组使用固定 0.5/0.5。详见 [消融运行与实现约定](docs/ABLATION.zh-CN.md)。

新版清单为 v7，源码快照保留 `experiments/`、`utils/` 的目录结构。支持经归档源码摘要验证的旧 v4/v5/v6 四组及 v3 完整模型断点迁移；原 `model.py` 和 `data.py` 未改动，旧完整模型仍以 `python train.py` 继续。
新消融实验不能接续其他组权重。验证说明见消融文档和 smoke_tests/results/timing_check/report.md。

## 耗时和恢复

每个 epoch 实时输出 loss、训练 ACC、源训练子集监测指标，以及以下五项时间；不再输出时域总计、monitor/save/plot 等其他时间。

| 日志名称 | 计时范围（所有仿真步的前向 + 反向） |
| --- | --- |
| Spectral Times | 频域分支，含其嵌入和辅助分类头 |
| Temporal Times1 | 电流映射 Conv/BN、LIF 直接编码及其 Dropout |
| Temporal Times2 | 全部堆叠的多尺度膨胀脉冲残差块，整体计时 |
| Temporal Times3 | 脉冲注意力、token 平均、嵌入/辅助分类头、跨仿真步平均读出 |
| Total Time | 整个训练 epoch：DataLoader、传输、前反向、融合、loss、Adam 与计时开销 |

`Total Time` 不含监测、测试、保存或绘图，不要求等于四个模块时间之和。`time_branches=True` 开启四个模块的计时；关闭时仍计 Total Time，存在模块显示 NA，消融删除的模块显示 0。CUDA 使用当前流上的事件，CPU 使用 perf_counter；不增加逐 batch 的强制同步。区间内包含调度开销，不是纯 kernel 时间，计时本身也有开销。

每项日志同时显示本 epoch 的秒数与平均 ms/trial。平均每 trial = 该项耗时 / 本 epoch 实际处理 trial 次数，包含末尾不足 batch_size 的 batch；这是 batch 均摊值，不是单 trial 独立运行延迟。完整训练和所有折/种子的平均值按对应 trial 次数加权，不能简单平均各 epoch 或任务的均值。

输出沿用现有文件：`training.log` 实时追加；`details/history.csv` 保存五项 epoch 秒数、`train_trials` 和五项 `_per_trial`（秒/trial）。每个窗口/被试/种子的 `report.md`、实验 `report.md`/`summary.csv`，以及消融批次 `report.md`/`comparison.csv` 保存五项累计与平均每 trial。汇总中 `total_seconds` 仅在计时完整时填写；`recorded_seconds`、`recorded_epochs/expected_epochs` 标明已记录部分，`seconds_per_trial` 使用同一部分的 trial 次数。

v6 及更早历史没有三个时域模块计时，不能由旧时域总计推算。升级后保留原频域和整体时间，三个模块从续训后的 epoch 开始记录；缺失显示 NA/空值及覆盖率，不补零。已有日志保留历史原文，新日志和更新后的 CSV/报告只使用五项时间。跨任务累计是各训练任务之和，不是双 GPU 经过的墙钟时长；未提交且需重做的 epoch 不计入。

每个任务开始时保存 epoch 0，之后每个完整 epoch 原子替换 `details/resume.pt`。该文件包含模型、Adam 状态、epoch、训练历史、Python/NumPy/Torch/CUDA 随机状态和 DataLoader 的打乱生成器状态。中途停止后，执行同一条 `python train.py` 即从最后成功保存的 epoch 后继续；当前未保存的 epoch 会重做。最终测试成功后删除恢复文件，只保留 `model.pt`。checkpoint 是恢复用途，测试仍只使用固定最后一轮模型。

恢复会校验数据指纹、trial 身份、模型/预处理和训练设置。可调整 N、M、起始 seed 和设备分配；学习率、batch size、epoch 总数、网络参数、确定性设置变化必须使用新实验名。窗口修改仍要求预处理指纹一致。新格式还校验训练与评估代码哈希；环境或 CPU/CUDA 后端变化不保证逐位相同，运行环境会记入恢复记录。

断点迁移先验证已归档源码与原哈希一致，再核对是否为已审查的 v2/v3/v4/v5/v6 源码（兼容 LF/CRLF 换行），才允许升级至 v7。旧消融模型按原名校验并映射到新路径；v5/v6 计时工具仅在固定摘要认证通过后升级。模型、预处理、优化器和训练数据顺序保持不变；监测前后恢复全部随机状态。未知代码修改不会绕过校验，有历史记录却缺失权重的新版任务也会停止报错，不会静默从头训练。

升级步骤：先停止服务器现有训练并确认子进程退出；同步新版代码；保留原 `experiment_name`、数据路径、N/M/seed 和科学参数，保持 `resume=True`；再次运行 `python train.py`。无需重命名、搬动或删除现有结果目录。已完成任务只补充展示文件，不重新训练；未完成任务从 checkpoint 的 epoch 继续。不要在训练仍运行时单独启动 `evaluate.py`。

旧训练历史可以重新绘图并在新日志中生成带 `RECOVERED` 标记的记录，但这不是原控制台日志。旧 epoch 的监测 loss/ACC 从未计算过，不能事后补造；监测曲线只从升级后的实际测量点开始，空缺不会填 0。新增监测和实时绘图会有额外耗时，可调整 `monitor_every`、`plot_every`，不改变优化过程。

## 为什么没有独立验证集

原训练协议将其余 34 名源被试的全部 8160 个 trial 用于优化，没有独立验证划分，留出被试只在固定最后一轮后测试。参考的 SSVEPformer 项目同样使用**训练集监测**而非验证集：每名源训练被试取第一个 block，共 1360 个 trial，在 eval 模式关闭 Dropout 后计算 loss/ACC。

现在按该方式增加监测线，默认在第 1、5、10、…、100 个 epoch 测量。监测子集仍属于训练集，仅用于观察拟合；蓝线是所有训练 batch 边更新边累计、Dropout 开启的数值，橙线是固定子集在当前权重下关闭 Dropout 的数值，两者口径不同。监测不更新 BatchNorm、不反向传播、不早停或选模，也不接触留出被试。

若需要真正独立的验证集，必须事先从源被试中划出不参与训练的数据并重新训练。当前模型已经见过全部源训练数据，中途再划分不能成为此前未见的独立验证集。本次为保留断点，未改变该划分。

从旧版升级时，完整任务会核验并复用；旧版中途任务没有权重时会明确打印 `no saved weights; restart from epoch 1`，把原曲线留到 `details/history_before_restart.csv` 后重训。旧版若已经写出末轮权重而测试中断，则只补完测试。旧版未保存的中途参数无法恢复。原始源码快照保留，恢复时执行的源码放在 `details/source_resume/`，变更配置及环境记录在原 `experiment.json` 中。

从 M=3 改为 M=1 时，保持起始 seed 不变即可只运行/统计该种子。已完成的该种子任务会跳过，其他种子的目录保留但不参与汇总；程序在恢复期间移除旧范围的 `summary.csv`，成功后重新生成。M=1 的跨重复方差为 0，仅表示没有独立重复，不能作为稳定性证据。

本轮性能检查发现：旧版逐 batch 扫描全部梯度、读取 loss/ACC 标量会产生额外 GPU 操作和同步。现在把梯度扫描设为调试开关，loss/ACC 在设备上累计至 epoch 末，再读取一次；保留逐 batch loss 有限性检查，并在 epoch 末检查全部参数。数据传输使用 `non_blocking=True`；测试的脉冲计数也在设备上累计，最后回传。该方向与 [PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 一致，实际收益需在服务器计时。模型结构、FP32、Ts=4、batch size 和确定性设置未改变，逐仿真步的脉冲计算仍有开销。

## 输出

```text
results/spiketfnet/
  report.md                           # 当前任务进度、已完成任务指标及报告链接
  summary.csv                         # 唯一汇总表，包含“均值 ± 方差”列
  details/
    experiment.log                    # 主进程调度、恢复和错误日志
    experiment.json                   # 配置、版本、用途、状态、代码哈希
    dataset.json                      # 预处理和数据文件指纹
    source/                           # 本轮源代码及实现假设快照
  window_0500ms/test_S01/seed_42/
    report.md                         # 可读报告；训练期间随曲线更新，完成后加入测试指标
    metrics.json                      # 本轮 ACC/ITR 的机器可读记录
    training.log                      # 实时追加的训练输出、耗时、恢复信息与错误
    training.png                      # 覆盖更新：训练与源训练子集监测 loss/ACC
    confusion.png                     # 仅最终测试后生成，基于已保存预测
    details/
      model.pt                        # 固定最后一轮的权重
      history.csv                     # 原始曲线、监测值与训练/监测秒数，旧监测值留空
      resume.pt                       # 训练中滚动覆盖；最终评估成功后删除
      provenance.json                 # 测试/训练被试、种子与完成状态
      evaluation.npz                  # 预测、身份、特征、脉冲计数及测量用输入
```

`evaluation.npz` 不使用 pickle。内容包括真实标签、三路 logits、融合特征、门控、准确的训练/测试 trial 身份、逐 LIF 层的发放总数与元素总数，以及少量已预处理时域/频域输入。混淆矩阵直接从它生成，不再额外保存一份预测 CSV。结合 checkpoint、代码和配置，可在服务器后续计算其他分类指标、参数量及 FLOPs/SOPs，或重新测量时延和吞吐量。时延须在指定硬件上重新 warm-up、batch size=1 并同步 CUDA 测量；本地仪器化测试时间不能替代论文的时延。

双进程运行时总进度报告约每 5 秒刷新；单进程在任务完成时刷新。未完成时报告明确标为部分结果，不将缺失任务当作 0；正式 `summary.csv` 仍在当前 N×M 全部完成后生成。日志是追加文件，曲线与报告覆盖更新，不生成每 epoch 一套文件。

## 统计口径

先对每个测试被试、每个种子计算 ACC 与 ITR。ACC 的单位是百分数；ITR 使用 40 类，决策时间为窗口秒数 + 0.5，P≤1/40 时为 0；P=1 按极限处理。

设指标矩阵为 `A[被试, 种子]`。`summary.csv` 的不同 scope 分开呈现：

- `across_repeats`：每个种子先对 N 名测试被试求均值，再对 M 个完整重复实验求均值和方差，作为主要多轮统计。
- `across_subjects`：先对每名被试的 M 个种子求均值，再统计 N 名被试的均值和方差，用于观察跨被试差异。
- `within_subject`：单名被试在 M 个种子间的均值和方差。
- `within_repeat`：单个种子在 N 名被试间的均值和方差。

默认 `variance_ddof=0`，确实输出方差而非标准差。方差单位是原指标单位的平方；“均值 ± 方差”是按用户要求的展示，不是置信区间。只有一个观测值时总体方差为 0，不代表结果稳定。若改为 ddof=1，单观测的方差显示 NA。ITR 先逐被试计算再汇总，不用平均 ACC 直接代入。

训练曲线不含逐 epoch 测试表现，避免据测试集选模型。重新对同一模型做确定性测试不会构成新的独立实验；多轮测试由不同 seed 的独立训练实现。

## 本地冒烟

```powershell
.\.venv\Scripts\python.exe -m pip install -r smoke_tests\requirements.txt
.\.venv\Scripts\python.exe smoke_tests\run_smoke.py
.\.venv\Scripts\python.exe smoke_tests\run_parallel_smoke.py
.\.venv\Scripts\python.exe smoke_tests\run_resume_smoke.py
.\.venv\Scripts\python.exe smoke_tests\run_reporting_smoke.py
.\.venv\Scripts\python.exe smoke_tests\run_layout_smoke.py
```

所有临时缓存、报告、训练及评估结果都进入 `smoke_tests/results/`。脚本先做数值与隔离检查，再对真实 S1–S3 数据进行预处理；集成部分取每人 8 个 trial、S1/S2 两折、42/43 两个种子、每次 2 epochs。完整网络尺寸、40 类输出和 Ts=4 保持不变；六个窗口另做形状检查。最后加载四个 checkpoint 重评估，确认 logits 和汇总完全一致。

集成目录已存在时会拒绝覆盖；需要重跑时先自行移走旧 `smoke_tests/results/integration/`。这些低轮数、小样本成绩仅验证软件流程，不能用于评价论文复现性能。`.gitignore` 已排除 `.venv`、缓存、正式和冒烟结果、原始 MAT 与 PDF。

`run_parallel_smoke.py` 的脚本与结果同样位于冒烟目录，结果写入 `smoke_tests/results/parallel_device_check/`，已存在时拒绝覆盖。它检查设备选择和任务完整性，并对真实数据小子集分别执行串行训练、两个 CPU `spawn` 进程并行训练，核对全部权重、训练历史、预测与汇总，再重放 checkpoint。本机 CPU 测试验证并行流程，不代表实际双 GPU 测试已完成。两个冒烟脚本均显式指定本地数据路径和 CPU，不继承服务器的设备设置。

`run_resume_smoke.py` 输出到 `smoke_tests/results/resume_check/`，模拟第 1 个 epoch 后中断，再以两个 CPU 进程恢复，比较权重、训练指标、预测和汇总（计时值不要求相同）。另验证完成任务跳过、M 缩减、旧版任务升级、写盘失败保留前一 checkpoint 及并发锁。所有冒烟结果均不作为 GPU 速度或正式复现性能依据。

`run_reporting_smoke.py` 使用 `smoke_tests/fixtures/` 中冻结的 v2 源码生成真实旧断点，与新版监测、实时输出及双进程恢复作对照。结果在 `smoke_tests/results/reporting_check/`。冻结源码仅供冒烟测试，不是服务器训练入口。

`run_layout_smoke.py` 默认验证本次修改前 v6 四组的真实断点迁移、三模块计时、trial 均值、旧历史缺失处理、权重/预测一致性和源码归档，结果在 `smoke_tests/results/module_timing_check/`。还可从 Python 调用 `main(output_name="新目录", version=5)` 验证 v5。`run_ablation_smoke.py` 覆盖四组双进程调度与 v3 恢复，`run_timing_smoke.py` 覆盖 v4 四组恢复。所有脚本都仅用于 CPU 小样本冒烟。
