# SpikeTFNet / Benchmark

PyTorch + SpikingJelly 实现，使用 40 类 Benchmark 数据做跨被试 LOSO。仅实现 SpikeTFNet，不包含论文的其他基线。原文未公开的参数已按用户同意的工程默认值补齐，详见 [实现决策](docs/IMPLEMENTATION_NOTES.zh-CN.md)。这属于带明确假设的复现，不宣称与作者未公开代码逐项相同。

## 文件分工

| 文件 | 功能 |
| --- | --- |
| `config.py` | 日常修改入口：实验范围、值得调节的超参数、数据路径和设备 |
| `utils/configuration.py` | 完整快照定义、固定数据/协议/运行默认值、校验及历史配置读取；不是第二个调参入口 |
| `data.py` | 数据读取、完整 trial 独立滤波、裁窗、FFT、缓存和 LOSO 划分 |
| `model.py` | 频域分支、脉冲时域分支、门控及联合损失 |
| `train.py` | N 折 × M 种子训练；只测试固定最后一轮权重 |
| `evaluate.py` | ACC/ITR、分层统计及已保存模型的重新评估 |
| `experiments/ablation.py` | 消融批次入口、组间汇总；复用根目录的训练与评估流程 |
| `experiments/variants.py` | 原始模型与消融组的构建、各组损失；复用 `model.py` 中的分支 |
| `utils/branch_timing.py`、`utils/timing.py` | 三项分支摘要、可选细粒度诊断及历史兼容 |
| `utils/operator_profile.py` | 第二模块底层算子采样、运行时配置记录 |
| `utils/provenance.py` | 源码归档清单、历史版本认证与断点兼容校验 |
| `docs/` | 复现计划、实现约定及消融运行说明 |
| `smoke_tests/` | 全部冒烟脚本与结果；与正式实验隔离 |

前期结构梳理归档见 [复现计划](docs/REPRODUCTION_PLAN.zh-CN.md)，当前运行参数以 `config.py` 和实现说明为准。

在项目根目录执行所有运行命令。普通训练用 `python train.py`，消融实验用 `python -m experiments.ablation`；不要直接运行包内文件。所有可修改参数仍集中在 `config.py`。后续分支改进可在 `experiments/` 扩展实验，继续共享数据、训练和评估流程。

## 运行

先修改 `config.py`。没有命令行参数解析，也不需要额外 YAML/JSON。文件中的 `CONFIG = Config(...)` 是当前实验的唯一参数入口；固定数据属性、预处理规则、LIF 实现约定、绘图和诊断默认值收在 `utils/configuration.py`，仍完整写入实验快照，以便复核和恢复旧实验。

本轮检查**32段时间分支变强后，联合训练的融合方式是否损害了跨被试表现**。冻结现有科学参数，仅运行两组：`spectral_only` 和 `equal_fusion`。窗口500 ms，N=3、M=2（S1–S3，种子42、43），每组6模型，共12模型。

`equal_fusion` 从训练开始固定使用 `0.5*zf + 0.5*zt`，删除自适应门控；保留两分支、融合投影、最终分类头和两项辅助损失。它使用现成消融实现，从头联合训练，不加载旧模型，不是在测试时临时改权重。

两个新组都使用 `temporal_readout_bins=32`。频域单分支最终结构不含时间分支，但当前构建流程先初始化完整模型，再删除组件：段数会影响后续融合投影/分类头的初值和训练随机数状态。因此补一组32段配置下的单频域模型，作为匹配初始化过程的对照；不宣称它的结构增加了时间信息。

其他参数不变：时间骨干dropout=0.1、头dropout=0.5、核31/7、宽度64、辅助损失各0.3、Ts=4、Adam学习率0.001、L2 0.001、batch128、100轮。共享BN、预处理及固定末轮评估不变。每折仍使用其他全部34名源被试的8160条试次。

**服务器入口：**

```bash
python -m experiments.ablation
```

结果目录为 `results/fusion_control_v1_GAP32_500ms/`，下面分别是 `spectral_only/` 和 `equal_fusion/`。现有五组结果全部保留，不重跑。`python train.py` 会运行自适应门控的完整模型，不是本轮入口。

配对对照均取S1–S3、种子42/43。原时间1段组的同范围ACC为29.65%，不能使用其7折汇总24.73%；时间32段53.82%，完整1段73.12%，完整32段64.44%，已有单频域75.56%。完整32段模型的频谱辅助头仍有78.26%，且六个模型均高于各自的融合结果，说明主要问题已转向融合后的表示和分类路径。详细证据与固定权重诊断见 [融合审查](smoke_tests/fusion_audit/results/report.md)。

下一轮先比较等权组与已有完整32段组的同折/同种子结果，再与新单频域组比较。等权明显回升说明训练中的门控分配值得继续改进；回升但未超过单频域不算获得融合收益；若仍差，则转向分支监督、共享分类头及表征配合，不继续盲目扫时间核宽或训练轮数。3折只能做开发筛查，不据此声称匹配论文35折性能；不按目标成绩选checkpoint。

数据路径保留服务器 `I:\jjh\SSVEP\data\benchmark`，本地为 `G:\SSVEP\data\benchmark`。沿用 `device="cuda:1"`，12个模型在该卡顺序执行；改为 `cuda` 才由 `gpu_ids=(0,1)` 分担独立模型。`resume=True` 恢复本轮新目录中的任务。本地仅进行软件冒烟。

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

`python -m experiments.ablation` 读取 `config.py` 的 `ABLATION_GROUPS`；当前只选时间分支。窗口直接使用 `CONFIG.windows_ms`，不再重复定义 `ABLATION_WINDOWS_MS`；批次路径由 `CONFIG.experiment_name` 派生，不再维护 `ABLATION_ROOT`。N/M/seed和其他参数同样直接继承。可选full / spectral_only / temporal_only / equal_fusion。

结果位于 `results/<experiment_name>/<组别>/`；各组独立恢复，总表为批次根目录的 `comparison.csv`。单分支保留对应融合投影和最终分类头，等权组使用固定0.5/0.5。详见 [消融说明](docs/ABLATION.zh-CN.md)。

当前清单v15记录确定性分段平均实现。v14的CUDA自适应平均池化反向不支持严格确定性；现按同样的floor/ceil边界逐段求平均，保持32段、参数量及确定性设置。浮点归约次序可不同，不承诺与旧多段实现逐位一致。已认证v14多段实验只有空历史、第0轮断点、空优化器状态且无末轮模型/测试结果时允许原目录重试；已有完成训练的多段实验需新目录。1段路径未变，旧快照缺读出字段仍恢复为1；其余已认证历史升级规则保留。

## 耗时和恢复

内部运行策略固定默认 `time_branches=True`、`t2_profile_batch=0`，每个 epoch 只显示三项时间，不显示每 trial 均摊值：

```text
Spectral Times=...s | Temporal Times=...s | Total Time=...s
```

前两项包含对应完整分支的前向与反向，CUDA 使用同流 Event，epoch 末同步后读取；区间可能包含主机提交间隙，不等于纯 kernel 执行时间。融合/损失/Adam 不算入两分支。Total Time 是训练循环墙钟时间，包含数据迭代、搬运、完整前反向、Adam、计时开销及末尾 GPU 等待；不含训练子集监测、测试、保存和绘图。两分支之和不要求等于总时间。多卡任务累计是各任务时长之和，不是两卡并行运行的实际经过时间。

`training.log` 实时追加；`details/history.csv` 保存 epoch 秒数及处理试次数；单次/实验/消融报告和原有汇总表保存三项累计、每 trial 均摊及覆盖率。均摊分母是对应已记录 epoch 内全部 batch 实际处理的样本次数之和，例如完整 35 被试 LOSO 每轮为 8160，而不是 batch size 128；跨轮同一 trial 每次参与训练都计一次。它不是 batch=1 推理延迟。缺失分支为 0，关闭计时的现存分支为 NA。

旧版只有 T2 局部计时，无法恢复完整分支/epoch 时长，因此保留未知。已存在的日志、细粒度历史不删改，新三项从后续 epoch 开始记录；报告明确已记录 epoch 数，不把部分累计称为完整总时长。

仅开发者再次排查性能时，在内部运行策略中设 `t2_profile_batch>0`：开启每轮 T2 逐层/逐步诊断，并在每次启动后首个新 epoch 的指定 batch 做一次 profiler 采样。设 0 同时关闭细粒度 hooks 和采样。详细记录集中在原 `details/t2_diagnostics.json`，不新增逐 batch 文件。`provenance.json` 继续记录实际 worker 环境和模型参数量。完整边界与服务器性能结果见 [第二模块诊断与排查](docs/TEMPORAL2_TIMING_AUDIT.zh-CN.md)。

每个任务开始时保存 epoch 0，之后每个完整 epoch 原子替换 `details/resume.pt`。该文件包含模型、Adam 状态、epoch、训练历史、Python/NumPy/Torch/CUDA 随机状态和 DataLoader 的打乱生成器状态。中途停止后，执行同一条 `python train.py` 即从最后成功保存的 epoch 后继续；当前未保存的 epoch 会重做。最终测试成功后删除恢复文件，只保留 `model.pt`。checkpoint 是恢复用途，测试仍只使用固定最后一轮模型。

恢复会校验数据指纹、trial 身份、模型/预处理和训练设置。可调整 N、M、起始 seed 和设备分配；学习率、batch size、epoch 总数、网络参数、确定性设置变化必须使用新实验名。窗口修改仍要求预处理指纹一致。新格式还校验训练与评估代码哈希；环境或 CPU/CUDA 后端变化不保证逐位相同，运行环境会记入恢复记录。

断点迁移先验证已归档源码与原哈希一致，再核对是否为已审查的 v2–v12 源码（兼容 LF/CRLF 换行），才允许升级至当前版本。旧消融模型按原名校验并映射到新路径；旧计时工具仅在固定摘要认证通过后升级。native 路径的模型计算、预处理、优化器和训练数据顺序保持不变；监测前后恢复全部随机状态。未知代码修改不会绕过校验，有历史记录却缺失权重的新版任务也会停止报错，不会静默从头训练。执行模式改变需新建实验，不能把参数可加载误认为数值轨迹完全一致。

升级步骤：先停止服务器现有训练并确认子进程退出；同步新版代码；保留原 `experiment_name`、数据路径、N/M/seed 和科学参数，保持 `resume=True`；不要使用本轮候选覆盖旧参数；再次运行 `python train.py`。无需重命名、搬动或删除现有结果目录。已完成任务只补充展示文件，不重新训练；未完成任务从 checkpoint 的 epoch 继续。不要在训练仍运行时单独启动 `evaluate.py`。

旧训练历史可以重新绘图并在新日志中生成带 `RECOVERED` 标记的记录，但这不是原控制台日志。旧 epoch 的监测 loss/ACC 从未计算过，不能事后补造；监测曲线只从升级后的实际测量点开始，空缺不会填 0。新增监测和实时绘图会有额外耗时，可调整 `monitor_every`、`plot_every`，不改变优化过程。

## 为什么没有独立验证集

原训练协议将其余 34 名源被试的全部 8160 个 trial 用于优化，没有独立验证划分，留出被试只在固定最后一轮后测试。参考的 SSVEPformer 项目同样使用**训练集监测**而非验证集：每名源训练被试取第一个 block，共 1360 个 trial，在 eval 模式关闭 Dropout 后计算 loss/ACC。

现在按该方式增加监测线，默认在第 1、5、10、…、100 个 epoch 测量。监测子集仍属于训练集，仅用于观察拟合；蓝线是所有训练 batch 边更新边累计、Dropout 开启的数值，橙线是固定子集在当前权重下关闭 Dropout 的数值，两者口径不同。监测不更新 BatchNorm、不反向传播、不早停或选模，也不接触留出被试。

若需要真正独立的验证集，必须事先从源被试中划出不参与训练的数据并重新训练。当前模型已经见过全部源训练数据，中途再划分不能成为此前未见的独立验证集。本次为保留断点，未改变该划分。

从旧版升级时，完整任务会核验并复用；旧版中途任务没有权重时会明确打印 `no saved weights; restart from epoch 1`，把原曲线留到 `details/history_before_restart.csv` 后重训。旧版若已经写出末轮权重而测试中断，则只补完测试。旧版未保存的中途参数无法恢复。原始源码快照保留，恢复时执行的源码放在 `details/source_resume/`，变更配置及环境记录在原 `experiment.json` 中。

从 M=3 改为 M=1 时，保持起始 seed 不变即可只运行/统计该种子。已完成的该种子任务会跳过，其他种子的目录保留但不参与汇总；程序在恢复期间移除旧范围的 `summary.csv`，成功后重新生成。M=1 的种子间标准差显示 NA；主报告与结束日志展示跨被试差异。

本轮性能检查发现：旧版逐 batch 扫描全部梯度、读取 loss/ACC 标量会产生额外 GPU 操作和同步。现在把梯度扫描设为调试开关，loss/ACC 在设备上累计至 epoch 末，再读取一次；保留逐 batch loss 有限性检查，并在 epoch 末检查全部参数。数据传输使用 `non_blocking=True`；测试的脉冲计数也在设备上累计，最后回传。该方向与 [PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 一致，实际收益需在服务器计时。模型结构、FP32、Ts=4、batch size 和确定性设置未改变，逐仿真步的脉冲计算仍有开销。

## 输出

```text
results/spiketfnet/
  report.md                           # 跨被试测试 ACC/ITR 汇总、折叠的分折进度及耗时
  summary.csv                         # 唯一汇总表，包含“均值 ± 标准差”列
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

已完成的旧实验无需重新训练或推理，停止训练后可直接从原指标重建报告（路径换成自己的实验根目录）：

```python
from evaluate import refresh_saved_reports
refresh_saved_reports(r"G:\SSVEP\SpikeTFNet\results_server\spiketfnet")
```

这只覆盖 report.md 和完整实验的 summary.csv，不改指标、日志、配置、权重或预测。单折报告首先呈现最终测试 ACC/ITR、正确试次数，随后保留训练表现、三项耗时与曲线；实验根报告首先呈现各种子的跨被试测试统计，多种子全部完成后补充种子平均后的跨被试统计及种子间统计。

## 统计口径

先对每个测试被试、每个种子计算 ACC 与 ITR。ACC 的单位是百分数；ITR 使用 40 类，决策时间为窗口秒数 + 0.5，P≤1/40 时为 0；P=1 按极限处理。

设指标矩阵为 `A[被试, 种子]`。`summary.csv` 的不同 scope 分开呈现：

- `across_repeats`：每个种子先对 N 名测试被试求均值，再对 M 个完整重复实验求均值和标准差，表示种子重复的稳定性；M=1 时标准差为 NA。
- `across_subjects`：先对每名被试的 M 个种子求均值，再统计 N 名被试的均值和标准差，作为结束日志的主要汇总，观察跨被试差异。
- `within_subject`：单名被试在 M 个种子间的均值和标准差。
- `within_repeat`：单个种子在 N 名被试间的均值和标准差。

当前输出标准差 SD，与均值单位一致；它不是标准误或置信区间。沿用 `variance_ddof=0`（总体标准差，分母 n），设为 1 则为样本标准差（分母 n−1）；字段名保留以兼容旧配置/断点。仅一个观测时 SD 记 NA，两个以上相同观测的真实零 SD 保留。参考 SSVEPformer 采用 ddof=1，严格对照时应统一此参数。CSV 使用 std、mean_plus_std 字段，不再使用 variance、mean_plus_variance。ITR 仍先逐被试计算再汇总。

训练曲线不含逐 epoch 测试表现，避免据测试集选模型。重新对同一模型做确定性测试不会构成新的独立实验；多轮测试由不同 seed 的独立训练实现。

## 本地冒烟

本轮CPU检查：`.venv/Scripts/python.exe smoke_tests/run_config_smoke.py`，结果位于 `smoke_tests/results/config_check/<experiment_name>/report.json`；71项检查及当前两个组的真实小样本训练、中断恢复、重评估通过。脚本跟随ABLATION_GROUPS逐组验证。冻结的旧模型只用于兼容检查，不是新训练入口。

分段读出的CUDA修复已用于本轮服务器结果。需要在新环境复核时，可运行 `python smoke_tests/run_cuda_readout_smoke.py`：无需EEG文件，按config.py选定GPU检查严格确定性下的前向/反向、参数更新及重复运行一致性。结果位于 `smoke_tests/results/cuda_readout_check/report.json`；CPU不能代替该设备检查。

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

`run_layout_smoke.py` 默认验证 v11 四组 split_bias 断点升级到当前版本 的训练轨迹、预测、历史、标准差汇总和源码认证，结果在 smoke_tests/results/std_reporting_check/。可从 Python 调用 main(output_name="新目录", version=5/6/7/8/9/10) 验证更早版本。全部为 CPU 小样本，不能代替服务器 GPU 性能测量。

## 膨胀卷积执行策略

新实验内部默认使用已验证的split_bias：d>1卷积先计算无偏置结果，再广播加上偏置，规避旧CUDA实现的性能问题。它是固定执行策略，不占日常调参区。历史快照按记录恢复；更早快照缺此字段时明确恢复为native。两条路径数学等价，但浮点轨迹可能不同，仍禁止切换路径后接续旧实验。

必要时用 `python smoke_tests/run_conv_benchmark.py` 重新测量服务器表现。详细历史见 [性能排查](docs/TEMPORAL2_TIMING_AUDIT.zh-CN.md)。
