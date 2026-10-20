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
| `smoke_tests/` | 全部冒烟脚本与结果；与正式实验隔离 |

## 运行

先修改 `config.py`。没有命令行参数解析，也不需要额外 YAML/JSON 配置。

```python
data_root = Path("/your/server/benchmark")
N = 35                    # 测试被试依次 S1,...,SN
M = 3                     # 每折独立训练 3 次
seed = 42                 # 种子 42、43、44
simulation_steps = 4       # Ts，可以修改
experiment_name = "benchmark_spiketfnet_v1"
```

N 只限制测试折数。即使 N=1，正式训练仍用除 S1 外的全部 34 名被试，不能把源被试池缩小为 N 个人。默认完整任务为 6 个窗口 × 35 折 × 3 种子 = 630 次独立训练。调试服务器配置时可先设 N=1、M=1、`windows_ms=(500,)`；不要把这种子集成绩当成完整 LOSO 结果。

服务器建议 Python 3.9–3.12（本地已测试 3.12）。先在服务器新建自己的环境；根据服务器驱动选择 PyTorch 2.5.1 的 CUDA 构建，然后安装其余依赖。以下 CUDA 12.1 命令依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)，仅在服务器驱动兼容时使用：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python train.py
```

默认自动选择可用的 CUDA，否则使用 CPU。项目当前 Windows `.venv` 是冒烟专用 CPU 环境，不上传、不拷贝到服务器。库版本与论文所述 Python 3.9/CUDA 11.6 环境不完全相同，运行时版本会归档。

`python data.py` 可单独准备缓存，`python train.py` 也会自动调用它。缓存只有 `cache/benchmark_prepared.npz` 一个文件，保存选通道后的最大窗片段与数据依据；每次核验原文件 SHA-256、预处理配置和数据脚本哈希，不把旧配置缓存误当新数据。原始数据从不被改写。

使用 `python evaluate.py` 可重新评估 `output_root / experiment_name` 下所有最终 checkpoint。模型结构和预处理设置读取训练时快照；当前配置仅提供实验位置、数据根路径与运行设备。重新评估覆盖同一份评估文件，不再创建一套重复指标。代码/原数据变化会触发一致性检查。

已存在的实验目录会被拒绝覆盖。更改 `experiment_name` 可启动新实验。本版本不实现中断训练恢复，只保存完成轮次的唯一最终 checkpoint；失败信息保存在该实验 `details/experiment.json` 中。

## 输出

```text
runs/benchmark_spiketfnet_v1/
  summary.csv                         # 唯一汇总表，包含“均值 ± 方差”列
  details/
    experiment.json                   # 配置、版本、用途、状态、代码哈希
    dataset.json                      # 预处理和数据文件指纹
    source/                           # 本轮源代码及实现假设快照
  window_0500ms/test_S01/seed_42/
    metrics.json                      # 本轮唯一 ACC/ITR 数值记录
    training.png                      # 一张图：训练联合损失与训练 ACC
    details/
      model.pt                        # 固定最后一轮的权重
      history.csv                     # 训练曲线的唯一原始数据
      provenance.json                 # 测试/训练被试、种子与完成状态
      evaluation.npz                  # 预测、身份、特征、脉冲计数及测量用输入
```

`evaluation.npz` 不使用 pickle。内容包括真实标签、三路 logits、融合特征、门控、准确的训练/测试 trial 身份、逐 LIF 层的发放总数与元素总数，以及少量已预处理时域/频域输入。结合 checkpoint、代码和配置，可在服务器后续计算混淆矩阵、其他分类指标、参数量及 FLOPs/SOPs，或重新测量时延和吞吐量。时延须在指定硬件上重新 warm-up、batch size=1 并同步 CUDA 测量；本地仪器化测试时间不能替代论文的时延。

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
.\.venv\Scripts\python.exe smoke_tests\run_smoke.py
```

所有临时缓存、报告、训练及评估结果都进入 `smoke_tests/results/`。脚本先做数值与隔离检查，再对真实 S1–S3 数据进行预处理；集成部分取每人 8 个 trial、S1/S2 两折、42/43 两个种子、每次 2 epochs。完整网络尺寸、40 类输出和 Ts=4 保持不变；六个窗口另做形状检查。最后加载四个 checkpoint 重评估，确认 logits 和汇总完全一致。

集成目录已存在时会拒绝覆盖；需要重跑时先自行移走旧 `smoke_tests/results/integration/`。这些低轮数、小样本成绩仅验证软件流程，不能用于评价论文复现性能。`.gitignore` 已排除 `.venv`、缓存、正式和冒烟结果、原始 MAT 与 PDF。
