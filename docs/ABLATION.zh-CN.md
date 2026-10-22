# SpikeTFNet 消融实验

仅使用 Benchmark 40 类数据，默认 500 ms、35 折 LOSO、每折一个种子 42、100 epochs。
四组共 140 个独立训练任务。500 ms 的完整模型须重新训练，1000 ms 结果不能替代该基线。

## 配置与运行

参数统一在 `config.py`。**只修改文件末尾的 `ABLATION_GROUPS` 来选择组别**；不再提供 MODEL_VARIANTS 这个易混淆的配置项。
`Config.model_variant` 是自动写入快照/断点的内部字段，不要手动修改。`ABLATION_CONFIG` 也是内部派生配置，无需修改。

```python
ABLATION_GROUPS = ("full", "spectral_only", "temporal_only", "equal_fusion")
ABLATION_WINDOWS_MS = (500,)
ABLATION_ROOT = CONFIG.output_root / "ablation"
```

确认 `CONFIG.data_root` 为服务器 Benchmark 路径。N、M、seed、优化器、网络参数和设备均继承 `CONFIG`，当前 N=35、M=1、seed=42、Ts=4。
服务器使用 `device="cuda"`、`gpu_ids=(0, 1)`，然后在项目根目录执行：

```bash
python -m experiments.ablation
```

在项目根目录执行。消融入口和模型变体集中在 `experiments/`，仍共享根目录的配置、数据、训练和评估脚本；结果路径与整理前相同。

按配置顺序逐组训练；每组内部两张 GPU 分担独立的窗口/被试/种子任务，每张卡一个进程。
单模型的 batch size、学习率和数据划分保持原样。`device="cuda:0"` 则仅使用第一张卡。

只训练一组时，例如：

```python
ABLATION_GROUPS = ("spectral_only",)
```

运行哪个组只需修改上述一行，然后执行同一个 `python -m experiments.ablation`。组名、窗口、被试和种子目录全部自动命名，不需要逐组修改 experiment_name。
`Config.experiment_name` 只控制 `python train.py` 的普通完整模型实验目录。

窗口仍可选择，例如 `ABLATION_WINDOWS_MS=(500, 1000)`，已选组统一应用这些窗口。
已经开始的实验若改变最大窗口，会改变预处理指纹；应改用新的 `ABLATION_ROOT`，例如 `CONFIG.output_root / "ablation_1000ms"`。
学习率、训练轮数、分支结构等科学参数改变也应使用新输出根目录，避免与旧结果混合。
以后改进某个分支时，四组依然从同一套配置和模块创建，不能只给某组额外调参后直接归因于组件消融。

## 模型与损失的约定

`experiments/variants.py` 复用 `model.py` 中的频域/时域编码器、融合空间投影和最终分类头。默认 `t2_conv_execution="native"` 保持原模型执行；可选 split_bias 仅将 d>1 卷积的偏置改为外置广播相加，参数和数学定义不变，但浮点轨迹可能变化。新消融批次应统一使用同一执行模式，并设置独立 `ABLATION_ROOT`，不能在已有批次中切换模式。
设 `zf`、`zt` 是原融合模块中各自投影后的特征，`C` 是原来的最终线性分类头。

| 组别 | 最终分类路径 | 损失 |
| --- | --- | --- |
| full | C((1-g)zf + gzt) | 主 CE + 0.3 频域辅助 CE + 0.3 时域辅助 CE |
| spectral_only | C(zf) | 主 CE + 0.3 频域辅助 CE |
| temporal_only | C(zt) | 主 CE + 0.3 时域辅助 CE |
| equal_fusion | C(0.5zf + 0.5zt) | 主 CE + 0.3 频域辅助 CE + 0.3 时域辅助 CE |

单分支组保留对应的融合投影和最终分类头，也保留该分支原有的辅助分类头；被移除分支及其投影、辅助头和门控不参与计算或优化。
单分支特征不乘 0.5。该分类路径和辅助损失处理是用户确认的工程约定，原文消融段未明确给出。

`equal_fusion` 直接删除门控网络，0.5 是所有样本和维度共享、不可学习的常数。
编码器、投影、分类头、辅助头均不改动，编码器及投影的执行顺序也与完整模型一致。

每次按完整模型原来的顺序随机初始化，然后删除指定组件，因此同一折/种子的公共层具有完全相同的初始参数，初始化后的随机状态也一致。
删除只发生在构建阶段，之后优化器只接收保留的参数；不加载已训练模型，不跨折继承权重。
训练采样器使用独立的种子生成器，四组样本顺序相同。不同结构会消耗不同数量的训练随机数，不承诺单分支组的所有 Dropout 掩码逐元素相同。

原分支接口仍为 `(embedding, auxiliary_logits)`。后续分支改进主要修改 `SpectralBranch` 或 `TemporalBranch`；保持该接口即可复用融合、训练与评估。
辅助权重、Ts、编码深度等仍由原配置控制。

## 数据与评估边界

预处理、LOSO 和原优化器流程共用现有脚本。N 只限制 S1 至 SN 的测试折数，不缩小源被试池；正式每折仍用其他 34 人训练。
源训练被试 block 1 的监测是训练集监测，不是独立验证集。留出被试只在固定末轮模型上测试，不用于早停或调参。
500 ms 对应 125 个 EEG 采样点；Ts=4 与采样点数无关。固定 FFT 分辨率及边界，因此频谱输入仍为 560 维。

## 输出与恢复

```text
results/ablation/
  report.md                  # 当前选择的所有组的训练时间累计及各组报告入口
  comparison.csv
  details/training.lock
  full/
  spectral_only/
  temporal_only/
  equal_fusion/
    report.md
    summary.csv
    details/                 # 配置、源码、数据依据和调度日志
    window_0500ms/test_S01/seed_42/
      report.md
      metrics.json
      training.log
      training.png
      confusion.png
      details/               # history.csv、model.pt、evaluation.npz、provenance.json
```

每组沿用实时日志、覆盖更新的曲线及每 epoch 滚动恢复文件 `details/resume.pt`。`resume=True` 时再次执行 `python -m experiments.ablation` 即可续训；完成的任务跳过，不重复训练。
窗口、折和种子仍各自独立。恢复校验包括组别、配置、源码和数据，不能将 full 的断点当成其他组的起点。

`comparison.csv` 的 ACC/ITR 行只汇总本次配置选中的、在该窗口下所有请求折/种子均完成的组。分别列出跨被试与跨种子的均值和方差。
当 full 组也完成时，额外计算其他组减 full 的配对差值，先按相同被试/种子相减再汇总。
M=1 时跨种子的总体方差为零，不代表结果稳定；可查看跨被试统计。
中途停止时可能只有部分组有总表行，各组 report.md 显示各自进度。
若此前逐组运行，将 `ABLATION_GROUPS` 恢复为上面的四组列表后再次启动，会跳过已完成任务并重新生成四组比较表。

仅保存存在的分支输出；无门控组不保存虚假的自适应 gate，频域组的脉冲统计为空数组。
真实逐 trial 预测、身份及辅助输出保留在 evaluation.npz，不另存重复预测文件。
ITR 沿用原公式，500 ms 时选择周期为 0.5 + 0.5 = 1.0 s。

新版实验清单为 v10。v9 四组已验证通过认证后保持 native 路径续训；更早版本已有迁移规则保留，旧模型代码必须通过固定摘要认证、恢复为 native，预处理和科学配置仍须兼容。结果目录不需要搬动。新版源码快照保留包目录，旧文件名按映射校验。详见 [膨胀卷积改善与数值边界](TEMPORAL2_TIMING_AUDIT.zh-CN.md)。
旧配置缺少 model_variant 时解释为 full，不放宽未知源码变更的校验。
升级服务器前应先停止旧进程及其子进程，再同步代码，保留结果目录和原配置，以 `python train.py` 继续原完整模型实验。
原 1000 ms 实验使用原输出路径，消融使用新根目录，不移动旧权重。

## 联合训练与第二模块诊断

完整模型对每个 batch 依次计算频域、时域分支，再融合；主 CE 与辅助 CE 相加后只调用一次 backward 和一次 Adam.step，两个分支及融合模块联合优化。
没有分别预训练后再融合，也没有把同一个模型的两个分支分放到两张卡。两张卡运行的是不同 LOSO 任务。

`CONFIG.time_branches=True` 当前只开启第二模块的算子诊断。先显示 T2 Conv Forward / T2 Conv Backward / T2 LIF Forward / T2 LIF Backward 四项秒数，再附逐层、逐仿真步、主机提交区间和调用数，不显示 trial 均值。full、temporal_only、equal_fusion 使用同一计时范围，spectral_only 四项为 0。

`CONFIG.t2_profile_batch=2` 在每个任务每次启动/续训后的首个新 epoch 对一个 batch 采样底层算子及 CUDA kernel；0 关闭此采样。采样有明显开销，后续 epoch 不重复采样。详细记录集中到该任务 `details/t2_diagnostics.json`，运行时配置写入 provenance.json；现有四项及汇总保留，旧 epoch 没有细节的不补造。模型与训练计算保持不变，计时、同步排查及研究依据详见 [第二模块诊断与排查](TEMPORAL2_TIMING_AUDIT.zh-CN.md)。

## 本地验证

```powershell
.venv/Scripts/python.exe smoke_tests/run_ablation_smoke.py
.venv/Scripts/python.exe smoke_tests/run_timing_smoke.py
.venv/Scripts/python.exe smoke_tests/run_layout_smoke.py
```

所有测试脚本、旧版验证样本和产物位于 smoke_tests；默认写 smoke_tests/results/ablation_check，已存在时拒绝覆盖。
此脚本使用本地 G:\SSVEP\data\benchmark 的 S1–S3 小样本，不执行正式 35 折训练。
覆盖公共层初始化、固定等权融合、单分支投影/分类头、梯度、500/1000 ms 输入、无脉冲层评估、四组中断恢复、独立重评估与旧 v3 完整模型迁移。
双 CPU 子进程验证任务调度；实际双 4090 的执行和性能仍需服务器验证。
本次没有新增依赖，继续使用已有 requirements.txt 和冒烟专用 smoke_tests/requirements.txt。
