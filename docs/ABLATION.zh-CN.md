# SpikeTFNet 消融实验

仅使用Benchmark 40类数据。当前为时间分支读出结构验证：500 ms、3折、每折种子42和43、100 epochs，共6个独立模型。每折仍由其余34名被试的全部试次训练。

## 配置与运行

唯一日常入口为 `config.py`：`CONFIG` 设置窗口、N/M/seed、超参数及设备，`ABLATION_GROUPS=("temporal_only",)` 设置当前组别。另可选full / spectral_only / equal_fusion。模型变体字段与派生配置由程序自动维护。

```bash
python -m experiments.ablation
```

窗口直接继承 `CONFIG.windows_ms`，批次目录为 `results/<experiment_name>/`，无需重复配置窗口或根目录。更换参数时更换experiment_name；若复用已存在目录，程序会检查配置、源码与数据一致性。

两张GPU分担独立的折/种子任务，每张卡一个进程。只改变 `temporal_readout_bins`：1是原整段平均，32是本轮按时间顺序展开的分段读出。其余科学设置沿用上一轮500 ms；复用旧服务器同三折、同两种子的结果作对照，不重跑旧组。范围、对照数值及判断规则见README，不把开发筛查当完整论文结果。

## 模型与损失的约定

`experiments/variants.py` 复用 `model.py` 中的频域/时域编码器、融合空间投影和最终分类头。新实验内部固定默认split_bias以规避已确认的GPU低效路径，旧快照保留原执行策略。不得在已有批次中更改执行策略或科学参数后继续训练。
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

固定同一套结构配置时，每次按完整模型原来的顺序随机初始化，然后删除指定组件，因此四种消融组同一折/种子的公共层具有完全相同的初始参数，初始化后的随机状态也一致。这个保证不跨越1段与32段两种结构：投影矩阵尺寸不同，会改变后续层初始化和随机数消耗。
删除只发生在构建阶段，之后优化器只接收保留的参数；不加载已训练模型，不跨折继承权重。
训练采样器使用独立的种子生成器，四组样本顺序相同。不同结构会消耗不同数量的训练随机数，不承诺单分支组的所有 Dropout 掩码逐元素相同。

原分支接口仍为 `(embedding, auxiliary_logits)`。后续分支改进主要修改 `SpectralBranch` 或 `TemporalBranch`；保持该接口即可复用融合、训练与评估。
辅助权重、Ts等仍由config.py控制；论文指定的频域2层、时域3个残差块作为固定实现约定保存。

## 数据与评估边界

预处理、LOSO 和原优化器流程共用现有脚本。N 只限制 S1 至 SN 的测试折数，不缩小源被试池；正式每折仍用其他 34 人训练。
源训练被试 block 1 的监测是训练集监测，不是独立验证集。留出被试只在固定末轮模型上测试，不用于早停或调参。
500 ms 对应 125 个 EEG 采样点；Ts=4 与采样点数无关。固定 FFT 分辨率及边界，因此频谱输入仍为 560 维。

## 输出与恢复

```text
results/temporal_readout32_v1_500ms/
  report.md                  # 当前选择的所有组的训练时间累计及各组报告入口
  comparison.csv
  details/training.lock
  temporal_only/
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

`comparison.csv` 的 ACC/ITR 行只汇总本次配置选中的、在该窗口下所有请求折/种子均完成的组。分别列出跨被试与跨种子的均值和标准差。
当 full 组也完成时，额外计算其他组减 full 的配对差值，先按相同被试/种子相减再汇总。
M=1 时种子间标准差记 NA；报告主表使用跨被试均值 ± 标准差（先对每名被试平均种子，ddof 沿用配置）。
中途停止时可能只有部分组有总表行，各组 report.md 显示各自进度。
若此前逐组运行，将 `ABLATION_GROUPS` 恢复为上面的四组列表后再次启动，会跳过已完成任务并重新生成四组比较表。

仅保存存在的分支输出；无门控组不保存虚假的自适应 gate，频域组的脉冲统计为空数组。
真实逐 trial 预测、身份及辅助输出保留在 evaluation.npz，不另存重复预测文件。
ITR 沿用原公式，500 ms 时选择周期为 0.5 + 0.5 = 1.0 s。

新版实验清单为 v14，保存读出段数；缺少该字段的历史配置恢复为1。已认证v13源码只允许保持原整段平均计算路径升级；更早版本已有迁移规则保留，预处理和科学配置仍须兼容。改变段数必须使用新实验目录，不能接着训练旧断点。源码快照保留包目录，旧文件名按映射校验。详见 [膨胀卷积改善与数值边界](TEMPORAL2_TIMING_AUDIT.zh-CN.md)。
旧配置缺少 model_variant 时解释为 full，不放宽未知源码变更的校验。
升级服务器前应先停止旧进程及其子进程，再同步代码，保留结果目录和原配置，以 `python train.py` 继续原完整模型实验。
原 1000 ms 实验使用原输出路径，消融使用新根目录，不移动旧权重。

## 联合训练与耗时

完整模型对每个 batch 依次计算频域、时域分支，再融合；主 CE 与辅助 CE 相加后只调用一次 backward 和一次 Adam.step，两个分支及融合模块联合优化。
没有分别预训练后再融合，也没有把同一个模型的两个分支分放到两张卡。两张卡运行的是不同 LOSO 任务。

`CONFIG.time_branches=True` 默认记录 Spectral Times、Temporal Times、Total Time。两分支分别含前向+反向，总时间含完整训练循环；日志不显示 trial 均值。各组共用同一口径，缺失分支为 0。沿用 report.md、summary.csv/comparison.csv 展示累计、均摊及覆盖率，旧局部计时不伪造为完整分支时间。

`CONFIG.t2_profile_batch=0` 默认关闭细粒度计时与 profiler。>0 才开启逐层/逐步诊断，并在每次启动后首个新 epoch 采样指定 batch；旧日志与诊断记录保留。模型和优化过程不变，详见 [第二模块诊断与排查](TEMPORAL2_TIMING_AUDIT.zh-CN.md)。

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
