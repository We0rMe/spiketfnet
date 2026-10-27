"""日常修改入口：实验范围、训练超参数和设备；完整快照定义见 utils/configuration.py。"""

from dataclasses import replace
from pathlib import Path

from utils.configuration import Config, PROJECT_ROOT

CONFIG = Config(
    data_root=Path(r"I:\jjh\SSVEP\data\benchmark"),
    experiment_name="temporal_v1_GAP32_500ms",  # 新结构单独存放，不恢复旧模型
    N=3,  # 先留出 S1～S3 筛查；每折仍由其他全部 34 人训练
    M=2,  # 每折两个独立种子：42、43
    seed=42,  # 起始种子
    windows_ms=(500,),  # 对照已有 500 ms 实验的同三折、同两种子
    device="cuda:1",  # 两张卡各运行一个独立模型
    gpu_ids=(0, 1),  # 可用 GPU 编号
    num_workers=0,  # 数据已在内存；需要时再调整加载进程数
    cpu_threads=4,  # 每个训练进程的 CPU 线程数
    resume=True,  # 从最后一个完整 epoch 恢复

    epochs=100,  # 固定最后一轮评估，不按测试成绩选轮数
    batch_size=128,  # 每个模型的训练批量
    learning_rate=0.001,  # Adam 学习率
    weight_decay=0.001,  # Adam L2 强度
    dropout=0.5,  # 频域分支和全部投影头；保留原设置
    temporal_dropout=0.1,  # 时域编码后及残差块的随机丢弃概率
    lambda_spectral=0.3,  # 频域辅助分类损失权重
    lambda_temporal=0.3,  # 时域辅助分类损失权重，本轮先保持
    temporal_width=64,  # 时域通道数
    stem_kernel=31,  # 初始时间卷积核：原 7 → 31
    residual_kernel=7,  # 残差时间卷积核：原 3 → 7
    temporal_readout_bins=32,  # 分类前保留的时间段数；1 为原整段平均，32 为结构探索
    attention_heads=4,  # 必须整除 temporal_width
    embedding_dim=128,  # 分支特征维度
    fusion_dim=128,  # 融合特征维度
    simulation_steps=4,  # 脉冲仿真步数，与 EEG 采样点数不同
)

# python -m experiments.ablation。
# 可选组：full / spectral_only / temporal_only / equal_fusion。
ABLATION_GROUPS = ("temporal_only",)
ABLATION_CONFIG = replace(
    CONFIG,
    output_root=CONFIG.output_root / CONFIG.experiment_name,
    cache_root=CONFIG.cache_root / "ablation",
)
