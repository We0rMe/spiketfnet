"""统一配置；P=论文，R=相关作者实现，A=已获同意的工程默认值。"""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"I:\jjh\SSVEP\data\benchmark")  # 服务器上修改数据路径
    subject_ids: tuple = tuple(range(1, 36))  # 数据属性：全部被试；N 不改变源被试池
    sampling_rate: int = 250  # 数据已降采样，Hz
    raw_channels: int = 64  # MAT 电极轴
    raw_samples: int = 1500  # MAT 时间轴
    num_classes: int = 40  # MAT target 轴，保留原始顺序
    num_blocks: int = 6  # 每个被试的 block 数
    channels: tuple = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "Pz", "PO5", "PO6")
    cue_seconds: float = 0.5  # 刺激提示时间
    latency_seconds: float = 0.14  # P：视觉延迟

    N: int = 35  # 留出 S1,...,SN；每折用其余全部被试训练
    M: int = 1  # 每折独立训练次数；多种子是用户要求的扩展
    seed: int = 42  # P：起始种子；使用 seed,...,seed+M-1
    windows_ms: tuple = (1000,)  # P：整数毫秒避免浮点截断
    epochs: int = 100  # P：固定训练轮数，最终 epoch 测试
    batch_size: int = 128  # P：训练批量
    learning_rate: float = 0.001  # P：Adam 学习率，不使用调度器
    weight_decay: float = 0.001  # P：Adam L2，不替换为 AdamW
    dropout: float = 0.5  # P：各 dropout 概率
    lambda_spectral: float = 0.3  # A：频域辅助 CE 权重
    lambda_temporal: float = 0.3  # A：时域辅助 CE 权重

    filter_order: int = 4  # P：Butterworth 原型阶数；带通转换后阶数为 8
    band_hz: tuple = (8.0, 64.0)  # P：带通范围
    zero_phase: bool = True  # A：完整 trial 上 SOS 双向滤波，再裁窗；离线协议
    extra_notch: bool = False  # A：数据采集已 notch，默认不重复施加
    notch_hz: float = 50.0  # 仅 extra_notch=True 时使用
    notch_q: float = 30.0  # A：可选 notch 的 Q
    fft_resolution: float = 0.2  # P：补零后的频点间隔，Hz
    spectrum_hz: tuple = (8.0, 64.0)  # R：[8,64)，560 个实虚拼接特征
    include_upper_bin: bool = False  # R：是否包含 64 Hz 频点
    fft_scale: str = "two_over_length"  # R：FFT×2/L；也可改为 "none"

    spectral_depth: int = 2  # P：图 2 的频域编码块数量
    spectral_kernel: int = 31  # A：频域卷积核宽
    spectral_head_multiplier: int = 6  # R：辅助分类头隐藏层 = 类别数×6
    spectral_init_std: float = 0.01  # R：频域 Conv/Linear 权重正态初始化
    temporal_width: int = 64  # A：时域通道数
    stem_kernel: int = 7  # A：初始时间卷积；stride 固定为 1
    residual_kernel: int = 3  # A：膨胀时间卷积核
    dilations: tuple = (1, 2, 4)  # 图 2 三块；A：正文示例的膨胀率
    attention_heads: int = 4  # A：脉冲注意力头数
    attention_scale: float = 0.125  # A：式 (16) gamma，不使用 softmax
    embedding_dim: int = 128  # A：两个分支各自的投影维度
    fusion_dim: int = 128  # A：共享空间维度
    gate_hidden_dim: int = 128  # A：门控 MLP 隐藏维度
    simulation_steps: int = 4  # P：Ts，独立于 EEG 采样点数，可修改
    lif_tau: float = 2.0  # P：LIF 时间常数，beta=1-1/tau
    lif_threshold: float = 1.0  # A：SpikingJelly 常用阈值
    lif_reset: float = 0.0  # A：硬重置电位
    lif_decay_input: bool = False  # A：匹配式 (11) 的 beta*U+I，避免默认 I/tau
    surrogate_alpha: float = 4.0  # A：Sigmoid surrogate 斜率
    detach_reset: bool = False  # A：保留重置操作梯度

    gaze_shift_seconds: float = 0.5  # P：ITR 分母 = Tl+0.5 s
    variance_ddof: int = 0  # 用户要求方差，不是标准差；默认总体方差
    device: str = "cuda"  # 训练：auto/cpu/cuda/cuda:0/cuda:1；auto 优先使用 gpu_ids 中可用卡
    gpu_ids: tuple = (0, 1)  # auto/cuda 时，一张卡一个独立任务进程；单卡可设为 (0,)
    evaluation_device: str = "auto"  # 独立 evaluate.py：auto/cpu/cuda:0/cuda:1；训练后的测试跟随任务设备
    num_workers: int = 0  # 可调 DataLoader 进程数；Windows 安全默认
    cpu_threads: int = 4  # CPU 运算线程数
    deterministic: bool = True  # 尽量启用确定性运算，不进行测试集调参
    t2_conv_execution: str = "native"  # native 保持旧轨迹；split_bias 将 d>1 卷积偏置外置，需新实验目录
    check_gradients: bool = False  # 调试开关：逐 batch 扫描全部梯度；正式训练关闭以减少开销
    time_branches: bool = True  # T2 卷积/LIF 前后向总计、逐层、逐仿真步诊断；有插桩开销
    t2_profile_batch: int = 2  # 首个新 epoch 采样第几个训练 batch 的底层算子；每次启动/续训一次，0 关闭采样
    resume: bool = True  # 同名实验自动跳过完成任务、从上个完整 epoch 恢复
    monitor_every: int = 5  # 每隔若干 epoch 监测源训练被试的 block 1；首/末轮也测，0 禁用
    plot_every: int = 1  # 每隔若干 epoch 覆盖更新曲线和单次报告；不产生逐轮图片
    cache_root: Path = PROJECT_ROOT / "cache"  # 仅一个可复用预处理缓存文件
    output_root: Path = PROJECT_ROOT / "results"  # 正式实验输出
    experiment_name: str = "spiketfnet"  # 同名恢复；换名称可开始全新实验
    probe_trials: int = 4  # 保存供未来效率测量使用的代表性输入数量

    # 内部记录字段，不作为运行选择项修改；experiments/ablation.py 自动填入并保存到断点。
    model_variant: str = field(default="full", repr=False)

    @property
    def seeds(self):
        return tuple(range(self.seed, self.seed + self.M))

    @property
    def test_subjects(self):
        return tuple(range(1, self.N + 1))

    @property
    def fft_length(self):
        return round(self.sampling_rate / self.fft_resolution)

    @property
    def spectral_features(self):
        lo, hi = (round(f / self.fft_resolution) for f in self.spectrum_hz)
        return 2 * (hi - lo + int(self.include_upper_bin))

    @property
    def start_sample(self):
        return round((self.cue_seconds + self.latency_seconds) * self.sampling_rate)

    def samples(self, window_ms):
        return self.sampling_rate * window_ms // 1000

    def validate(self):
        if self.model_variant not in ("full", "spectral_only", "temporal_only", "equal_fusion"):
            raise ValueError(f"Unknown model variant: {self.model_variant}")
        integer_fields = ("N", "M", "seed", "epochs", "batch_size", "sampling_rate", "simulation_steps",
                          "temporal_width", "attention_heads", "spectral_depth", "embedding_dim", "fusion_dim", "gate_hidden_dim")
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ValueError("Counts, seeds and dimensions must be integers")
        if len(set(self.subject_ids)) != len(self.subject_ids) or len(self.subject_ids) < 2:
            raise ValueError("subject_ids must contain at least two unique subjects")
        if not 1 <= self.N <= len(self.subject_ids) or not set(self.test_subjects) <= set(self.subject_ids):
            raise ValueError("N must select S1,...,SN within the available subjects")
        if self.M < 1 or self.epochs < 1 or self.batch_size < 1 or self.seed < 0:
            raise ValueError("M, epochs and batch_size must be positive; seed nonnegative")
        if not self.windows_ms or len(set(self.windows_ms)) != len(self.windows_ms):
            raise ValueError("windows_ms must be nonempty and unique")
        if any(not isinstance(w, int) or w <= 0 for w in self.windows_ms):
            raise ValueError("Window lengths must be positive integer milliseconds")
        if self.start_sample < 0 or self.start_sample + self.samples(max(self.windows_ms)) > self.raw_samples:
            raise ValueError("Requested windows exceed the stored trial")
        if self.fft_length < self.samples(max(self.windows_ms)):
            raise ValueError("FFT length cannot truncate the EEG window")
        if self.attention_heads < 1 or self.temporal_width % self.attention_heads:
            raise ValueError("temporal_width must be divisible by attention_heads")
        if min(self.temporal_width, self.spectral_depth, self.embedding_dim, self.fusion_dim, self.gate_hidden_dim) < 1:
            raise ValueError("Network dimensions and depth must be positive")
        if self.learning_rate <= 0 or min(self.weight_decay, self.lambda_spectral, self.lambda_temporal) < 0:
            raise ValueError("Invalid optimizer or auxiliary loss settings")
        if self.cpu_threads < 1 or self.num_workers < 0 or self.gaze_shift_seconds < 0:
            raise ValueError("Invalid runtime or ITR settings")
        if any(type(value) is not bool for value in (self.resume, self.check_gradients, self.time_branches)):
            raise ValueError("resume, check_gradients and time_branches must be bool")
        if type(self.monitor_every) is not int or self.monitor_every < 0 or type(self.plot_every) is not int or self.plot_every < 1:
            raise ValueError("monitor_every must be nonnegative; plot_every must be positive integers")
        if type(self.t2_profile_batch) is not int or self.t2_profile_batch < 0:
            raise ValueError("t2_profile_batch must be a nonnegative integer")
        if self.t2_conv_execution not in ("native", "split_bias"):
            raise ValueError("t2_conv_execution must be native or split_bias")
        for name in (self.device, self.evaluation_device):
            if name not in ("auto", "cpu", "cuda") and not (
                    isinstance(name, str) and name.startswith("cuda:") and name[5:].isdigit()):
                raise ValueError("device/evaluation_device must be auto, cpu, cuda or cuda:<index>")
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids) or any(
                type(index) is not int or index < 0 for index in self.gpu_ids):
            raise ValueError("gpu_ids must contain unique, nonnegative integer GPU indices")
        if self.lif_tau <= 1 or self.simulation_steps < 1 or not self.dilations:
            raise ValueError("Invalid LIF configuration or residual block count")
        if any(k < 1 or k % 2 == 0 for k in (self.stem_kernel, self.residual_kernel, self.spectral_kernel)):
            raise ValueError("Convolution kernels must be positive and odd")
        if any(d < 1 for d in self.dilations) or not 0 <= self.dropout < 1:
            raise ValueError("Invalid dilations or dropout")
        if self.variance_ddof not in (0, 1) or self.fft_scale not in ("none", "two_over_length"):
            raise ValueError("Unsupported variance or FFT convention")
        if not 0 < self.band_hz[0] < self.band_hz[1] < self.sampling_rate / 2:
            raise ValueError("Invalid bandpass frequencies")
        if not 0 < self.spectrum_hz[0] < self.spectrum_hz[1] < self.sampling_rate / 2 or self.fft_resolution <= 0:
            raise ValueError("Invalid spectral frequencies")
        grid_values = [self.sampling_rate / self.fft_resolution, *(f / self.fft_resolution for f in self.spectrum_hz)]
        if any(not math.isclose(v, round(v), abs_tol=1e-8, rel_tol=0) for v in grid_values):
            raise ValueError("Sampling rate and spectral boundaries must align with the FFT grid")
        if self.spectral_features <= 0 or self.probe_trials < 1:
            raise ValueError("Invalid spectrum range or probe count")
        if Path(self.experiment_name).name != self.experiment_name or self.experiment_name in ("", ".", ".."):
            raise ValueError("experiment_name must be a directory name")
        return self

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, values):
        defaults = cls()
        # v9 及更早归档没有此字段，不能继承后来手动改过的运行默认值。
        restored = {"t2_conv_execution": values.get("t2_conv_execution", "native")}
        for key, value in values.items():
            original = getattr(defaults, key)
            restored[key] = Path(value) if isinstance(original, Path) else tuple(value) if isinstance(original, tuple) else value
        return cls(**restored).validate()


CONFIG = Config()

# 消融运行设置：python -m experiments.ablation 只运行下面列出的组，这是唯一的组别选择入口。
# 单组示例：ABLATION_GROUPS = ("spectral_only",)
ABLATION_GROUPS = ("full", "spectral_only", "temporal_only", "equal_fusion")
ABLATION_WINDOWS_MS = (500,)  # 可选窗口，例如 (500, 1000)；对所有已选组一致
ABLATION_ROOT = CONFIG.output_root / "ablation"  # 批次根目录；组名/窗口/被试/种子子目录自动生成

# N/M/seed、设备和其余参数继承 CONFIG。
ABLATION_CONFIG = replace(
    CONFIG,
    windows_ms=ABLATION_WINDOWS_MS,
    output_root=ABLATION_ROOT,
    cache_root=CONFIG.cache_root / "ablation",  # 与普通训练缓存隔离
)
