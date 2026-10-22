"""SpikeTFNet：频域卷积编码、SpikingJelly 时域分支及特征门控。"""

import torch
from torch import nn
from spikingjelly.activation_based import functional, neuron, surrogate

from config import Config


def projection(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU(), nn.Dropout(dropout))


def lif_node(cfg):
    return neuron.LIFNode(tau=cfg.lif_tau, decay_input=cfg.lif_decay_input,
                          v_threshold=cfg.lif_threshold, v_reset=cfg.lif_reset,
                          surrogate_function=surrogate.Sigmoid(alpha=cfg.surrogate_alpha),
                          detach_reset=cfg.detach_reset, step_mode="s", backend="torch")


class SpectralBlock(nn.Module):
    def __init__(self, channels, features, cfg):
        super().__init__()
        self.pre_conv = nn.LayerNorm(features)
        self.conv = nn.Sequential(nn.Conv1d(channels, channels, cfg.spectral_kernel, padding=cfg.spectral_kernel // 2),
                                  nn.LayerNorm(features), nn.GELU(), nn.Dropout(cfg.dropout))
        self.pre_ff = nn.LayerNorm(features)
        # 原 SSVEPformer 的 channel MLP 在每个通道的频谱长度轴上变换。
        self.ff = nn.Sequential(nn.Linear(features, features), nn.GELU(), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.conv(self.pre_conv(x))
        return x + self.ff(self.pre_ff(x))


class SpectralBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels, features = 2 * len(cfg.channels), cfg.spectral_features
        self.combine = nn.Sequential(nn.Conv1d(len(cfg.channels), channels, 1), nn.LayerNorm(features),
                                      nn.GELU(), nn.Dropout(cfg.dropout))
        self.blocks = nn.Sequential(*(SpectralBlock(channels, features, cfg) for _ in range(cfg.spectral_depth)))
        flat, hidden = channels * features, cfg.num_classes * cfg.spectral_head_multiplier
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(flat, hidden), nn.LayerNorm(hidden),
                                         nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(hidden, cfg.num_classes))
        self.embedding = projection(flat, cfg.embedding_dim, cfg.dropout)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, std=cfg.spectral_init_std)

    def forward(self, spectrum):
        flat = self.blocks(self.combine(spectrum)).flatten(1)
        return self.embedding(flat), self.classifier(flat)


class SplitBiasConv1d(nn.Conv1d):
    """保留原 weight/bias 参数，避开旧 CUDA 膨胀实现的逐通道偏置填充。"""

    def forward(self, x):
        output = self._conv_forward(x, self.weight, None)
        if self.bias is not None:
            # 数学等价，浮点累加顺序可不同；不承诺与 native 的训练轨迹逐位相同。
            shape = (1, -1, 1) if output.ndim == 3 else (-1, 1)
            output = output + self.bias.view(shape)
        return output


class SpikingResidualBlock(nn.Module):
    def __init__(self, cfg, dilation):
        super().__init__()
        width = cfg.temporal_width
        conv = SplitBiasConv1d if cfg.t2_conv_execution == "split_bias" and dilation > 1 else nn.Conv1d
        self.dilated = nn.Sequential(conv(width, width, cfg.residual_kernel,
                                               padding=dilation * (cfg.residual_kernel // 2), dilation=dilation),
                                     nn.BatchNorm1d(width), lif_node(cfg))
        self.pointwise = nn.Sequential(nn.Conv1d(width, width, 1), nn.BatchNorm1d(width), lif_node(cfg))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return x + self.dropout(self.pointwise(self.dilated(x)))


class SpikeAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.scale = cfg.attention_heads, cfg.attention_scale
        width = cfg.temporal_width
        self.qkv = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, 1), nn.BatchNorm1d(width), lif_node(cfg))
                                  for _ in range(3)])
        self.attended = nn.Sequential(nn.BatchNorm1d(width), lif_node(cfg))
        self.output = nn.Sequential(nn.Conv1d(width, width, 1), nn.BatchNorm1d(width), lif_node(cfg))

    def forward(self, x):
        batch, width, length = x.shape
        q, k, v = [layer(x).reshape(batch, self.heads, width // self.heads, length).transpose(-1, -2)
                   for layer in self.qkv]
        # 每个仿真步内沿 EEG token 轴做 spike coincidence，不进行 softmax。
        current = self.scale * ((q @ k.transpose(-1, -2)) @ v)
        current = current.transpose(-1, -2).reshape(batch, width, length)
        return x + self.output(self.attended(current))


class TemporalBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.steps = cfg.simulation_steps
        self.current = nn.Sequential(nn.Conv1d(len(cfg.channels), cfg.temporal_width, cfg.stem_kernel,
                                               padding=cfg.stem_kernel // 2), nn.BatchNorm1d(cfg.temporal_width))
        self.encoder = lif_node(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*(SpikingResidualBlock(cfg, dilation) for dilation in cfg.dilations))
        self.attention = SpikeAttention(cfg)
        self.embedding = projection(cfg.temporal_width, cfg.embedding_dim, cfg.dropout)
        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_classes)

    def forward(self, time):
        current = self.current(time)  # 恒定电流只计算一次，stem BN 也只更新一次。
        embeddings, logits = [], []
        for _ in range(self.steps):
            spikes = self.dropout(self.encoder(current))
            features = self.attention(self.blocks(spikes))
            embedding = self.embedding(features.mean(dim=-1))
            embeddings.append(embedding)
            logits.append(self.classifier(embedding))
        return torch.stack(embeddings).mean(0), torch.stack(logits).mean(0)


class GatedFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.spectral = projection(cfg.embedding_dim, cfg.fusion_dim, cfg.dropout)
        self.temporal = projection(cfg.embedding_dim, cfg.fusion_dim, cfg.dropout)
        self.gate = nn.Sequential(nn.LayerNorm(4 * cfg.fusion_dim), nn.Linear(4 * cfg.fusion_dim, cfg.gate_hidden_dim),
                                  nn.GELU(), nn.Linear(cfg.gate_hidden_dim, cfg.fusion_dim), nn.Sigmoid())
        self.classifier = nn.Linear(cfg.fusion_dim, cfg.num_classes)

    @staticmethod
    def blend(spectral, temporal, gate):
        return (1 - gate) * spectral + gate * temporal

    def forward(self, spectral, temporal):
        spectral, temporal = self.spectral(spectral), self.temporal(temporal)
        relations = torch.cat((spectral, temporal, spectral * temporal, (spectral - temporal).abs()), dim=-1)
        gate = self.gate(relations)
        fused = self.blend(spectral, temporal, gate)
        return self.classifier(fused), gate, fused


class SpikeTFNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        cfg.validate()
        self.spectral = SpectralBranch(cfg)
        self.temporal = TemporalBranch(cfg)
        self.fusion = GatedFusion(cfg)

    def forward(self, time, spectrum):
        functional.reset_net(self.temporal)
        try:
            hf, of = self.spectral(spectrum)
            ht, ot = self.temporal(time)
            logits, gate, embedding = self.fusion(hf, ht)
            return {"logits": logits, "spectral_logits": of, "temporal_logits": ot,
                    "gate": gate, "embedding": embedding}
        finally:
            # 清理状态不切断输出的计算图；不同 trial/batch 不共享膜电位。
            functional.reset_net(self.temporal)


def joint_loss(outputs, labels, cfg):
    ce = nn.functional.cross_entropy
    return (ce(outputs["logits"], labels) + cfg.lambda_spectral * ce(outputs["spectral_logits"], labels)
            + cfg.lambda_temporal * ce(outputs["temporal_logits"], labels))
