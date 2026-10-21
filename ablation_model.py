"""复用原编码器、投影及分类头；只删除消融指定的组件。"""

from torch.nn import functional as F
from spikingjelly.activation_based import functional

from model import SpikeTFNet, joint_loss as full_joint_loss


class AblationSpikeTFNet(SpikeTFNet):
    def __init__(self, cfg):
        # 先按原顺序初始化，再删除组件，使同种子的公共层初值和 RNG 状态一致。
        super().__init__(cfg)
        self.model_variant = cfg.model_variant
        if self.model_variant == "full":
            raise ValueError("Use the original SpikeTFNet for the full baseline")
        del self.fusion.gate
        if self.model_variant == "spectral_only":
            del self.temporal, self.fusion.temporal
        elif self.model_variant == "temporal_only":
            del self.spectral, self.fusion.spectral

    def forward(self, time, spectrum):
        outputs = {}
        if hasattr(self, "temporal"):
            functional.reset_net(self.temporal)
        try:
            if hasattr(self, "spectral"):
                hf, outputs["spectral_logits"] = self.spectral(spectrum)
            if hasattr(self, "temporal"):
                ht, outputs["temporal_logits"] = self.temporal(time)
            if hasattr(self, "spectral"):
                zf = self.fusion.spectral(hf)
            if hasattr(self, "temporal"):
                zt = self.fusion.temporal(ht)
            if self.model_variant == "equal_fusion":
                fused = 0.5 * zf + 0.5 * zt  # 所有 trial/维度共享的不可学习常数。
            elif self.model_variant == "spectral_only":
                fused = zf
            else:
                fused = zt
            outputs.update(logits=self.fusion.classifier(fused), embedding=fused)
            return outputs
        finally:
            if hasattr(self, "temporal"):
                functional.reset_net(self.temporal)


def build_model(cfg):
    return SpikeTFNet(cfg) if cfg.model_variant == "full" else AblationSpikeTFNet(cfg)


def joint_loss(outputs, labels, cfg):
    if cfg.model_variant in ("full", "equal_fusion"):
        return full_joint_loss(outputs, labels, cfg)
    loss = F.cross_entropy(outputs["logits"], labels)
    if cfg.model_variant == "spectral_only":
        return loss + cfg.lambda_spectral * F.cross_entropy(outputs["spectral_logits"], labels)
    return loss + cfg.lambda_temporal * F.cross_entropy(outputs["temporal_logits"], labels)
