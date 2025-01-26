from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from omegaconf import DictConfig
import torch.utils.checkpoint as checkpoint
from magical_drones.models.base_gan.generator import BaseGenerator


class EfficientBlock(nn.Module):
    """Efficient block using depthwise separable convolutions and channel attention"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.dwconv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            spectral_norm(nn.Conv2d(channels, channels, 1, bias=False)),
        )

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.dwconv(out)
        out = out * self.se(out)
        return out + identity


class Generator(BaseGenerator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg.channels)
        self.cfg = cfg
        channels = cfg.num_features  # base number of channels, e.g., 32

        # Initial conv
        self.init_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(cfg.channels, channels, 3, 1, 1)), nn.GELU()
        )

        # Encoder blocks with shared channel count
        self.enc1 = self._make_encoder_block(channels)  # 128x128
        self.enc2 = self._make_encoder_block(channels)  # 64x64
        self.enc3 = self._make_encoder_block(channels)  # 32x32
        self.enc4 = self._make_encoder_block(channels)  # 16x16
        self.enc5 = self._make_encoder_block(channels)  # 8x8

        # Final processing
        self.final_process = nn.Sequential(
            EfficientBlock(channels),
            EfficientBlock(channels),
            spectral_norm(nn.Conv2d(channels, cfg.channels, 3, 1, 1)),
            nn.Tanh(),
        )

        self.init_weights()

    def _make_encoder_block(self, channels):
        return nn.Sequential(EfficientBlock(channels), nn.AvgPool2d(2, 2))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _upsample_add(self, x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        # Initial feature extraction
        x = self.init_conv(x)

        # Encoder path with checkpointing
        e1 = checkpoint.checkpoint(self.enc1, x)
        e2 = checkpoint.checkpoint(self.enc2, e1)
        e3 = checkpoint.checkpoint(self.enc3, e2)
        e4 = checkpoint.checkpoint(self.enc4, e3)
        e5 = checkpoint.checkpoint(self.enc5, e4)

        # Upsample and add features
        size = (x.shape[2], x.shape[3])
        out = (
            x
            + self._upsample_add(e1, size)
            + self._upsample_add(e2, size)
            + self._upsample_add(e3, size)
            + self._upsample_add(e4, size)
            + self._upsample_add(e5, size)
        )

        # Final processing
        return checkpoint.checkpoint(self.final_process, out)
