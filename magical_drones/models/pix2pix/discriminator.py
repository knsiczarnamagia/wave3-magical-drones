from torch import nn, Tensor
import torch
from magical_drones.models.base_gan.discriminator import BaseDiscriminator
from omegaconf import DictConfig


class Discriminator(BaseDiscriminator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg.channels)
        self.cfg = cfg
        self.model = self._construct_model()

    def _construct_model(self):
        layers = []
        in_channels = self.cfg.channels * 2

        for i in range(self.cfg.depth):
            out_channels = self.cfg.num_features * min(2**i, 8)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2 if i < self.cfg.depth - 1 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(p=self.cfg.dropout),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            )
            in_channels = out_channels

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, bias=False)
        )

        return nn.Sequential(*layers)

    def forward(self, sat: Tensor, map: Tensor) -> Tensor:
        # noise injection
        mean, std = 0.0, self.cfg.noise_std
        with torch.no_grad():
            map = map + (torch.randn_like(map) + mean) * std

        xy = torch.cat([sat, map], dim=1)
        return self.model(xy)
