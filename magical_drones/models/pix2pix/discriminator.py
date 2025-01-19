from torch import nn, Tensor
import torch
from magical_drones.models.base_gan.discriminator import BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(self, channels: int = 3, num_features: int = 64, depth: int = 3, **kwargs):
        super().__init__(channels)
        self.num_features = num_features    
        self.depth = 5 #TODO: change in config to disc_depth and gen_depth
        self.model = self._construct_model()

    def _construct_model(self):
        layers = []
        in_channels = self.channels * 2

        for i in range(self.depth):
            out_channels = self.num_features * min(2**i, 8)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2 if i < self.depth - 1 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(p=0.1)
                )
            )
            in_channels = out_channels

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, bias=False)
        )

        return nn.Sequential(*layers)

    def forward(self, sat: Tensor, map: Tensor) -> Tensor:
        # noise injection
        mean, std = 0.5, 0.1
        with torch.no_grad():
            map =+ (torch.randn_like(map)+mean)*std

        xy = torch.cat([sat, map], dim=1)
        return self.model(xy)