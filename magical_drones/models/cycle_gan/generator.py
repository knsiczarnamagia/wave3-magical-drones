from .models.gan import BaseGenerator
from torch import tanh
from torch import Tensor


class Generator(BaseGenerator):
    def __init__(self, input_channels: int = 3, num_features: int = 64, num_residuals: int = 2):
        super().__init__()
        # self.initial =
        # self.down_blocks =
        # self.residual_blocks =
        # self.up_blocks =
        # self.last_block  =
        pass

    def forward(self, noise: Tensor) -> Tensor:
        x = self.initial(noise)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)

        return tanh(self.last_block(x))


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, down=True, use_activation=True, **kwargs
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(
                channels, channels, use_activation=False, kernel_size=3, padding=1
            ),
        )

    def forward(self, x):
        return x + self.block(x)