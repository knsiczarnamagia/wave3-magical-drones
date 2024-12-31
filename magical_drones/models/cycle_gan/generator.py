from torch import tanh
from torch import Tensor
import torch.nn as nn

from magical_drones.models.base_gan.generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.model = self._construct_model()

    def _construct_model(self):
        initial_layer = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

        down_blocks = nn.Sequential(
                ConvBlock(
                    self.num_features,
                    self.num_features * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ConvBlock(
                    self.num_features * 2,
                    self.num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
        )

        residual_blocks = nn.Sequential(
            *[ResidualBlock(self.num_features * 4) for _ in range(self.num_features)]
        )

        up_blocks = nn.Sequential(
                ConvBlock(
                    self.num_features * 4,
                    self.num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    self.num_features * 2,
                    self.num_features,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
        )

        last_block = nn.Conv2d(
            self.num_features,
            self.input_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

        model = nn.Sequential(
            initial_layer, down_blocks, residual_blocks, up_blocks, last_block
        )
        return model

    def forward(self, noise: Tensor) -> Tensor:
        x = self.model(noise)

        return tanh(self.model(x))


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

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(
                channels, channels, use_activation=False, kernel_size=3, padding=1
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)
