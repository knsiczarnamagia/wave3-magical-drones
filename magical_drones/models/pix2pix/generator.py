from torch import nn, Tensor
from magical_drones.models.base_gan.generator import BaseGenerator
from omegaconf import DictConfig


class Generator(BaseGenerator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg.generator.channels)
        self.cfg = cfg.generator
        self.model = self._construct_model()

    def _construct_model(self):
        input_block = nn.Sequential(
            nn.Conv2d(
                self.cfg.channels,
                self.cfg.num_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        encoder_layers = []
        for d in range(1, self.cfg.depth):
            in_channels = self.cfg.num_features * 2 ** (d - 1)
            out_channels = self.cfg.num_features * 2**d
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        encoder = nn.Sequential(*encoder_layers)

        residual_blocks = nn.Sequential(
            *[
                ResidualBlock(self.cfg.num_features * 2 ** (self.cfg.depth - 1))
                for _ in range(self.cfg.num_residuals)
            ]
        )

        decoder_layers = []
        for d in range(self.cfg.depth - 1, 0, -1):
            in_channels = self.cfg.num_features * 2**d
            out_channels = self.cfg.num_features * 2 ** (d - 1)
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        decoder = nn.Sequential(*decoder_layers)

        output_block = nn.Sequential(
            nn.ConvTranspose2d(
                self.cfg.num_features,
                self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),  # Ensure output is in the range [-1, 1]
        )

        return nn.Sequential(
            input_block, encoder, residual_blocks, decoder, output_block
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)
