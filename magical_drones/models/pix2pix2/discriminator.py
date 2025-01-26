from torch import nn, Tensor
import torch
from torch.nn.utils import spectral_norm
from magical_drones.models.base_gan.discriminator import BaseDiscriminator
from omegaconf import DictConfig


class Discriminator(BaseDiscriminator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg.channels)
        self.cfg = cfg
        self.model = self._construct_model()

    def _construct_model(self):
        # For 256x256 input, use ndf=64 to get 32x32 PatchGAN
        # Adjust these parameters based on your input size
        ndf = self.cfg.num_features  # number of discriminator filters

        layers = []
        # input is (channels*2) x 256 x 256
        in_channels = self.cfg.channels * 2

        # First layer doesn't use normalization
        layers.append(
            spectral_norm(
                nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1, bias=False)
            )
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Defining multipliers for the number of filters
        # Using smaller multipliers to reduce parameters
        mult_list = [1, 2, 4, 8]

        # Middle layers
        current_dim = ndf
        for i, mult in enumerate(mult_list[:-1]):
            next_dim = ndf * mult_list[i + 1]
            layers.extend(
                [
                    spectral_norm(
                        nn.Conv2d(
                            current_dim, next_dim, 4, stride=2, padding=1, bias=False
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(p=self.cfg.dropout)
                    if self.cfg.dropout > 0
                    else nn.Identity(),
                ]
            )
            current_dim = next_dim

        # Final layers to produce patch output
        layers.extend(
            [
                spectral_norm(
                    nn.Conv2d(
                        current_dim, current_dim, 4, stride=1, padding=1, bias=False
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(
                    nn.Conv2d(current_dim, 1, 4, stride=1, padding=1, bias=False)
                ),
            ]
        )

        return nn.Sequential(*layers)

    def forward(self, sat: Tensor, map: Tensor) -> Tensor:
        # Instance noise injection (helps with training stability)
        if self.training and self.cfg.noise_std > 0:
            with torch.no_grad():
                noise = torch.randn_like(map) * self.cfg.noise_std
                map = map + noise

        xy = torch.cat([sat, map], dim=1)
        return self.model(xy)
