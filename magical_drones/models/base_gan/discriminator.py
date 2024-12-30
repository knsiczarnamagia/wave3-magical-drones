from abc import ABC, abstractmethod

from torch import nn, Tensor


class BaseDiscriminator(ABC, nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            features: list = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        if features is None:
            self.features = [64, 128, 256, 512]

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        pass
