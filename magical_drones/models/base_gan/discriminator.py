from abc import ABC, abstractmethod

from torch import nn, Tensor


class BaseDiscriminator(ABC, nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
