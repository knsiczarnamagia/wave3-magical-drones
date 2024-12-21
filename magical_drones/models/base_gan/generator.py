from abc import ABC, abstractmethod
from torch import nn, Tensor


class BaseGenerator(ABC, nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_features: int = 64,
        num_residuals: int = 2
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_features = num_features
        self.num_residuals = num_residuals

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        pass
