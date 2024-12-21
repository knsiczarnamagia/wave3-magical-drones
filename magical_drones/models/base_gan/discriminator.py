from abc import ABC, abstractmethod

from torch import nn


class BaseDiscriminator(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self):
        pass
