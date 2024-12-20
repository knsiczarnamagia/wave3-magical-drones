from abc import ABC, abstractmethod
from pytorch_lightning import LightningModule


class BaseDiscriminator(ABC, LightningModule):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass
