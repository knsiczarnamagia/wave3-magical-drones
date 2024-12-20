from abc import ABC, abstractmethod
from pytorch_lightning import LightningModule




class BaseGenerator(ABC, LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim