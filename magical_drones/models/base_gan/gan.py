from pytorch_lightning import LightningModule

from abc import ABC, abstractmethod
from torch import Tensor


class BaseGAN(ABC, LightningModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def training_step(self, batch: Tensor):
        pass

    @abstractmethod
    def validation_step(self, batch: Tensor, batch_idx: int):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def on_validation_epoch_end(self):
        pass

    def on_train_epoch_end(self):
        self.log(
            "epoch", self.current_epoch
        )  # for wandb last 3 checkpoint saving (using epoch as metric hack)
