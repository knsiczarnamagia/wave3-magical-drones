from pytorch_lightning import LightningModule

from abc import ABC, abstractmethod
from torch import Tensor
from torch.functional import F


class BaseGAN(ABC, LightningModule):
    def __init__(
        self,
        channels: int = 3,
        width: int = 224,
        height: int = 224,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        pass

    def adversarial_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.binary_cross_entropy(y_hat, y)

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
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
