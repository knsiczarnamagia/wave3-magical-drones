from pytorch_lightning import LightningModule

<<<<<<< HEAD
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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            channels, width, height, latent_dim, lr, b1, b2, kwargs
        )

    def adversarial_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.binary_cross_entropy(y_hat, y)
=======
from generator import BaseGenerator
from abc import ABC, abstractmethod
from torch import Tensor


class BaseGAN(ABC, LightningModule):
    def __init__(self, channels, width, height, latent_dim: int = 100, lr: float = 0.0002, b1: float = 0.5,
                 b2: float = 0.999, batch_size: int = 32, **kwargs):
        super().__init__()
        pass
>>>>>>> 3cf493b (First steps of Lightning structure)

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        pass

<<<<<<< HEAD
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
=======

>>>>>>> 3cf493b (First steps of Lightning structure)
