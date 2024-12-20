from torch import Tensor

from magical_drones.CycleGAN.generator import Generator
from magical_drones.models import BaseGAN
from magical_drones.models.cycle_gan.discriminator import Discriminator


class CycleGAN(BaseGAN):
    def __init__(
            self,
            channels,
            width,
            height,
            latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)


    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
