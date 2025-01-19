from torch import Tensor
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from magical_drones.models.base_gan.gan import BaseGAN
from magical_drones.models.pix2pix.discriminator import Discriminator
from magical_drones.models.pix2pix.generator import Generator
from pytorch_lightning.loggers import WandbLogger
import wandb


class Pix2Pix(BaseGAN):
    def __init__(
        self,
        channels: int = 3,
        num_features: int = 64,
        num_residuals: int = 9,
        depth: int = 3,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        lambda_l1: float = 100.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_l1 = torch.linspace(lambda_l1, 1, 10_000) # TODO: num of steps based on n_epochs*len(dataloader)
        self.automatic_optimization = False
        self.generator = Generator(channels, num_features, num_residuals, depth)
        self.discriminator = Discriminator(channels, num_features, depth)
        self.val_step_images = 0

        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self, log="gradients", log_graph=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.generator(x)

    def _train_discriminator(self, x: Tensor, y: Tensor, optim_disc):
        y_fake = self.generator(x)

        disc_real = self.discriminator(x, y)
        disc_real_loss = F.mse_loss(disc_real, torch.ones_like(disc_real))

        disc_fake = self.discriminator(x, y_fake.detach())
        disc_fake_loss = F.mse_loss(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_real_loss + disc_fake_loss) / 2

        optim_disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()

        self.log("disc_loss", disc_loss.item())

    def _train_generator(self, x: Tensor, y: Tensor, optim_gen):
        y_fake = self.generator(x)

        disc_fake = self.discriminator(x, y_fake)
        gen_gan_loss = F.mse_loss(disc_fake, torch.ones_like(disc_fake)) # GAN loss
        gen_l1_loss = F.l1_loss(y_fake, y) * self.lambda_l1[self.global_step] # L1 loss TODO: add lambda schedule? slow transition to only GAN loss (it would let model to create sharp edges?)
        gen_loss = gen_gan_loss + gen_l1_loss

        optim_gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()

        self.log("gen_loss", gen_loss.item())

    def training_step(self, batch: Tensor) -> None:
        sat, map = batch
        optim_gen, optim_disc = self.optimizers()
        self._train_discriminator(sat, map, optim_disc)
        self._train_generator(sat, map, optim_gen)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        sat, map = batch
        map_fake = self.generator(sat)

        images = {
            "sat_real": make_grid(sat, nrow=4, normalize=True),
            "map_real": make_grid(map, nrow=4, normalize=True),
            "map_fake": make_grid(map_fake, nrow=4, normalize=True),
        }

        if isinstance(self.logger, WandbLogger):
            wandb_images = {}
            for name, img in images.items():
                wandb_images[name] = wandb.Image(img.to(device="cpu", dtype=torch.float32))
            self.logger.experiment.log(wandb_images)

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.05)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.1)
        return [opt_g, opt_d], []