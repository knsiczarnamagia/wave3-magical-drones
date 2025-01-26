from torch import Tensor
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from magical_drones.models.base_gan.gan import BaseGAN
from .discriminator import Discriminator
from .generator import Generator
from pytorch_lightning.loggers import WandbLogger
import wandb
from omegaconf import DictConfig


class CycleGAN2(BaseGAN):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.gan
        self.automatic_optimization = False
        self.gen_sat = Generator(cfg.generator)
        self.gen_map = Generator(cfg.generator)
        self.disc_sat = Discriminator(cfg.discriminator)
        self.disc_map = Discriminator(cfg.discriminator)
        self.val_step_images = 0
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self, log="gradients", log_graph=False)

    def forward(self, sat: Tensor) -> Tensor:
        return self.gen_map(sat)

    def _train_discriminators(self, sat, map, optim_disc):
        sat_fake = self.gen_sat(map)
        disc_sat_real = self.disc_sat(sat)
        disc_sat_fake = self.disc_sat(sat_fake.detach())
        disc_sat_loss = F.mse_loss(
            disc_sat_real, torch.ones_like(disc_sat_real)
        ) + F.mse_loss(disc_sat_fake, torch.zeros_like(disc_sat_fake))

        map_fake = self.gen_map(sat)
        disc_map_real = self.disc_map(map)
        disc_map_fake = self.disc_map(map_fake.detach())
        disc_map_loss = F.mse_loss(
            disc_map_real, torch.ones_like(disc_map_real)
        ) + F.mse_loss(disc_map_fake, torch.zeros_like(disc_map_fake))

        disc_loss = (disc_sat_loss + disc_map_loss) / 2
        optim_disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()
        self.log("disc_loss", disc_loss.item())
        return sat_fake, map_fake

    def _train_generators(self, sat, map, sat_fake, map_fake, optim_gen):
        disc_sat_fake = self.disc_sat(sat_fake)
        disc_map_fake = self.disc_map(map_fake)
        gen_loss = F.mse_loss(
            disc_sat_fake, torch.ones_like(disc_sat_fake)
        ) + F.mse_loss(
            disc_map_fake, torch.ones_like(disc_map_fake)
        )  # adversarial GAN los

        cycle_map = self.gen_map(sat_fake)
        cycle_sat = self.gen_sat(map_fake)
        cycle_loss = (
            F.l1_loss(map, cycle_map) + F.l1_loss(sat, cycle_sat)
        ) * self.cfg.lambda_cycle
        gen_loss += cycle_loss

        l1_loss = F.l1_loss(map_fake, map) + F.l1_loss(
            sat_fake, sat
        )  # pixel-wise L1 loss (from Pix2Pix)
        gen_loss += l1_loss * self.cfg.lambda_l1

        optim_gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()
        self.log("gen_loss", gen_loss.item())

    def training_step(self, batch: Tensor) -> None:
        sat, map = batch
        optim_gen, optim_disc = self.optimizers()
        sat_fake, map_fake = self._train_discriminators(sat, map, optim_disc)
        self._train_generators(sat, map, sat_fake, map_fake, optim_gen)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        sat, map = batch
        sat, map = sat[:32], map[:32]  # first 32 images because more isn't readable
        map_fake = self.gen_map(sat)
        sat_fake = self.gen_sat(map)
        images = {
            "sat_real": make_grid(sat, nrow=4, normalize=True),
            "sat_fake": make_grid(sat_fake, nrow=4, normalize=True),
            "map_real": make_grid(map, nrow=4, normalize=True),
            "map_fake": make_grid(map_fake, nrow=4, normalize=True),
        }
        if isinstance(self.logger, WandbLogger):
            wandb_images = {}
            for name, img in images.items():
                wandb_images[name] = wandb.Image(
                    img.to(device="cpu", dtype=torch.float32)
                )
            self.logger.experiment.log(wandb_images)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.gen_sat.parameters()) + list(self.gen_map.parameters()),
            lr=self.cfg.lr_g,
            betas=(self.cfg.b1, self.cfg.b2),
            fused=True,
        )
        opt_d = torch.optim.Adam(
            list(self.disc_sat.parameters()) + list(self.disc_map.parameters()),
            lr=self.cfg.lr_d,
            betas=(self.cfg.b1, self.cfg.b2),
            fused=True,
        )
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        pass
