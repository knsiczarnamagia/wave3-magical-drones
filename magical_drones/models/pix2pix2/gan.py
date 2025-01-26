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


class Pix2Pix2(BaseGAN):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.gan
        self.automatic_optimization = False
        self.generator = Generator(cfg.generator)
        self.discriminator = Discriminator(cfg.discriminator)

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
        # `self.global_step` is doubled because it increases after each optim.step() and we have 2
        lambda_l1 = (
            self.lambda_l1_sched[self.global_step // 2]
            if self.global_step // 2 < len(self.lambda_l1_sched)
            else torch.tensor(1.0)
        )

        disc_fake = self.discriminator(x, y_fake)
        gen_gan_loss = F.mse_loss(disc_fake, torch.ones_like(disc_fake))
        gen_l1_loss = F.l1_loss(y_fake, y) * lambda_l1
        gen_loss = gen_gan_loss + gen_l1_loss

        optim_gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()

        self.log("gen_loss", gen_loss.item())
        self.log("l1_lambda", lambda_l1.item())

    def on_train_start(self):
        # create l1_lambda schedule on the start of training
        steps = (
            len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        )
        self.lambda_l1_sched = torch.linspace(self.cfg.lambda_l1, 1, steps)  # [LINEAR]
        # self.lambda_l1_sched = (0.5*(1+torch.cos(torch.linspace(0, torch.pi, steps)))*(self.cfg.lambda_l1 - 1) + 1)  # [COSINE]
        # self.lambda_l1_sched = self.inverse_exponential_schedule(self.cfg.lambda_l1, 1, steps) # inverse exp

    def inverse_exponential_schedule(self, initial_value, final_value, num_steps):
        decay_rate = -torch.log(torch.tensor(final_value / initial_value)) / num_steps
        steps = torch.arange(num_steps, dtype=torch.float32)
        schedule = initial_value * torch.exp(-decay_rate * steps)
        return schedule

    def training_step(self, batch: Tensor) -> None:
        sat, map = batch
        optim_gen, optim_disc = self.optimizers()
        self._train_discriminator(sat, map, optim_disc)
        self._train_generator(sat, map, optim_gen)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        sat, map = batch
        sat, map = sat[:32], map[:32]  # first 32 images because more isn't readable
        map_fake = self.generator(sat)

        images = {
            "sat_real": make_grid(sat, nrow=4, normalize=True),
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

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        b1 = self.cfg.b1
        b2 = self.cfg.b2

        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.cfg.lr_g,
            betas=(b1, b2),
            weight_decay=self.cfg.gen_wd,
            eps=1e-8,
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.lr_d,
            betas=(b1, b2),
            weight_decay=self.cfg.disc_wd,
            eps=1e-8,
        )
        return [opt_g, opt_d], []
