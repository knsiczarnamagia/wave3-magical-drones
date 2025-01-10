from torch import Tensor
import torch
import torch.nn.functional as F
from magical_drones.models.base_gan.gan import BaseGAN
from magical_drones.models.cycle_gan.discriminator import Discriminator
from magical_drones.models.cycle_gan.generator import Generator


class CycleGAN(BaseGAN):
    def __init__(
        self,
        channels: int = 3,
        width: int = 224,
        height: int = 224,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        lambda_cycle=10,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.gen_sat = Generator()
        self.gen_map = Generator()
        self.disc_sat = Discriminator()
        self.disc_map = Discriminator()

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)
    
    def _train_discriminators(self, sat, map, optim_disc):
        # sat discriminator training
        sat_fake = self.gen_sat(map)

        disc_sat_real = self.disc_sat(sat)
        disc_sat_fake = self.disc_sat(sat_fake.detach())

        disc_sat_real_loss = F.mse_loss(
            disc_sat_real, torch.ones_like(disc_sat_real)
        )
        disc_sat_fake_loss = F.mse_loss(
            disc_sat_fake, torch.zeros_like(disc_sat_fake)
        )

        disc_sat_loss = disc_sat_real_loss + disc_sat_fake_loss

        # map discriminator training
        map_fake = self.gen_map(sat)

        disc_map_real = self.disc_map(map)
        disc_map_fake = self.disc_map(map_fake.detach())

        disc_map_real_loss = F.mse_loss(disc_map_real, torch.ones_like(disc_map_real))
        disc_map_fake_loss = F.mse_loss(disc_map_fake, torch.zeros_like(disc_map_fake))

        disc_map_loss = disc_map_real_loss + disc_map_fake_loss

        # put loss together
        disc_loss = (disc_sat_loss + disc_map_loss) / 2

        optim_disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()

        return disc_loss, sat_fake, map_fake

    def _train_generators(self, sat, map, sat_fake, map_fake, optim_gen):
        # generators training
        # adversarial loss for generators
        disc_sat_fake = self.disc_sat(sat_fake)
        disc_map_fake = self.disc_map(map_fake)

        gen_sat_loss = F.mse_loss(disc_sat_fake, torch.ones_like(disc_sat_fake))
        gen_map_loss = F.mse_loss(disc_map_fake, torch.ones_like(disc_map_fake))

        # cycle loss
        cycle_map = self.gen_map(sat_fake)
        cycle_sat = self.gen_sat(map_fake)

        cycle_map_loss = F.l1_loss(map, cycle_map)
        cycle_sat_loss = F.l1_loss(sat, cycle_sat)

        # put loss together
        gen_loss = (
            gen_map_loss
            + gen_sat_loss
            + cycle_map_loss * self.hparams.lambda_cycle
            + cycle_sat_loss * self.hparams.lambda_cycle
        )

        optim_gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()

        return gen_loss

    def training_step(self, batch: Tensor) -> None:
        sat, map = batch
        optim_gen, optim_disc = self.optimizers()
        disc_loss, sat_fake, map_fake = self._train_discriminators(sat, map, optim_disc)
        gen_loss = self._train_generators(sat, map, sat_fake, map_fake, optim_gen)
        # TODO: logging disc and gen losses

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        pass

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(
            list(self.gen_sat.parameters()) + list(self.gen_map.parameters()),
            lr=lr,
            betas=(b1, b2),
        )
        opt_d = torch.optim.Adam(
            list(self.disc_sat.parameters()) + list(self.disc_map.parameters()),
            lr=lr,
            betas=(b1, b2),
        )
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        pass
