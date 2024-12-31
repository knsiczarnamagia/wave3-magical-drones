from torch import Tensor
import torch
from magical_drones.models.base_gan.gan import BaseGAN
from magical_drones.models.cycle_gan.discriminator import Discriminator

from magical_drones.models.cycle_gan.generator import Generator


class CycleGAN(BaseGAN):
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
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator()
        self.discriminator = Discriminator()  # TODO: Implement

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)

    def training_step(self, batch: Tensor) -> None:
        images, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        z = torch.randn(images.shape[0], self.hparams.latent_dim)
        z = z.type_as(images)

        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # sample_images = self.generated_imgs[:6]
        # grid = torchvision.utils.make_grid(sample_images)
        # self.logger.experiment.add_image("train/generated_images", grid, self.current_epoch)

        valid = torch.ones(images.size(0), 1)
        valid = valid.type_as(images)

        # TODO: Wrapp loss calculation in helper function?
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(images.size(0), 1)
        valid = valid.type_as(images)

        real_loss = self.adversarial_loss(self.discriminator(images), valid)

        fake = torch.zeros(images.size(0), 1)
        fake = fake.type_as(images)

        fake_loss = self.adversarial_loss(
            self.discriminator(self.generated_imgs.detach()), fake
        )

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        pass

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        pass

        optimizer_g, optimizer_d = self.optimizers()

        z = torch.randn(images.shape[0], self.hparams.latent_dim)
        z = z.type_as(images)

        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # sample_images = self.generated_imgs[:6]
        # grid = torchvision.utils.make_grid(sample_images)
        # self.logger.experiment.add_image("train/generated_images", grid, self.current_epoch)

        valid = torch.ones(images.size(0), 1)
        valid = valid.type_as(images)

        # TODO: Wrapp loss calculation in helper function?
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(images.size(0), 1)
        valid = valid.type_as(images)

        real_loss = self.adversarial_loss(self.discriminator(images), valid)

        fake = torch.zeros(images.size(0), 1)
        fake = fake.type_as(images)

        fake_loss = self.adversarial_loss(
            self.discriminator(self.generated_imgs.detach()), fake
        )

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        pass

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # sample_images = self(z)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image("validation/generated_images", grid, self.current_epoch)