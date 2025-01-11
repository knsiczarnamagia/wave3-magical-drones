import torch
from pathlib import Path
from pytorch_lightning import Trainer, LightningDataModule, LightningModule
import os
import yaml
from magical_drones.models.cycle_gan.gan import CycleGAN
from magical_drones.datasets.magmap import MagMapV1, augmentations
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint


class TrainerHandler:
    def __init__(
        self,
        model_class: LightningModule,
        datamodule_class: LightningDataModule,
        config_path: str | Path = Path("magical_drones/config/train_config.yaml"),
    ):
        self.logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        self.trainer_config = config.get("trainer", {})
        self.other_config = config.get("other", {})

        self.model = model_class(**self.model_config)
        self.datamodule = datamodule_class(**self.data_config, transform=augmentations)

        torch.set_float32_matmul_precision(
            "high"
            if self.other_config.get("use_TF32", False)
            else "highest"  # for Ampere and later (RTX 30 or A100)
        )

    def training(self):
        checkpoint_callback = ModelCheckpoint(
            monitor="disc_loss",
            dirpath="./checkpoints",
            filename="model-{epoch:02d}-{disc_loss:.2f}",
            save_top_k=3,
            mode="min",
            every_n_train_steps=1000,
        )
        trainer = Trainer(
            logger=self.logger, **self.trainer_config, callbacks=[checkpoint_callback]
        )
        trainer.fit(self.model, self.datamodule)

    def debug(self):
        trainer = Trainer(fast_dev_run=True, profiler=SimpleProfiler())
        trainer.fit(self.model, self.datamodule)


if __name__ == "__main__":
    handler = TrainerHandler(CycleGAN, MagMapV1)
    handler.training()
