import torch
from pathlib import Path
from pytorch_lightning import Trainer, LightningDataModule, LightningModule
import os
import yaml
from magical_drones.models.cycle_gan.gan import CycleGAN
from magical_drones.datasets.magmap import MagMapV1, make_tfms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint


class TrainerHandler:
    def __init__(
        self,
        model_class: LightningModule,
        datamodule_class: LightningDataModule,
        config_path: str | Path = Path("magical_drones/config/train_config.yaml"),
    ):
        # self.logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs")
        self.logger = WandbLogger(
            save_dir=os.getcwd(), project="magical-drones", log_model=True
        )

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        self.trainer_config = config.get("trainer", {})
        self.other_config = config.get("other", {})

        self.model = model_class(**self.model_config)
        self.datamodule = datamodule_class(
            **self.data_config,
            train_transform=make_tfms(**self.data_config["train_tfms"]),
            valid_transform=make_tfms(**self.data_config["valid_tfms"]),
            test_transform=make_tfms(**self.data_config["test_tfms"]),
        )

        torch.set_float32_matmul_precision(
            "high"
            if self.other_config.get("use_TF32", False)
            else "highest"  # for Ampere and later (RTX 30 or A100)
        )

    def train(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath="./checkpoints",
            filename="model-{epoch:02d}-{disc_loss:.2f}",
            every_n_train_steps=self.other_config.get("checkpoint_interval", 1000),
            save_last=True,
        )
        trainer = Trainer(
            logger=self.logger, **self.trainer_config, callbacks=[checkpoint_callback]
        )
        trainer.fit(self.model, self.datamodule)
        trainer.logger.version

    def debug(self):
        trainer = Trainer(fast_dev_run=True, profiler=SimpleProfiler())
        trainer.fit(self.model, self.datamodule)


if __name__ == "__main__":
    handler = TrainerHandler(CycleGAN, MagMapV1)
    handler.train()
