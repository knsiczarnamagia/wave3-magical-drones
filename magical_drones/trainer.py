import torch
from pytorch_lightning import Trainer, LightningDataModule, LightningModule
import os
from magical_drones.models.pix2pix2.gan import Pix2Pix2
from magical_drones.models.cycle_gan.gan import CycleGAN
from magical_drones.datasets.magmap import MagMapV1
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from uuid import uuid4
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


class TrainerHandler:
    def __init__(
        self,
        model_class: LightningModule,
        datamodule_class: LightningDataModule,
        trainer_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
    ):
        self.trainer_cfg = trainer_cfg
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.run_name = self.trainer_cfg.get(
            "run_name", f"{model_class.__name__}-{str(uuid4())[:8]}"
        )
        self.logger = WandbLogger(
            save_dir=os.getcwd(),
            project="magical-drones",
            name=self.run_name,
            log_model=True,
        )
        self.model = model_class(model_cfg)
        self.datamodule = datamodule_class(data_cfg)

        torch.set_float32_matmul_precision(
            "high" if self.trainer_cfg.get("use_TF32", False) else "highest"
        )

    def train(self):
        self.logger.log_hyperparams({**self.trainer_cfg, **self.model_cfg, **self.data_cfg})
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/{self.run_name}",
            filename="{epoch}epoch",
            auto_insert_metric_name=False,
            save_last=True,
        )
        trainer = Trainer(
            logger=self.logger, **self.trainer_cfg.trainer, callbacks=[checkpoint_callback]
        )
        trainer.fit(self.model, self.datamodule)

    def debug(self):
        debug_profiler = PyTorchProfiler(
            dirpath="./profiler",
            filename="profile",
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            export_to_chrome=True
        )
        trainer = Trainer(
            max_steps=5,
            profiler=debug_profiler,
            limit_train_batches=2,
            num_sanity_val_steps=1
        )
        trainer.fit(self.model, self.datamodule)


@hydra.main(version_base=None, config_path="../conf", config_name="trainer")
def main(trainer_cfg: DictConfig):
    model_cfg = OmegaConf.load("conf/models.yaml")
    data_cfg = OmegaConf.load("conf/data.yaml")

    handler = TrainerHandler(CycleGAN, MagMapV1, trainer_cfg, model_cfg, data_cfg)
    if trainer_cfg.mode == "train":
        handler.train()
    elif trainer_cfg.mode == "debug":
        handler.debug()
    else:
        raise ValueError(
            f"Invalid mode: {trainer_cfg.mode}. Supported modes are 'train' and 'debug'."
        )


if __name__ == "__main__":
    main()
