from pathlib import Path
from pytorch_lightning import Trainer, LightningDataModule, LightningModule
import os
import yaml
from magical_drones.models.cycle_gan.gan import CycleGAN
from magical_drones.datasets.magmap import MagMapV1
from pytorch_lightning.loggers import TensorBoardLogger


class TrainerHandler:
    def __init__(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
    ):
        self.logger = TensorBoardLogger(
            save_dir=os.getcwd(), version=1, name="lightning_logs"
        )

        file_name = "train_config.yaml"
        config_path = Path("magical_drones/config", file_name)
        self.data_config, self.model_config, self.training_config = (
            self._parse_arguments(config_path)
        )
        self.model = model(**self.model_config)
        self.datamodule = datamodule(**self.data_config)

    @staticmethod
    def _parse_arguments(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return (
            config.get("data", None),
            config.get("model", None),
            config.get("trainer", None),
        )

    def training(self):
        trainer = Trainer(logger=self.logger, **self.training_config)
        trainer.fit(self.model, self.datamodule)

    def debug(self):
        trainer = Trainer(fast_dev_run=True)
        trainer.fit(self.model, self.datamodule)


if __name__ == "__main__":
    handler = TrainerHandler(CycleGAN, MagMapV1)
    handler.training()
