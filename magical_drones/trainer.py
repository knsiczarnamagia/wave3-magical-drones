from pathlib import Path
from pytorch_lightning import Trainer, LightningDataModule, LightningModule
import os
import yaml
from magical_drones.models.cycle_gan.gan import CycleGAN
from magical_drones.datasets.magmap import MagMapV1
from pytorch_lightning.loggers import TensorBoardLogger


class TrainerHandler:
    def __init__(self, model_instance: LightningModule, datamodule: LightningDataModule, max_epochs_param: int,
                 lr_param: float):
        self.model = model_instance
        self.datamodule = datamodule
        self.max_epochs = max_epochs_param
        self.lr = lr_param
        self.logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

        file_name = 'train_config.yaml'
        config_path = Path('magical_drones/config', file_name)
        self.arguments, self.epochs, self.lr = self._parse_arguments(config_path)

    @staticmethod
    def _parse_arguments(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config, config.get('epochs', 1), config.get('learning_rate', 0.001)

    def training(self):
        trainer = Trainer(max_epochs=self.max_epochs, logger=self.logger)
        trainer.fit(self.model, self.datamodule)

    def debug(self):
        trainer = Trainer(fast_dev_run=True)
        trainer.fit(self.model, self.datamodule)


if __name__ == "__main__":
    model = CycleGAN()
    dm = MagMapV1(data_link='magical_drones/datasets/magmap', batch_size=32, transform=None)
    max_epochs = 1
    lr = 0.001

    handler = TrainerHandler(model, dm, max_epochs, lr)
    handler.training()