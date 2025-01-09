import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path
from datasets import load_dataset


class MagMapDataset(Dataset):
    def __init__(self, data, transforms: transforms.Compose):
        self.data = data
        self.transform = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.data[idx]

        sat_image = sample["sat_image"].convert("RGB")
        map_image = sample["map_image"].convert("RGB")

        filename = sample["filename"]

        sat_image = pil_to_tensor(sat_image).float() / 255.0
        map_image = pil_to_tensor(map_image).float() / 255.0

        if self.transform:
            sat_image = self.transform(sat_image)
            map_image = self.transform(map_image)

        return (sat_image, map_image, filename)


class MagMapV1(LightningDataModule):
    def __init__(
        self,
        data_link: str | Path,
        data_dir: list[str | Path] = ["./data/train", "./data/val", "./data/test"],
        batch_size: int = 32,
        train_transform: transforms.Compose = None,
        val_transform: transforms.Compose = None,
        test_transform: transforms.Compose = None,
        split_for_upload: list[str] = ["train[:80%]", "train[80%:90%]", "train[90%:]"],
    ):
        super().__init__()
        self.data_link = data_link
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.split_for_upload = split_for_upload

        self.train_data_dict = None
        self.val_data_dict = None
        self.test_data_dict = None

    def prepare_data(self):
        try:
            self.train_data_dict = load_dataset(
                self.data_link,
                split=self.split_for_upload[0],
                cache_dir=self.data_dir[0],
            )

            self.val_data_dict = load_dataset(
                self.data_link,
                split=self.split_for_upload[1],
                cache_dir=self.data_dir[1],
            )

            self.test_data_dict = load_dataset(
                self.data_link,
                split=self.split_for_upload[2],
                cache_dir=self.data_dir[2],
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    def setup(self, stage: str = None):
        self.train_dataset = MagMapDataset(
            self.train_data_dict, transforms=self.train_transform
        )

        self.val_dataset = MagMapDataset(
            self.val_data_dict, transforms=self.val_transform
        )

        self.test_dataset = MagMapDataset(
            self.test_data_dict, transforms=self.test_transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
