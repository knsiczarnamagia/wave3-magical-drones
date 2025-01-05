import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as transforms
from pathlib import Path
from datasets import load_dataset


class MagMapDataset(Dataset):
    def __init__(self, data, transform: transforms.Compose):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        sample = self.data[idx]

        sat_image = (
            read_image(sample["sat_image"])
            if isinstance(sample["sat_image"], str)
            else sample["sat_image"]
        )
        map_image = (
            read_image(sample["map_image"])
            if isinstance(sample["map_image"], str)
            else sample["map_image"]
        )
        sat_image, map_image = self.transform([sat_image, map_image])

        return (sat_image, map_image)


class MagMapV1(LightningDataModule):
    def __init__(
        self,
        data_link: str | Path,
        batch_size: int,
        train_transform: transforms,
        test_transform: transforms,
        data_dir: str | Path = "./data",
    ):
        super().__init__()
        self.data_link = data_link
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.data_dir = data_dir
        self.data_dict = None

    def prepare_data(self):
        self.data_dict = load_dataset(self.data_link, cache_dir=self.data_dir)

    def setup(self, stage: str = None):
        if not self.data_dict:
            raise RuntimeError("Data has not been prepared. Call prepare_data() first.")

        data = self.data_dict["train"]
        total_len = len(data)

        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        train_data, val_data, test_data = random_split(
            data,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dataset = MagMapDataset(train_data, transform=self.train_transform)
        self.val_dataset = MagMapDataset(val_data, transform=self.test_transform)
        self.test_dataset = MagMapDataset(test_data, transform=self.test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
