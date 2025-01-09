import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path
from datasets import load_dataset


<<<<<<< HEAD
class MagMapDataset(Dataset):
    def __init__(self, data, transforms: transforms.Compose):
=======
class MagMapDataSet(Dataset):
    def __init__(self, data, transform):
>>>>>>> main
        self.data = data
        self.transform = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.data[idx]
<<<<<<< HEAD
=======
        try:
            sat_image = (
                Image.open(sample["sat_image"])
                if isinstance(sample["sat_image"], str)
                else sample["sat_image"]
            )
            map_image = (
                Image.open(sample["map_image"])
                if isinstance(sample["map_image"], str)
                else sample["map_image"]
            )
>>>>>>> main

        sat_image = sample["sat_image"].convert("RGB")
        map_image = sample["map_image"].convert("RGB")

        filename = sample["filename"]

        sat_image = pil_to_tensor(sat_image).float() / 255.0 
        map_image = pil_to_tensor(map_image).float() / 255.0  
    
        if self.transform:
            sat_image = self.transform(sat_image)
            map_image = self.transform(map_image)
<<<<<<< HEAD
=======

        except Exception as e:
            raise ValueError(f"Error loading image at index {idx}: {e}")

        return {"sat_image": sat_image, "map_image": map_image}
>>>>>>> main

        return (sat_image, map_image, filename)
    

class MagMapV1(LightningDataModule):
<<<<<<< HEAD
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
=======
    def __init__(self, data_link, batch_size, transform):
>>>>>>> main
        super().__init__()
        self.data_link = data_link
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

<<<<<<< HEAD
        self.split_for_upload = split_for_upload
        
        self.train_data_dict = None
        self.val_data_dict = None
        self.test_data_dict = None
=======
    def setup(self, stage: str = None):
        # data_dict = load_dataset(self.data_link, split="train", data_files="train-00000-of-00015.parquet")
        data_dict = load_dataset(
            self.data_link,
        )
        data = data_dict["train"]
>>>>>>> main

    def prepare_data(self):
        try:
            self.train_data_dict = load_dataset(self.data_link, 
                         split=self.split_for_upload[0], 
                         cache_dir=self.data_dir[0])
            
            self.val_data_dict = load_dataset(self.data_link, 
                         split=self.split_for_upload[1], 
                         cache_dir=self.data_dir[1])
            
            self.test_data_dict = load_dataset(self.data_link, 
                         split=self.split_for_upload[2], 
                         cache_dir=self.data_dir[2])

<<<<<<< HEAD
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
=======
        self.train_dataset = MagMapDataSet(
            data.select(range(0, train_len)), transform=self.transform
        )
        self.val_dataset = MagMapDataSet(
            data.select(range(train_len, train_len + val_len)), transform=self.transform
        )
        self.test_dataset = MagMapDataSet(
            data.select(range(train_len + val_len, total_len)), transform=self.transform
        )
>>>>>>> main

    def setup(self, stage: str = None):
        self.train_dataset = MagMapDataset(self.train_data_dict, 
                                           transforms=self.train_transform)
        
        self.val_dataset = MagMapDataset(self.val_data_dict, 
                                           transforms=self.val_transform)
        
        self.test_dataset = MagMapDataset(self.test_data_dict, 
                                           transforms=self.test_transform)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
