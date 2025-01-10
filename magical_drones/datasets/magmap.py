from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms


class MagMapDataSet(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            sat_image = self.transform(sample["sat_image"].convert("RGB"))
            map_image = self.transform(sample["map_image"].convert("RGB"))
        except Exception as e:
            raise ValueError(f"Error loading or transforming image at index {idx}: {e}")

        return (sat_image, map_image)


class MagMapV1(LightningDataModule):
    def __init__(self, data_link, batch_size, transform):
        super().__init__()
        self.data_link = data_link
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: str = None):
        data_dict = load_dataset(self.data_link)
        data = data_dict["train"]

        total_len = len(data)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)

        self.train_dataset = MagMapDataSet(
            data.select(range(0, train_len)), transform=self.transform
        )
        self.val_dataset = MagMapDataSet(
            data.select(range(train_len, train_len + val_len)), transform=self.transform
        )
        self.test_dataset = MagMapDataSet(
            data.select(range(train_len + val_len, total_len)), transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)
