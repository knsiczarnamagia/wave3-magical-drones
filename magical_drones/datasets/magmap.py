from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision.transforms import v2
import torch


class MagMapDataSet(Dataset):
    def __init__(self, data, transform=None, sat_transform=None):
        self.data = data
        self.transform = transform if transform is not None else self._to_tensor()
        self.sat_transform = sat_transform if sat_transform is not None else v2.Identity()

    def _to_tensor(self):
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            sat_image, map_image = sample["sat_image"].convert("RGB"), sample["map_image"].convert("RGB")
            sat_image = self.sat_transform(sat_image)
            sat_image, map_image = self.transform(sat_image, map_image)
        except Exception as e:
            raise ValueError(f"Error loading or transforming image at index {idx}: {e}")

        return (sat_image, map_image)


class MagMapV1(LightningDataModule):
    def __init__(
        self,
        data_link,
        batch_size,
        train_transform=None,
        valid_transform=None,
        test_transform=None,
        num_workers=0,
        **kwargs,
    ):
        super().__init__()
        self.data_link = data_link
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        data_dict = load_dataset(self.data_link)
        data = data_dict["train"]

        total_len = len(data)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)

        self.train_dataset = MagMapDataSet(
            data.select(range(0, train_len)), transform=self.train_transform, sat_transform=sat_tfms
        )
        self.val_dataset = MagMapDataSet(
            data.select(range(train_len, train_len + val_len)),
            transform=self.valid_transform,
        )
        self.test_dataset = MagMapDataSet(
            data.select(range(train_len + val_len, total_len)),
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # shuffle=True,
            pin_memory=True,
            prefetch_factor=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 8,
            num_workers=self.num_workers,
            prefetch_factor=8,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 8,
            num_workers=self.num_workers,
            prefetch_factor=8,
        )


def make_tfms(
    size: int = 256,
    degrees: int = 0,
    translate: tuple[float] | None = None,
    flip_p: float = 0.0,
    scale: tuple[float] | None = None,
    shear: tuple[float] | None = None,
):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=size),
            v2.ToDtype(torch.float32, scale=True), # scale to 0,1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # scale -1,+1
            v2.RandomHorizontalFlip(flip_p) if flip_p > 0 else v2.Identity(),
            v2.RandomAffine(
                degrees=degrees, translate=translate, scale=scale, shear=shear
            )
            if degrees or translate or scale or shear
            else v2.Identity(),
        ]
    )

sat_tfms = v2.Compose([
    v2.ColorJitter(brightness=(.5, 1.5), contrast=(.5, 1.5), saturation=(.5, 1.5)),
    v2.RandomAdjustSharpness(1.5, p=0.5)
])