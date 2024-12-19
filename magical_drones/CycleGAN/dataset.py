from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class MapDataset(Dataset):
    def __init__(self, root_map, root_aerial, transform=None):
        self.root_map = root_map
        self.root_aerial = root_aerial
        self.transform = transform

        self.map_images = os.listdir(root_map)
        self.aerial_images = os.listdir(root_aerial)

        self.map_len = len(self.map_images)
        self.aerial_len = len(self.aerial_images)

    def __len__(self):
        return max(self.map_len, self.aerial_len)

    def __getitem__(self, index):
        map_img = self.map_images[index % self.map_len]
        aerial_img = self.aerial_images[index % self.aerial_len]

        map_path = os.path.join(self.root_map, map_img)
        aerial_path = os.path.join(self.root_aerial, aerial_img)

        map_img = np.array(Image.open(map_path).convert("RGB"))
        aerial_img = np.array(Image.open(aerial_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=map_img, image0=aerial_img)
            map_img = augmentations["image"]
            aerial_img = augmentations["image0"]

        return map_img, aerial_img
