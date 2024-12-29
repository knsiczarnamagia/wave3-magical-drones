from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


class MagMapDataSet(Dataset):

    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),  
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            sat_image = Image.open(sample["sat_image"]) if isinstance(sample["sat_image"], str) else sample["sat_image"]
            map_image = Image.open(sample["map_image"]) if isinstance(sample["map_image"], str) else sample["map_image"]

            sat_image = self.transform(sat_image)
            map_image = self.transform(map_image)
            
        except Exception as e:
            raise ValueError(f"Error loading image at index {idx}: {e}")
        
        return {"sat_image": sat_image, "map_image": map_image}




class MagMapV1(LightningDataModule): 

    def __init__(self, data_link, batch_size):
        super().__init__()
        self.data_link = data_link
        self.batch_size = batch_size

    def setup(self, stage: str=None):
        data_dict = load_dataset(self.data_link)

        data = data_dict["train"] 

        total_len = len(data)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)

        self.train_dataset = MagMapDataSet(data.select(range(0, train_len)))
        self.val_dataset = MagMapDataSet(data.select(range(train_len, train_len + val_len)))
        self.test_dataset = MagMapDataSet(data.select(range(train_len + val_len, total_len)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    data_link = "Kiwinicki/sat2map_poland"
    batch_size = 32

    magmap = MagMapV1(data_link, batch_size=batch_size)

    print("Setting up datasets...")
    magmap.setup()

    print("Testing train_dataloader...")
    train_loader = magmap.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2: 
            break

    print("\nTesting val_dataloader...")
    val_loader = magmap.val_dataloader()
    for i, batch in enumerate(val_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2: 
            break

    print("\nTesting test_dataloader...")
    test_loader = magmap.test_dataloader()
    for i, batch in enumerate(test_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2:  
            break

    for i, batch in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"  sat_image shape: {batch['sat_image'].shape}")
        print(f"  map_image shape: {batch['map_image'].shape}")
        if i == 2:
            break

