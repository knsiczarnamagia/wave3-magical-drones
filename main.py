from magical_drones.models.cycle_gan.gan import CycleGAN
from magical_drones.datasets.magmap import MagMapV1, augmentations
from tqdm import tqdm
import torch

# test model
model = CycleGAN(channels=3, width=256, height=256)
fake_input = torch.randn(1, 3, 224, 224)
for i in range(100):
    prediction = model(fake_input)
print(f"out shape: {prediction.shape}")  # should be (1, 3, 224, 224)

# test datamodule speed
datamodule = MagMapV1(
    data_link="czarna-magia/mag-map",
    batch_size=4,
    num_workers=8,
    train_transform=augmentations,
)
datamodule.setup()
dataloader = datamodule.train_dataloader()

for batch in tqdm(dataloader):  # ~30it/s (fast enough)
    sat, map = batch
