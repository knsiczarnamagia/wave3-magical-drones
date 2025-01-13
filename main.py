from magical_drones.models.cycle_gan.gan import CycleGAN
from magical_drones.datasets.magmap import MagMapV1, make_tfms
from tqdm import tqdm
import torch

# test model
model = CycleGAN(channels=3)
fake_input = torch.randn(1, 3, 256, 256)
prediction = model(fake_input)
print(f"out shape: {prediction.shape}")  # should be (1, 3, 256, 256)

# test datamodule speed
datamodule = MagMapV1(
    data_link="czarna-magia/mag-map",
    batch_size=4,
    num_workers=8,  # shouldn't be bigger than number of CPU cores
    train_transform=make_tfms(size=256),
)
datamodule.setup()
dataloader = datamodule.train_dataloader()

for batch in tqdm(dataloader):  # ~30it/s (fast enough)
    sat, map = batch
