from magical_drones.datasets.magmap import MagMapV1
from magical_drones.models.pix2pix2.gan import Pix2Pix2
from magical_drones.models.cycle_gan.gan import CycleGAN
from omegaconf import OmegaConf
import wandb
import os

trainer_cfg = OmegaConf.load("conf/trainer.yaml")
model_cfg = OmegaConf.load("conf/models.yaml")
data_cfg = OmegaConf.load("conf/data.yaml")

datamodule = MagMapV1(data_cfg)
datamodule.setup()
train_dl = datamodule.train_dataloader()

sat, map = next(iter(train_dl))
sat = sat[:4]

# print('sat mean', sat.mean(), 'map mean', map.mean())

# grid_image = Image.fromarray(
#     make_grid(sat, normalize=True, scale_each=True).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
# )
# grid_image.show()


gan = Pix2Pix2(model_cfg)
# disc = gan.discriminator
# print(sat.shape, map.shape)
# out = disc(sat, map)  # outputs "patch-wise discrimination"
# print('disc out shape:', out.shape) # patches so different than img size

gen = gan.generator
print("sat in shape:", sat.shape)
out2 = gen(sat)
print("gen out shape:", out2.shape)


### test checkpoint loading ###

# from magical_drones.models.cycle_gan.gan import CycleGAN
# from pathlib import Path

# gan = CycleGAN.load_from_checkpoint(Path('checkpoints/model-epoch=90-disc_loss=0.43.ckpt')).eval().bfloat16()
# print(gan)

# out = gan.gen_map(sat.to(device='cuda', dtype=torch.bfloat16))
# grid_image = Image.fromarray(
#     make_grid(out, normalize=True, scale_each=True).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
# )
# grid_image.show()

# print(gan.optimizers())


### dataloader speed ###
# from tqdm import tqdm
# from time import time

# datamodule = MagMapV1(cfg=data_cfg)
# datamodule.setup()
# train_dl = datamodule.train_dataloader()

# start = time()
# for batch in tqdm(train_dl):
#     pass

# print(f"TIME SPENT: {time()-start}")


### load checkpoint from W&B ###

run = wandb.init(project="magical-drones")

# "Run path" from W&B run overveiew panel
artifact = run.use_artifact(
    "szefek24/magical-drones/hdoulm7c", type="model"
)  # its for "CycleGAN-eb993c73" run
artifact_dir = (
    artifact.download()
)  # it will download to current dir (does it take path argument?)

# The downloaded directory will contain any checkpoints that were saved
# Replace "my-model-epoch.ckpt" with the name of the saved checkpoint file
checkpoint_path = os.path.join(artifact_dir, "my-model-00.ckpt")

# Load the model
model = CycleGAN.load_from_checkpoint(checkpoint_path)
model.eval()
