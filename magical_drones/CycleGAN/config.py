import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "maps_aerial_dataset/"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
SIZE = 256
CHECKPOINT_GEN_M = "genh.pth.tar"
CHECKPOINT_GEN_A = "genz.pth.tar"
CHECKPOINT_CRITIC_M = "critich.pth.tar"
CHECKPOINT_CRITIC_A = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=SIZE, height=SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
