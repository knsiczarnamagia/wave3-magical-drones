import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from discriminator import Discriminator


def test_discriminator():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    predicts = model(x)
    print("Discriminator output shape:", predicts.shape)


if __name__ == "__main__":
    test_discriminator()
