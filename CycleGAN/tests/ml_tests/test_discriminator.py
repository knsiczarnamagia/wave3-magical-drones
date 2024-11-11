import torch
from discriminator import Discriminator  

def test_discriminator():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    predicts = model(x)
    print("Discriminator output shape:", predicts.shape)

if __name__ == "__main__":
    test_discriminator()