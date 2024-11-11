from generator import Generator
import torch

def test_generator():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    generator = Generator(img_channels, 9)
    output = generator(x)
    print("Generator output shape:", output.shape)

if __name__ == "__main__":
    test_generator()
