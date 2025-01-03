from magical_drones.models.cycle_gan.gan import CycleGAN
import torch

print(torch.randn(1, 3, 224, 224))

model = CycleGAN(channels=3, width=224, height=224)

prediction = model(torch.randn(1, 3, 224, 224))
