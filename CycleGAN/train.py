import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from discriminator import Discriminator
from generator import Generator
from dataset import MapDataset
from utils import save_checkpoint, load_checkpoint
import config

import sys
from tqdm import tqdm

if config.DEVICE == 'cpu': use_amp = False
else:
  use_amp = True 

def train_function(
          Discriminator_Map,
          Discriminator_Aerial,
          Generator_Aerial,
          Generator_Map,
          dataloader,
          optim_discriminator,
          optim_generator,
          L1_loss,
          MSE_loss,
          generator_scaler,
          discriminator_scaler,
          epoch,
      ):
    loop = tqdm(dataloader, leave=True)

    for index, (map, aerial_photo) in enumerate(loop):
      map = map.to(config.DEVICE)
      aerial_photo = aerial_photo.to(config.DEVICE)

      # Discriminators training
      with torch.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=use_amp):
        # Aerial discriminator train
        aerial_fake = Generator_Aerial(map)

        disc_aerial_real = Discriminator_Aerial(aerial_photo)
        disc_aerial_fake = Discriminator_Aerial(aerial_fake.detach())

        disc_aerial_real_loss = MSE_loss(disc_aerial_real, torch.ones_like(disc_aerial_real))
        disc_aerial_fake_loss = MSE_loss(disc_aerial_fake, torch.zeros_like(disc_aerial_fake))

        disc_aerial_loss = disc_aerial_real_loss + disc_aerial_fake_loss

        # Map discriminator train
        map_fake = Generator_Map(aerial_photo)

        disc_map_real = Discriminator_Map(map)
        disc_map_fake = Discriminator_Map(map_fake.detach())

        disc_map_real_loss = MSE_loss(disc_map_real, torch.ones_like(disc_map_real))
        disc_map_fake_loss = MSE_loss(disc_map_fake, torch.zeros_like(disc_map_fake))

        disc_map_loss = disc_map_real_loss + disc_map_fake_loss

        # Put loss together
        Discriminator_loss = (disc_aerial_loss + disc_map_loss)/2
          

      optim_discriminator.zero_grad()
      discriminator_scaler.scale(Discriminator_loss).backward()
      discriminator_scaler.step(optim_discriminator)
      discriminator_scaler.update()

      # Generators training
      with torch.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=use_amp):
        # Adversarial loss for generators
        disc_aerial_fake = Discriminator_Aerial(aerial_fake)
        disc_map_fake = Discriminator_Map(map_fake)

        Generator_loss_aerial = MSE_loss(disc_aerial_fake, torch.ones_like(disc_aerial_fake))
        Generator_loss_map = MSE_loss(disc_map_fake, torch.ones_like(disc_map_fake))

        #Cycle loss
        cycle_map = Generator_Map(aerial_fake)
        cycle_aerial = Generator_Aerial(map_fake)

        cycle_map_loss = L1_loss(map, cycle_map)
        cycle_aerial_loss = L1_loss(aerial_photo, cycle_aerial)

        # Put loss together
        Generator_loss = (
            Generator_loss_map
            + Generator_loss_aerial
            + cycle_map_loss * config.LAMBDA_CYCLE
            + cycle_aerial_loss * config.LAMBDA_CYCLE
        )

      optim_generator.zero_grad()
      generator_scaler.scale(Generator_loss).backward()
      generator_scaler.step(optim_generator)
      generator_scaler.update()
      

    print(f"Epoch {epoch} | Generator loss: {Generator_loss} | Discriminator loss: {Discriminator_loss}")
    if epoch % 5 == 0:
      cycle_first = torch.cat((map, aerial_fake * 0.5 + 0.5, cycle_map), dim=2)
      save_image(cycle_first, f"saved_images/cycle_first_{epoch}.png")

      cycle_second = torch.cat((aerial_photo, map_fake * 0.5 + 0.5, cycle_aerial), dim=2)
      save_image(cycle_second, f"saved_images/cycle_second_{epoch}.png")



if __name__ == "__main__":
  # Discriminators initialization
  Discriminator_Map = Discriminator(in_channels=3).to(config.DEVICE)
  Discriminator_Aerial = Discriminator(in_channels=3).to(config.DEVICE)

  # Generators initialization
  Generator_Map = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
  Generator_Aerial = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

  # Optimizators initialization
  optim_discriminator = optim.Adam(
      list(Discriminator_Map.parameters()) + list(Discriminator_Aerial.parameters()),
      lr = config.LEARNING_RATE,
      betas = (0.5, 0.999),
  )

  optim_generator = optim.Adam(
      list(Generator_Map.parameters()) + list(Generator_Aerial.parameters()),
      lr = config.LEARNING_RATE,
      betas = (0.5, 0.999),
  )

  # Loss funtions initialization
  L1_loss = nn.L1Loss()
  MSE_loss = nn.MSELoss()

  if config.LOAD_MODEL:
    load_checkpoint(
        config.CHECKPOINT_GEN_M, Generator_Map, optim_generator ,config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_GEN_A, Generator_Aerial, optim_generator ,config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_CRITIC_M, Discriminator_Map, optim_discriminator ,config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_CRITIC_A, Discriminator_Aerial, optim_discriminator ,config.LEARNING_RATE,
    )

  # Dataset and Dataloader initialization
  dataset = MapDataset(
      root_map = config.TRAIN_DIR + "trainB", root_aerial = config.TRAIN_DIR + "trainA", transform=config.transforms
  )

  dataloader = DataLoader(
      dataset,
      batch_size=config.BATCH_SIZE,
      shuffle=True,
      num_workers=config.NUM_WORKERS,
      pin_memory=True,
  )

  # Scalers initialization
  generator_scaler =  torch.amp.GradScaler(enabled=use_amp)
  discriminator_scaler =  torch.amp.GradScaler(enabled=use_amp)

  # Train loop
  for epoch in range(config.NUM_EPOCHS):
    train_function(
        Discriminator_Map,
        Discriminator_Aerial,
        Generator_Aerial,
        Generator_Map,
        dataloader,
        optim_discriminator,
        optim_generator,
        L1_loss,
        MSE_loss,
        generator_scaler,
        discriminator_scaler,
        epoch,
    )

    if config.SAVE_MODEL:
      save_checkpoint(Generator_Map, optim_generator, file_name=config.CHECKPOINT_GEN_M)
      save_checkpoint(Generator_Aerial, optim_generator, file_name=config.CHECKPOINT_GEN_A)
      save_checkpoint(Discriminator_Map, optim_discriminator, file_name=config.CHECKPOINT_CRITIC_M)
      save_checkpoint(Discriminator_Aerial, optim_discriminator, file_name=config.CHECKPOINT_CRITIC_A)