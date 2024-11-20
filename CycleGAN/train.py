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

import pathlib
from pathlib import Path
import os
import zipfile
import shutil
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
          loss_info,
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
      
    loss_info.append(
       {
          "Generator loss" : Generator_loss,
          "Discriminator loss" : Discriminator_loss,
       }
    )

    print(f"Epoch {epoch} | Generator loss: {Generator_loss} | Discriminator loss: {Discriminator_loss}")
    if epoch % 5 == 0:
      pathlib.Path('saved_images').mkdir(parents=True, exist_ok=True)
      
      cycle_first = torch.cat((map, aerial_fake * 0.5 + 0.5, cycle_map), dim=2)
      save_image(cycle_first, f"saved_images/cycle_first_{epoch}.png")

      cycle_second = torch.cat((aerial_photo, map_fake * 0.5 + 0.5, cycle_aerial), dim=2)
      save_image(cycle_second, f"saved_images/cycle_second_{epoch}.png")

def Get_data(image_path,
             data_path,
             kaggle_dataset,
             zip_file_name,
             kaggle_folder_name):
    
    image_path = Path(image_path)  
    data_path = Path(data_path)

    if not image_path.is_dir():
        print(f"Creating directory: {image_path}")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download the dataset from Kaggle using Kaggle API
    if not (data_path / zip_file_name).exists():  
        print("Downloading dataset from Kaggle...")
        os.makedirs(data_path, exist_ok=True)  
        os.system(f"kaggle datasets download -d {kaggle_dataset} -p {data_path}")

    # Unzip the downloaded file
    with zipfile.ZipFile(data_path / zip_file_name, "r") as zip_ref:
        print(f"Unzipping {zip_file_name}...")
        zip_ref.extractall(data_path)

    print("Extracted files:")
    for root, dirs, files in os.walk(data_path):
        print(f"Directory: {root}")
        for dir_name in dirs:
            print(f"  Sub-directory: {dir_name}")
        for file_name in files:
            print(f"  File: {file_name}")

    # Locate trainA and trainB folders
    source_path = data_path / kaggle_folder_name  
    if source_path.exists():
        for folder in ["trainA", "trainB"]: 
            source_folder = source_path / folder
            destination_folder = image_path / folder
            if source_folder.exists():
                print(f"Moving {folder} to {destination_folder}...")
                shutil.move(str(source_folder), str(destination_folder))
            else:
                print(f"Folder {folder} not found in {source_path}.")  # Handle missing folders
    else:
        print(f"Source folder {kaggle_folder_name} not found after unzipping.")

    # Remove 'data/' directory
    print("Cleaning up...")
    if data_path.exists():
        shutil.rmtree(data_path) 

    print("Dataset is ready!")

if __name__ == "__main__":
  
  image_path = "maps_aerial_dataset"
  data_path = Path("data/")               
  kaggle_dataset = "suyashdamle/cyclegan" 
  zip_file_name = "cyclegan.zip"
  kaggle_folder_name = "maps/maps"
  
  Get_data(image_path, 
           data_path, 
           kaggle_dataset, 
           zip_file_name, 
           kaggle_folder_name)

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

  loss_info = []

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
        loss_info,
        epoch,
    )


    if config.SAVE_MODEL:
      save_checkpoint(Generator_Map, optim_generator, file_name=config.CHECKPOINT_GEN_M)
      save_checkpoint(Generator_Aerial, optim_generator, file_name=config.CHECKPOINT_GEN_A)
      save_checkpoint(Discriminator_Map, optim_discriminator, file_name=config.CHECKPOINT_CRITIC_M)
      save_checkpoint(Discriminator_Aerial, optim_discriminator, file_name=config.CHECKPOINT_CRITIC_A)

  generator_losses = [info["Generator loss"] for info in loss_info]
  discriminator_losses = [info["Discriminator loss"] for info in loss_info]

  plt.figure(figsize=(10, 5))
  plt.plot(generator_losses, label='Generator Loss')
  plt.plot(discriminator_losses, label='Discriminator Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.title('Generator and Discriminator Losses')
  plt.savefig("gen_disc_loss.png")  
  plt.close()