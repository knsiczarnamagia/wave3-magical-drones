from magical_drones.datasets.magmap import MagMapV1
from torchvision import transforms

if __name__ == "__main__":
    
    data_link = "Kiwinicki/sat2map_poland"
    batch_size = 32
    transform = transforms.Compose([
            transforms.ToTensor(),  
        ])
    
    magmap = MagMapV1(data_link, 
                      batch_size=batch_size,
                      transform=transform)

    print("Setting up datasets...")
    magmap.setup()

    print("Testing train_dataloader...")
    train_loader = magmap.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2: 
            break

    print("\nTesting val_dataloader...")
    val_loader = magmap.val_dataloader()
    for i, batch in enumerate(val_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2: 
            break

    print("\nTesting test_dataloader...")
    test_loader = magmap.test_dataloader()
    for i, batch in enumerate(test_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2:  
            break

    for i, batch in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"  sat_image shape: {batch['sat_image'].shape}")
        print(f"  map_image shape: {batch['map_image'].shape}")
        if i == 2:
            break