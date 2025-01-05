import pytest
from magical_drones.datasets.magmap import MagMapV1
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomCrop, Resize, ToTensor


@pytest.fixture
def magmap_dataset():
    data_link = "czarna-magia/mag-map"
    batch_size = 32

    train_transform = transforms.Compose(
        [RandomCrop(size=(224, 224)), RandomHorizontalFlip(p=0.5), ToTensor()]
    )

    test_transform = transforms.Compose([Resize(size=(224, 224)), ToTensor()])

    magmap = MagMapV1(
        data_link,
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
    )
    return magmap


def test_prepare_data(magmap_dataset):
    print("Preparing data...")
    magmap_dataset.prepare_data()
    assert magmap_dataset.data_dict is not None, "The data has not been downloaded."


def test_setup_datasets(magmap_dataset):
    print("Setting up datasets...")
    magmap_dataset.prepare_data()
    magmap_dataset.setup()

    assert magmap_dataset.train_dataset is not None, "Train dataset not created"
    assert magmap_dataset.val_dataset is not None, "Validation dataset not created"
    assert magmap_dataset.test_dataset is not None, "Test dataset not created"


def test_train_dataloader(magmap_dataset):
    print("Testing train_dataloader...")
    magmap_dataset.prepare_data()
    magmap_dataset.setup()
    train_loader = magmap_dataset.train_dataloader()

    for i, batch in enumerate(train_loader):
        assert batch is not None, f"Batch {i + 1} is empty"
        assert len(batch) > 0, f"Batch {i + 1} contains no data"
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break


def test_val_dataloader(magmap_dataset):
    print("Testing val_dataloader...")
    magmap_dataset.prepare_data()
    magmap_dataset.setup()
    val_loader = magmap_dataset.val_dataloader()

    for i, batch in enumerate(val_loader):
        assert batch is not None, f"Batch {i + 1} is empty"
        assert len(batch) > 0, f"Batch {i + 1} contains no data"
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break


def test_test_dataloader(magmap_dataset):
    print("Testing test_dataloader...")
    magmap_dataset.prepare_data()
    magmap_dataset.setup()
    test_loader = magmap_dataset.test_dataloader()

    for i, batch in enumerate(test_loader):
        assert batch is not None, f"Batch {i + 1} is empty"
        assert len(batch) > 0, f"Batch {i + 1} contains no data"
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break
