import pytest
import torch
from unittest.mock import patch, MagicMock
from magical_drones.datasets.magmap import MagMapV1
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomCrop, Resize, ToTensor


@pytest.fixture
def magmap_dataset_mocked():
    data_link = "czarna-magia/mag-map"
    batch_size = 32

    train_transform = MagicMock(return_value=("mock_sat_tensor", "mock_map_tensor"))
    test_transform = MagicMock(return_value=("mock_sat_tensor", "mock_map_tensor"))

    magmap = MagMapV1(
        data_link,
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    with patch.object(magmap, "prepare_data") as mock_prepare_data:
        mock_prepare_data.return_value = None
        magmap.data_dict = {
            "train": [
                {"sat_image": "fake_sat1", "map_image": "fake_map1"},
                {"sat_image": "fake_sat2", "map_image": "fake_map2"},
                {"sat_image": "fake_sat3", "map_image": "fake_map3"},
            ]
        }

        with patch("magical_drones.datasets.magmap.read_image") as mock_read_image:
            mock_read_image.side_effect = lambda path: torch.rand(3, 256, 256)
            yield magmap


def test_prepare_data(magmap_dataset_mocked):
    print("Preparing data...")
    magmap_dataset_mocked.prepare_data()
    assert magmap_dataset_mocked.data_dict is not None, "The data has not been mocked."


def test_setup_datasets(magmap_dataset_mocked):
    print("Setting up datasets...")
    magmap_dataset_mocked.prepare_data()
    magmap_dataset_mocked.setup()

    assert magmap_dataset_mocked.train_dataset is not None, "Train dataset not created"
    assert (
        magmap_dataset_mocked.val_dataset is not None
    ), "Validation dataset not created"
    assert magmap_dataset_mocked.test_dataset is not None, "Test dataset not created"


def test_train_dataloader(magmap_dataset_mocked):
    print("Testing train_dataloader...")
    magmap_dataset_mocked.prepare_data()
    magmap_dataset_mocked.setup()
    train_loader = magmap_dataset_mocked.train_dataloader()

    for i, batch in enumerate(train_loader):
        assert batch is not None, f"Batch {i + 1} is empty"
        assert len(batch) > 0, f"Batch {i + 1} contains no data"
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break


def test_val_dataloader(magmap_dataset_mocked):
    print("Testing val_dataloader...")
    magmap_dataset_mocked.prepare_data()
    magmap_dataset_mocked.setup()
    val_loader = magmap_dataset_mocked.val_dataloader()

    for i, batch in enumerate(val_loader):
        assert batch is not None, f"Batch {i + 1} is empty"
        assert len(batch) > 0, f"Batch {i + 1} contains no data"
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break


def test_test_dataloader(magmap_dataset_mocked):
    print("Testing test_dataloader...")
    magmap_dataset_mocked.prepare_data()
    magmap_dataset_mocked.setup()
    test_loader = magmap_dataset_mocked.test_dataloader()

    for i, batch in enumerate(test_loader):
        assert batch is not None, f"Batch {i + 1} is empty"
        assert len(batch) > 0, f"Batch {i + 1} contains no data"
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break
