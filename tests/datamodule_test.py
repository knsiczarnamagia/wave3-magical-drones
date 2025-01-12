import pytest
import torch
from unittest.mock import patch, MagicMock
from magical_drones.datasets.magmap import MagMapV1
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomCrop, Resize, ToTensor
import structlog


@pytest.fixture
def magmap():
    data_link = "czarna-magia/mag-map"
    data_files = "data/train-00000-of-00018.parquet"
    split_for_upload = [80, 10, 10, "abs"]
    batch_size = 32

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(180),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return MagMapV1(
        data_link,
        data_files=data_files,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        # split_for_upload=split_for_upload,
    )


def test_prepare_data(magmap):
    print("Preparing data...")
    magmap.prepare_data()
    assert magmap.train_data_dict is not None, "The train data has not been uploaded."
    assert (
        magmap.val_data_dict is not None
    ), "The validation data has not been uploaded."
    assert magmap.test_data_dict is not None, "The test data has not been uploaded."


def test_setup_datasets(magmap):
    print("Setting up datasets...")
    magmap.prepare_data()
    magmap.setup()

    assert magmap.train_dataset is not None, "Train dataset not created"
    assert magmap.val_dataset is not None, "Validation dataset not created"
    assert magmap.test_dataset is not None, "Test dataset not created"


def test_train_dataloader(magmap):
    print("Testing train_dataloader...")
    magmap.prepare_data()
    magmap.setup()
    train_loader = magmap.train_dataloader()

    logger = structlog.get_logger()

    batch = next(iter(train_loader), None)
    assert batch is not None, "Train dataloader batch is empty"
    assert len(batch) > 0, "Train dataloader batch contains no data"
    logger.info("Sampled Val Batch", batch=batch)


def test_val_dataloader(magmap):
    print("Testing val_dataloader...")
    magmap.prepare_data()
    magmap.setup()
    val_loader = magmap.val_dataloader()

    logger = structlog.get_logger()

    batch = next(iter(val_loader), None)
    assert batch is not None, "Validation dataloader batch is empty"
    assert len(batch) > 0, "Validation dataloader batch contains no data"
    logger.info("Sampled Val Batch", batch=batch)


def test_test_dataloader(magmap):
    print("Testing test_dataloader...")
    magmap.prepare_data()
    magmap.setup()
    test_loader = magmap.test_dataloader()

    logger = structlog.get_logger()

    batch = next(iter(test_loader), None)
    assert batch is not None, "Test dataloader batch is empty"
    assert len(batch) > 0, "Test dataloader batch contains no data"
    logger.info("Sampled Val Batch", batch=batch)
