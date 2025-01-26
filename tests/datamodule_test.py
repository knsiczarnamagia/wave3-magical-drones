import pytest
from omegaconf import OmegaConf
from magical_drones.datasets.magmap import MagMapV1
import structlog


@pytest.fixture
def magmap_cfg():
    return OmegaConf.create(
        {
            "data_link": "czarna-magia/mag-map",
            "data_files": "data/train-00000-of-00018.parquet",
            "split_for_upload": [80, 10, 10, "abs"],
            "batch_size": 4,
            "num_workers": 2,
            "prefetch_factor": 2,
            "data_dir": "./data",
            "train_transforms": {"size": 256, "degrees": 180, "flip_p": 0.5},
            "valid_transforms": {"size": 256},
            "test_transforms": {"size": 256},
        }
    )


@pytest.fixture
def magmap(magmap_cfg):
    dm = MagMapV1(magmap_cfg)
    dm.setup()
    return dm


def test_setup_datasets(magmap):
    assert magmap.train_dataset is not None
    assert magmap.val_dataset is not None
    assert magmap.test_dataset is not None
    print("\nDataset sizes:")
    print(f"Train: {len(magmap.train_dataset)}")
    print(f"Val: {len(magmap.val_dataset)}")
    print(f"Test: {len(magmap.test_dataset)}")


def test_dataloaders(magmap):
    logger = structlog.get_logger()

    for name, loader in [
        ("Train", magmap.train_dataloader()),
        ("Val", magmap.val_dataloader()),
        ("Test", magmap.test_dataloader()),
    ]:
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape[0] in {4, 10}, f"Wrong batch size in {name}"
        assert batch[0].shape[1:] == (3, 256, 256)
        logger.info(f"{name} batch OK", shapes=[t.shape for t in batch])
