import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from magical_drones.datasets.magmap import MagMapV1
import structlog


@pytest.fixture
def mock_dataset():
    return {
        "train": [
            {
                "sat_image": MagicMock(convert=MagicMock(return_value=MagicMock())),
                "map_image": MagicMock(convert=MagicMock(return_value=MagicMock())),
            }
            for _ in range(100)
        ]
    }


@pytest.fixture
def magmap_cfg():
    return OmegaConf.create(
        {
            "data_link": "dummy/data_link",
            "data_files": "dummy/data.parquet",
            "split_for_upload": [80, 10, 10, "%"],
            "batch_size": 32,
            "num_workers": 4,
            "prefetch_factor": 2,
            "data_dir": "./data",
            "train_transforms": {"size": 256, "degrees": 180, "flip_p": 0.5},
            "valid_transforms": {"size": 256},
            "test_transforms": {"size": 256},
        }
    )


@pytest.fixture
def magmap(magmap_cfg, mock_dataset):
    with patch("datasets.load_dataset", return_value=mock_dataset):
        mm = MagMapV1(magmap_cfg)
        mm.setup()
        return mm


def test_setup_datasets(magmap):
    assert magmap.train_dataset is not None
    assert magmap.val_dataset is not None
    assert magmap.test_dataset is not None


def test_dataloaders(magmap):
    logger = structlog.get_logger()

    for loader in [
        magmap.train_dataloader(),
        magmap.val_dataloader(),
        magmap.test_dataloader(),
    ]:
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape == (32, 3, 256, 256)
        logger.info("Batch check passed", batch_shape=batch[0].shape)
