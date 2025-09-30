import itertools
import os
import shutil

import pytest
import torch

from autoencoder.data.datasets.mcap_loader import McapImgLoader
from autoencoder.enums import DataSplit


@pytest.fixture
def staged_mcap(tmp_path):
    """
    Stage assets/test.mcap into a temporary root_dir
    with train/ and val/ subdirs using hardlinks.
    Cleans up automatically after the test.
    """
    asset_file = os.path.join("assets", "test.mcap")
    assert os.path.exists(asset_file), f"Missing test.mcap at {asset_file}"

    # Create train/ and val/ dirs inside tmp_path
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    shutil.copy(asset_file, train_dir / "test.mcap")
    shutil.copy(asset_file, val_dir / "test.mcap")

    return tmp_path  # acts as root_dir for the loader


@pytest.mark.parametrize("split", [DataSplit.TRAIN, DataSplit.VAL])
def test_mcap_img_loader(staged_mcap, split):
    dataset = McapImgLoader(
        target_height=768,
        target_width=1920,
        crop_bottom=0,
        crop_top=300,
        scale=2.0,
        root_dir=str(staged_mcap),
        data_split=split,
        topics=[
            "/camera/front/image_compressed",
            "/camera/front_left/image_compressed",
            "/camera/front_right/image_compressed",
        ],
        buffer_size=5,
    )

    # Assert dataset not empty
    assert len(dataset) > 0

    for img in itertools.islice(dataset, 10):
        # Check image tensor properties
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.ndim == 3
        assert img.shape[0] == 3  # C,H,W
