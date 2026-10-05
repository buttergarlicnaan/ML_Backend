"""
Pytest fixtures and test helpers for generating synthetic GeoTIFF files
and initializing the FastAPI TestClient.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.inference.predictor import predictor


@pytest.fixture(scope="session", autouse=True)
def init_model():
    """Ensure predictor is loaded before running tests."""
    if not predictor.is_loaded:
        predictor.load(
            checkpoint_path=settings.MODEL_CHECKPOINT,
            device_name="cpu",
            feat_channels=64,
        )
    return predictor


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def create_synthetic_geotiff(
    file_path: str,
    num_channels: int = 17,
    height: int = 32,
    width: int = 32,
    crs: str = "EPSG:32632",
    origin_x: float = 500000.0,
    origin_y: float = 4000000.0,
    pixel_size: float = 10.0,
    scale_16bit: bool = True,
    inject_nan: bool = False,
) -> str:
    """
    Creates a valid synthetic multi-band GeoTIFF with geospatial metadata.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    # Generate random raster data
    if scale_16bit:
        # Simulate Sentinel-2 16-bit DN values (0 - 10000)
        data = np.random.uniform(200.0, 4500.0, size=(num_channels, height, width)).astype(np.float32)
    else:
        data = np.random.uniform(0.02, 0.45, size=(num_channels, height, width)).astype(np.float32)

    # If 17 channels:
    # 0..11: reflectance
    # 12: valid mask (1.0 = valid)
    # 13..16: angles (degrees)
    if num_channels == 17:
        data[12] = 1.0  # Clear valid mask
        data[13] = 32.5  # Solar Zenith
        data[14] = 135.0  # Solar Azimuth
        data[15] = 4.2  # View Zenith
        data[16] = 85.0  # View Azimuth

    if inject_nan:
        data[0, 0, 0] = np.nan
        data[0, 0, 1] = np.inf

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": num_channels,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
    }

    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    d = tempfile.mkdtemp(prefix="mfsr_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)
