"""
Unit tests for geospatial transformations and GeoTIFF output creation.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from app.utils.geospatial import (
    compute_upscaled_transform,
    save_geotiff,
    read_geospatial_metadata,
)


def test_upscaled_transform_bounds():
    """Verify that compute_upscaled_transform preserves the geospatial envelope."""
    lr_h, lr_w = 159, 158
    hr_h, hr_w = 1060, 1053
    left, bottom, right, top = 500000.0, 4000000.0, 501580.0, 4001590.0

    lr_transform = from_bounds(left, bottom, right, top, lr_w, lr_h)

    class MockBounds:
        pass

    bounds = MockBounds()
    bounds.left = left
    bounds.bottom = bottom
    bounds.right = right
    bounds.top = top

    hr_transform = compute_upscaled_transform(
        lr_transform=lr_transform,
        lr_width=lr_w,
        lr_height=lr_h,
        hr_width=hr_w,
        hr_height=hr_h,
        bounds=bounds,
    )

    # Pixel (0, 0) should correspond to top-left (left, top)
    x_orig, y_orig = hr_transform * (0, 0)
    assert abs(x_orig - left) < 1e-4
    assert abs(y_orig - top) < 1e-4

    # Pixel (hr_w, hr_h) should correspond to bottom-right (right, bottom)
    x_end, y_end = hr_transform * (hr_w, hr_h)
    assert abs(x_end - right) < 1e-4
    assert abs(y_end - bottom) < 1e-4


def test_save_geotiff_multiband(temp_dir):
    """Verify multi-band GeoTIFF creation and band descriptions."""
    out_file = os.path.join(temp_dir, "test_sr.tif")
    data = np.random.uniform(0.0, 1.0, size=(4, 64, 64)).astype(np.float32)
    transform = from_bounds(100, 100, 200, 200, 64, 64)
    crs = "EPSG:4326"

    saved_path = save_geotiff(
        output_path=out_file,
        data=data,
        crs=crs,
        transform=transform,
        descriptions=["Red", "Green", "Blue", "NIR"],
    )

    assert os.path.isfile(saved_path)
    with rasterio.open(saved_path) as src:
        assert src.count == 4
        assert src.height == 64
        assert src.width == 64
        assert src.descriptions == ("Red", "Green", "Blue", "NIR")
        assert src.dtypes[0] == "float32"
