"""
Geospatial utilities for reading raster metadata and generating geospatially aligned GeoTIFFs.
"""

import os
from typing import Any, Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds
from rasterio.crs import CRS


class GeospatialMetadata:
    def __init__(
        self,
        crs: Optional[CRS],
        transform: Affine,
        width: int,
        height: int,
        count: int,
        dtypes: tuple[str, ...],
        bounds: Any,
    ):
        self.crs = crs
        self.transform = transform
        self.width = width
        self.height = height
        self.count = count
        self.dtypes = dtypes
        self.bounds = bounds

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": str(self.crs) if self.crs else None,
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "bounds": {
                "left": self.bounds.left if self.bounds else None,
                "bottom": self.bounds.bottom if self.bounds else None,
                "right": self.bounds.right if self.bounds else None,
                "top": self.bounds.top if self.bounds else None,
            },
        }


def read_geospatial_metadata(tiff_path: str) -> GeospatialMetadata:
    """
    Reads CRS, Affine transform, dimensions, and bounding box from a GeoTIFF.
    """
    with rasterio.open(tiff_path) as src:
        return GeospatialMetadata(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            count=src.count,
            dtypes=src.dtypes,
            bounds=src.bounds,
        )


def compute_upscaled_transform(
    lr_transform: Affine,
    lr_width: int,
    lr_height: int,
    hr_width: int,
    hr_height: int,
    bounds: Optional[Any] = None,
) -> Affine:
    """
    Computes the scaled affine transform so that the high-resolution output
    preserves exact geospatial bounding box and alignment.
    """
    if bounds is not None and bounds.left is not None:
        # Use rasterio from_bounds for exact bounding box preservation
        return from_bounds(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            hr_width,
            hr_height,
        )

    # Scale transform using pixel size ratios: dx_hr = dx_lr * (W_lr / W_hr)
    scale_x = float(lr_width) / float(hr_width)
    scale_y = float(lr_height) / float(hr_height)
    return lr_transform * Affine.scale(scale_x, scale_y)


def save_geotiff(
    output_path: str,
    data: np.ndarray,
    crs: Optional[CRS],
    transform: Affine,
    nodata: Optional[float] = None,
    descriptions: Optional[list[str]] = None,
) -> str:
    """
    Writes a NumPy array to a GeoTIFF file.

    Args:
        output_path: Destination file path
        data: Array of shape [channels, height, width] or [height, width]
        crs: Coordinate Reference System
        transform: Updated Affine transform
        nodata: Optional NoData value
        descriptions: Optional list of band description strings

    Returns:
        Absolute output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if data.ndim == 2:
        # Single band: [height, width] -> [1, height, width]
        count = 1
        height, width = data.shape
        write_data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        if data.shape[0] > data.shape[2] and data.shape[2] <= 16:
            # If passed as [height, width, channels], transpose to [channels, height, width]
            write_data = np.transpose(data, (2, 0, 1))
        else:
            write_data = data
        count, height, width = write_data.shape
    else:
        raise ValueError(f"Unsupported data shape for GeoTIFF writing: {data.shape}")

    write_data = np.nan_to_num(write_data, nan=0.0).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(write_data)
        if descriptions and len(descriptions) == count:
            for idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(idx, desc)

    return os.path.abspath(output_path)
