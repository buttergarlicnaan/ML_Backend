"""
Geospatial utilities for reading raster metadata and generating geospatially aligned GeoTIFFs,
RGB visual previews, and ZIP archives.
"""

import json
import os
from typing import Any, Optional, Tuple
import zipfile
import numpy as np
from PIL import Image
import matplotlib.cm as cm
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
    if bounds is not None and bounds.left is not None:
        return from_bounds(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            hr_width,
            hr_height,
        )
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if data.ndim == 2:
        count = 1
        height, width = data.shape
        write_data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        if data.shape[0] > data.shape[2] and data.shape[2] <= 16:
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


def save_rgb_png(
    output_path: str,
    data: np.ndarray,
    rgb_indices: Tuple[int, int, int] = (0, 1, 2),
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if data.ndim == 2:
        gray = np.nan_to_num(data, nan=0.0).astype(np.float32)
        p2, p98 = np.percentile(gray, (2, 98))
        norm = np.clip((gray - p2) / (p98 - p2 + 1e-7), 0.0, 1.0)
        rgb_uint8 = np.stack([(norm * 255.0).astype(np.uint8)] * 3, axis=-1)
    else:
        max_idx = data.shape[0] - 1
        r_idx = min(rgb_indices[0], max_idx)
        g_idx = min(rgb_indices[1], max_idx)
        b_idx = min(rgb_indices[2], max_idx)

        rgb = np.stack([data[r_idx], data[g_idx], data[b_idx]], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0).astype(np.float32)

        p2, p98 = np.percentile(rgb, (2, 98))
        norm = np.clip((rgb - p2) / (p98 - p2 + 1e-7), 0.0, 1.0)
        rgb_uint8 = (norm * 255.0).astype(np.uint8)

    img = Image.fromarray(rgb_uint8, mode="RGB")
    img.save(output_path, format="PNG")
    return os.path.abspath(output_path)


def save_uncertainty_png(
    output_path: str,
    data: np.ndarray,
    cmap_name: str = "inferno",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    u_data = np.nan_to_num(data, nan=0.0).astype(np.float32)
    p1, p99 = np.percentile(u_data, (1, 99))
    if p99 > p1:
        norm = np.clip((u_data - p1) / (p99 - p1), 0.0, 1.0)
    else:
        norm = np.zeros_like(u_data)

    try:
        import matplotlib
        colormap = matplotlib.colormaps[cmap_name]
    except (AttributeError, KeyError):
        colormap = cm.get_cmap(cmap_name)

    colored_rgba = colormap(norm)
    colored_rgb = (colored_rgba[..., :3] * 255.0).astype(np.uint8)

    img = Image.fromarray(colored_rgb, mode="RGB")
    img.save(output_path, format="PNG")
    return os.path.abspath(output_path)


def create_results_zip(
    output_zip_path: str,
    file_map: dict[str, str],
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for arcname, filepath in file_map.items():
            if os.path.isfile(filepath):
                zipf.write(filepath, arcname=arcname)
        if metadata is not None:
            metadata_str = json.dumps(metadata, indent=2, default=str)
            zipf.writestr("metadata.json", metadata_str)

    return os.path.abspath(output_zip_path)
