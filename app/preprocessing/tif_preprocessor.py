"""
TIFF Preprocessor for Multi-Frame Super-Resolution (MFSR).

Loads 8 temporal GeoTIFF files, validates them, normalizes Sentinel-2 reflectance,
constructs the exact 17-channel representation, and outputs a PyTorch tensor
of shape [1, 8, 17, H, W].
"""

import logging
from typing import Any, List, Optional, Tuple
from fastapi import HTTPException
import numpy as np
import rasterio
import torch

from app.config import settings
from app.preprocessing.channel_mapping import (
    EXPECTED_NUM_CHANNELS,
    NUM_REFLECTANCE_CHANNELS,
    MASK_CHANNEL_INDEX,
    NUM_ANGLE_CHANNELS,
    ALL_17_CHANNEL_NAMES,
)
from app.utils.geospatial import read_geospatial_metadata, GeospatialMetadata
from app.utils.validation import validate_spatial_consistency

logger = logging.getLogger(__name__)


def process_single_frame(
    tiff_path: str,
    frame_idx: int,
    allow_synthetic: bool = False,
) -> np.ndarray:
    """
    Reads a single GeoTIFF, validates channel counts, performs Sentinel-2 normalization,
    and formats into exactly 17 channels [17, H, W].

    Args:
        tiff_path: Path to the GeoTIFF file
        frame_idx: 1-indexed temporal frame number
        allow_synthetic: If True, permits synthesizing missing mask/angles for testing

    Returns:
        np.ndarray of shape [17, H, W], dtype float32
    """
    with rasterio.open(tiff_path) as src:
        num_channels = src.count
        height = src.height
        width = src.width
        raw_data = src.read()  # [C, H, W]

    # Clean NaN / Inf values immediately
    data = np.nan_to_num(raw_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Channel count validation and 17-channel tensor construction
    # ------------------------------------------------------------------
    if num_channels == EXPECTED_NUM_CHANNELS:
        # Exactly 17 channels:
        # [0:12] = 12 reflectance bands
        # [12:13] = 1 valid/cloud mask
        # [13:17] = 4 angle channels
        channels_17 = data

    elif num_channels == 16:
        if allow_synthetic or settings.ALLOW_SYNTHETIC_ANGLES:
            logger.warning(
                f"Frame {frame_idx}: 16 channels detected (12 reflectance + 4 angles). "
                f"Synthesizing all-valid mask (1.0) for channel 12."
            )
            # Insert valid mask = 1.0 at index 12
            ref = data[0:12]
            mask = np.ones((1, height, width), dtype=np.float32)
            angles = data[12:16]
            channels_17 = np.concatenate([ref, mask, angles], axis=0)
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing valid/cloud mask channel",
                    "message": (
                        f"Frame {frame_idx} has 16 channels. MFSR requires 17 channels: "
                        f"12 reflectance bands (0-11), 1 valid/cloud mask (12), "
                        f"and 4 solar/view angle channels (13-16). "
                        f"Channel 12 (valid/cloud mask) is missing."
                    ),
                    "frame_index": frame_idx,
                    "received_channels": num_channels,
                    "expected_channels": EXPECTED_NUM_CHANNELS,
                },
            )

    elif num_channels == 12 or num_channels == 13:
        if allow_synthetic or settings.ALLOW_SYNTHETIC_ANGLES:
            logger.warning(
                f"Frame {frame_idx}: {num_channels} channels detected. "
                f"Synthesizing auxiliary mask and neutral angle channels (Sun Zenith 30°, Sun Azimuth 120°, View Zenith 0°, View Azimuth 0°)."
            )
            ref = data[0:12]
            if num_channels == 13:
                mask = data[12:13]
            else:
                mask = np.ones((1, height, width), dtype=np.float32)
            # Neutral angles: typical solar zenith 30 deg, azimuth 120 deg, nadir sensor (0 deg)
            sun_zenith = np.full((1, height, width), 30.0, dtype=np.float32)
            sun_azimuth = np.full((1, height, width), 120.0, dtype=np.float32)
            view_zenith = np.zeros((1, height, width), dtype=np.float32)
            view_azimuth = np.zeros((1, height, width), dtype=np.float32)
            angles = np.concatenate([sun_zenith, sun_azimuth, view_zenith, view_azimuth], axis=0)
            channels_17 = np.concatenate([ref, mask, angles], axis=0)
        else:
            missing_info = "4 solar/viewing angle channels (Sun Zenith, Sun Azimuth, View Zenith, View Azimuth)"
            if num_channels == 12:
                missing_info = "1 valid/cloud mask channel and " + missing_info

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Incomplete satellite image channels",
                    "message": (
                        f"Frame {frame_idx} has {num_channels} channels. MFSR expects 17 channels: "
                        f"12 Sentinel-2 surface reflectance bands (0-11), "
                        f"1 valid/cloud mask (12), and 4 solar/view angle channels (13-16). "
                        f"Missing: {missing_info}."
                    ),
                    "frame_index": frame_idx,
                    "received_channels": num_channels,
                    "expected_channels": EXPECTED_NUM_CHANNELS,
                },
            )
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid channel count",
                "message": (
                    f"Frame {frame_idx} contains {num_channels} channels, which is incompatible with MFSR. "
                    f"Expected exactly 17 channels: 12 Sentinel-2 reflectance bands, "
                    f"1 cloud/valid mask, and 4 solar/viewing angle channels."
                ),
                "frame_index": frame_idx,
                "received_channels": num_channels,
                "expected_channels": EXPECTED_NUM_CHANNELS,
            },
        )

    # ------------------------------------------------------------------
    # Normalization (Exact match to training pipeline)
    # Sentinel-2 raw L2A reflectance is 16-bit integer (scaled by 10000.0)
    # If max reflectance > 1.0, divide reflectance bands (0..11) by 10000.0
    # ------------------------------------------------------------------
    ref_bands = channels_17[0:12]
    max_ref = float(ref_bands.max())
    if max_ref > 1.0:
        channels_17[0:12] = np.clip(ref_bands / settings.NORMALIZE_DIVISOR, 0.0, 1.0)
    else:
        channels_17[0:12] = np.clip(ref_bands, 0.0, 1.0)

    # Valid mask (channel 12): ensure valid pixel representation
    mask_band = channels_17[12:13]
    if mask_band.max() > 1.0:
        # Non-normalized mask (e.g. 0 or 255)
        channels_17[12:13] = (mask_band > 0).astype(np.float32)
    else:
        # Already in [0, 1]
        channels_17[12:13] = np.clip(mask_band, 0.0, 1.0)

    return channels_17.astype(np.float32)


def preprocess_temporal_tiffs(
    tiff_paths: List[str],
    allow_synthetic: bool = False,
) -> Tuple[torch.Tensor, GeospatialMetadata, dict[str, Any]]:
    """
    Validates and preprocesses 8 temporal GeoTIFF files into an MFSR input tensor.

    Args:
        tiff_paths: List of 8 paths to the temporal GeoTIFFs
        allow_synthetic: If True, permits fallback angle/mask generation for testing

    Returns:
        tensor: PyTorch tensor of shape [1, 8, 17, H, W] on CPU
        ref_geospatial: GeospatialMetadata object of reference frame 1
        debug_info: Processing summary dict
    """
    if len(tiff_paths) != 8:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid frame count",
                "message": f"Expected exactly 8 temporal frames, received {len(tiff_paths)}.",
            },
        )

    # Validate spatial consistency across all 8 frames
    (height, width), crs_str = validate_spatial_consistency(tiff_paths)

    # Read reference geospatial metadata from frame 1
    ref_metadata = read_geospatial_metadata(tiff_paths[0])

    # Process all 8 frames
    frames_list = []
    for idx, path in enumerate(tiff_paths, start=1):
        frame_17ch = process_single_frame(
            path,
            frame_idx=idx,
            allow_synthetic=allow_synthetic,
        )
        frames_list.append(frame_17ch)

    # Stack along temporal dimension T=8: [8, 17, H, W]
    stacked_np = np.stack(frames_list, axis=0)

    # Convert to PyTorch tensor and add batch dimension B=1: [1, 8, 17, H, W]
    input_tensor = torch.from_numpy(stacked_np).unsqueeze(0).float()

    debug_info = {
        "num_frames": 8,
        "input_tensor_shape": list(input_tensor.shape),
        "spatial_dimensions": {"height": height, "width": width},
        "crs": crs_str,
        "channels_per_frame": EXPECTED_NUM_CHANNELS,
        "channel_ordering": ALL_17_CHANNEL_NAMES,
    }

    return input_tensor, ref_metadata, debug_info
