"""
TIFF Preprocessor for Multi-Frame Super-Resolution (MFSR).
Incorporates the complete WorldStrat data preprocessing pipeline from
worldstrat-data-preprocessing(1).ipynb.
"""

from collections import defaultdict
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
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


def load_raster(file_path: str) -> np.ndarray:
    with rasterio.open(file_path) as src:
        data = src.read()
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return data


def normalize_reflectance(l2a_data: np.ndarray, divisor: float = 10000.0) -> np.ndarray:
    data = np.nan_to_num(l2a_data.astype(np.float32), nan=0.0, posinf=2.0, neginf=0.0)
    if data.max() > 10.0:
        return np.clip(data / divisor, 0.0, 2.0)
    return np.clip(data, 0.0, 2.0)


def standardize_shape(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    elif arr.ndim == 3 and arr.shape[2] == 12 and arr.shape[0] != 12:
        return np.transpose(arr, (2, 0, 1))
    return arr


def process_validity_mask(
    clm_data: Optional[np.ndarray] = None,
    clp_data: Optional[np.ndarray] = None,
    datamask: Optional[np.ndarray] = None,
    target_hw: Tuple[int, int] = (16, 16),
) -> np.ndarray:
    H, W = target_hw

    if clp_data is not None:
        clp = standardize_shape(clp_data)
        validity_mask = (clp < 50.0).astype(np.float32)
    elif datamask is not None:
        mask = standardize_shape(datamask).astype(np.float32)
        if mask.max() > 1.0:
            validity_mask = (mask > 0).astype(np.float32)
        else:
            validity_mask = np.clip(mask, 0.0, 1.0)
    elif clm_data is not None:
        clm = standardize_shape(clm_data)
        validity_mask = (clm == 0).astype(np.float32)
    else:
        validity_mask = np.ones((1, H, W), dtype=np.float32)

    if np.max(validity_mask) == 0.0:
        validity_mask = np.ones((1, H, W), dtype=np.float32)

    return validity_mask


def normalize_angles(
    sun_az: np.ndarray,
    sun_zen: np.ndarray,
    view_az: np.ndarray,
    view_zen: np.ndarray,
) -> np.ndarray:
    sun_az_norm = np.clip(sun_az / 360.0 if sun_az.max() > 1.0 else sun_az, 0.0, 1.0)
    sun_zen_norm = np.clip(sun_zen / 180.0 if sun_zen.max() > 1.0 else sun_zen, 0.0, 1.0)
    view_az_norm = np.clip(view_az / 360.0 if view_az.max() > 1.0 else view_az, 0.0, 1.0)
    view_zen_norm = np.clip(view_zen / 180.0 if view_zen.max() > 1.0 else view_zen, 0.0, 1.0)

    return np.concatenate([sun_az_norm, sun_zen_norm, view_az_norm, view_zen_norm], axis=0)


def safe_load_angle(
    file_path: Optional[str],
    denominator: float,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    if file_path and os.path.exists(file_path):
        raster = load_raster(file_path)
        std_raster = standardize_shape(raster)
        if std_raster.max() > 1.0:
            return np.clip(std_raster / denominator, 0.0, 1.0)
        return np.clip(std_raster, 0.0, 1.0)

    H, W = target_hw
    return np.full((1, H, W), 0.5, dtype=np.float32)


def group_files_by_frame(folder_or_files: Union[str, List[str]]) -> List[Dict[str, str]]:
    if isinstance(folder_or_files, str) and os.path.isdir(folder_or_files):
        filenames = os.listdir(folder_or_files)
        dir_path = folder_or_files
        full_paths = [os.path.join(dir_path, f) for f in filenames]
    else:
        full_paths = list(folder_or_files)

    frames = defaultdict(dict)
    regex = re.compile(r'-(\d+)-([A-Za-z0-9_]+)\.(?:tiff|tif)$', re.IGNORECASE)

    for filepath in full_paths:
        filename = os.path.basename(filepath)
        match = regex.search(filename)
        if match:
            frame_idx = int(match.group(1))
            param_name = match.group(2)

            if param_name == 'L2A_data':
                frames[frame_idx]['L2A'] = filepath
            elif param_name == 'dataMask':
                frames[frame_idx]['dataMask'] = filepath
            elif param_name == 'CLM':
                frames[frame_idx]['CLM'] = filepath
            elif param_name == 'CLP':
                frames[frame_idx]['CLP'] = filepath
            elif param_name == 'sunAzimuthAngles':
                frames[frame_idx]['sunAzimuth'] = filepath
            elif param_name == 'sunZenithAngles':
                frames[frame_idx]['sunZenith'] = filepath
            elif param_name == 'viewAzimuthMean':
                frames[frame_idx]['viewAzimuth'] = filepath
            elif param_name == 'viewZenithMean':
                frames[frame_idx]['viewZenith'] = filepath

    valid_frames = []
    for frame_idx in sorted(frames.keys()):
        frame_files = frames[frame_idx]
        if 'L2A' in frame_files:
            valid_frames.append(frame_files)

    return valid_frames


def process_single_frame(
    source: Union[str, Dict[str, str]],
    frame_idx: int = 1,
    allow_synthetic: bool = True,
) -> np.ndarray:
    if isinstance(source, dict):
        l2a_raw = load_raster(source['L2A'])
        l2a = standardize_shape(normalize_reflectance(l2a_raw))
        H, W = l2a.shape[1], l2a.shape[2]

        datamask_raw = load_raster(source['dataMask']) if 'dataMask' in source else None
        clp_raw = load_raster(source['CLP']) if 'CLP' in source else None
        clm_raw = load_raster(source['CLM']) if 'CLM' in source else None
        valid_mask = process_validity_mask(clm_raw, clp_raw, datamask_raw, target_hw=(H, W))

        sun_az = safe_load_angle(source.get('sunAzimuth'), 360.0, (H, W))
        sun_zen = safe_load_angle(source.get('sunZenith'), 180.0, (H, W))
        view_az = safe_load_angle(source.get('viewAzimuth'), 360.0, (H, W))
        view_zen = safe_load_angle(source.get('viewZenith'), 180.0, (H, W))

        return np.concatenate([l2a, valid_mask, sun_az, sun_zen, view_az, view_zen], axis=0).astype(np.float32)

    raw_data = load_raster(source)
    raw_data = standardize_shape(raw_data)
    num_channels, H, W = raw_data.shape

    if num_channels == EXPECTED_NUM_CHANNELS:
        ref = normalize_reflectance(raw_data[0:12])
        mask = process_validity_mask(datamask=raw_data[12:13], target_hw=(H, W))

        angles_raw = raw_data[13:17]
        sun_az = angles_raw[0:1]
        sun_zen = angles_raw[1:2]
        view_az = angles_raw[2:3]
        view_zen = angles_raw[3:4]
        angles = normalize_angles(sun_az, sun_zen, view_az, view_zen)

        channels_17 = np.concatenate([ref, mask, angles], axis=0)

    elif num_channels == 16:
        if allow_synthetic or settings.ALLOW_SYNTHETIC_ANGLES:
            ref = normalize_reflectance(raw_data[0:12])
            mask = np.ones((1, H, W), dtype=np.float32)
            angles_raw = raw_data[12:16]
            angles = normalize_angles(angles_raw[0:1], angles_raw[1:2], angles_raw[2:3], angles_raw[3:4])
            channels_17 = np.concatenate([ref, mask, angles], axis=0)
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing valid/cloud mask channel",
                    "message": f"Frame {frame_idx} has 16 channels. MFSR requires 17 channels.",
                    "frame_index": frame_idx,
                    "received_channels": num_channels,
                    "expected_channels": EXPECTED_NUM_CHANNELS,
                },
            )

    elif num_channels in (12, 13):
        if allow_synthetic or settings.ALLOW_SYNTHETIC_ANGLES:
            ref = normalize_reflectance(raw_data[0:12])
            if num_channels == 13:
                mask = process_validity_mask(datamask=raw_data[12:13], target_hw=(H, W))
            else:
                mask = np.ones((1, H, W), dtype=np.float32)

            sun_az = np.full((1, H, W), 0.5, dtype=np.float32)
            sun_zen = np.full((1, H, W), 0.5, dtype=np.float32)
            view_az = np.full((1, H, W), 0.5, dtype=np.float32)
            view_zen = np.full((1, H, W), 0.5, dtype=np.float32)
            angles = np.concatenate([sun_az, sun_zen, view_az, view_zen], axis=0)

            channels_17 = np.concatenate([ref, mask, angles], axis=0)
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Incomplete satellite image channels",
                    "message": f"Frame {frame_idx} has {num_channels} channels. MFSR expects 17 channels.",
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
                "message": f"Frame {frame_idx} contains {num_channels} channels. Expected 17.",
                "frame_index": frame_idx,
                "received_channels": num_channels,
                "expected_channels": EXPECTED_NUM_CHANNELS,
            },
        )

    return channels_17.astype(np.float32)


def preprocess_temporal_tiffs(
    tiff_paths: List[str],
    allow_synthetic: bool = True,
) -> Tuple[torch.Tensor, GeospatialMetadata, dict[str, Any]]:
    ws_regex = re.compile(r'-(CLM|CLP|L2A_data|dataMask|sunAzimuthAngles|sunZenithAngles|viewAzimuthMean|viewZenithMean)\.(?:tiff|tif)$', re.IGNORECASE)
    is_worldstrat_split = any(ws_regex.search(os.path.basename(p)) for p in tiff_paths)

    if is_worldstrat_split:
        grouped_frames = group_files_by_frame(tiff_paths)
        if len(grouped_frames) == 0:
            raise HTTPException(
                status_code=400,
                detail={"error": "No valid frames found", "message": "No valid L2A frames found in uploaded files."},
            )

        temporal_frames = []
        ref_metadata = None
        for idx, frame_info in enumerate(grouped_frames[:8], start=1):
            if ref_metadata is None:
                ref_metadata = read_geospatial_metadata(frame_info['L2A'])
            frame_tensor = process_single_frame(frame_info, frame_idx=idx, allow_synthetic=allow_synthetic)
            temporal_frames.append(frame_tensor)

        _, H, W = temporal_frames[0].shape
        while len(temporal_frames) < 8:
            blank_frame = np.zeros((EXPECTED_NUM_CHANNELS, H, W), dtype=np.float32)
            temporal_frames.append(blank_frame)

        stacked_np = np.stack(temporal_frames, axis=0)
        input_tensor = torch.from_numpy(stacked_np).unsqueeze(0).float()

        debug_info = {
            "num_frames": 8,
            "detected_valid_frames": len(grouped_frames),
            "input_tensor_shape": list(input_tensor.shape),
            "spatial_dimensions": {"height": H, "width": W},
            "crs": str(ref_metadata.crs) if ref_metadata else None,
            "channel_ordering": ALL_17_CHANNEL_NAMES,
        }
        return input_tensor, ref_metadata, debug_info

    if len(tiff_paths) != 8:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid frame count",
                "message": f"Expected exactly 8 temporal frames, received {len(tiff_paths)}.",
                "required_count": 8,
                "received_count": len(tiff_paths),
            },
        )

    (height, width), crs_str = validate_spatial_consistency(tiff_paths)
    ref_metadata = read_geospatial_metadata(tiff_paths[0])

    frames_list = []
    for idx, path in enumerate(tiff_paths, start=1):
        frame_17ch = process_single_frame(
            path,
            frame_idx=idx,
            allow_synthetic=allow_synthetic,
        )
        frames_list.append(frame_17ch)

    stacked_np = np.stack(frames_list, axis=0)
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
