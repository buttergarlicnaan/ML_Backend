"""
Unit tests for the TIFF preprocessing pipeline.
"""

import os
import torch
import pytest
from app.preprocessing.tif_preprocessor import (
    preprocess_temporal_tiffs,
    process_single_frame,
)
from app.model.mfsr_model import MFSR_Network
from tests.conftest import create_synthetic_geotiff


def test_preprocessing_8_tiffs_pipeline(temp_dir):
    """
    Test requirement:
    Input: 8 TIFF frames
    Produces: torch.Size([1, 8, 17, H, W])
    and inference produces:
    prediction: [1, out_channels, HR_H, HR_W]
    uncertainty: [1, 1, HR_H, HR_W]
    """
    H, W = 24, 24
    tiff_paths = []
    for i in range(8):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(
            file_path=path,
            num_channels=17,
            height=H,
            width=W,
            scale_16bit=True,
        )
        tiff_paths.append(path)

    # 1. Preprocessing
    tensor, ref_meta, debug_info = preprocess_temporal_tiffs(tiff_paths)

    # Verify tensor shape: [1, 8, 17, H, W]
    assert tensor.shape == torch.Size([1, 8, 17, H, W]), f"Unexpected tensor shape: {tensor.shape}"
    assert tensor.dtype == torch.float32

    # Verify normalization: 16-bit reflectance (>1.0) scaled by 10000.0 into [0, 1]
    reflectance_channels = tensor[0, :, 0:12, :, :]
    assert reflectance_channels.max() <= 1.0 + 1e-5
    assert reflectance_channels.min() >= 0.0

    # 2. Run inference with the preprocessed tensor
    model = MFSR_Network(feat_channels=64, out_channels=4)
    model.eval()
    hr_target = (48, 48)

    with torch.inference_mode():
        prediction, uncertainty = model(tensor, target_size=hr_target)

    assert prediction.shape == torch.Size([1, 4, 48, 48])
    assert uncertainty.shape == torch.Size([1, 1, 48, 48])


def test_nan_inf_cleaning(temp_dir):
    """Verify that NaN and Inf values are cleanly handled."""
    path = os.path.join(temp_dir, "frame_nan.tif")
    create_synthetic_geotiff(path, num_channels=17, height=16, width=16, inject_nan=True)

    frame = process_single_frame(path, frame_idx=1)
    assert not torch.isnan(torch.from_numpy(frame)).any()
    assert not torch.isinf(torch.from_numpy(frame)).any()
