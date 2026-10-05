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


def test_worldstrat_notebook_preprocessing_components():
    """Verify specific components from worldstrat-data-preprocessing(1).ipynb."""
    from app.preprocessing.tif_preprocessor import (
        normalize_reflectance,
        process_validity_mask,
        normalize_angles,
        standardize_shape,
        safe_load_angle,
    )
    import numpy as np

    # 1. Test normalize_reflectance: WorldStrat clipping to [0.0, 2.0]
    raw_l2a = np.array([[[ -0.5, 0.5, 2.5 ]]], dtype=np.float32)
    norm_l2a = normalize_reflectance(raw_l2a)
    assert norm_l2a.min() == 0.0
    assert norm_l2a.max() == 2.0

    # 2. Test raw 16-bit DN scaling
    raw_dn = np.array([[[ 5000.0, 10000.0, 15000.0 ]]], dtype=np.float32)
    norm_dn = normalize_reflectance(raw_dn)
    assert norm_dn[0, 0, 0] == 0.5
    assert norm_dn[0, 0, 1] == 1.0
    assert norm_dn[0, 0, 2] == 1.5

    # 3. Test standardize_shape
    arr_2d = np.ones((16, 16), dtype=np.float32)
    assert standardize_shape(arr_2d).shape == (1, 16, 16)
    arr_3d = np.ones((16, 16, 12), dtype=np.float32)
    assert standardize_shape(arr_3d).shape == (12, 16, 16)

    # 4. Test process_validity_mask with cloud probability (CLP < 50)
    clp = np.array([[10.0, 60.0]], dtype=np.float32)
    mask = process_validity_mask(clp_data=clp, target_hw=(1, 2))
    assert mask[0, 0, 0] == 1.0  # valid (<50)
    assert mask[0, 0, 1] == 0.0  # invalid (>50)

    # 5. Test mask safety net (all black -> forced all white)
    clp_all_cloud = np.full((1, 4, 4), 99.0, dtype=np.float32)
    safe_mask = process_validity_mask(clp_data=clp_all_cloud, target_hw=(4, 4))
    assert (safe_mask == 1.0).all()

    # 6. Test normalize_angles (Sun Az / 360, Sun Zen / 180, View Az / 360, View Zen / 180)
    sun_az = np.array([[[180.0]]], dtype=np.float32)
    sun_zen = np.array([[[90.0]]], dtype=np.float32)
    view_az = np.array([[[360.0]]], dtype=np.float32)
    view_zen = np.array([[[0.0]]], dtype=np.float32)
    angles = normalize_angles(sun_az, sun_zen, view_az, view_zen)
    assert angles.shape == (4, 1, 1)
    assert np.isclose(angles[0, 0, 0], 0.5)   # 180/360
    assert np.isclose(angles[1, 0, 0], 0.5)   # 90/180
    assert np.isclose(angles[2, 0, 0], 1.0)   # 360/360
    assert np.isclose(angles[3, 0, 0], 0.0)   # 0/180

    # 7. Test safe_load_angle fallback (missing file -> 0.5)
    fallback_angle = safe_load_angle(None, 360.0, (8, 8))
    assert fallback_angle.shape == (1, 8, 8)
    assert (fallback_angle == 0.5).all()
