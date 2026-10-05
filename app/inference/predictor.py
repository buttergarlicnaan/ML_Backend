"""
Inference predictor module for the MFSR Model.
"""

import logging
import os
import uuid
from typing import Any, Optional, Tuple
import numpy as np
import torch

from app.config import settings
from app.model.checkpoint_loader import load_mfsr_checkpoint, get_device
from app.model.mfsr_model import MFSR_Network
from app.utils.geospatial import (
    GeospatialMetadata,
    compute_upscaled_transform,
    save_geotiff,
    save_rgb_png,
    save_uncertainty_png,
    create_results_zip,
)

logger = logging.getLogger(__name__)


class MFSRPredictor:
    def __init__(self):
        self.model: Optional[MFSR_Network] = None
        self.device: Optional[torch.device] = None
        self.metadata: dict[str, Any] = {}
        self.is_loaded: bool = False

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        device_name: Optional[str] = None,
        feat_channels: int = 64,
        out_channels: Optional[int] = None,
    ) -> None:
        ckpt_path = checkpoint_path or settings.MODEL_CHECKPOINT
        dev_str = device_name or settings.DEVICE
        self.device = get_device(dev_str)

        logger.info(f"Initializing MFSR Model on device: {self.device}")
        self.model, self.device, self.metadata = load_mfsr_checkpoint(
            checkpoint_path=ckpt_path,
            device=self.device,
            feat_channels=feat_channels,
            out_channels=out_channels,
        )
        self.is_loaded = True

    def calculate_target_size(
        self,
        lr_height: int,
        lr_width: int,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        scale_factor: Optional[float] = None,
    ) -> Tuple[int, int]:
        if target_height is not None and target_width is not None:
            return int(target_height), int(target_width)

        factor = scale_factor if scale_factor is not None else settings.DEFAULT_SCALE_FACTOR
        hr_h = int(round(lr_height * factor))
        hr_w = int(round(lr_width * factor))
        return hr_h, hr_w

    def predict(
        self,
        input_tensor: torch.Tensor,
        ref_geospatial: GeospatialMetadata,
        request_id: Optional[str] = None,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        scale_factor: Optional[float] = None,
    ) -> dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("MFSR Model is not loaded. Call predictor.load() before predict().")

        req_id = request_id or str(uuid.uuid4())
        req_out_dir = os.path.join(settings.OUTPUT_DIR, req_id)
        os.makedirs(req_out_dir, exist_ok=True)

        _, T, C, lr_h, lr_w = input_tensor.shape

        hr_h, hr_w = self.calculate_target_size(
            lr_height=lr_h,
            lr_width=lr_w,
            target_height=target_height,
            target_width=target_width,
            scale_factor=scale_factor,
        )

        logger.info(
            f"[{req_id}] Running MFSR inference: LR ({lr_h}, {lr_w}) -> HR ({hr_h}, {hr_w}) "
            f"on {self.device}"
        )

        device_tensor = input_tensor.to(
            self.device,
            non_blocking=(self.device.type == "cuda"),
        )

        with torch.inference_mode():
            y_pred, uncertainty = self.model(device_tensor, target_size=(hr_h, hr_w))

        y_pred_np = y_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        uncertainty_np = uncertainty.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        out_channels = y_pred_np.shape[0]

        u_min = float(np.min(uncertainty_np))
        u_max = float(np.max(uncertainty_np))
        u_mean = float(np.mean(uncertainty_np))

        hr_transform = compute_upscaled_transform(
            lr_transform=ref_geospatial.transform,
            lr_width=lr_w,
            lr_height=lr_h,
            hr_width=hr_w,
            hr_height=hr_h,
            bounds=ref_geospatial.bounds,
        )

        if out_channels == 4:
            descriptions = ["Red", "Green", "Blue", "NIR"]
        elif out_channels == 12:
            descriptions = [f"Band_{i + 1}" for i in range(12)]
        else:
            descriptions = [f"Channel_{i + 1}" for i in range(out_channels)]

        sr_filename = "super_resolved.tif"
        sr_path = os.path.join(req_out_dir, sr_filename)
        save_geotiff(
            output_path=sr_path,
            data=y_pred_np,
            crs=ref_geospatial.crs,
            transform=hr_transform,
            descriptions=descriptions,
        )

        rgb_indices = (3, 2, 1) if out_channels >= 12 else (0, 1, 2)
        sr_png_filename = "super_resolved.png"
        sr_png_path = os.path.join(req_out_dir, sr_png_filename)
        save_rgb_png(
            output_path=sr_png_path,
            data=y_pred_np,
            rgb_indices=rgb_indices,
        )

        uq_filename = "uncertainty_map.tif"
        uq_path = os.path.join(req_out_dir, uq_filename)
        save_geotiff(
            output_path=uq_path,
            data=uncertainty_np,
            crs=ref_geospatial.crs,
            transform=hr_transform,
            descriptions=["Uncertainty_Variance_SigmaSq"],
        )

        uq_png_filename = "uncertainty_map.png"
        uq_png_path = os.path.join(req_out_dir, uq_png_filename)
        save_uncertainty_png(
            output_path=uq_png_path,
            data=uncertainty_np,
            cmap_name="inferno",
        )

        zip_filename = "mfsr_results.zip"
        zip_path = os.path.join(req_out_dir, zip_filename)
        metadata_dict = {
            "request_id": req_id,
            "scale_factor": scale_factor or settings.DEFAULT_SCALE_FACTOR,
            "input_shape": [T, C, lr_h, lr_w],
            "output_shape": [out_channels, hr_h, hr_w],
            "band_descriptions": descriptions,
            "uncertainty_stats": {
                "min": u_min,
                "max": u_max,
                "mean": u_mean,
            },
            "crs": str(ref_geospatial.crs) if ref_geospatial.crs else None,
        }
        create_results_zip(
            output_zip_path=zip_path,
            file_map={
                "super_resolved.tif": sr_path,
                "super_resolved.png": sr_png_path,
                "uncertainty_map.tif": uq_path,
                "uncertainty_map.png": uq_png_path,
            },
            metadata=metadata_dict,
        )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"[{req_id}] Inference, GeoTIFF, PNG, and ZIP export completed successfully.")

        return {
            "request_id": req_id,
            "success": True,
            "message": "Super-resolution completed successfully",
            "input": {
                "num_frames": T,
                "shape": [T, C, lr_h, lr_w],
            },
            "output": {
                "shape": [out_channels, hr_h, hr_w],
                "num_bands": out_channels,
            },
            "uncertainty": {
                "shape": [hr_h, hr_w],
                "min": u_min,
                "max": u_max,
                "mean": u_mean,
            },
            "files": {
                "super_resolved": f"/api/v1/download/{req_id}/{sr_filename}",
                "uncertainty_map": f"/api/v1/download/{req_id}/{uq_filename}",
                "super_resolved_tif": f"/api/v1/download/{req_id}/{sr_filename}",
                "super_resolved_png": f"/api/v1/download/{req_id}/{sr_png_filename}",
                "uncertainty_map_tif": f"/api/v1/download/{req_id}/{uq_filename}",
                "uncertainty_map_png": f"/api/v1/download/{req_id}/{uq_png_filename}",
                "zip_archive": f"/api/v1/download/{req_id}/{zip_filename}",
                "local_paths": {
                    "super_resolved": sr_path,
                    "uncertainty_map": uq_path,
                    "super_resolved_tif": sr_path,
                    "super_resolved_png": sr_png_path,
                    "uncertainty_map_tif": uq_path,
                    "uncertainty_map_png": uq_png_path,
                    "zip_archive": zip_path,
                },
            },
            "device": str(self.device),
        }


predictor = MFSRPredictor()
