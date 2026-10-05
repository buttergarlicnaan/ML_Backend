"""
Inference predictor module for the MFSR Model.
Manages the loaded model instance, runs GPU/CPU inference, computes uncertainty metrics,
and writes out geospatially aligned GeoTIFF files.
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
)

logger = logging.getLogger(__name__)


class MFSRPredictor:
    """
    Singleton-style predictor holder for the MFSR model.
    Loads once on startup and handles inference requests.
    """

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
        """
        Loads the MFSR model from checkpoint. Called once on FastAPI startup.
        """
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
        """
        Determines the target HR spatial dimensions.
        Prefers explicit dimensions if provided; otherwise applies the scale factor.
        """
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
        """
        Runs inference on the preprocessed [1, 8, 17, H, W] tensor and writes GeoTIFF outputs.

        Args:
            input_tensor: [1, 8, 17, H, W] PyTorch tensor
            ref_geospatial: GeospatialMetadata of the reference frame
            request_id: Unique request identifier
            target_height: Optional explicit HR height
            target_width: Optional explicit HR width
            scale_factor: Optional custom upscaling factor

        Returns:
            Dictionary with prediction shapes, uncertainty stats, file paths, and geospatial info.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("MFSR Model is not loaded. Call predictor.load() before predict().")

        req_id = request_id or str(uuid.uuid4())
        req_out_dir = os.path.join(settings.OUTPUT_DIR, req_id)
        os.makedirs(req_out_dir, exist_ok=True)

        _, T, C, lr_h, lr_w = input_tensor.shape

        # Calculate target HR size
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

        # Move tensor to device (using non_blocking transfer if GPU)
        device_tensor = input_tensor.to(
            self.device,
            non_blocking=(self.device.type == "cuda"),
        )

        # Execute model forward pass
        with torch.inference_mode():
            y_pred, uncertainty = self.model(device_tensor, target_size=(hr_h, hr_w))

        # Transfer back to CPU as NumPy arrays
        # y_pred: [1, out_channels, hr_h, hr_w] -> [out_channels, hr_h, hr_w]
        y_pred_np = y_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        # uncertainty: [1, 1, hr_h, hr_w] -> [hr_h, hr_w]
        uncertainty_np = uncertainty.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        out_channels = y_pred_np.shape[0]

        # Uncertainty metrics
        u_min = float(np.min(uncertainty_np))
        u_max = float(np.max(uncertainty_np))
        u_mean = float(np.mean(uncertainty_np))

        # Compute updated Affine transform for the HR grid
        hr_transform = compute_upscaled_transform(
            lr_transform=ref_geospatial.transform,
            lr_width=lr_w,
            lr_height=lr_h,
            hr_width=hr_w,
            hr_height=hr_h,
            bounds=ref_geospatial.bounds,
        )

        # Band descriptions for super-resolved GeoTIFF
        if out_channels == 4:
            descriptions = ["Red", "Green", "Blue", "NIR"]
        elif out_channels == 12:
            descriptions = [f"Band_{i + 1}" for i in range(12)]
        else:
            descriptions = [f"Channel_{i + 1}" for i in range(out_channels)]

        # Save Super-Resolved Multi-band GeoTIFF
        sr_filename = "super_resolved.tif"
        sr_path = os.path.join(req_out_dir, sr_filename)
        save_geotiff(
            output_path=sr_path,
            data=y_pred_np,
            crs=ref_geospatial.crs,
            transform=hr_transform,
            descriptions=descriptions,
        )

        # Save Uncertainty Map Single-band GeoTIFF
        uq_filename = "uncertainty_map.tif"
        uq_path = os.path.join(req_out_dir, uq_filename)
        save_geotiff(
            output_path=uq_path,
            data=uncertainty_np,
            crs=ref_geospatial.crs,
            transform=hr_transform,
            descriptions=["Uncertainty_Variance_SigmaSq"],
        )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"[{req_id}] Inference and GeoTIFF export completed successfully.")

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
                "local_paths": {
                    "super_resolved": sr_path,
                    "uncertainty_map": uq_path,
                },
            },
            "device": str(self.device),
        }


predictor = MFSRPredictor()
