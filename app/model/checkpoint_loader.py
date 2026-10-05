"""
Checkpoint loading utilities for the MFSR Model.
Handles checkpoint metadata inspection, dynamic head channel adaptation,
and device placement.
"""

import logging
import os
from typing import Any, Optional, Tuple
import torch

from app.model.mfsr_model import MFSR_Network

logger = logging.getLogger(__name__)


def get_device(device_setting: str = "auto") -> torch.device:
    """
    Resolve device string ("auto", "cuda", "cpu") to a torch.device.
    """
    device_setting = (device_setting or "auto").strip().lower()
    if device_setting == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif device_setting == "cpu":
        return torch.device("cpu")
    else:  # "auto"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mfsr_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    feat_channels: int = 64,
    out_channels: Optional[int] = None,
) -> Tuple[MFSR_Network, torch.device, dict[str, Any]]:
    """
    Loads trained MFSR checkpoint and returns (model, device, metadata).

    Args:
        checkpoint_path: Path to the .pth or .pt checkpoint file
        device: Target torch.device (defaults to auto-detected device)
        feat_channels: Number of internal feature channels (default 64)
        out_channels: Expected output channels (4 or 12). If None, auto-detected from checkpoint.

    Returns:
        model: Loaded MFSR_Network in eval mode on target device
        device: The device on which the model is loaded
        metadata: Dict containing epoch, loss values, config, and channel info
    """
    if device is None:
        device = get_device("auto")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at: {checkpoint_path}. "
            f"Please verify the MODEL_CHECKPOINT environment variable or path."
        )

    logger.info(f"Loading checkpoint from: {checkpoint_path} onto {device}")
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Inspect state dict format
    metadata: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "device": str(device),
    }

    if isinstance(raw_checkpoint, dict):
        if "model_state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["model_state_dict"]
        elif "state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["state_dict"]
        else:
            state_dict = raw_checkpoint

        for key in ["epoch", "best_val_charb", "best_val_sam", "config"]:
            if key in raw_checkpoint:
                metadata[key] = raw_checkpoint[key]
    else:
        state_dict = raw_checkpoint

    # Strip potential DataParallel prefix 'module.'
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_k] = v

    # Inspect reconstruction head for output channels and feature channels
    recon_weight_key = "sr_head.reconstruction_head.weight"
    if recon_weight_key in cleaned_state_dict:
        recon_weight = cleaned_state_dict[recon_weight_key]
        ckpt_out_channels = recon_weight.shape[0]
        ckpt_in_channels = recon_weight.shape[1]
        logger.info(
            f"Checkpoint inspection: detected out_channels={ckpt_out_channels}, "
            f"feat_channels={ckpt_in_channels}"
        )
        if out_channels is None:
            out_channels = ckpt_out_channels
        elif out_channels != ckpt_out_channels:
            logger.warning(
                f"Requested out_channels={out_channels} differs from checkpoint "
                f"out_channels={ckpt_out_channels}. Using checkpoint value {ckpt_out_channels}."
            )
            out_channels = ckpt_out_channels

        feat_channels = ckpt_in_channels
    else:
        if out_channels is None:
            out_channels = 12  # Standard default if not detectable
        logger.info(f"Reconstruction head weight not directly matched; using out_channels={out_channels}")

    metadata["out_channels"] = out_channels
    metadata["feat_channels"] = feat_channels

    # Instantiate model
    model = MFSR_Network(feat_channels=feat_channels, out_channels=out_channels)

    # Load weights
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys during state_dict load: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys during state_dict load: {unexpected}")

    model.to(device)
    model.eval()

    logger.info(
        f"MFSR_Network successfully initialized and set to eval mode on {device} "
        f"with feat_channels={feat_channels}, out_channels={out_channels}."
    )
    return model, device, metadata
