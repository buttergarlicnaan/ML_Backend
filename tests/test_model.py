"""
Unit tests for MFSR Model architecture, forward pass, and checkpoint loader.
"""

import torch
import pytest
from app.model.mfsr_model import MFSR_Network
from app.model.checkpoint_loader import load_mfsr_checkpoint
from app.config import settings


def test_mfsr_architecture_12_channels():
    """Verify standard 12-channel model forward pass and output shapes."""
    model = MFSR_Network(feat_channels=64, out_channels=12)
    model.eval()

    # Simulate 8 temporal observations of size 16x16 with 17 channels
    dummy_input = torch.randn(1, 8, 17, 16, 16)
    target_size = (32, 32)

    with torch.inference_mode():
        y_pred, uncertainty = model(dummy_input, target_size=target_size)

    assert y_pred.shape == torch.Size([1, 12, 32, 32]), f"Expected [1, 12, 32, 32], got {y_pred.shape}"
    assert uncertainty.shape == torch.Size([1, 1, 32, 32]), f"Expected [1, 1, 32, 32], got {uncertainty.shape}"
    assert torch.all(uncertainty > 0), "Uncertainty variance must be strictly positive"


def test_mfsr_architecture_4_channels():
    """Verify 4-channel model (matching PlanetScope HR training) forward pass."""
    model = MFSR_Network(feat_channels=64, out_channels=4)
    model.eval()

    dummy_input = torch.randn(1, 8, 17, 20, 20)
    target_size = (40, 40)

    with torch.inference_mode():
        y_pred, uncertainty = model(dummy_input, target_size=target_size)

    assert y_pred.shape == torch.Size([1, 4, 40, 40]), f"Expected [1, 4, 40, 40], got {y_pred.shape}"
    assert uncertainty.shape == torch.Size([1, 1, 40, 40]), f"Expected [1, 1, 40, 40], got {uncertainty.shape}"


def test_checkpoint_loader():
    """Verify that load_mfsr_checkpoint successfully loads the trained checkpoint."""
    model, device, meta = load_mfsr_checkpoint(
        checkpoint_path=settings.MODEL_CHECKPOINT,
        device=torch.device("cpu"),
    )

    assert model is not None
    assert str(device) == "cpu"
    assert "epoch" in meta
    assert meta["out_channels"] == 4
    assert meta["feat_channels"] == 64
    assert meta["best_val_charb"] is not None
