"""
MFSR Model Architecture exactly reproducing the implementation in mfsr-model(1).ipynb.

Includes:
- InputDecoupler
- AngleConditioner
- FeatureExtractor
- AlignmentModule (with deformable convolutions)
- MaskedTemporalFusion
- SuperResolutionHead
- MFSR_Network
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class InputDecoupler(nn.Module):
    """Splits the 17-channel input into reflectance, mask, and angles."""

    def forward(self, x: torch.Tensor):
        # x shape: [B, T=8, C=17, H, W]
        reflectance = x[:, :, 0:12, :, :]  # Channels 0-11: 12 reflectance bands
        valid_mask = x[:, :, 12:13, :, :]  # Channel 12: valid/cloud mask
        angles = x[:, :, 13:17, :, :]  # Channels 13-16: 4 solar/view angles
        return reflectance, valid_mask, angles


class AngleConditioner(nn.Module):
    """Embeds the 4 angle channels into spatial feature maps."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.angle_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # Merge B and T dimensions for 2D convolutions
        B, T, C, H, W = angles.shape
        angles = angles.reshape(B * T, C, H, W)
        emb = self.angle_conv(angles)
        return emb.reshape(B, T, -1, H, W)


class FeatureExtractor(nn.Module):
    """Extracts features from reflectance and fuses angle embeddings."""

    def __init__(self, in_channels: int = 12, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, ref: torch.Tensor, angle_emb: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = ref.shape
        ref = ref.reshape(B * T, C, H, W)
        angle_emb = angle_emb.reshape(B * T, -1, H, W)

        feat = self.conv(ref)
        # Concatenate image features with solar/view angle embeddings
        fused = self.fusion(torch.cat([feat, angle_emb], dim=1))
        return fused.reshape(B, T, -1, H, W)


class AlignmentModule(nn.Module):
    """Aligns target frames to a reference frame using Deformable Convolutions."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2 * 3 * 3, kernel_size=3, padding=1),  # 2D offsets for 3x3 kernel
        )
        self.weight = nn.Parameter(torch.empty(channels, channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feats: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = feats.shape

        # Dynamically select the best reference frame (frame with most valid pixels)
        valid_counts = valid_mask.reshape(B, T, -1).sum(dim=2)  # [B, T]
        ref_idx = valid_counts.argmax(dim=1)  # [B]

        aligned_feats = []
        for b in range(B):
            ref_feat = feats[b, ref_idx[b]]  # [C, H, W]
            batch_aligned = []
            for t in range(T):
                target_feat = feats[b, t]  # [C, H, W]

                # Predict offsets based on target and reference differences
                concat_feat = torch.cat([target_feat, ref_feat], dim=0).unsqueeze(0)  # [1, 2C, H, W]
                offsets = self.offset_conv(concat_feat)  # [1, 18, H, W]

                # Apply deformable convolution to align target to reference
                aligned = deform_conv2d(target_feat.unsqueeze(0), offsets, self.weight, padding=1)
                batch_aligned.append(aligned.squeeze(0))
            aligned_feats.append(torch.stack(batch_aligned))  # [T, C, H, W]

        return torch.stack(aligned_feats)  # [B, T, C, H, W]


class MaskedTemporalFusion(nn.Module):
    """Fuses temporal frames using self-attention and zeroes out cloud/invalid pixels."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(channels // 2, 1, kernel_size=1),  # Outputs [B, 1, T, H, W]
        )

    def forward(self, aligned_feats: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = aligned_feats.shape

        # Calculate attention weights across time: [B, C, T, H, W] for Conv3d
        x = aligned_feats.permute(0, 2, 1, 3, 4)
        attn_logits = self.attention(x)  # [B, 1, T, H, W]
        attn_logits = attn_logits.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]

        # Mask out clouds: set logits of invalid pixels to highly negative value
        invalid_mask = (valid_mask == 0)
        attn_logits = attn_logits.masked_fill(invalid_mask, -1e9)

        # Softmax over the temporal dimension
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, T, 1, H, W]

        # Weighted sum across time
        fused = torch.sum(aligned_feats * attn_weights, dim=1)  # [B, C, H, W]
        return fused


class SuperResolutionHead(nn.Module):
    """Upscales features and branches into Reconstruction and Uncertainty predictions."""

    def __init__(self, in_channels: int = 64, out_channels: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Continuous refinement blocks for non-integer scaling (e.g. 6.67x)
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Dual Heads
        self.reconstruction_head = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.uncertainty_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, fused_feat: torch.Tensor, target_size: tuple[int, int]):
        # 1. Bilinear upsample to exact target grid (e.g. 159 * 6.67 = 1060)
        upsampled = F.interpolate(fused_feat, size=target_size, mode="bilinear", align_corners=False)

        # 2. Refine high-frequency details post-upsampling
        hr_feat = self.refinement(upsampled)

        # 3. Predict outputs
        y_pred = self.reconstruction_head(hr_feat)

        # Compute variance (sigma^2). Use Softplus to guarantee positive variance.
        sigma_sq = F.softplus(self.uncertainty_head(hr_feat)) + 1e-6

        return y_pred, sigma_sq


class MFSR_Network(nn.Module):
    """The complete Multi-Frame Super-Resolution (MFSR) Model."""

    def __init__(self, feat_channels: int = 64, out_channels: int = 12):
        super().__init__()
        self.feat_channels = feat_channels
        self.out_channels = out_channels

        self.decoupler = InputDecoupler()
        self.angle_cond = AngleConditioner(out_channels=feat_channels)
        self.extractor = FeatureExtractor(in_channels=12, out_channels=feat_channels)
        self.aligner = AlignmentModule(channels=feat_channels)
        self.fusion = MaskedTemporalFusion(channels=feat_channels)
        self.sr_head = SuperResolutionHead(in_channels=feat_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int] = (1060, 1053)):
        """
        Forward pass.
        Args:
            x: Input tensor of shape [B, 8, 17, H, W]
            target_size: Desired (HR_H, HR_W)
        Returns:
            y_pred: Super-resolved reconstruction [B, out_channels, HR_H, HR_W]
            sigma_sq: Predicted pixel-wise uncertainty [B, 1, HR_H, HR_W]
        """
        # Step 1: Decouple & Embed
        ref, mask, angles = self.decoupler(x)
        angle_emb = self.angle_cond(angles)

        # Step 2: Extract Features
        feats = self.extractor(ref, angle_emb)

        # Step 3: Align temporal features to best reference frame
        aligned_feats = self.aligner(feats, mask)

        # Step 4: Mask-guided Temporal Fusion
        fused_feats = self.fusion(aligned_feats, mask)

        # Step 5: Upscale & predict Dual Outputs
        y_pred, sigma_sq = self.sr_head(fused_feats, target_size=target_size)

        return y_pred, sigma_sq
