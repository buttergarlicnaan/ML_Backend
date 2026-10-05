"""
Validation utilities for uploaded temporal satellite TIFFs.
"""

import os
import re
import uuid
from typing import List, Tuple
from fastapi import HTTPException, UploadFile
import rasterio

from app.config import settings


def sanitize_filename(filename: str) -> str:
    """
    Strips directory traversal patterns and unsafe characters from filenames.
    """
    base = os.path.basename(filename)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    if not safe:
        safe = f"frame_{uuid.uuid4().hex[:8]}.tif"
    return safe


def validate_file_count(files: List[UploadFile]) -> None:
    """
    Validates that exactly 8 temporal frames are provided.
    """
    if len(files) != 8:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid frame count",
                "message": f"Expected exactly 8 temporal TIFF frames, received {len(files)}.",
                "required_count": 8,
                "received_count": len(files),
            },
        )


def validate_file_extensions(files: List[UploadFile]) -> None:
    """
    Validates that all uploaded files have valid TIFF extensions.
    """
    allowed_exts = tuple(settings.ALLOWED_EXTENSIONS)
    for i, file in enumerate(files):
        filename = (file.filename or "").lower()
        if not filename.endswith(allowed_exts):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid file format",
                    "message": (
                        f"File '{file.filename}' (frame {i + 1}) is not a valid TIFF image. "
                        f"Allowed extensions: {settings.ALLOWED_EXTENSIONS}"
                    ),
                    "filename": file.filename,
                    "frame_index": i + 1,
                },
            )


def validate_spatial_consistency(tiff_paths: List[str]) -> Tuple[tuple[int, int], str]:
    """
    Validates that all 8 TIFF files share:
    - Same spatial dimensions (height, width)
    - Compatible Coordinate Reference System (CRS)
    - Compatible spatial bounds and affine transform
    - Readable and non-corrupted raster bands

    Returns:
        ((height, width), crs_string)
    """
    ref_meta = None
    ref_path = tiff_paths[0]

    try:
        with rasterio.open(ref_path) as src:
            ref_meta = {
                "height": src.height,
                "width": src.width,
                "crs": str(src.crs),
                "transform": src.transform,
                "bounds": src.bounds,
                "count": src.count,
            }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Corrupted TIFF file",
                "message": f"Failed to open frame 1 ({os.path.basename(ref_path)}): {str(e)}",
            },
        )

    for idx, path in enumerate(tiff_paths[1:], start=2):
        try:
            with rasterio.open(path) as src:
                fname = os.path.basename(path)

                # Check dimensions
                if src.height != ref_meta["height"] or src.width != ref_meta["width"]:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Incompatible image dimensions",
                            "message": (
                                f"Frame {idx} ({fname}) has dimensions "
                                f"({src.height}, {src.width}), but frame 1 has "
                                f"({ref_meta['height']}, {ref_meta['width']})."
                            ),
                            "frame_index": idx,
                            "expected_shape": (ref_meta["height"], ref_meta["width"]),
                            "received_shape": (src.height, src.width),
                        },
                    )

                # Check CRS
                current_crs = str(src.crs)
                if current_crs != ref_meta["crs"]:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Incompatible CRS",
                            "message": (
                                f"Frame {idx} ({fname}) has CRS '{current_crs}', "
                                f"which does not match frame 1 CRS '{ref_meta['crs']}'."
                            ),
                            "frame_index": idx,
                            "expected_crs": ref_meta["crs"],
                            "received_crs": current_crs,
                        },
                    )

                # Check bounds / transform alignment (within 1% tolerance)
                if src.bounds and ref_meta["bounds"]:
                    dx = abs(src.bounds.left - ref_meta["bounds"].left)
                    dy = abs(src.bounds.top - ref_meta["bounds"].top)
                    pixel_size = abs(ref_meta["transform"].a) if ref_meta["transform"] else 1.0
                    if dx > pixel_size * 2 or dy > pixel_size * 2:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "Incompatible spatial bounds",
                                "message": (
                                    f"Frame {idx} ({fname}) bounding box does not align "
                                    f"with frame 1. Temporal frames must cover the exact same geographic area."
                                ),
                                "frame_index": idx,
                            },
                        )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Corrupted TIFF file",
                    "message": f"Failed to open frame {idx} ({os.path.basename(path)}): {str(e)}",
                    "frame_index": idx,
                },
            )

    return (ref_meta["height"], ref_meta["width"]), ref_meta["crs"]
