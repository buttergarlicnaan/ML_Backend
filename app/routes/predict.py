"""
FastAPI route handlers for MFSR inference and output file downloads.
"""

import logging
import os
import re
import shutil
import uuid
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.inference.predictor import predictor
from app.preprocessing.tif_preprocessor import preprocess_temporal_tiffs
from app.utils.validation import (
    sanitize_filename,
    validate_file_count,
    validate_file_extensions,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Multi-Frame Super-Resolution"])

SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")
ALLOWED_OUTPUT_FILENAMES = {
    "super_resolved.tif",
    "super_resolved.png",
    "uncertainty_map.tif",
    "uncertainty_map.png",
    "mfsr_results.zip",
}


@router.post("/predict")
async def predict_super_resolution(
    files: List[UploadFile] = File(
        ...,
        description="Exactly 8 temporal GeoTIFF files representing observations of the same region.",
    ),
    target_height: Optional[int] = Form(
        None,
        description="Optional custom HR height (e.g. 1060). If omitted, derived via scale_factor.",
    ),
    target_width: Optional[int] = Form(
        None,
        description="Optional custom HR width (e.g. 1053). If omitted, derived via scale_factor.",
    ),
    scale_factor: Optional[float] = Form(
        None,
        description="Optional custom super-resolution upscaling factor (default: 6.6667).",
    ),
    allow_synthetic: Optional[bool] = Form(
        False,
        description="If True, allows synthetic auxiliary angles/mask for testing with 12/13/16 channel data.",
    ),
):
    validate_file_count(files)
    validate_file_extensions(files)

    request_id = str(uuid.uuid4())
    upload_temp_dir = os.path.join(settings.UPLOAD_DIR, request_id)
    os.makedirs(upload_temp_dir, exist_ok=True)

    saved_paths = []
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    try:
        for idx, file_obj in enumerate(files, start=1):
            safe_name = sanitize_filename(file_obj.filename or f"frame_{idx}.tif")
            dest_path = os.path.join(upload_temp_dir, f"{idx:02d}_{safe_name}")

            total_bytes = 0
            with open(dest_path, "wb") as f_out:
                while chunk := await file_obj.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "error": "File size exceeds limit",
                                "message": f"File '{file_obj.filename}' exceeds limit of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                            },
                        )
                    f_out.write(chunk)
            saved_paths.append(dest_path)

        logger.info(f"[{request_id}] Successfully saved {len(saved_paths)} uploaded temporal TIFFs.")

        input_tensor, ref_geospatial, debug_info = preprocess_temporal_tiffs(
            tiff_paths=saved_paths,
            allow_synthetic=allow_synthetic or settings.ALLOW_SYNTHETIC_ANGLES,
        )

        result = predictor.predict(
            input_tensor=input_tensor,
            ref_geospatial=ref_geospatial,
            request_id=request_id,
            target_height=target_height,
            target_width=target_width,
            scale_factor=scale_factor,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during MFSR pipeline processing: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "message": f"An error occurred during super-resolution processing: {str(e)}",
            },
        )
    finally:
        try:
            shutil.rmtree(upload_temp_dir, ignore_errors=True)
            logger.info(f"[{request_id}] Upload temporary directory cleaned up.")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to clean upload directory: {e}")


@router.post("/predict/zip")
async def predict_super_resolution_zip(
    files: List[UploadFile] = File(
        ...,
        description="Exactly 8 temporal GeoTIFF files representing observations of the same region.",
    ),
    target_height: Optional[int] = Form(
        None,
        description="Optional custom HR height (e.g. 1060). If omitted, derived via scale_factor.",
    ),
    target_width: Optional[int] = Form(
        None,
        description="Optional custom HR width (e.g. 1053). If omitted, derived via scale_factor.",
    ),
    scale_factor: Optional[float] = Form(
        None,
        description="Optional custom super-resolution upscaling factor (default: 6.6667).",
    ),
    allow_synthetic: Optional[bool] = Form(
        False,
        description="If True, allows synthetic auxiliary angles/mask for testing with 12/13/16 channel data.",
    ),
):
    validate_file_count(files)
    validate_file_extensions(files)

    request_id = str(uuid.uuid4())
    upload_temp_dir = os.path.join(settings.UPLOAD_DIR, request_id)
    os.makedirs(upload_temp_dir, exist_ok=True)

    saved_paths = []
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    try:
        for idx, file_obj in enumerate(files, start=1):
            safe_name = sanitize_filename(file_obj.filename or f"frame_{idx}.tif")
            dest_path = os.path.join(upload_temp_dir, f"{idx:02d}_{safe_name}")

            total_bytes = 0
            with open(dest_path, "wb") as f_out:
                while chunk := await file_obj.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "error": "File size exceeds limit",
                                "message": f"File '{file_obj.filename}' exceeds limit of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                            },
                        )
                    f_out.write(chunk)
            saved_paths.append(dest_path)

        logger.info(f"[{request_id}] (ZIP route) Successfully saved {len(saved_paths)} uploaded temporal TIFFs.")

        input_tensor, ref_geospatial, debug_info = preprocess_temporal_tiffs(
            tiff_paths=saved_paths,
            allow_synthetic=allow_synthetic or settings.ALLOW_SYNTHETIC_ANGLES,
        )

        result = predictor.predict(
            input_tensor=input_tensor,
            ref_geospatial=ref_geospatial,
            request_id=request_id,
            target_height=target_height,
            target_width=target_width,
            scale_factor=scale_factor,
        )

        zip_local_path = result["files"]["local_paths"]["zip_archive"]
        return FileResponse(
            path=zip_local_path,
            media_type="application/zip",
            filename=f"mfsr_{request_id}_results.zip",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during MFSR ZIP processing: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "message": f"An error occurred during super-resolution processing: {str(e)}",
            },
        )
    finally:
        try:
            shutil.rmtree(upload_temp_dir, ignore_errors=True)
            logger.info(f"[{request_id}] Upload temporary directory cleaned up.")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to clean upload directory: {e}")


@router.get("/download/{request_id}/{filename}")
async def download_output_file(request_id: str, filename: str):
    if not SAFE_REQUEST_ID_REGEX.match(request_id):
        raise HTTPException(status_code=400, detail="Invalid request ID format.")

    if filename not in ALLOWED_OUTPUT_FILENAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file request. Allowed files: {list(ALLOWED_OUTPUT_FILENAMES)}",
        )

    file_path = os.path.abspath(os.path.join(settings.OUTPUT_DIR, request_id, filename))

    expected_prefix = os.path.abspath(settings.OUTPUT_DIR)
    if not file_path.startswith(expected_prefix):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Requested file '{filename}' for request '{request_id}' not found.",
        )

    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".zip"):
        media_type = "application/zip"
    else:
        media_type = "image/tiff"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )
