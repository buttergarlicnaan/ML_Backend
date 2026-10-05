# Multi-Frame Super-Resolution (MFSR) Satellite Imagery Backend

A production-ready FastAPI backend for Multi-Frame Super-Resolution (MFSR) satellite image processing. The backend receives 8 temporal GeoTIFF observations of the same geographic region, constructs an exact 17-channel temporal tensor representation `[1, 8, 17, H, W]`, passes it through the trained MFSR deep neural network with deformable alignment and attention-guided temporal fusion, and exports geospatially aligned multi-band super-resolved GeoTIFFs alongside uncertainty maps.

---

## 1. Project Overview

Satellite constellations like Sentinel-2 capture multi-spectral imagery at 10m–60m spatial resolution at frequent revisit intervals. High-resolution sensors (such as PlanetScope or SPOT) capture 1.5m–3.0m resolution.

The **Multi-Frame Super-Resolution (MFSR)** network fuses information across **8 temporal observations** of the same geographic area to reconstruct high-resolution features that cannot be resolved from a single image alone.

This backend provides:
- **FastAPI REST API**: High-throughput asynchronous endpoints with validation and error handling.
- **Geospatial Integrity**: Strict CRS, bounding box, and affine transform preservation using `rasterio`.
- **Dual Outputs**:
  - Super-resolved multi-band GeoTIFF (`super_resolved.tif`)
  - Pixel-wise uncertainty variance map GeoTIFF (`uncertainty_map.tif`)
- **Single-Model Lifetime**: Trained model checkpoint is loaded only once on startup into GPU/CPU memory.
- **Robust Security**: File validation, sanitized paths, isolated request folders, and automatic cleanup of temporary files.

---

## 2. Architecture & Data Flow

```
8 Temporal GeoTIFFs (Uploaded via /api/v1/predict)
              │
              ▼
   Spatial & File Validation (Validation Module)
    - 8 frames check
    - TIFF format validation
    - CRS & bounding box alignment
    - Dimensions compatibility
              │
              ▼
    TIFF Preprocessing & Normalization
    - NaN / Inf sanitization
    - Sentinel-2 16-bit reflectance normalization (/ 10000.0)
    - Construction of 17 channels per frame
              │
              ▼
  PyTorch Tensor Construction: [B=1, T=8, C=17, H, W]
              │
              ▼
      MFSR_Network (Trained Neural Network)
    ┌───────────────────────────────────────────┐
    │ 1. InputDecoupler (Reflectance, Mask, Ang)│
    │ 2. AngleConditioner (Solar & Sensor emb)  │
    │ 3. FeatureExtractor (Fused with angles)   │
    │ 4. AlignmentModule (Deformable Convolut.) │
    │ 5. MaskedTemporalFusion (Attention 3D)    │
    │ 6. SuperResolutionHead (Continuous Upsamp)│
    └───────────────────────────────────────────┘
              │
              ├──► Super-Resolved Output: [1, out_channels, HR_H, HR_W]
              └──► Uncertainty Map (Sigma^2): [1, 1, HR_H, HR_W]
              │
              ▼
     Geospatial Post-Processing (Rasterio)
    - Scaled affine transform calculation (from_bounds)
    - Export `super_resolved.tif`
    - Export `uncertainty_map.tif`
              │
              ▼
    API Response & Download Links (/api/v1/download/...)
```

---

## 3. Exact 17-Channel Input Representation

Each of the 8 temporal observations must provide 17 channels mapped as follows:

| Channel Index | Channel Name | Description | Value Range |
| :--- | :--- | :--- | :--- |
| **0** | `B01` | Coastal Aerosol (443 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **1** | `B02` | Blue (490 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **2** | `B03` | Green (560 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **3** | `B04` | Red (665 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **4** | `B05` | Vegetation Red Edge 1 (705 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **5** | `B06` | Vegetation Red Edge 2 (740 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **6** | `B07` | Vegetation Red Edge 3 (783 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **7** | `B08` | NIR Broadband (842 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **8** | `B8A` | NIR Narrow (865 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **9** | `B09` | Water Vapour (940 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **10** | `B11` | SWIR 1 (1610 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **11** | `B12` | SWIR 2 (2190 nm) | `[0.0, 1.0]` reflectance (DN / 10000) |
| **12** | `CLM` | Valid / Cloud Mask | `1.0` = valid/clear, `0.0` = cloud/invalid |
| **13** | `Sun_Zenith` | Solar Zenith Angle | Continuous degrees/radians |
| **14** | `Sun_Azimuth` | Solar Azimuth Angle | Continuous degrees/radians |
| **15** | `View_Zenith` | Sensor Viewing Zenith Angle | Continuous degrees/radians |
| **16** | `View_Azimuth` | Sensor Viewing Azimuth Angle | Continuous degrees/radians |

> **Important**: If uploaded TIFFs contain fewer than 17 channels, the API returns a clear HTTP 400 error detailing the missing channels. For testing environments without angle bands, set `ALLOW_SYNTHETIC_ANGLES=true` in `.env` to allow the preprocessor to generate neutral angles.

---

## 4. Installation & Setup

### Option A: Conda Environment (Recommended)

```bash
# 1. Create Conda environment with Python 3.12
conda create -n mfsr_env python=3.12 -y
conda activate mfsr_env

# 2. Install PyTorch with CUDA support (or CPU-only if no GPU)
# For CUDA 12.1+:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Option B: Python Virtual Environment (`venv`)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Model Checkpoint Setup

The checkpoints extracted from `results.zip` are located in `mfsr_checkpoints/`:
- `best_val_charbonnier.pth` (recommended, best Charbonnier metric)
- `best_val_sam.pth` (best Spectral Angle Mapper metric)
- `mfsr_epoch_10.pth` ... `mfsr_epoch_50.pth`

Set the desired checkpoint in your `.env` file or environment:

```bash
cp .env.example .env
```

Configuration variables in `.env`:
```ini
MODEL_CHECKPOINT=mfsr_checkpoints/best_val_charbonnier.pth
DEVICE=auto
FEAT_CHANNELS=64
DEFAULT_SCALE_FACTOR=6.66666667
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
MAX_UPLOAD_SIZE_MB=250
NORMALIZE_DIVISOR=10000.0
ALLOW_SYNTHETIC_ANGLES=false
```

---

## 6. How to Start the FastAPI Server

```bash
# Production / Local development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 7. API Reference

### Health Check

**`GET /health`**

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "cuda_available": false,
  "model_metadata": {
    "epoch": 42,
    "best_val_charb": 0.02496,
    "best_val_sam": 0.14107,
    "out_channels": 4,
    "feat_channels": 64
  }
}
```

---

### Super-Resolution Inference

**`POST /api/v1/predict`**

Accepts `multipart/form-data`:
- `files`: Exactly 8 `.tif` / `.tiff` files.
- `target_height` *(optional)*: Explicit HR height (integer).
- `target_width` *(optional)*: Explicit HR width (integer).
- `scale_factor` *(optional)*: Upscaling multiplier (default: `6.6667`).
- `allow_synthetic` *(optional)*: Boolean flag for testing mode.

Example Response (`200 OK`):
```json
{
  "request_id": "7f8b9e6a-1234-4567-89ab-cdef01234567",
  "success": true,
  "message": "Super-resolution completed successfully",
  "input": {
    "num_frames": 8,
    "shape": [8, 17, 159, 158]
  },
  "output": {
    "shape": [4, 1060, 1053],
    "num_bands": 4
  },
  "uncertainty": {
    "shape": [1060, 1053],
    "min": 0.000105,
    "max": 0.042180,
    "mean": 0.012340
  },
  "files": {
    "super_resolved": "/api/v1/download/7f8b9e6a-1234-4567-89ab-cdef01234567/super_resolved.tif",
    "uncertainty_map": "/api/v1/download/7f8b9e6a-1234-4567-89ab-cdef01234567/uncertainty_map.tif",
    "local_paths": {
      "super_resolved": "/path/to/outputs/7f8b9e6a-1234-4567-89ab-cdef01234567/super_resolved.tif",
      "uncertainty_map": "/path/to/outputs/7f8b9e6a-1234-4567-89ab-cdef01234567/uncertainty_map.tif"
    }
  },
  "device": "cpu"
}
```

---

### Download Output GeoTIFF

**`GET /api/v1/download/{request_id}/{filename}`**

- `filename`: Either `super_resolved.tif` or `uncertainty_map.tif`.
- Returns binary GeoTIFF file stream with `Content-Type: image/tiff`.

---

## 8. Example `curl` Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "files=@frame_01.tif" \
  -F "files=@frame_02.tif" \
  -F "files=@frame_03.tif" \
  -F "files=@frame_04.tif" \
  -F "files=@frame_05.tif" \
  -F "files=@frame_06.tif" \
  -F "files=@frame_07.tif" \
  -F "files=@frame_08.tif" \
  -F "scale_factor=6.66666667"
```

To download the result:
```bash
curl -O "http://localhost:8000/api/v1/download/<REQUEST_ID>/super_resolved.tif"
curl -O "http://localhost:8000/api/v1/download/<REQUEST_ID>/uncertainty_map.tif"
```

---

## 9. GPU Setup & Performance

- **Automatic Device Resolution**: `DEVICE=auto` detects if CUDA is available via `torch.cuda.is_available()`.
- **Non-blocking transfers**: `tensor.to(device, non_blocking=True)` optimizes host-to-device transfers.
- **Inference Mode**: Inference is run under `torch.inference_mode()`, preventing gradient overhead and conserving memory.
- **Cache Management**: `torch.cuda.empty_cache()` is called conditionally upon request completion.

---

## 10. Running the Test Suite

Execute the unit and integration tests using pytest:

```bash
pytest -v tests/
```

Test coverage includes:
- Health check endpoint
- 8-frame super-resolution pipeline
- Input validation (rejection of 7 frames, 9 frames, non-TIFF files)
- Spatial consistency checks (dimensions and CRS matching)
- Affine transform scaling and bounding box preservation
- Checkpoint loading and channel adaptation
- NaN / Inf sanitization
