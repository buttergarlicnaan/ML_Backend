"""
End-to-end integration tests for FastAPI MFSR backend.
Covers:
- Health check
- 8 valid TIFFs upload and super-resolution
- Rejection of 7 TIFFs
- Rejection of 9 TIFFs
- Rejection of invalid file types
- Rejection of mismatched dimensions
- Rejection of mismatched CRS
- Download endpoint
"""

import io
import os
import pytest
from tests.conftest import create_synthetic_geotiff


def test_health_endpoint(client):
    """1. Test health endpoint returns status 200 and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "device" in data
    assert "cuda_available" in data


def test_predict_success_8_valid_tiffs(client, temp_dir):
    """2. Test uploading exactly 8 valid TIFFs produces super-resolution output."""
    files_to_upload = []
    file_handles = []

    for i in range(8):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16)
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    try:
        response = client.post(
            "/api/v1/predict",
            files=files_to_upload,
            data={"target_height": "32", "target_width": "32"},
        )
        assert response.status_code == 200, f"Error: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert data["input"]["num_frames"] == 8
        assert data["input"]["shape"] == [8, 17, 16, 16]
        assert data["output"]["shape"] == [4, 32, 32]
        assert "uncertainty" in data
        assert data["uncertainty"]["shape"] == [32, 32]
        assert "min" in data["uncertainty"]
        assert "max" in data["uncertainty"]
        assert "mean" in data["uncertainty"]
        assert "files" in data
        assert "super_resolved" in data["files"]
        assert "uncertainty_map" in data["files"]

        # Test downloading generated GeoTIFFs
        sr_url = data["files"]["super_resolved"]
        dl_resp = client.get(sr_url)
        assert dl_resp.status_code == 200
        assert dl_resp.headers["content-type"] == "image/tiff"
        assert len(dl_resp.content) > 0

        uq_url = data["files"]["uncertainty_map"]
        uq_resp = client.get(uq_url)
        assert uq_resp.status_code == 200
        assert len(uq_resp.content) > 0
    finally:
        for fh in file_handles:
            fh.close()


def test_predict_reject_7_tiffs(client, temp_dir):
    """3. Test rejecting 7 TIFFs (HTTP 400)."""
    files_to_upload = []
    file_handles = []

    for i in range(7):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16)
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    try:
        response = client.post("/api/v1/predict", files=files_to_upload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "8" in str(data)
    finally:
        for fh in file_handles:
            fh.close()


def test_predict_reject_9_tiffs(client, temp_dir):
    """4. Test rejecting 9 TIFFs (HTTP 400)."""
    files_to_upload = []
    file_handles = []

    for i in range(9):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16)
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    try:
        response = client.post("/api/v1/predict", files=files_to_upload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
    finally:
        for fh in file_handles:
            fh.close()


def test_predict_reject_invalid_file_type(client, temp_dir):
    """5. Test rejecting non-TIFF files (HTTP 400)."""
    files_to_upload = []
    file_handles = []

    for i in range(7):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16)
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    # 8th file is a text file
    txt_path = os.path.join(temp_dir, "frame_8.txt")
    with open(txt_path, "w") as f:
        f.write("not an image")
    fh_txt = open(txt_path, "rb")
    file_handles.append(fh_txt)
    files_to_upload.append(("files", ("frame_8.txt", fh_txt, "text/plain")))

    try:
        response = client.post("/api/v1/predict", files=files_to_upload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "not a valid TIFF" in str(data)
    finally:
        for fh in file_handles:
            fh.close()


def test_predict_reject_different_dimensions(client, temp_dir):
    """6. Test rejecting frames with mismatched spatial dimensions (HTTP 400)."""
    files_to_upload = []
    file_handles = []

    for i in range(7):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16)
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    # 8th frame has different dimensions (24x24 instead of 16x16)
    path_mismatch = os.path.join(temp_dir, "frame_8_mismatch.tif")
    create_synthetic_geotiff(path_mismatch, num_channels=17, height=24, width=24)
    fh_mismatch = open(path_mismatch, "rb")
    file_handles.append(fh_mismatch)
    files_to_upload.append(("files", ("frame_8.tif", fh_mismatch, "image/tiff")))

    try:
        response = client.post("/api/v1/predict", files=files_to_upload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "dimensions" in str(data).lower()
    finally:
        for fh in file_handles:
            fh.close()


def test_predict_reject_different_crs(client, temp_dir):
    """7. Test rejecting frames with different CRS (HTTP 400)."""
    files_to_upload = []
    file_handles = []

    for i in range(7):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16, crs="EPSG:32632")
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    # 8th frame has different CRS (EPSG:4326)
    path_diff_crs = os.path.join(temp_dir, "frame_8_diff_crs.tif")
    create_synthetic_geotiff(path_diff_crs, num_channels=17, height=16, width=16, crs="EPSG:4326")
    fh_diff_crs = open(path_diff_crs, "rb")
    file_handles.append(fh_diff_crs)
    files_to_upload.append(("files", ("frame_8.tif", fh_diff_crs, "image/tiff")))

    try:
        response = client.post("/api/v1/predict", files=files_to_upload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "crs" in str(data).lower()
    finally:
        for fh in file_handles:
            fh.close()


def test_predict_all_download_assets(client, temp_dir):
    """8. Test that predict endpoint generates super_resolved.png, uncertainty_map.png, and mfsr_results.zip."""
    import zipfile
    files_to_upload = []
    file_handles = []

    for i in range(8):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16)
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    try:
        response = client.post(
            "/api/v1/predict",
            files=files_to_upload,
            data={"target_height": "32", "target_width": "32"},
        )
        assert response.status_code == 200
        data = response.json()
        files_dict = data["files"]

        # Check PNG downloads
        sr_png_url = files_dict["super_resolved_png"]
        sr_png_resp = client.get(sr_png_url)
        assert sr_png_resp.status_code == 200
        assert "image/png" in sr_png_resp.headers["content-type"]

        uq_png_url = files_dict["uncertainty_map_png"]
        uq_png_resp = client.get(uq_png_url)
        assert uq_png_resp.status_code == 200
        assert "image/png" in uq_png_resp.headers["content-type"]

        # Check ZIP download
        zip_url = files_dict["zip_archive"]
        zip_resp = client.get(zip_url)
        assert zip_resp.status_code == 200
        assert "application/zip" in zip_resp.headers["content-type"]
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
            names = z.namelist()
            assert "super_resolved.tif" in names
            assert "super_resolved.png" in names
            assert "uncertainty_map.tif" in names
            assert "uncertainty_map.png" in names
            assert "metadata.json" in names
    finally:
        for fh in file_handles:
            fh.close()


def test_predict_zip_stream_endpoint(client, temp_dir):
    """9. Test POST /api/v1/predict/zip direct ZIP stream endpoint."""
    import zipfile
    files_to_upload = []
    file_handles = []

    for i in range(8):
        path = os.path.join(temp_dir, f"frame_{i + 1}.tif")
        create_synthetic_geotiff(path, num_channels=17, height=16, width=16)
        fh = open(path, "rb")
        file_handles.append(fh)
        files_to_upload.append(("files", (f"frame_{i + 1}.tif", fh, "image/tiff")))

    try:
        response = client.post(
            "/api/v1/predict/zip",
            files=files_to_upload,
            data={"target_height": "32", "target_width": "32"},
        )
        assert response.status_code == 200
        assert "application/zip" in response.headers["content-type"]
        assert "attachment" in response.headers.get("content-disposition", "")

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            names = z.namelist()
            assert "super_resolved.tif" in names
            assert "super_resolved.png" in names
            assert "uncertainty_map.tif" in names
            assert "uncertainty_map.png" in names
            assert "metadata.json" in names
    finally:
        for fh in file_handles:
            fh.close()
