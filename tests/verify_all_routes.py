"""
Comprehensive End-to-End Route Verification Script.
Tests every API route with synthetic multi-frame satellite GeoTIFF inputs.
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from app.main import app

def create_synthetic_frame(
    path: str,
    width: int = 16,
    height: int = 16,
    channels: int = 12,
    crs_epsg: int = 32630,
):
    """
    Creates a valid synthetic multi-band GeoTIFF with geospatial metadata.
    """
    transform = from_origin(500000.0, 4500000.0, 10.0, 10.0)
    crs = CRS.from_epsg(crs_epsg)
    
    # Random realistic reflectance values in [0, 3000] (Sentinel-2 L2A DN scale)
    data = (np.random.rand(channels, height, width) * 3000.0).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


def run_full_verification():
    print("=" * 70)
    print("🚀 STARTING FULL MFSR API END-TO-END ROUTE VERIFICATION")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp(prefix="mfsr_test_inputs_")
    
    try:
        # Step 1: Generate 8 temporal GeoTIFF files
        print("\n[Step 1] Generating 8 temporal synthetic GeoTIFF files...")
        test_files = []
        for i in range(1, 9):
            fpath = os.path.join(temp_dir, f"frame_{i:02d}.tif")
            create_synthetic_frame(fpath, width=16, height=16, channels=12)
            test_files.append(fpath)
            print(f"  ✓ Created {os.path.basename(fpath)} (16x16, 12 channels, EPSG:32630)")

        # Step 2: Use TestClient with lifespan context
        with TestClient(app) as client:
            print("\n[Step 2] Testing System Endpoints:")
            
            # 2.1 GET /
            r_root = client.get("/")
            print(f"  --> GET /: Status {r_root.status_code}")
            assert r_root.status_code == 200, f"Root failed: {r_root.text}"
            root_data = r_root.json()
            print(f"      Response: {json.dumps(root_data, indent=2)}")

            # 2.2 GET /health
            r_health = client.get("/health")
            print(f"\n  --> GET /health: Status {r_health.status_code}")
            assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
            health_data = r_health.json()
            print(f"      Response: {json.dumps(health_data, indent=2)}")
            assert health_data["model_loaded"] is True, "Model is not loaded!"

            # Step 3: Test POST /api/v1/predict
            print("\n[Step 3] Testing POST /api/v1/predict:")
            opened_files = [("files", (os.path.basename(p), open(p, "rb"), "image/tiff")) for p in test_files]
            
            r_predict = client.post(
                "/api/v1/predict",
                files=opened_files,
                data={
                    "target_height": "32",
                    "target_width": "32",
                    "allow_synthetic": "true",
                },
            )
            # Close file descriptors
            for _, (_, f, _) in opened_files:
                f.close()

            print(f"  --> POST /api/v1/predict: Status {r_predict.status_code}")
            assert r_predict.status_code == 200, f"Predict failed: {r_predict.text}"
            pred_data = r_predict.json()
            req_id = pred_data["request_id"]
            print(f"      Request ID: {req_id}")
            print(f"      Output Shape: {pred_data['output']['shape']}")
            print(f"      Uncertainty Stats: min={pred_data['uncertainty']['min']:.6f}, max={pred_data['uncertainty']['max']:.6f}, mean={pred_data['uncertainty']['mean']:.6f}")
            print("      Generated Download URLs:")
            for k, url in pred_data["files"].items():
                if isinstance(url, str):
                    print(f"        • {k}: {url}")

            # Step 4: Test GET /api/v1/download/{request_id}/{filename} for all 5 assets
            print("\n[Step 4] Testing GET /api/v1/download endpoints:")
            expected_assets = [
                ("super_resolved.tif", "image/tiff"),
                ("super_resolved.png", "image/png"),
                ("uncertainty_map.tif", "image/tiff"),
                ("uncertainty_map.png", "image/png"),
                ("mfsr_results.zip", "application/zip"),
            ]

            for fname, expected_type in expected_assets:
                dl_url = f"/api/v1/download/{req_id}/{fname}"
                r_dl = client.get(dl_url)
                print(f"  --> GET {dl_url}: Status {r_dl.status_code}, Media-Type: {r_dl.headers.get('content-type')}")
                assert r_dl.status_code == 200, f"Failed to download {fname}: {r_dl.text}"
                assert expected_type in r_dl.headers.get("content-type", "")

                # Deep integrity checks on the downloaded content
                content_bytes = r_dl.content
                if fname.endswith(".tif"):
                    with rasterio.open(io.BytesIO(content_bytes)) as ds:
                        print(f"      [Verified GeoTIFF] Shape: (bands={ds.count}, H={ds.height}, W={ds.width}), CRS: {ds.crs}")
                elif fname.endswith(".png"):
                    img = Image.open(io.BytesIO(content_bytes))
                    print(f"      [Verified PNG] Size: {img.size}, Mode: {img.mode}, Format: {img.format}")
                elif fname.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
                        zip_names = z.namelist()
                        print(f"      [Verified ZIP] Contents ({len(zip_names)} items): {zip_names}")
                        assert "super_resolved.tif" in zip_names
                        assert "super_resolved.png" in zip_names
                        assert "uncertainty_map.tif" in zip_names
                        assert "uncertainty_map.png" in zip_names
                        assert "metadata.json" in zip_names
                        meta_bytes = z.read("metadata.json")
                        print(f"      [Metadata in ZIP]: {meta_bytes.decode('utf-8')[:150]}...")

            # Step 5: Test POST /api/v1/predict/zip (Direct 1-Click ZIP Response)
            print("\n[Step 5] Testing POST /api/v1/predict/zip (Direct ZIP Stream):")
            opened_files_zip = [("files", (os.path.basename(p), open(p, "rb"), "image/tiff")) for p in test_files]
            r_zip_route = client.post(
                "/api/v1/predict/zip",
                files=opened_files_zip,
                data={
                    "target_height": "32",
                    "target_width": "32",
                    "allow_synthetic": "true",
                },
            )
            for _, (_, f, _) in opened_files_zip:
                f.close()

            print(f"  --> POST /api/v1/predict/zip: Status {r_zip_route.status_code}")
            print(f"      Headers: Content-Type={r_zip_route.headers.get('content-type')}, Content-Disposition={r_zip_route.headers.get('content-disposition')}")
            assert r_zip_route.status_code == 200, f"Predict/zip failed: {r_zip_route.text}"
            assert "application/zip" in r_zip_route.headers.get("content-type", "")
            
            with zipfile.ZipFile(io.BytesIO(r_zip_route.content)) as direct_zip:
                direct_zip_contents = direct_zip.namelist()
                print(f"      [Direct ZIP Verified] Successfully unpacked {len(direct_zip_contents)} files:")
                for item in direct_zip_contents:
                    info = direct_zip.getinfo(item)
                    print(f"        ✓ {item:<22} ({info.file_size:,} bytes)")
                assert "super_resolved.tif" in direct_zip_contents
                assert "super_resolved.png" in direct_zip_contents
                assert "uncertainty_map.tif" in direct_zip_contents
                assert "uncertainty_map.png" in direct_zip_contents
                assert "metadata.json" in direct_zip_contents

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nCleaned up temporary test directory.")

    print("\n" + "=" * 70)
    print("🎯 ALL ROUTES SUCCESSFULLY VERIFIED END-TO-END! (100% PASS)")
    print("=" * 70)


if __name__ == "__main__":
    run_full_verification()
