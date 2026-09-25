import os,sys,uuid,shutil,tempfile
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

WORLDSTRAT_PATH="./worldstrat"
MODEL_PATH="./worldstrat/pretrained_model/model.ckpt"
OUTPUT_DIR="./outputs"

os.makedirs(OUTPUT_DIR,exist_ok=True)
sys.path.insert(0,WORLDSTRAT_PATH)

from worldstrat.src.lightning_modules import LitModel

app=FastAPI(
    title="AI-SRM Backend",
    description="Satellite Image Super-Resolution API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading AI-SRM model...")
print("Using device:",DEVICE)

model=LitModel.load_from_checkpoint(
    MODEL_PATH,
    weights_only=False
)
model=model.to(DEVICE)
model.eval()

print("AI-SRM model loaded successfully.")


def global_percentile_stretch(img,is_prediction=False):
    img=np.nan_to_num(img,nan=0.0)

    if is_prediction:
        valid_pixels=img
    else:
        valid_pixels=img[img>0]

    if valid_pixels.size==0:
        return np.clip(img,0,1)

    p2,p98=np.percentile(valid_pixels,(2,98))

    if p98-p2>1e-5:
        img=(img-p2)/(p98-p2)

    if not is_prediction:
        img[img==0]=0

    return np.clip(img,0,1)


def pred(list_of_8_images):
    if len(list_of_8_images)!=8:
        raise ValueError(
            f"Expected exactly 8 temporal images,received {len(list_of_8_images)}"
        )

    input_sequence=np.stack(list_of_8_images)
    input_sequence=np.nan_to_num(
        input_sequence,
        nan=0.0
    ).astype(np.float32)

    input_tensor=torch.tensor(
        input_sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction_tensor=model(input_tensor)

    predicted_array=(
        prediction_tensor
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    return np.nan_to_num(
        predicted_array,
        nan=0.0
    )


def generate_map(
    predicted_array,
    reference_tiff,
    output_dir,
    request_id
):
    predicted_array=np.nan_to_num(
        predicted_array,
        nan=0.0
    )

    if predicted_array.ndim==2:
        rgb=np.stack(
            [predicted_array]*3,
            axis=-1
        )

    elif predicted_array.ndim==3:

        if predicted_array.shape[0]<=10:

            if predicted_array.shape[0]>=3:
                rgb=predicted_array[
                    [2,1,0]
                ].transpose(1,2,0)
            else:
                rgb=np.stack(
                    [predicted_array[0]]*3,
                    axis=-1
                )
        else:
            rgb=predicted_array

    else:
        raise ValueError(
            f"Invalid prediction shape: {predicted_array.shape}"
        )

    rgb=global_percentile_stretch(
        rgb,
        is_prediction=True
    )

    with rasterio.open(reference_tiff) as src:
        crs=src.crs
        bounds=src.bounds

    height,width,channels=rgb.shape

    transform=from_bounds(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        width,
        height
    )

    # SR GEOTIFF

    sr_tiff=f"sr_{request_id}.tif"
    sr_tiff_path=os.path.join(
        output_dir,
        sr_tiff
    )

    metadata={
        "driver":"GTiff",
        "height":height,
        "width":width,
        "count":channels,
        "dtype":"float32",
        "crs":crs,
        "transform":transform
    }

    with rasterio.open(
        sr_tiff_path,
        "w",
        **metadata
    ) as dst:

        dst.write(
            rgb.transpose(
                2,
                0,
                1
            ).astype(np.float32)
        )

    # SR PNG

    sr_png=f"sr_{request_id}.png"

    sr_png_path=os.path.join(
        output_dir,
        sr_png
    )

    sr_uint8=(
        rgb*255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    Image.fromarray(
        sr_uint8
    ).save(
        sr_png_path
    )

    # HEATMAP

    intensity=(
        0.299*rgb[:,:,0]+
        0.587*rgb[:,:,1]+
        0.114*rgb[:,:,2]
    )

    intensity=global_percentile_stretch(
        intensity,
        is_prediction=True
    )

    heatmap=f"heatmap_{request_id}.png"

    heatmap_path=os.path.join(
        output_dir,
        heatmap
    )

    plt.figure(
        figsize=(8,8)
    )

    plt.imshow(
        intensity,
        cmap="jet"
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        heatmap_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=150
    )

    plt.close()

    return {
        "sr_png":sr_png,
        "sr_tiff":sr_tiff,
        "heatmap_png":heatmap,
        "width":width,
        "height":height,
        "channels":channels,
        "crs":str(crs),
        "bounds":{
            "left":bounds.left,
            "bottom":bounds.bottom,
            "right":bounds.right,
            "top":bounds.top
        }
    }


@app.get("/")
async def root():
    return {
        "message":"AI-SRM Backend Running",
        "status":"success"
    }


@app.post("/api/super-resolve")
async def super_resolve(
    images:list[UploadFile]=File(...)
):

    if len(images)!=8:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Exactly 8 temporal TIFF images "
                f"are required. Received {len(images)}."
            )
        )

    for image in images:
        if not image.filename.lower().endswith(
            (".tif",".tiff")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{image.filename} is not a TIFF image."
            )

    request_id=str(uuid.uuid4())

    temp_dir=os.path.join(
        tempfile.gettempdir(),
        f"ai_srm_{request_id}"
    )

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    try:

        # SAVE 8 INPUT TIFFs

        input_paths=[]

        for i,image in enumerate(images):

            path=os.path.join(
                temp_dir,
                f"temporal_{i+1}.tif"
            )

            with open(path,"wb") as f:
                shutil.copyfileobj(
                    image.file,
                    f
                )

            input_paths.append(path)

        # READ TIFFs

        temporal_images=[]

        for path in input_paths:

            with rasterio.open(path) as src:

                data=src.read()

                data=np.nan_to_num(
                    data,
                    nan=0.0
                )

                temporal_images.append(
                    data.astype(np.float32)
                )

        # CHECK SHAPES

        first_shape=temporal_images[0].shape

        for i,img in enumerate(temporal_images):

            if img.shape!=first_shape:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Temporal image {i+1} has shape "
                        f"{img.shape},expected {first_shape}."
                    )
                )

        # PREDICTION

        predicted_array=pred(
            temporal_images
        )

        # GENERATE SR + HEATMAP

        map_info=generate_map(
            predicted_array=predicted_array,
            reference_tiff=input_paths[0],
            output_dir=OUTPUT_DIR,
            request_id=request_id
        )

        # RESPONSE

        base_url="http://localhost:8000"

        return {
            "status":"success",
            "message":(
                "Super-resolution completed "
                "using 8 temporal images."
            ),
            "prediction":{
                "shape":list(
                    predicted_array.shape
                )
            },
            "outputs":{
                "sr_png":(
                    f"{base_url}/api/output/"
                    f"{map_info['sr_png']}"
                ),
                "sr_tiff":(
                    f"{base_url}/api/output/"
                    f"{map_info['sr_tiff']}"
                ),
                "heatmap_png":(
                    f"{base_url}/api/output/"
                    f"{map_info['heatmap_png']}"
                )
            },
            "map":{
                "width":map_info["width"],
                "height":map_info["height"],
                "channels":map_info["channels"],
                "crs":map_info["crs"],
                "bounds":map_info["bounds"]
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


@app.get("/api/output/{filename}")
async def get_output(filename:str):

    file_path=os.path.join(
        OUTPUT_DIR,
        filename
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Output file not found"
        )

    if filename.lower().endswith(
        (".tif",".tiff")
    ):
        media_type="image/tiff"

    elif filename.lower().endswith(".png"):
        media_type="image/png"

    else:
        media_type="application/octet-stream"

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )


if __name__=="__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )