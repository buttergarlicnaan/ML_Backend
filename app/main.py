"""
FastAPI Main Application for Multi-Frame Super-Resolution (MFSR).
Loads the model checkpoint once at startup and configures security, logging,
exception handlers, and route endpoints.
"""

from contextlib import asynccontextmanager
import logging
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

from app.config import settings
from app.inference.predictor import predictor
from app.routes.predict import router as predict_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mfsr_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: Loads the MFSR model once when the application starts,
    and handles graceful shutdown.
    """
    logger.info("==================================================")
    logger.info("Initializing MFSR Satellite Super-Resolution Service")
    logger.info(f"Target Device Config: {settings.DEVICE}")
    logger.info(f"Checkpoint Target: {settings.MODEL_CHECKPOINT}")
    logger.info("==================================================")

    try:
        predictor.load(
            checkpoint_path=settings.MODEL_CHECKPOINT,
            device_name=settings.DEVICE,
            feat_channels=settings.FEAT_CHANNELS,
            out_channels=settings.OUT_CHANNELS,
        )
        logger.info("Model loaded successfully on startup.")
    except Exception as e:
        logger.error(f"Failed to load MFSR checkpoint during startup: {e}")
        logger.error("API will start, but inference requests may fail until a valid checkpoint is provided.")

    yield

    logger.info("Shutting down MFSR Satellite Super-Resolution Service.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready FastAPI backend for Multi-Frame Super-Resolution (MFSR) "
        "satellite imagery. Accepts 8 temporal observations, preserves geospatial metadata, "
        "and produces super-resolved imagery alongside pixel-wise uncertainty maps."
    ),
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": detail.get("error", "Bad Request"),
                "detail": detail.get("message", detail),
                **{k: v for k, v in detail.items() if k not in ("error", "message")},
            },
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": "HTTP Error",
            "detail": str(detail),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception during request processing: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred during image processing. Please check server logs.",
        },
    )


# Health Check Endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    System and model health check endpoint.
    """
    cuda_avail = torch.cuda.is_available()
    device_name = str(predictor.device) if predictor.device else ("cuda" if cuda_avail else "cpu")
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "device": device_name,
        "cuda_available": cuda_avail,
        "model_metadata": predictor.metadata if predictor.is_loaded else None,
    }


# Root Endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_endpoint": "/api/v1/predict",
        "predict_zip_endpoint": "/api/v1/predict/zip",
    }


# Include inference routes
app.include_router(predict_router)
