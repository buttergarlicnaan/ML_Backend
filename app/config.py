import os
from pathlib import Path
from typing import Optional
from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "MFSR Satellite Super-Resolution API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model configuration
    MODEL_CHECKPOINT: str = str(BASE_DIR / "mfsr_checkpoints" / "best_val_charbonnier.pth")
    DEVICE: str = "auto"  # "auto", "cuda", or "cpu"
    FEAT_CHANNELS: int = 64
    OUT_CHANNELS: Optional[int] = None  # None for auto-detection from checkpoint (typically 4 or 12)
    
    # Spatial configuration
    DEFAULT_SCALE_FACTOR: float = 6.66666667  # WorldStrat LR (10m) to HR (~1.5m, 159 -> 1060)
    
    # Upload and output paths
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    OUTPUT_DIR: str = str(BASE_DIR / "outputs")
    
    # Security and upload constraints
    MAX_UPLOAD_SIZE_MB: int = 250
    ALLOWED_EXTENSIONS: list[str] = [".tif", ".tiff"]
    
    # Preprocessing flags
    NORMALIZE_DIVISOR: float = 10000.0  # Sentinel-2 L2A 16-bit to reflectance [0, 1]
    ALLOW_SYNTHETIC_ANGLES: bool = False  # Strict by default: requires all 17 channels
    
    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )
    else:
        class Config:
            env_file = ".env"
            extra = "ignore"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
