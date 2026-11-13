
from pathlib import Path

# Project paths
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Application settings
PROJECT_NAME = "TerraSpectra"
PROJECT_VERSION = "0.1.0"
PROJECT_DESCRIPTION = (
    "Hyperspectral Crop Disease Detection and Forecasting System"
)

# Frontend configuration
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
