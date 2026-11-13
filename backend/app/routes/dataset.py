
from fastapi import APIRouter
from pathlib import Path

from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"]
)


@router.get("/info")
def dataset_info():

    raw_files = []
    processed_files = []

    if RAW_DATA_DIR.exists():
        raw_files = [
            file.name
            for file in RAW_DATA_DIR.rglob("*")
            if file.is_file()
        ]

    if PROCESSED_DATA_DIR.exists():
        processed_files = [
            file.name
            for file in PROCESSED_DATA_DIR.rglob("*")
            if file.is_file()
        ]

    return {
        "raw_data_directory": str(RAW_DATA_DIR),
        "processed_data_directory": str(PROCESSED_DATA_DIR),
        "raw_files_count": len(raw_files),
        "processed_files_count": len(processed_files),
        "raw_files": raw_files,
        "processed_files": processed_files
    }