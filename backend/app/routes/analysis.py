
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.data_loader import load_hyperspectral_data
from app.services.dataset_analysis import analyze_hyperspectral_dataset


router = APIRouter(
    prefix="/analysis",
    tags=["Dataset Analysis"]
)

SUPPORTED_EXTENSIONS = {".npy", ".npz", ".tif", ".tiff"}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB


@router.get("/status")
def analysis_status():
    return {
        "service": "Dataset Analysis",
        "status": "operational",
        "supported_formats": sorted(SUPPORTED_EXTENSIONS),
        "max_upload_size_mb": 100
    }


@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...)
):
    """
    Upload and analyze a hyperspectral dataset.

    Supported formats:
    - NPY
    - NPZ (must contain exactly one array)
    - TIFF
    - TIF
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported file format.",
                "supported_formats": sorted(SUPPORTED_EXTENSIONS)
            }
        )

    temp_path = None

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(file_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 100 MB upload limit."
            )

        with NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        # Load the hyperspectral cube
        cube = load_hyperspectral_data(temp_path)

        # Perform dataset quality analysis
        analysis_result = analyze_hyperspectral_dataset(cube)

        return {
            "success": True,
            "filename": Path(file.filename).name,
            "analysis": analysis_result
        }

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {str(error)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        await file.close()
