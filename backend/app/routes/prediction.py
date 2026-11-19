
"""
TerraSpectra - Prediction API Routes

Accepts hyperspectral files and returns crop disease
classification results using a trained ML model.
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Query

from app.config import PROJECT_ROOT
from app.services.data_loader import load_hyperspectral_data
from app.services.preprocessing import preprocess_cube
from app.services.prediction import HyperspectralPredictor


router = APIRouter(
    prefix="/prediction",
    tags=["Prediction"]
)

MODEL_PATH = Path(
    os.getenv(
        "TERRASPECTRA_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "crop_disease_model.joblib")
    )
)

SUPPORTED_EXTENSIONS = {
    ".npy",
    ".npz",
    ".tif",
    ".tiff"
}


@router.get("/status")
def prediction_status():
    """Check whether the prediction model is available."""

    return {
        "service": "TerraSpectra Prediction Service",
        "status": "ready" if MODEL_PATH.exists() else "model_not_found",
        "model_path": str(MODEL_PATH),
        "model_available": MODEL_PATH.exists()
    }


@router.post("/predict")
async def predict_crop_disease(
    file: UploadFile = File(...),
    normalization: str = Query(
        default="minmax",
        pattern="^(minmax|standard|maxabs)$"
    )
):
    """
    Upload a hyperspectral cube and predict crop disease classes.

    Supported formats:
        .npy
        .npz
        .tif
        .tiff
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
            detail=(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained model not found. "
                "Train and save the model before making predictions."
            )
        )

    temporary_path = None

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temporary_file:

            temporary_path = Path(temporary_file.name)

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                temporary_file.write(chunk)

        # Load hyperspectral cube
        cube = load_hyperspectral_data(
            temporary_path
        )

        # Preprocess input data
        processed_cube = preprocess_cube(
            cube,
            normalization=normalization,
            smoothing=False
        )

        # Load trained model
        predictor = HyperspectralPredictor(
            MODEL_PATH
        )

        predictor.load_model()

        # Generate predictions
        results = predictor.predict_cube(
            processed_cube
        )

        return {
            "success": True,
            "project": "TerraSpectra",
            "message": "Prediction completed successfully.",
            "filename": Path(file.filename).name,
            "normalization": normalization,
            "results": results
        }

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        ) from exc

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

        await file.close()
