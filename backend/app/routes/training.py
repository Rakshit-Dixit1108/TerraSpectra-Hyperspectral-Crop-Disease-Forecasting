
"""
TerraSpectra - Model Training API

Accepts labelled hyperspectral features, trains a
Random Forest classifier and saves the trained model.
"""

import os
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, Query

from app.config import PROJECT_ROOT
from app.services.model_training import train_model, save_model


router = APIRouter(
    prefix="/training",
    tags=["Model Training"]
)

MODEL_PATH = Path(
    os.getenv(
        "TERRASPECTRA_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "crop_disease_model.joblib")
    )
)

MAX_UPLOAD_SIZE = 100 * 1024 * 1024


@router.get("/status")
def training_status():
    """Check model training service status."""

    return {
        "service": "TerraSpectra Model Training",
        "status": "ready",
        "trained_model_available": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH)
    }


@router.post("/train")
async def train_crop_model(
    file: UploadFile = File(...),
    test_size: float = Query(default=0.2, gt=0, lt=1),
    n_estimators: int = Query(default=200, ge=10, le=1000),
    random_state: int = Query(default=42, ge=0)
):
    """
    Train a Random Forest model using an NPZ dataset.

    Required NPZ arrays:
        X = Feature matrix (samples, features)
        y = Class labels (samples,)
    """

    if not file.filename or Path(file.filename).suffix.lower() != ".npz":
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid .npz training dataset."
        )

    try:
        content = await file.read(MAX_UPLOAD_SIZE + 1)

        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Dataset exceeds the 100 MB upload limit."
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded dataset is empty."
            )

        # Load dataset safely
        try:
            with np.load(
                BytesIO(content),
                allow_pickle=False
            ) as archive:

                if not {"X", "y"}.issubset(archive.files):
                    raise HTTPException(
                        status_code=422,
                        detail="NPZ must contain both X and y arrays."
                    )

                X = np.asarray(archive["X"], dtype=np.float32)
                y = np.asarray(archive["y"])

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Unable to read NPZ dataset: {exc}"
            ) from exc

        # Validate dataset
        if X.ndim != 2:
            raise HTTPException(
                status_code=422,
                detail="X must be a 2D feature matrix."
            )

        if y.ndim != 1:
            raise HTTPException(
                status_code=422,
                detail="y must be a 1D label array."
            )

        if len(X) != len(y):
            raise HTTPException(
                status_code=422,
                detail="X and y must have equal sample counts."
            )

        # Train model
        training_result = train_model(
            X=X,
            y=y,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators
        )

        # Save trained model
        saved_path = save_model(
            training_result["model"],
            MODEL_PATH
        )

        return {
            "success": True,
            "message": "Model trained and saved successfully.",
            "dataset": {
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "classes": int(len(np.unique(y)))
            },
            "training_parameters": {
                "test_size": test_size,
                "n_estimators": n_estimators,
                "random_state": random_state
            },
            "metrics": training_result["metrics"],
            "model": {
                "path": str(saved_path),
                "saved": True
            }
        }

    except HTTPException:
        raise

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {exc}"
        ) from exc

    finally:
        await file.close()
