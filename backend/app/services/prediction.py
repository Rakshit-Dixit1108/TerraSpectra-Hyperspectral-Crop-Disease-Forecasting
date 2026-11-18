
"""
TerraSpectra - Prediction Service

Provides inference utilities for trained hyperspectral crop
disease classification models.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class PredictionError(Exception):
    """Raised when prediction cannot be completed."""
    pass


class HyperspectralPredictor:

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = None

    def load_model(self):
        """Load a trained classification model."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise PredictionError(
                f"Could not load model: {exc}"
            ) from exc

        if not hasattr(self.model, "predict"):
            raise PredictionError(
                "Loaded object does not support prediction."
            )

        return self.model

    @staticmethod
    def validate_features(features: np.ndarray) -> np.ndarray:
        """Validate a feature matrix."""

        features = np.asarray(features, dtype=np.float32)

        if features.ndim != 2:
            raise PredictionError(
                "Features must be a 2D array."
            )

        if features.shape[0] == 0 or features.shape[1] == 0:
            raise PredictionError(
                "Feature matrix cannot be empty."
            )

        if not np.isfinite(features).all():
            raise PredictionError(
                "Features contain NaN or infinite values."
            )

        return features

    def predict_features(
        self,
        features: np.ndarray,
        batch_size: int = 10000,
    ) -> dict[str, Any]:
        """
        Predict classes for a feature matrix in batches.
        """

        if self.model is None:
            raise PredictionError(
                "Model is not loaded. Call load_model() first."
            )

        features = self.validate_features(features)

        if batch_size < 1:
            raise PredictionError(
                "batch_size must be greater than zero."
            )

        expected_features = getattr(
            self.model,
            "n_features_in_",
            None
        )

        if (
            expected_features is not None
            and features.shape[1] != expected_features
        ):
            raise PredictionError(
                f"Model expects {expected_features} features, "
                f"but received {features.shape[1]}."
            )

        predictions = []
        probabilities = []

        for start in range(0, len(features), batch_size):

            batch = features[start:start + batch_size]

            try:
                batch_predictions = self.model.predict(batch)
            except Exception as exc:
                raise PredictionError(
                    f"Model prediction failed: {exc}"
                ) from exc

            predictions.extend(batch_predictions.tolist())

            if hasattr(self.model, "predict_proba"):
                batch_probabilities = self.model.predict_proba(batch)
                probabilities.extend(
                    batch_probabilities.tolist()
                )

        confidence = []

        if probabilities:
            confidence = np.max(
                np.asarray(probabilities),
                axis=1
            ).tolist()

        return {
            "predictions": predictions,
            "confidence": confidence,
            "classes": (
                np.asarray(self.model.classes_).tolist()
                if hasattr(self.model, "classes_")
                else None
            ),
        }

    def predict_cube(
        self,
        cube: np.ndarray,
        batch_size: int = 10000,
    ) -> dict[str, Any]:
        """
        Classify every pixel in a hyperspectral image cube.

        Input shape:
            (height, width, spectral_bands)
        """

        cube = np.asarray(cube, dtype=np.float32)

        if cube.ndim != 3:
            raise PredictionError(
                "Expected cube shape (height, width, bands)."
            )

        if not np.isfinite(cube).all():
            raise PredictionError(
                "Hyperspectral cube contains invalid values. "
                "Preprocess the cube before prediction."
            )

        height, width, bands = cube.shape

        pixel_features = cube.reshape(-1, bands)

        result = self.predict_features(
            pixel_features,
            batch_size=batch_size
        )

        prediction_map = np.asarray(
            result["predictions"]
        ).reshape(height, width)

        confidence_map = None

        if result["confidence"]:
            confidence_map = np.asarray(
                result["confidence"],
                dtype=np.float32
            ).reshape(height, width)

        unique_classes, counts = np.unique(
            prediction_map,
            return_counts=True
        )

        total_pixels = prediction_map.size

        disease_distribution = {}

        for class_label, count in zip(
            unique_classes,
            counts
        ):
            disease_distribution[str(class_label)] = {
                "pixel_count": int(count),
                "percentage": round(
                    float(count / total_pixels * 100),
                    2
                )
            }

        return {
            "prediction_map": prediction_map.tolist(),
            "confidence_map": (
                confidence_map.tolist()
                if confidence_map is not None
                else None
            ),
            "disease_distribution": disease_distribution,
            "image_metadata": {
                "height": int(height),
                "width": int(width),
                "spectral_bands": int(bands),
                "total_pixels": int(total_pixels)
            }
        }


def predict_hyperspectral_cube(
    cube: np.ndarray,
    model_path: str | Path,
    batch_size: int = 10000,
) -> dict[str, Any]:
    """Convenience function for complete cube prediction."""

    predictor = HyperspectralPredictor(model_path)
    predictor.load_model()

    return predictor.predict_cube(
        cube,
        batch_size=batch_size
    )
