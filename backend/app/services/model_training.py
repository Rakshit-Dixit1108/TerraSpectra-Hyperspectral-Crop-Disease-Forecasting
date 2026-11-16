
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """
    Train a Random Forest crop disease classifier.

    X: Feature matrix (samples, features)
    y: Labels (samples,)
    """

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    # Validate input data
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D label array.")

    if len(X) != len(y):
        raise ValueError("X and y must contain equal samples.")

    if len(X) < 2:
        raise ValueError("At least two samples are required.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    if len(np.unique(y)) < 2:
        raise ValueError("At least two classes are required.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Create classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return {
        "model": model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_model(
    model: RandomForestClassifier,
    file_path: str | Path,
) -> Path:
    """
    Save trained model using joblib.
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, file_path)

    return file_path


def load_model(
    file_path: str | Path,
) -> RandomForestClassifier:
    """
    Load a previously trained model.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Model not found: {file_path}"
        )

    return joblib.load(file_path)


def predict_disease(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> dict[str, Any]:
    """
    Predict disease classes and confidence scores.
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")

    if not np.isfinite(X).all():
        raise ValueError("X contains invalid values.")

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    confidence = np.max(probabilities, axis=1)

    return {
        "predictions": predictions.tolist(),
        "confidence": confidence.tolist(),
    }
