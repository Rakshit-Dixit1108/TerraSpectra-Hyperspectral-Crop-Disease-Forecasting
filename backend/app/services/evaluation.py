
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate a trained classification model.
    """

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if X_test.ndim != 2:
        raise ValueError("X_test must be a 2D array.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test sizes do not match.")

    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains invalid values.")

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    # ROC-AUC for binary and multiclass classification
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = model.classes_

        try:
            if len(classes) == 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities[:, 1],
                    )
                )
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                        labels=classes,
                    )
                )
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(
        y_test,
        predictions,
        labels=getattr(model, "classes_", None),
    ).tolist()

    return metrics


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str | Path,
) -> Path:
    """
    Generate and save a confusion matrix visualization.
    """

    predictions = model.predict(X_test)

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=getattr(model, "classes_", None),
        cmap="Blues",
        xticks_rotation="vertical",
    )

    display.ax_.set_title("TerraSpectra - Confusion Matrix")

    display.figure_.tight_layout()
    display.figure_.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(display.figure_)

    return save_path


def plot_feature_importance(
    model: Any,
    feature_names: list[str] | None,
    save_path: str | Path,
    top_n: int = 20,
) -> Path:
    """
    Plot the most important features from a trained model.

    Works with models exposing feature_importances_.
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "This model does not support feature importance."
        )

    importances = np.asarray(
        model.feature_importances_
    )

    if feature_names is None:
        feature_names = [
            f"Band_{i}"
            for i in range(len(importances))
        ]

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature names and importance values do not match."
        )

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    indices = np.argsort(importances)[-top_n:]

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        np.array(feature_names)[indices],
        importances[indices],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Spectral Features")
    plt.title("TerraSpectra - Feature Importance")

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return save_path
