
"""
TerraSpectra - Hyperspectral Dataset Analysis

Provides dataset validation, spectral statistics,
and data quality assessment.
"""

from typing import Any

import numpy as np


class DatasetAnalysisError(Exception):
    """Raised when dataset analysis fails."""
    pass


def validate_hyperspectral_cube(
    cube: np.ndarray
) -> dict[str, Any]:
    """
    Validate a hyperspectral cube.

    Expected shape:
        (height, width, spectral_bands)
    """

    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise DatasetAnalysisError(
            f"Expected a 3D cube, received {cube.ndim} dimensions."
        )

    height, width, bands = cube.shape

    if min(height, width, bands) < 1:
        raise DatasetAnalysisError(
            "Cube dimensions must all be greater than zero."
        )

    if not np.issubdtype(cube.dtype, np.number):
        raise DatasetAnalysisError(
            "Dataset must contain numerical values."
        )

    finite_mask = np.isfinite(cube)

    invalid_count = int(cube.size - np.count_nonzero(finite_mask))

    return {
        "valid": bool(np.any(finite_mask)),
        "shape": [int(height), int(width), int(bands)],
        "height": int(height),
        "width": int(width),
        "spectral_bands": int(bands),
        "total_pixels": int(height * width),
        "total_values": int(cube.size),
        "invalid_values": invalid_count,
        "invalid_percentage": round(
            invalid_count / cube.size * 100, 4
        ),
        "data_type": str(cube.dtype),
    }


def calculate_band_statistics(
    cube: np.ndarray
) -> dict[str, list]:
    """
    Calculate statistics independently for every spectral band.
    """

    cube = np.asarray(cube, dtype=np.float64)

    if cube.ndim != 3:
        raise DatasetAnalysisError(
            "Expected a 3D hyperspectral cube."
        )

    height, width, bands = cube.shape

    flattened = cube.reshape(-1, bands)

    statistics = {
        "mean": [],
        "std": [],
        "minimum": [],
        "maximum": [],
        "median": [],
    }

    for band_index in range(bands):

        values = flattened[:, band_index]
        values = values[np.isfinite(values)]

        if values.size == 0:
            for key in statistics:
                statistics[key].append(None)
            continue

        statistics["mean"].append(
            float(np.mean(values))
        )

        statistics["std"].append(
            float(np.std(values))
        )

        statistics["minimum"].append(
            float(np.min(values))
        )

        statistics["maximum"].append(
            float(np.max(values))
        )

        statistics["median"].append(
            float(np.median(values))
        )

    return statistics


def identify_unusable_bands(
    statistics: dict[str, list]
) -> list[int]:
    """
    Identify bands with no valid values or zero variance.
    """

    unusable_bands = []

    means = statistics["mean"]
    standard_deviations = statistics["std"]

    for index, (mean, std) in enumerate(
        zip(means, standard_deviations)
    ):

        if mean is None or std is None or std == 0:
            unusable_bands.append(index)

    return unusable_bands


def calculate_data_quality(
    cube: np.ndarray
) -> dict[str, Any]:
    """
    Calculate overall hyperspectral dataset quality metrics.
    """

    cube = np.asarray(cube)

    if cube.ndim != 3 or cube.size == 0:
        raise DatasetAnalysisError(
            "Invalid or empty hyperspectral cube."
        )

    total_values = cube.size

    finite_mask = np.isfinite(cube)

    valid_values = int(np.count_nonzero(finite_mask))

    invalid_values = int(
        total_values - valid_values
    )

    if valid_values == 0:
        raise DatasetAnalysisError(
            "Dataset contains no valid numerical values."
        )

    valid_data = cube[finite_mask].astype(
        np.float64
    )

    return {
        "valid_values": valid_values,
        "invalid_values": invalid_values,
        "valid_percentage": round(
            valid_values / total_values * 100, 4
        ),
        "invalid_percentage": round(
            invalid_values / total_values * 100, 4
        ),
        "minimum_value": float(np.min(valid_data)),
        "maximum_value": float(np.max(valid_data)),
        "mean_value": float(np.mean(valid_data)),
        "standard_deviation": float(np.std(valid_data)),
        "has_nan": bool(np.isnan(cube).any()),
        "has_infinity": bool(np.isinf(cube).any()),
    }


def analyze_hyperspectral_dataset(
    cube: np.ndarray
) -> dict[str, Any]:
    """
    Run the complete hyperspectral dataset analysis pipeline.
    """

    cube = np.asarray(cube)

    validation = validate_hyperspectral_cube(cube)

    if not validation["valid"]:
        raise DatasetAnalysisError(
            "Dataset contains no valid numerical values."
        )

    statistics = calculate_band_statistics(cube)

    quality = calculate_data_quality(cube)

    unusable_bands = identify_unusable_bands(
        statistics
    )

    total_bands = validation["spectral_bands"]

    usable_bands = total_bands - len(unusable_bands)

    if quality["invalid_values"] > 0 or unusable_bands:
        status = "needs_preprocessing"
    else:
        status = "good"

    return {
        "success": True,
        "dataset_status": status,
        "validation": validation,
        "quality": quality,
        "band_statistics": statistics,
        "band_analysis": {
            "total_bands": total_bands,
            "usable_bands": usable_bands,
            "unusable_bands": unusable_bands,
        },
        "recommendations": [
            "Replace or handle invalid numerical values."
            if quality["invalid_values"] > 0 else None,

            "Review constant or unusable spectral bands."
            if unusable_bands else None,

            "Normalize spectral values before model training."
        ]
    }
