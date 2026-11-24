
"""
TerraSpectra - Hyperspectral Data Visualization

Generates spectral graphs, band visualizations,
and vegetation index maps.

All generated plots are returned as Base64 PNG images,
ready to be displayed by a frontend.
"""

import base64
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


class VisualizationError(Exception):
    """Raised when hyperspectral visualization fails."""
    pass


def validate_cube(cube: np.ndarray) -> np.ndarray:
    """Validate and convert a hyperspectral cube."""

    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise VisualizationError(
            f"Expected a 3D hyperspectral cube, got {cube.ndim}D."
        )

    if not np.issubdtype(cube.dtype, np.number):
        raise VisualizationError(
            "Hyperspectral data must contain numerical values."
        )

    if cube.size == 0 or min(cube.shape) < 1:
        raise VisualizationError("Hyperspectral cube is empty.")

    if not np.isfinite(cube).any():
        raise VisualizationError(
            "Dataset contains no valid numerical values."
        )

    return cube.astype(np.float64, copy=False)


def figure_to_base64(fig) -> str:
    """Convert a Matplotlib figure into a Base64 PNG string."""

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    buffer.seek(0)

    encoded = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    plt.close(fig)

    return encoded


def plot_mean_spectral_signature(
    cube: np.ndarray
) -> str:
    """
    Plot the average reflectance across spectral bands.

    Returns:
        Base64-encoded PNG image.
    """

    cube = validate_cube(cube)

    with np.errstate(invalid="ignore"):
        mean_signature = np.nanmean(
            np.where(np.isfinite(cube), cube, np.nan),
            axis=(0, 1)
        )

    if not np.isfinite(mean_signature).any():
        raise VisualizationError(
            "Unable to calculate mean spectral signature."
        )

    bands = np.arange(1, cube.shape[2] + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        bands,
        mean_signature,
        linewidth=2,
        label="Mean spectral signature"
    )

    ax.set_title("Mean Spectral Signature")
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel("Mean Reflectance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return figure_to_base64(fig)


def plot_band_statistics(
    cube: np.ndarray
) -> str:
    """
    Plot mean and standard deviation for every band.

    Returns:
        Base64-encoded PNG image.
    """

    cube = validate_cube(cube)

    valid_cube = np.where(
        np.isfinite(cube),
        cube,
        np.nan
    )

    with np.errstate(invalid="ignore"):
        means = np.nanmean(valid_cube, axis=(0, 1))
        stds = np.nanstd(valid_cube, axis=(0, 1))

    band_numbers = np.arange(1, cube.shape[2] + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        band_numbers,
        means,
        label="Mean",
        linewidth=2
    )

    ax.fill_between(
        band_numbers,
        means - stds,
        means + stds,
        alpha=0.2,
        label="Standard deviation"
    )

    ax.set_title("Spectral Band Statistics")
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel("Reflectance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return figure_to_base64(fig)


def plot_band_image(
    cube: np.ndarray,
    band_index: int
) -> str:
    """
    Generate a spatial image for one spectral band.

    band_index is zero-based.
    """

    cube = validate_cube(cube)

    if not isinstance(band_index, (int, np.integer)):
        raise VisualizationError(
            "Band index must be an integer."
        )

    if not 0 <= band_index < cube.shape[2]:
        raise VisualizationError(
            f"Band index must be between 0 and {cube.shape[2] - 1}."
        )

    band = cube[:, :, band_index]

    if not np.isfinite(band).any():
        raise VisualizationError(
            "Selected band contains no valid values."
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        band,
        cmap="viridis"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Reflectance"
    )

    ax.set_title(
        f"Hyperspectral Band {band_index + 1}"
    )

    ax.set_xlabel("Pixel Width")
    ax.set_ylabel("Pixel Height")

    return figure_to_base64(fig)


def calculate_ndvi(
    cube: np.ndarray,
    red_band_index: int,
    nir_band_index: int
) -> np.ndarray:
    """
    Calculate NDVI from selected red and NIR bands.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    Band indices are zero-based.
    """

    cube = validate_cube(cube)

    bands = cube.shape[2]

    for index, name in [
        (red_band_index, "Red"),
        (nir_band_index, "NIR")
    ]:
        if not isinstance(index, (int, np.integer)):
            raise VisualizationError(
                f"{name} band index must be an integer."
            )

        if not 0 <= index < bands:
            raise VisualizationError(
                f"{name} band index must be between 0 and {bands - 1}."
            )

    red = cube[:, :, red_band_index]
    nir = cube[:, :, nir_band_index]

    denominator = nir + red

    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & (denominator != 0)
    )

    ndvi = np.full(
        red.shape,
        np.nan,
        dtype=np.float32
    )

    np.divide(
        nir - red,
        denominator,
        out=ndvi,
        where=valid
    )

    return ndvi


def plot_ndvi_map(
    cube: np.ndarray,
    red_band_index: int,
    nir_band_index: int
) -> str:
    """
    Generate an NDVI vegetation map.

    Important:
    Correct red and NIR band indices must be supplied
    according to the dataset's wavelength information.
    """

    ndvi = calculate_ndvi(
        cube,
        red_band_index,
        nir_band_index
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        ndvi,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1
    )

    fig.colorbar(
        image,
        ax=ax,
        label="NDVI"
    )

    ax.set_title("Vegetation NDVI Map")
    ax.set_xlabel("Pixel Width")
    ax.set_ylabel("Pixel Height")

    return figure_to_base64(fig)


def generate_visualization_report(
    cube: np.ndarray,
    red_band_index: int | None = None,
    nir_band_index: int | None = None,
    selected_band_index: int = 0
) -> dict[str, Any]:
    """
    Generate a complete visualization report.

    NDVI is generated only when both red and NIR
    band indices are supplied.
    """

    cube = validate_cube(cube)

    if not 0 <= selected_band_index < cube.shape[2]:
        raise VisualizationError(
            "Selected band index is outside the cube."
        )

    report = {
        "image_format": "PNG",
        "encoding": "Base64",
        "cube_shape": list(cube.shape),
        "visualizations": {
            "mean_spectral_signature":
                plot_mean_spectral_signature(cube),

            "band_statistics":
                plot_band_statistics(cube),

            "selected_band":
                plot_band_image(cube, selected_band_index),
        }
    }

    if red_band_index is not None and nir_band_index is not None:
        report["visualizations"]["ndvi_map"] = plot_ndvi_map(
            cube,
            red_band_index,
            nir_band_index
        )

    return report
