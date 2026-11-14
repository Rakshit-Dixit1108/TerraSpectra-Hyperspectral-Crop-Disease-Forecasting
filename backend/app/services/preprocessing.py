
import numpy as np
from scipy.signal import savgol_filter


def handle_invalid_values(
    cube: np.ndarray,
    replacement_value: float = 0.0
) -> np.ndarray:
    """
    Replace NaN and infinite values in a hyperspectral cube.
    """

    cube = np.asarray(cube, dtype=np.float32)

    cube = np.nan_to_num(
        cube,
        nan=replacement_value,
        posinf=replacement_value,
        neginf=replacement_value
    )

    return cube


def normalize_spectra(
    cube: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """
    Normalize spectral values.

    Supported methods:
    - minmax
    - standard
    - maxabs
    """

    cube = np.asarray(cube, dtype=np.float32)

    if not np.isfinite(cube).all():
        raise ValueError(
            "Cube contains invalid values. "
            "Run handle_invalid_values() first."
        )

    if method == "minmax":

        minimum = np.min(cube)
        maximum = np.max(cube)

        if maximum == minimum:
            return np.zeros_like(cube)

        normalized = (
            (cube - minimum) / (maximum - minimum)
        )

    elif method == "standard":

        mean = np.mean(cube)
        std = np.std(cube)

        if std == 0:
            return np.zeros_like(cube)

        normalized = (cube - mean) / std

    elif method == "maxabs":

        maximum = np.max(np.abs(cube))

        if maximum == 0:
            return np.zeros_like(cube)

        normalized = cube / maximum

    else:
        raise ValueError(
            f"Unsupported normalization method: {method}"
        )

    return normalized.astype(np.float32)


def smooth_spectra(
    cube: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing along the spectral axis.

    Expected input shape:
    (height, width, spectral_bands)
    """

    cube = np.asarray(cube, dtype=np.float32)

    bands = cube.shape[-1]

    if window_length % 2 == 0:
        raise ValueError(
            "window_length must be an odd number."
        )

    if window_length > bands:
        raise ValueError(
            "window_length cannot exceed spectral bands."
        )

    if polyorder >= window_length:
        raise ValueError(
            "polyorder must be smaller than window_length."
        )

    if not np.isfinite(cube).all():
        raise ValueError(
            "Cube contains invalid values. "
            "Handle invalid values first."
        )

    smoothed = savgol_filter(
        cube,
        window_length=window_length,
        polyorder=polyorder,
        axis=-1
    )

    return smoothed.astype(np.float32)


def preprocess_cube(
    cube: np.ndarray,
    normalization: str = "minmax",
    smoothing: bool = False
) -> np.ndarray:
    """
    Complete preprocessing pipeline.
    """

    cube = handle_invalid_values(cube)

    if smoothing:
        cube = smooth_spectra(cube)

    cube = normalize_spectra(
        cube,
        method=normalization
    )

    return cube