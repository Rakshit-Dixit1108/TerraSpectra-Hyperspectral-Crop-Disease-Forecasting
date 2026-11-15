
import numpy as np


def extract_mean_spectral_signature(
    cube: np.ndarray
) -> np.ndarray:
    """
    Calculate the mean reflectance for each spectral band.

    Input:
        (height, width, bands)

    Output:
        (bands,)
    """

    return np.mean(cube, axis=(0, 1))


def extract_std_spectral_signature(
    cube: np.ndarray
) -> np.ndarray:
    """
    Calculate standard deviation for every spectral band.
    """

    return np.std(cube, axis=(0, 1))


def extract_band_statistics(
    cube: np.ndarray
) -> dict:
    """
    Calculate statistical features for each spectral band.
    """

    return {
        "mean": np.mean(cube, axis=(0, 1)),
        "std": np.std(cube, axis=(0, 1)),
        "min": np.min(cube, axis=(0, 1)),
        "max": np.max(cube, axis=(0, 1)),
    }


def extract_pixel_features(
    cube: np.ndarray
) -> np.ndarray:
    """
    Convert every pixel spectrum into a feature vector.

    Output:
        (height * width, spectral_bands)
    """

    height, width, bands = cube.shape

    return cube.reshape(height * width, bands)


def extract_spectral_features(
    cube: np.ndarray
) -> dict:
    """
    Extract a complete set of basic spectral features.
    """

    return {
        "mean_signature": extract_mean_spectral_signature(cube),
        "std_signature": extract_std_spectral_signature(cube),
        "band_statistics": extract_band_statistics(cube),
        "pixel_features": extract_pixel_features(cube),
    }


def calculate_ndvi(
    nir: np.ndarray,
    red: np.ndarray
) -> np.ndarray:
    """
    Calculate NDVI using NIR and red reflectance.

    NDVI = (NIR - Red) / (NIR + Red)
    """

    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )

    return ndvi


def calculate_spectral_ratio(
    band1: np.ndarray,
    band2: np.ndarray
) -> np.ndarray:
    """
    Calculate the ratio between two spectral bands.
    """

    band1 = np.asarray(band1, dtype=np.float32)
    band2 = np.asarray(band2, dtype=np.float32)

    return np.divide(
        band1,
        band2,
        out=np.zeros_like(band1),
        where=band2 != 0
    )
