"""Reduce the 200 spectral bands of the mock cube with PCA."""

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def reduce_bands(cube, n_components=10):
    """Reduce (H, W, bands) to (H, W, n_components) using PCA."""
    height, width, bands = cube.shape
    pixels = cube.reshape(-1, bands)  # one row per pixel
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(pixels)
    return reduced.reshape(height, width, n_components), pca


if __name__ == "__main__":
    cube = np.load(DATA_DIR / "mock_cube.npy")
    reduced, pca = reduce_bands(cube)
    np.save(DATA_DIR / "mock_cube_pca.npy", reduced)
    print("Original shape:", cube.shape)
    print("Reduced shape:", reduced.shape)
    print("Variance kept:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")