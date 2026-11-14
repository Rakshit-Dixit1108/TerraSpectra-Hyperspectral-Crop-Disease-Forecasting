
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from spectral import open_image


SUPPORTED_FORMATS = {
    ".hdr",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
}


def load_hyperspectral_data(file_path: str | Path) -> np.ndarray:
    """
    Load hyperspectral data from supported file formats.

    Returns:
        NumPy array containing spectral data.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".hdr":
        image = open_image(str(file_path))
        data = image.load()

    elif extension in {".tif", ".tiff"}:
        data = tifffile.imread(file_path)

    elif extension == ".npy":
        data = np.load(file_path, allow_pickle=False)

    elif extension == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(
                    "NPZ file must contain exactly one array. "
                    "Select the required array explicitly."
                )

            data = archive[archive.files[0]]

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("Dataset is empty.")

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D hyperspectral cube, "
            f"but received shape {data.shape}"
        )

    return data


def get_cube_metadata(data: np.ndarray) -> dict[str, Any]:
    """
    Extract basic metadata from a hyperspectral cube.
    """

    height, width, bands = data.shape

    return {
        "height": int(height),
        "width": int(width),
        "spectral_bands": int(bands),
        "total_pixels": int(height * width),
        "total_values": int(data.size),
        "data_type": str(data.dtype),
        "minimum_value": float(np.nanmin(data)),
        "maximum_value": float(np.nanmax(data)),
        "mean_value": float(np.nanmean(data)),
    }
