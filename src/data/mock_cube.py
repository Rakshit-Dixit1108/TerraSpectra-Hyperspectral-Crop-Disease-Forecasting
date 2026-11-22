"""Create a mock hyperspectral cube for testing the pipeline."""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

HEIGHT, WIDTH, BANDS = 50, 50, 200


def create_mock_cube(seed=42):
    """Return a cube (H, W, bands) and a label map (H, W). 1 = stressed."""
    rng = np.random.default_rng(seed)

    # Healthy spectrum: a smooth curve plus small noise per pixel
    bands = np.linspace(0, 1, BANDS)
    healthy = 0.3 + 0.2 * np.sin(2 * np.pi * bands)
    cube = healthy + rng.normal(0, 0.02, size=(HEIGHT, WIDTH, BANDS))

    # Stressed zone: 10 x 10 block with a shifted spectrum
    labels = np.zeros((HEIGHT, WIDTH), dtype=int)
    labels[20:30, 20:30] = 1
    cube[20:30, 20:30, 80:120] += 0.08

    return cube, labels


if __name__ == "__main__":
    cube, labels = create_mock_cube()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "mock_cube.npy", cube)
    np.save(OUTPUT_DIR / "mock_labels.npy", labels)
    print("Cube shape:", cube.shape)
    print("Stressed pixels:", labels.sum())