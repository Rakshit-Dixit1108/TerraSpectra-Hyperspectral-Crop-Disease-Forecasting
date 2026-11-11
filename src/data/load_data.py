"""Dataset loading utilities (to be implemented after dataset format is confirmed)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_raw_data():
    """List files in data/raw without attempting to interpret them."""
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    files = sorted(path for path in RAW_DATA_DIR.rglob("*") if path.is_file())
    for path in files:
        print(path.relative_to(PROJECT_ROOT), f"({path.stat().st_size:,} bytes)")
    return files


if __name__ == "__main__":
    inspect_raw_data()
