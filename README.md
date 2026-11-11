# TerraSpectra — Hyperspectral Crop Disease Forecasting

Starter scaffold for learning and developing the internship project step by step.

## Project status
This is an initial structure only. No dataset, trained model, or forecasting claims are included yet.
Dataset choice and target definition must be validated before model development.

## Structure
- `data/raw/`: original dataset (do not commit large/private data)
- `data/processed/`: cleaned/derived data
- `notebooks/`: exploratory notebooks
- `src/data/`: data loading and preprocessing
- `src/features/`: spectral feature engineering
- `src/models/`: training and evaluation
- `backend/`: API (later)
- `frontend/`: UI (later)
- `reports/`: figures and project documentation

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Ubuntu/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Next steps
1. Confirm dataset access, format, labels, wavelengths, and observation dates.
2. Build and validate data loading.
3. Perform EDA and establish a baseline.
4. Define a valid temporal split if forecasting is supported by the data.
