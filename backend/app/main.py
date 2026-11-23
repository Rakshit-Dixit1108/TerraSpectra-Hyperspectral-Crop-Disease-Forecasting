
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_DESCRIPTION,
    ALLOWED_ORIGINS,
)

# Import API routers
from app.routes.health import router as health_router
from app.routes.dataset import router as dataset_router
from app.routes.prediction import router as prediction_router
from app.routes.training import router as training_router
from app.routes.analysis import router as analysis_router


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=PROJECT_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REGISTER API ROUTERS
# --------------------------------------------------

app.include_router(health_router)
app.include_router(dataset_router)
app.include_router(prediction_router)
app.include_router(training_router)
app.include_router(analysis_router)


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/", tags=["Root"])
def root():
    return {
        "project": PROJECT_NAME,
        "description": PROJECT_DESCRIPTION,
        "version": PROJECT_VERSION,
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health/",
            "dataset": "/dataset/info",
            "prediction_status": "/prediction/status",
            "prediction": "/prediction/predict",
            "training_status": "/training/status",
            "training": "/training/train",
            "analysis_status": "/analysis/status",
            "analysis": "/analysis/analyze",
        },
    }


# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

@app.get("/api/status", tags=["System"])
def api_status():
    return {
        "api": "TerraSpectra",
        "status": "operational",
        "version": PROJECT_VERSION,
        "modules": {
            "health": "registered",
            "dataset": "registered",
            "prediction": "registered",
            "training": "registered",
            "analysis": "registered",
        },
    }
