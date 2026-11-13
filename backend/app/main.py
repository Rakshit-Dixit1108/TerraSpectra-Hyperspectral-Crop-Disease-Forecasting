
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_DESCRIPTION,
    ALLOWED_ORIGINS
)

from app.routes.health import router as health_router
from app.routes.dataset import router as dataset_router


app = FastAPI(
    title=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=PROJECT_DESCRIPTION
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Register API routes
app.include_router(health_router)
app.include_router(dataset_router)


@app.get("/")
def root():
    return {
        "project": "TerraSpectra",
        "message": "Hyperspectral Crop Disease Analysis API",
        "version": PROJECT_VERSION,
        "documentation": "/docs",
        "health": "/health/"
    }
