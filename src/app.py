"""
app.py
------
FastAPI application that serves energy demand predictions using the latest
model registered in the MLflow Model Registry.

Endpoints:
    GET  /          — Welcome message and API info
    GET  /health    — Service health check + model load status
    POST /predict   — Accept weather features, return demand prediction

The model is loaded once on application startup via a lifespan context
manager to avoid per-request I/O overhead.

Usage:
    uvicorn src.app:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "mlruns"
MODEL_REGISTRY_NAME = "energy-demand-model"
MODEL_STAGE = "latest"  # Loads the most recently registered model version

# ---------------------------------------------------------------------------
# Shared Application State
# ---------------------------------------------------------------------------
app_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan: Load Model Once at Startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.

    Loads the latest registered MLflow model on startup so it is ready
    for inference without per-request loading latency.
    """
    logger.info("Starting up — loading model from MLflow registry...")
    try:
        import os
        portable_model_path = os.path.join("models", MODEL_REGISTRY_NAME)

        if os.path.exists(portable_model_path):
            # Load from portable directory (best for Docker)
            logger.info("Loading portable model copy from '%s'...", portable_model_path)
            app_state["model"] = mlflow.pyfunc.load_model(portable_model_path)
            app_state["model_version"] = "portable-local"
        else:
            # Fallback for MLflow Model Registry
            logger.info("Portable copy not found. Loading from MLflow registry...")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MODEL_REGISTRY_NAME}/{MODEL_STAGE}"
            app_state["model"] = mlflow.pyfunc.load_model(model_uri)
            app_state["model_version"] = MODEL_STAGE

        logger.info("Model '%s' loaded successfully.", MODEL_REGISTRY_NAME)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        app_state["model"] = None
        app_state["model_version"] = "unavailable"
        app_state["load_error"] = str(exc)

    yield  # Application runs here

    # Teardown (cleanup if needed)
    logger.info("Shutting down — releasing model from memory.")
    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Energy Grid Demand Prediction API",
    description=(
        "Production-grade MLOps API for predicting daily energy grid demand "
        "based on weather features. Powered by a Random Forest model tracked "
        "via MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Input schema for the /predict endpoint."""

    temperature: float = Field(
        ...,
        description="Ambient temperature in degrees Celsius.",
        examples=[22.5],
    )
    humidity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity as a percentage (0–100).",
        examples=[65.0],
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of the week (0 = Monday, 6 = Sunday).",
        examples=[1],
    )

    @field_validator("temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        """Reject physically implausible temperature values."""
        if v < -60 or v > 60:
            raise ValueError("Temperature must be between -60°C and 60°C.")
        return v


class PredictionResponse(BaseModel):
    """Output schema for the /predict endpoint."""

    predicted_demand_mwh: float = Field(
        ..., description="Predicted daily energy demand in Megawatt-hours (MWh)."
    )
    model_name: str = Field(..., description="Registered model name.")
    model_version: str = Field(..., description="Model version used for inference.")


class HealthResponse(BaseModel):
    """Output schema for the /health endpoint."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=JSONResponse, tags=["General"])
async def root() -> dict:
    """
    Welcome endpoint. Returns API metadata.
    """
    return {
        "api": "Energy Grid Demand Prediction",
        "version": "1.0.0",
        "docs": "/docs",
        "predict": "/predict",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Reports whether the model was successfully loaded and the service
    is ready to serve predictions.
    """
    model_loaded = app_state.get("model") is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=MODEL_REGISTRY_NAME,
        model_version=app_state.get("model_version", "unknown"),
        detail=app_state.get("load_error") if not model_loaded else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict daily energy grid demand from weather features.

    Args:
        request: JSON body with temperature (°C), humidity (%), day_of_week (0–6).

    Returns:
        JSON with predicted_demand_mwh, model_name, and model_version.

    Raises:
        503: If the model failed to load at startup.
        500: If an unexpected error occurs during inference.
    """
    # Guard: model must be available
    if app_state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not available. "
                f"Load error: {app_state.get('load_error', 'Unknown')}. "
                "Please run `python src/train.py` and restart the server."
            ),
        )

    try:
        import pandas as pd
        feature_df = pd.DataFrame(
            [{
                "temperature": request.temperature,
                "humidity": request.humidity,
                "day_of_week": request.day_of_week
            }]
        )

        prediction = app_state["model"].predict(feature_df)
        predicted_value = float(round(prediction[0], 2))

        logger.info(
            "Prediction — temp=%.1f, humidity=%.1f, dow=%d → %.2f MWh",
            request.temperature,
            request.humidity,
            request.day_of_week,
            predicted_value,
        )

        return PredictionResponse(
            predicted_demand_mwh=predicted_value,
            model_name=MODEL_REGISTRY_NAME,
            model_version=app_state.get("model_version", "latest"),
        )

    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed due to an internal error: {exc}",
        ) from exc
