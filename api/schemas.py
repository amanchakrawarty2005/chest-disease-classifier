"""Pydantic schemas for the Chest Disease FastAPI service."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    disease: str = Field(..., description="Disease class name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability")
    predicted: bool = Field(..., description="True when probability >= threshold")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    classes: int
    version: str


class PredictResponse(BaseModel):
    filename: Optional[str] = None
    threshold: float = Field(..., ge=0.0, le=1.0)
    top_k: int = Field(..., ge=1)
    predicted_labels: List[str]
    top_predictions: List[PredictionItem]
    all_predictions: List[PredictionItem]
    inference_ms: float = Field(..., ge=0.0)
