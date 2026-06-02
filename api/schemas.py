from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class PredictionItem(BaseModel):
    disease: str
    probability: float
    predicted: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    classes: int
    version: str


class PredictResponse(BaseModel):
    filename: Optional[str] = None
    threshold: float
    top_k: int
    predicted_labels: List[str]
    top_predictions: List[PredictionItem]
    all_predictions: List[PredictionItem]
    inference_ms: float
