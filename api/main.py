from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DISEASE_CLASSES,
    IMAGE_SIZE,
    MODEL_DIR,
)

from api.inference import ChestXrayInferenceService  
from api.schemas import HealthResponse, PredictResponse  

LOGGER = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

inference_service = ChestXrayInferenceService(
    model_dir=Path(MODEL_DIR),
    image_size=IMAGE_SIZE,
    disease_classes=DISEASE_CLASSES,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = inference_service
    try:
        model_path = inference_service.load_model()
        LOGGER.info("Startup complete. Loaded model: %s", model_path)
    except Exception as exc:  
        LOGGER.error("Startup warning: model could not be loaded yet (%s)", exc)
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "message": "Chest Disease Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service: ChestXrayInferenceService = request.app.state.inference_service
    return HealthResponse(
        status="ok",
        model_loaded=service.model_loaded,
        model_path=str(service.model_path) if service.model_path else None,
        classes=len(DISEASE_CLASSES),
        version=API_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    top_k: int = Query(5, ge=1, le=len(DISEASE_CLASSES)),
) -> PredictResponse:
    service: ChestXrayInferenceService = request.app.state.inference_service

    allowed_non_image_types = {"application/octet-stream"}
    incoming_type = (file.content_type or "").lower()
    if incoming_type and (not incoming_type.startswith("image/")) and incoming_type not in allowed_non_image_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}. Please upload an image file.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        start = time.perf_counter()
        probabilities = service.predict_from_bytes(raw_bytes)
        inference_ms = (time.perf_counter() - start) * 1000.0
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {exc}") from exc

    top_predictions, all_predictions, predicted_labels = service.format_predictions(
        probabilities=probabilities,
        threshold=threshold,
        top_k=top_k,
    )

    return PredictResponse(
        filename=file.filename,
        threshold=threshold,
        top_k=top_k,
        predicted_labels=predicted_labels,
        top_predictions=top_predictions,
        all_predictions=all_predictions,
        inference_ms=round(float(inference_ms), 3),
    )