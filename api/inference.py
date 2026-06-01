"""Inference helpers for FastAPI prediction endpoint."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("api.inference")


class ChestXrayInferenceService:
    """Loads trained model and serves single-image predictions."""

    def __init__(self, model_dir: Path, image_size: int, disease_classes: Sequence[str]) -> None:
        self.model_dir = Path(model_dir)
        self.image_size = int(image_size)
        self.disease_classes = list(disease_classes)

        self._model: tf.keras.Model | None = None
        self._model_path: Path | None = None
        self._predict_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Path | None:
        return self._model_path

    def resolve_model_path(self) -> Path:
        for file_name in ("final_model.keras", "best_model.keras"):
            path = self.model_dir / file_name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No model file found in {self.model_dir}. "
            "Expected final_model.keras or best_model.keras."
        )

    def load_model(self) -> Path:
        if self._model is not None and self._model_path is not None:
            return self._model_path

        path = self.resolve_model_path()
        LOGGER.info("Loading model from %s", path)
        self._model = tf.keras.models.load_model(path)
        self._model_path = path
        LOGGER.info("Model loaded successfully.")
        return path

    def preprocess_image_bytes(self, raw_bytes: bytes) -> np.ndarray:
        """Decode image bytes and transform to (1, H, W, 3) float32 batch."""
        if not raw_bytes:
            raise ValueError("Uploaded file is empty.")

        try:
            image = tf.io.decode_image(raw_bytes, channels=1, expand_animations=False)
        except Exception as exc:  # pragma: no cover - TensorFlow raises several subclasses
            raise ValueError("Uploaded file is not a valid image.") from exc

        image = tf.image.resize(image, [self.image_size, self.image_size], method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.grayscale_to_rgb(image)
        image.set_shape([self.image_size, self.image_size, 3])

        batch = tf.expand_dims(image, axis=0).numpy()
        return batch

    def predict_from_bytes(self, raw_bytes: bytes) -> np.ndarray:
        if self._model is None:
            self.load_model()

        if self._model is None:
            raise RuntimeError("Model failed to load.")

        model_input = self.preprocess_image_bytes(raw_bytes)

        with self._predict_lock:
            probabilities = self._model.predict(model_input, verbose=0)[0]

        return probabilities.astype(np.float32)

    def format_predictions(
        self, probabilities: Iterable[float], threshold: float, top_k: int
    ) -> Tuple[List[dict], List[dict], List[str]]:
        scores: List[dict] = []

        for disease, prob in zip(self.disease_classes, probabilities):
            probability = float(prob)
            scores.append(
                {
                    "disease": disease,
                    "probability": probability,
                    "predicted": probability >= threshold,
                }
            )

        sorted_scores = sorted(scores, key=lambda x: x["probability"], reverse=True)
        predicted_labels = [item["disease"] for item in sorted_scores if item["predicted"]]
        top_predictions = sorted_scores[:top_k]
        return top_predictions, sorted_scores, predicted_labels
