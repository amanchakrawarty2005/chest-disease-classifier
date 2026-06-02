from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("api.inference")


class ChestXrayInferenceService:

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
        for name in ("final_model.keras", "best_model.keras"):
            path = self.model_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No model in {self.model_dir}. Need final_model.keras or best_model.keras."
        )

    def load_model(self) -> Path:
        if self._model is not None and self._model_path is not None:
            return self._model_path

        path = self.resolve_model_path()
        LOGGER.info("Loading model from %s", path)
        self._model = tf.keras.models.load_model(path)
        self._model_path = path
        return path

    def preprocess_image_bytes(self, raw_bytes: bytes) -> np.ndarray:
        if not raw_bytes:
            raise ValueError("Empty file.")

        try:
            image = tf.io.decode_image(raw_bytes, channels=1, expand_animations=False)
        except Exception as exc:
            raise ValueError("Not a valid image.") from exc

        image = tf.image.resize(image, [self.image_size, self.image_size], method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.grayscale_to_rgb(image)
        image.set_shape([self.image_size, self.image_size, 3])
        return tf.expand_dims(image, axis=0).numpy()

    def predict_from_bytes(self, raw_bytes: bytes) -> np.ndarray:
        if self._model is None:
            self.load_model()
        if self._model is None:
            raise RuntimeError("Model failed to load.")

        batch = self.preprocess_image_bytes(raw_bytes)
        with self._predict_lock:
            out = self._model.predict(batch, verbose=0)[0]
        return out.astype(np.float32)

    def format_predictions(
        self, probabilities: Iterable[float], threshold: float, top_k: int
    ) -> Tuple[List[dict], List[dict], List[str]]:
        scores = [
            {
                "disease": disease,
                "probability": float(prob),
                "predicted": float(prob) >= threshold,
            }
            for disease, prob in zip(self.disease_classes, probabilities)
        ]
        ranked = sorted(scores, key=lambda x: x["probability"], reverse=True)
        labels = [row["disease"] for row in ranked if row["predicted"]]
        return ranked[:top_k], ranked, labels
