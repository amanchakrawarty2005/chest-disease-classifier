from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        # Priority 1: per-class thresholds tuned on val set (F1-maximisation)
        self._tuned_thresholds: Dict[str, float] | None = None
        # Priority 2: class weights for formula-based threshold adjustment
        self._class_weights: Dict[str, float] | None = None
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
        raise FileNotFoundError(f"No model in {self.model_dir}.")

    def _load_tuned_thresholds(self) -> None:
        """Load per-class F1-tuned thresholds saved by train.py."""
        path = self.model_dir / "thresholds.json"
        if not path.exists():
            LOGGER.info("No thresholds.json found — will use formula-based fallback.")
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                self._tuned_thresholds = data
                LOGGER.info("Loaded tuned per-class thresholds from %s", path)
        except Exception as exc:
            LOGGER.warning("Could not load thresholds.json: %s", exc)

    def _load_class_weights(self) -> None:
        """
        Load inverse_frequency class weights for formula-based threshold adjustment.
        Fallback when thresholds.json does not exist (e.g. before retraining).
        """
        parent = self.model_dir.parent
        path = parent / "data" / "processed" / "class_weights.json"
        if not path.exists():
            LOGGER.warning("class_weights.json not found at %s", path)
            return
        try:
            with open(path) as f:
                all_w = json.load(f)
            if isinstance(all_w, dict):
                for key in ("inverse_frequency", "recommended"):
                    if key in all_w and isinstance(all_w[key], dict):
                        self._class_weights = all_w[key]
                        LOGGER.info("Loaded '%s' class weights", key)
                        return
                self._class_weights = all_w  # legacy flat format
        except Exception as exc:
            LOGGER.warning("Could not load class_weights.json: %s", exc)

    def load_model(self) -> Path:
        if self._model is not None and self._model_path is not None:
            return self._model_path

        path = self.resolve_model_path()
        LOGGER.info("Loading model from %s", path)
        self._model = tf.keras.models.load_model(path, compile=False)
        self._model_path = path
        self._load_tuned_thresholds()   # Priority 1
        self._load_class_weights()      # Priority 2 (fallback)
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
        batch = self.preprocess_image_bytes(raw_bytes)
        with self._predict_lock:
            out = self._model.predict(batch, verbose=0)[0]
        return out.astype(np.float32)

    def format_predictions(
        self, probabilities: Iterable[float], threshold: float, top_k: int
    ) -> Tuple[List[dict], List[dict], List[str]]:
        """
        Enhanced threshold strategy to prevent bias toward common diseases.
        
        Threshold priority:
          1. Tuned per-class threshold from thresholds.json (F1-optimized on val set)
          2. Class-weight-based formula: applies higher weights to rare diseases
          3. Global threshold (default 0.5)
        
        For rare diseases like Cardiomegaly & Hernia:
        - Use LOWER thresholds to ensure detection despite class imbalance
        - Rare disease signals (even if weak) should not be suppressed
        
        Calibrated probability ranking ensures:
        - Rare diseases that barely clear threshold rank high
        - Common diseases must have stronger signal to outrank rare ones
        """
        scores: List[dict] = []

        for disease, prob in zip(self.disease_classes, probabilities):
            probability = float(prob)

            # Identify rare diseases (prevalence < 10%)
            is_rare_disease = disease in ("Cardiomegaly", "Pneumonia", "Hernia", "Fibrosis", "Edema")

            # Priority 1 — tuned thresholds (most reliable)
            if self._tuned_thresholds and disease in self._tuned_thresholds:
                class_threshold = float(self._tuned_thresholds[disease])

            # Priority 2 — formula-based from class weights
            elif self._class_weights and disease in self._class_weights:
                w = float(np.clip(self._class_weights[disease], 0.1, 15.0))
                # Lower threshold for rare diseases: threshold / (1 + w^0.5)
                # This makes rare diseases easier to predict
                if is_rare_disease:
                    class_threshold = max(0.05, threshold / (2.0 + w**0.5 / 2.0))
                else:
                    class_threshold = max(0.05, threshold / (1.0 + w**0.5 / 4.0))

            # Priority 3 — global with rare disease adjustment
            else:
                class_threshold = 0.35 if is_rare_disease else threshold

            # Calibrated probability for ranking
            # Amplify rare disease signals in ranking
            cal_amplification = 1.5 if is_rare_disease else 1.0
            cal = min(probability * (threshold / max(class_threshold, 1e-6)) * cal_amplification, 1.0)

            scores.append({
                "disease": disease,
                "probability": probability,
                "_cal": cal,
                "_threshold": class_threshold,
                "predicted": probability >= class_threshold,
            })

        ranked = sorted(scores, key=lambda x: x["_cal"], reverse=True)
        labels = [r["disease"] for r in ranked if r["predicted"]]
        top = ranked[:top_k]

        # Clean output (remove internal fields)
        for r in ranked:
            r.pop("_cal", None)
            r.pop("_threshold", None)

        return top, ranked, labels