from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from config import DISEASE_CLASSES, IMAGE_SIZE, MODEL_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
from balancing import get_class_balance_report

LOGGER = logging.getLogger("evaluate")
AUTOTUNE = tf.data.AUTOTUNE


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_label_cell(cell) -> np.ndarray:
    if isinstance(cell, str):
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    return np.array(cell, dtype=np.float32)


def load_test_split(
    processed_data_dir: Path,
    max_samples: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = processed_data_dir / "test_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing test split file: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"image_path", "labels"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Invalid test split format in {csv_path}. Missing: {sorted(missing)}")

    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    image_paths = df["image_path"].astype(str).values
    labels = np.stack(df["labels"].apply(parse_label_cell).values).astype(np.float32)

    valid_mask = np.array([Path(path).exists() and len(path) > 0 for path in image_paths])
    image_paths = image_paths[valid_mask]
    labels = labels[valid_mask]

    if len(image_paths) == 0:
        raise RuntimeError("No valid test images found after path filtering.")

    LOGGER.info("Loaded test split with %s usable samples", f"{len(image_paths):,}")
    return image_paths, labels


def decode_image(path: tf.Tensor, label: tf.Tensor, image_size: int):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.grayscale_to_rgb(image)
    image.set_shape([image_size, image_size, 3])
    return image, label


def build_dataset(image_paths: np.ndarray, labels: np.ndarray, batch_size: int, image_size: int):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(lambda p, y: decode_image(p, y, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic, np.integer, np.floating)):
        return obj.item()
    return str(obj)


def load_trained_model(model_dir: Path) -> Tuple[tf.keras.Model, Path]:
    final_path = model_dir / "final_model.keras"
    best_path = model_dir / "best_model.keras"

    if final_path.exists():
        return tf.keras.models.load_model(final_path), final_path
    if best_path.exists():
        return tf.keras.models.load_model(best_path), best_path

    raise FileNotFoundError("No trained model found. Expected final_model.keras or best_model.keras.")


def compute_per_class_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for idx, class_name in enumerate(DISEASE_CLASSES):
        y_true_col = y_true[:, idx]
        y_prob_col = y_prob[:, idx]

        if np.unique(y_true_col).size < 2:
            result[class_name] = None
            continue

        result[class_name] = float(metric_fn(y_true_col, y_prob_col))

    return result


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(np.float32)

    per_class_auc = compute_per_class_metric(y_true, y_prob, roc_auc_score)
    per_class_ap = compute_per_class_metric(y_true, y_prob, average_precision_score)

    valid_aucs = [value for value in per_class_auc.values() if value is not None]
    valid_aps = [value for value in per_class_ap.values() if value is not None]

    class_weights = np.mean(y_true, axis=0)
    class_weights = class_weights / np.sum(class_weights)
    
    weighted_auc_scores = [
        per_class_auc[class_name] * class_weights[idx]
        for idx, class_name in enumerate(DISEASE_CLASSES)
        if per_class_auc[class_name] is not None
    ]
    weighted_auc = float(np.sum(weighted_auc_scores)) if weighted_auc_scores else None

    metrics = {
        "micro_auc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
        "weighted_auc": weighted_auc,
        "macro_ap": float(np.mean(valid_aps)) if valid_aps else None,
        "subset_accuracy": float(np.mean(np.all(y_pred == y_true, axis=1))),
        "hamming_accuracy": float(np.mean(y_pred == y_true)),
    }

    minority_indices = np.where(np.mean(y_true, axis=0) < 0.05)[0]
    if len(minority_indices) > 0:
        minority_auc = np.mean([
            per_class_auc[DISEASE_CLASSES[idx]]
            for idx in minority_indices
            if per_class_auc[DISEASE_CLASSES[idx]] is not None
        ])
        metrics["minority_class_auc"] = float(minority_auc) if not np.isnan(minority_auc) else None
    
    return {
        "threshold": threshold,
        "metrics": metrics,
        "per_class_auc": per_class_auc,
        "per_class_average_precision": per_class_ap,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained chest disease classifier on test split.")
    parser.add_argument("--processed-data-dir", type=Path, default=Path(PROCESSED_DATA_DIR), help="Directory containing test_labels.csv")
    parser.add_argument("--model-dir", type=Path, default=Path(MODEL_DIR), help="Directory containing trained model files")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional explicit output JSON path")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Evaluation image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for model.predict")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting probabilities to binary predictions")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick evaluation runs")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for sampling")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    LOGGER.info("Loading test split...")
    image_paths, y_true = load_test_split(args.processed_data_dir, args.max_samples, args.seed)

    dataset = build_dataset(
        image_paths,
        y_true,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    model, model_path = load_trained_model(args.model_dir)
    LOGGER.info("Loaded model: %s", model_path)

    LOGGER.info("Running predictions...")
    y_prob = model.predict(dataset, verbose=1)

    report = evaluate_predictions(y_true, y_prob, threshold=args.threshold)
    report["test_samples"] = int(len(image_paths))
    report["model_path"] = str(model_path)
    
    balance_report = get_class_balance_report(y_true, DISEASE_CLASSES, "test")
    report["class_balance"] = balance_report

    output_path = args.output_path or (args.processed_data_dir / "evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=to_json_serializable)

    LOGGER.info("Evaluation complete")
    LOGGER.info("=" * 70)
    LOGGER.info("METRICS SUMMARY")
    LOGGER.info("=" * 70)
    LOGGER.info("Micro AUC: %.4f", report["metrics"]["micro_auc"])
    macro_auc = report["metrics"]["macro_auc"]
    if macro_auc is None:
        LOGGER.info("Macro AUC: N/A (insufficient positives for all classes)")
    else:
        LOGGER.info("Macro AUC: %.4f", macro_auc)
    
    weighted_auc = report["metrics"].get("weighted_auc")
    if weighted_auc is not None:
        LOGGER.info("Weighted AUC (handles class imbalance): %.4f", weighted_auc)
    
    minority_auc = report["metrics"].get("minority_class_auc")
    if minority_auc is not None:
        LOGGER.info("Minority Class AUC: %.4f", minority_auc)
    
    LOGGER.info("Subset Accuracy: %.4f", report["metrics"]["subset_accuracy"])
    LOGGER.info("Hamming Accuracy: %.4f", report["metrics"]["hamming_accuracy"])
    LOGGER.info("=" * 70)
    LOGGER.info("CLASS BALANCE IN TEST SET")
    LOGGER.info("=" * 70)
    LOGGER.info("Total Samples: %d", balance_report["total_samples"])
    LOGGER.info("Healthy Samples: %d (%.2f%%)", balance_report["healthy_samples"], balance_report["healthy_percentage"])
    LOGGER.info("Diseased Samples: %d (%.2f%%)", balance_report["diseased_samples"], 100 - balance_report["healthy_percentage"])
    LOGGER.info("Imbalance Ratio: %.2fx", balance_report["imbalance_ratio"])
    LOGGER.info("Report saved: %s", output_path)


if __name__ == "__main__":
    main()