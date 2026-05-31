"""Evaluation script for chest disease multi-label model."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

from config import PROCESSED_DATA_DIR, MODEL_DIR, IMAGE_SIZE, DISEASE_CLASSES


def _parse_label_cell(cell):
    if isinstance(cell, str):
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    return np.array(cell, dtype=np.float32)


def _load_test_split():
    path = Path(PROCESSED_DATA_DIR) / "test_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test split file: {path}")

    df = pd.read_csv(path)
    if "image_path" not in df.columns or "labels" not in df.columns:
        raise ValueError(f"Invalid file format in {path}")

    image_paths = df["image_path"].astype(str).values
    labels = np.stack(df["labels"].apply(_parse_label_cell).values)

    valid_mask = np.array([Path(p).exists() and len(p) > 0 for p in image_paths])
    image_paths = image_paths[valid_mask]
    labels = labels[valid_mask]

    if len(image_paths) == 0:
        raise RuntimeError("No valid test images found.")

    return image_paths, labels


def _decode_image(path, label):
    bytes_ = tf.io.read_file(path)
    image = tf.io.decode_image(bytes_, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.grayscale_to_rgb(image)
    return image, label


def _build_dataset(image_paths, labels, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(_decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _load_model():
    model_dir = Path(MODEL_DIR)
    final_path = model_dir / "final_model.keras"
    best_path = model_dir / "best_model.keras"

    if final_path.exists():
        return tf.keras.models.load_model(final_path), final_path
    if best_path.exists():
        return tf.keras.models.load_model(best_path), best_path

    raise FileNotFoundError("No trained model found in saved_model/. Expected final_model.keras or best_model.keras")


def _safe_auc(y_true, y_prob):
    aucs = {}
    for i, name in enumerate(DISEASE_CLASSES):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]
        if np.unique(y_t).size < 2:
            aucs[name] = None
        else:
            aucs[name] = float(roc_auc_score(y_t, y_p))
    return aucs


def _safe_ap(y_true, y_prob):
    aps = {}
    for i, name in enumerate(DISEASE_CLASSES):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]
        if np.unique(y_t).size < 2:
            aps[name] = None
        else:
            aps[name] = float(average_precision_score(y_t, y_p))
    return aps


def main():
    print("Loading test split...")
    image_paths, y_true = _load_test_split()
    print(f"Test samples: {len(image_paths):,}")

    ds = _build_dataset(image_paths, y_true, batch_size=16)

    model, model_path = _load_model()
    print(f"Loaded model: {model_path}")

    print("Running predictions...")
    y_prob = model.predict(ds, verbose=1)

    y_pred = (y_prob >= 0.5).astype(np.float32)

    micro_auc = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
    macro_auc = float(np.nanmean([v for v in _safe_auc(y_true, y_prob).values() if v is not None]))

    subset_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))
    hamming_acc = float(np.mean(y_pred == y_true))

    per_class_auc = _safe_auc(y_true, y_prob)
    per_class_ap = _safe_ap(y_true, y_prob)

    report = {
        "test_samples": int(len(image_paths)),
        "model_path": str(model_path),
        "threshold": 0.5,
        "metrics": {
            "micro_auc": micro_auc,
            "macro_auc": macro_auc,
            "subset_accuracy": subset_acc,
            "hamming_accuracy": hamming_acc,
        },
        "per_class_auc": per_class_auc,
        "per_class_average_precision": per_class_ap,
    }

    out_path = Path(PROCESSED_DATA_DIR) / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nEvaluation complete")
    print(f"micro_auc: {micro_auc:.4f}")
    print(f"macro_auc: {macro_auc:.4f}")
    print(f"subset_accuracy: {subset_acc:.4f}")
    print(f"hamming_accuracy: {hamming_acc:.4f}")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
