from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    DISEASE_CLASSES,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    FOCAL_GAMMA,
    IMAGE_SIZE,
    LEARNING_RATE_PHASE1,
    LEARNING_RATE_PHASE2,
    LR_WARMUP_EPOCHS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from balancing import compute_sample_weights
from model import WarmupCosineDecay, build_model, focal_binary_crossentropy, unfreeze_last_layers

LOGGER = logging.getLogger("train")
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


def load_split_csv(
    split_name: str,
    processed_data_dir: Path,
    max_samples: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = processed_data_dir / f"{split_name}_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"image_path", "labels"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Invalid split format in {csv_path}. Missing: {sorted(missing)}")

    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    paths = df["image_path"].astype(str).values
    labels = np.stack(df["labels"].apply(parse_label_cell).values).astype(np.float32)

    valid_mask = np.array([Path(p).exists() and len(p) > 0 for p in paths])
    paths, labels = paths[valid_mask], labels[valid_mask]

    if len(paths) == 0:
        raise RuntimeError(f"No valid image paths found for split: {split_name}")

    LOGGER.info("Loaded %s split with %s usable samples", split_name, f"{len(paths):,}")
    return paths, labels


def decode_image(path: tf.Tensor, label: tf.Tensor, sample_weight: tf.Tensor, image_size: int):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.grayscale_to_rgb(image)
    image.set_shape([image_size, image_size, 3])
    return image, label, sample_weight


def build_dataset(
    image_paths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    image_size: int,
    training: bool,
    sample_weights: Optional[np.ndarray] = None,
):
    """Build dataset with proper sample weighting for class balance."""
    if sample_weights is None:
        sample_weights = np.ones(len(image_paths), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, sample_weights))
    if training:
        # Larger shuffle buffer for better randomness with oversampled data
        ds = ds.shuffle(min(len(image_paths), 30000), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y, w: decode_image(p, y, w, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_callbacks(model_dir: Path, base_lr: float, total_epochs: int,
                    warmup_epochs: int, patience: int = 5):
    model_dir.mkdir(parents=True, exist_ok=True)
    return [
        WarmupCosineDecay(base_lr=base_lr, total_epochs=total_epochs, warmup_epochs=warmup_epochs),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
    ]


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile with focal BCE + multi-label AUC."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_binary_crossentropy(gamma=FOCAL_GAMMA),
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )


def tune_thresholds(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    val_labels: np.ndarray,
    class_names: list,
    sweep: np.ndarray = np.arange(0.05, 0.95, 0.025),
) -> Dict[str, float]:
    """
    Find per-class threshold maximising F1 on validation set.
    Special handling for rare diseases to ensure detection.
    """
    LOGGER.info("Tuning per-class thresholds on validation set...")
    y_prob = model.predict(val_ds, verbose=0)
    best: Dict[str, float] = {}
    
    # Identify rare diseases
    class_prevalence = np.mean(val_labels, axis=0)
    rare_threshold = 0.10  # Classes < 10% prevalence
    rare_classes = {class_names[i] for i in range(len(class_names)) if class_prevalence[i] < rare_threshold}

    for idx, cls in enumerate(class_names):
        y_true_col = val_labels[:, idx]
        if y_true_col.sum() == 0:
            best[cls] = 0.5
            continue

        # For rare diseases, use wider sweep with lower minimum threshold
        if cls in rare_classes:
            sweep_for_class = np.arange(0.05, 0.70, 0.02)
        else:
            sweep_for_class = sweep

        best_f1, best_thr = -1.0, 0.5
        for thr in sweep_for_class:
            y_pred = (y_prob[:, idx] >= thr).astype(np.float32)
            tp = float(np.sum((y_pred == 1) & (y_true_col == 1)))
            fp = float(np.sum((y_pred == 1) & (y_true_col == 0)))
            fn = float(np.sum((y_pred == 0) & (y_true_col == 1)))
            prec = tp / max(tp + fp, 1e-6)
            rec  = tp / max(tp + fn, 1e-6)
            f1   = 2 * prec * rec / max(prec + rec, 1e-6)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

        best[cls] = best_thr
        rare_marker = " [RARE]" if cls in rare_classes else ""
        LOGGER.info("  %-25s threshold=%.3f  (val F1=%.4f)%s", cls, best_thr, best_f1, rare_marker)

    return best


def save_training_summary(path: Path, args: argparse.Namespace,
                          train_size: int, val_size: int,
                          thresholds: Optional[Dict] = None) -> None:
    payload = {
        "train_samples": int(train_size),
        "val_samples": int(val_size),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "epochs_phase1": int(args.epochs_phase1),
        "epochs_phase2": int(args.epochs_phase2),
        "lr_phase1": float(args.lr_phase1),
        "lr_phase2": float(args.lr_phase2),
        "warmup_epochs": int(args.warmup_epochs),
        "focal_gamma": float(args.focal_gamma),
        "sample_weighting": str(args.sample_weighting),
        "per_class_thresholds": thresholds or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-data-dir", type=Path, default=Path(PROCESSED_DATA_DIR))
    p.add_argument("--model-dir", type=Path, default=Path(MODEL_DIR))
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--epochs-phase1", type=int, default=EPOCHS_PHASE1)
    p.add_argument("--epochs-phase2", type=int, default=EPOCHS_PHASE2)
    p.add_argument("--lr-phase1", type=float, default=LEARNING_RATE_PHASE1)
    p.add_argument("--lr-phase2", type=float, default=LEARNING_RATE_PHASE2)
    p.add_argument("--warmup-epochs", type=int, default=LR_WARMUP_EPOCHS)
    p.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    p.add_argument("--sample-weighting", type=str, default="inverse_frequency",
                   choices=["inverse_frequency", "effective_num"])
    p.add_argument("--tune-thresholds", action="store_true", default=True)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)
    tf.keras.utils.set_random_seed(args.seed)

    LOGGER.info("Loading processed splits...")
    train_paths, y_train = load_split_csv("train", args.processed_data_dir, args.max_train_samples, args.seed)
    val_paths, y_val     = load_split_csv("val",   args.processed_data_dir, args.max_val_samples,   args.seed)

    LOGGER.info("Computing sample weights (%s)...", args.sample_weighting)
    sw = compute_sample_weights(y_train, method=args.sample_weighting)

    train_ds = build_dataset(train_paths, y_train, args.batch_size, args.image_size, training=True,  sample_weights=sw)
    val_ds   = build_dataset(val_paths,   y_val,   args.batch_size, args.image_size, training=False)

    # Phase 1 — train head only
    LOGGER.info("Phase 1 | Frozen backbone | epochs=%d lr=%.1e warmup=%d gamma=%.1f",
                args.epochs_phase1, args.lr_phase1, args.warmup_epochs, args.focal_gamma)
    model = build_model(freeze_backbone=True)
    compile_model(model, args.lr_phase1)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_phase1,
              callbacks=build_callbacks(args.model_dir, args.lr_phase1, args.epochs_phase1, args.warmup_epochs),
              verbose=1)

    # Phase 2 — fine-tune backbone
    LOGGER.info("Phase 2 | Fine-tune backbone | epochs=%d lr=%.1e", args.epochs_phase2, args.lr_phase2)
    model = unfreeze_last_layers(model)
    compile_model(model, args.lr_phase2)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_phase2,
              callbacks=build_callbacks(args.model_dir, args.lr_phase2, args.epochs_phase2,
                                        min(2, args.epochs_phase2 // 3), patience=4),
              verbose=1)

    final_path = args.model_dir / "final_model.keras"
    model.save(final_path)
    LOGGER.info("Saved: %s", final_path)

    # Per-class threshold tuning
    thresholds: Optional[Dict] = None
    if args.tune_thresholds:
        tune_ds = build_dataset(val_paths, y_val, args.batch_size, args.image_size, training=False)
        thresholds = tune_thresholds(model, tune_ds, y_val, DISEASE_CLASSES)
        thr_path = args.model_dir / "thresholds.json"
        with open(thr_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        LOGGER.info("Saved tuned thresholds: %s", thr_path)

    save_training_summary(args.model_dir / "training_summary.json", args,
                          len(train_paths), len(val_paths), thresholds)


if __name__ == "__main__":
    main()