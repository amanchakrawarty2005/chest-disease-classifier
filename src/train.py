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
    IMAGE_SIZE,
    LEARNING_RATE_PHASE1,
    LEARNING_RATE_PHASE2,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from balancing import compute_sample_weights
from model import build_model, unfreeze_last_layers

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

    valid_mask = np.array([Path(path).exists() and len(path) > 0 for path in paths])
    filtered_paths = paths[valid_mask]
    filtered_labels = labels[valid_mask]

    if len(filtered_paths) == 0:
        raise RuntimeError(f"No valid image paths found for split: {split_name}")

    LOGGER.info("Loaded %s split with %s usable samples", split_name, f"{len(filtered_paths):,}")
    return filtered_paths, filtered_labels


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
    if sample_weights is None:
        sample_weights = np.ones((len(image_paths),), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, sample_weights))

    if training:
        ds = ds.shuffle(min(len(image_paths), 20000), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y, w: decode_image(p, y, w, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_callbacks(model_dir: Path, patience: int = 4):
    model_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
    ]


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )


def build_sample_weights(y_train: np.ndarray, method: str = "inverse_frequency") -> np.ndarray:
    return compute_sample_weights(y_train, method=method)


def load_class_weights(processed_data_dir: Path, method: str = "recommended") -> Optional[Dict[str, float]]:
    weights_path = processed_data_dir / "class_weights.json"
    
    if not weights_path.exists():
        LOGGER.warning("Class weights file not found: %s", weights_path)
        return None
    
    try:
        with open(weights_path, "r", encoding="utf-8") as f:
            all_weights = json.load(f)
        
        if isinstance(all_weights, dict):
            if method in all_weights:
                weights = all_weights[method]
                LOGGER.info("Loaded %s class weights from %s", method, weights_path)
                return weights
            elif "recommended" in all_weights:
                weights = all_weights["recommended"]
                LOGGER.info("Loaded recommended class weights (fallback from requested method '%s')", method)
                return weights
            else:
                LOGGER.info("Loaded class weights (legacy format)")
                return all_weights
        
        return None
    except Exception as exc:
        LOGGER.error("Failed to load class weights: %s", exc)
        return None


def save_training_summary(summary_path: Path, args: argparse.Namespace, train_size: int, val_size: int) -> None:
    payload = {
        "train_samples": int(train_size),
        "val_samples": int(val_size),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "epochs_phase1": int(args.epochs_phase1),
        "epochs_phase2": int(args.epochs_phase2),
        "lr_phase1": float(args.lr_phase1),
        "lr_phase2": float(args.lr_phase2),
        "sample_weighting_method": str(args.sample_weighting),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train chest disease classifier with 2-stage fine-tuning.")
    parser.add_argument("--processed-data-dir", type=Path, default=Path(PROCESSED_DATA_DIR), help="Directory containing train_labels.csv and val_labels.csv")
    parser.add_argument("--model-dir", type=Path, default=Path(MODEL_DIR), help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Input image size")
    parser.add_argument("--epochs-phase1", type=int, default=EPOCHS_PHASE1, help="Epochs for frozen-backbone stage")
    parser.add_argument("--epochs-phase2", type=int, default=EPOCHS_PHASE2, help="Epochs for fine-tuning stage")
    parser.add_argument("--lr-phase1", type=float, default=LEARNING_RATE_PHASE1, help="Learning rate for stage 1")
    parser.add_argument("--lr-phase2", type=float, default=LEARNING_RATE_PHASE2, help="Learning rate for stage 2")
    parser.add_argument("--sample-weighting", type=str, default="inverse_frequency", choices=["effective_num", "inverse_frequency"], help="Sample weighting method for imbalanced data")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick smoke runs")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap for quick smoke runs")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    tf.keras.utils.set_random_seed(args.seed)

    LOGGER.info("Loading processed splits...")
    train_paths, y_train = load_split_csv(
        "train",
        processed_data_dir=args.processed_data_dir,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_paths, y_val = load_split_csv(
        "val",
        processed_data_dir=args.processed_data_dir,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )

    LOGGER.info("Computing sample weights using '%s' method...", args.sample_weighting)
    train_sample_weights = build_sample_weights(y_train, method=args.sample_weighting)

    train_ds = build_dataset(
        train_paths,
        y_train,
        batch_size=args.batch_size,
        image_size=args.image_size,
        training=True,
        sample_weights=train_sample_weights,
    )
    val_ds = build_dataset(
        val_paths,
        y_val,
        batch_size=args.batch_size,
        image_size=args.image_size,
        training=False,
    )

    callbacks = build_callbacks(args.model_dir)

    LOGGER.info("Phase 2.1 | Training classifier head (frozen backbone)")
    model = build_model(freeze_backbone=True)
    compile_model(model, args.lr_phase1)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_phase1, callbacks=callbacks, verbose=1)

    LOGGER.info("Phase 2.2 | Fine-tuning last backbone layers")
    model = unfreeze_last_layers(model)
    compile_model(model, args.lr_phase2)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_phase2, callbacks=callbacks, verbose=1)

    final_path = args.model_dir / "final_model.keras"
    model.save(final_path)
    LOGGER.info("Training complete. Saved final model: %s", final_path)

    summary_path = args.model_dir / "training_summary.json"
    save_training_summary(summary_path, args, train_size=len(train_paths), val_size=len(val_paths))
    LOGGER.info("Saved training summary: %s", summary_path)


if __name__ == "__main__":
    main()