"""Training script for chest disease classifier (Phase 2)."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LEARNING_RATE_PHASE1,
    LEARNING_RATE_PHASE2,
)
from model import build_model, unfreeze_last_layers


AUTOTUNE = tf.data.AUTOTUNE


def _parse_label_cell(cell):
    if isinstance(cell, str):
        arr = np.array(ast.literal_eval(cell), dtype=np.float32)
    else:
        arr = np.array(cell, dtype=np.float32)
    return arr


def _load_split_csv(split_name):
    path = Path(PROCESSED_DATA_DIR) / f"{split_name}_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    if "image_path" not in df.columns or "labels" not in df.columns:
        raise ValueError(f"Invalid split format in {path}")

    image_paths = df["image_path"].astype(str).values
    labels = np.stack(df["labels"].apply(_parse_label_cell).values).astype(np.float32)

    valid_mask = np.array([Path(p).exists() and len(p) > 0 for p in image_paths])
    image_paths = image_paths[valid_mask]
    labels = labels[valid_mask]

    return image_paths, labels


def _decode_image(path, label, sample_weight):
    bytes_ = tf.io.read_file(path)
    image = tf.io.decode_image(bytes_, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.grayscale_to_rgb(image)
    return image, label, sample_weight


def _build_dataset(image_paths, labels, sample_weights=None, training=False):
    if sample_weights is None:
        sample_weights = np.ones((len(image_paths),), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, sample_weights))

    if training:
        ds = ds.shuffle(min(len(image_paths), 20000), reshuffle_each_iteration=True)

    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def _callbacks():
    model_dir = Path(MODEL_DIR)
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
            patience=4,
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


def _compile(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )


def _sample_weights(y):
    pos_fraction = np.clip(np.mean(y, axis=0), 1e-6, 1 - 1e-6)
    class_importance = 1.0 / pos_fraction
    class_importance = class_importance / np.mean(class_importance)
    weights = np.sum(y * class_importance[None, :], axis=1)
    weights = np.where(weights > 0, weights, 1.0)
    return weights.astype(np.float32)


def main():
    print("Loading processed splits...")
    x_train_paths, y_train = _load_split_csv("train")
    x_val_paths, y_val = _load_split_csv("val")

    print(f"Train samples: {len(x_train_paths):,}")
    print(f"Val samples: {len(x_val_paths):,}")

    sample_weights_train = _sample_weights(y_train)

    train_ds = _build_dataset(x_train_paths, y_train, sample_weights_train, training=True)
    val_ds = _build_dataset(x_val_paths, y_val, sample_weights=None, training=False)

    print("Phase 2.1: Train classifier head (frozen backbone)")
    model = build_model(freeze_backbone=True)
    _compile(model, LEARNING_RATE_PHASE1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=_callbacks(),
        verbose=1,
    )

    print("Phase 2.2: Fine-tune last backbone layers")
    model = unfreeze_last_layers(model)
    _compile(model, LEARNING_RATE_PHASE2)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=_callbacks(),
        verbose=1,
    )

    final_path = Path(MODEL_DIR) / "final_model.keras"
    model.save(final_path)
    print(f"Training complete. Final model saved: {final_path}")


if __name__ == "__main__":
    main()
