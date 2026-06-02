from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import DISEASE_CLASSES, IMAGE_SIZE, MODEL_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

LOGGER = logging.getLogger("gradcam")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_label_cell(cell) -> np.ndarray:
    if isinstance(cell, str):
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    return np.array(cell, dtype=np.float32)


def load_test_dataframe(processed_data_dir: Path) -> pd.DataFrame:
    csv_path = processed_data_dir / "test_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing test split file: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"image_path", "labels"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Invalid test split format in {csv_path}. Missing: {sorted(missing)}")

    df["labels_arr"] = df["labels"].apply(parse_label_cell)
    df = df[df["image_path"].apply(lambda p: Path(str(p)).exists())].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No valid test images found for Grad-CAM.")

    return df


def load_trained_model(model_dir: Path) -> tf.keras.Model:
    final_path = model_dir / "final_model.keras"
    best_path = model_dir / "best_model.keras"

    if final_path.exists():
        return tf.keras.models.load_model(final_path)
    if best_path.exists():
        return tf.keras.models.load_model(best_path)

    raise FileNotFoundError("No trained model found in saved_model/.")


def preprocess_image(image_path: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to read image: {image_path}")

    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = gray.astype(np.float32) / 255.0
    rgb = np.stack([normalized, normalized, normalized], axis=-1)
    return gray, rgb


def find_backbone_model(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise RuntimeError("Backbone model not found in trained model.")


def find_last_conv_layer(backbone: tf.keras.Model) -> tf.keras.layers.Layer:
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    def search(layers_list):
        for layer in reversed(layers_list):
            if isinstance(layer, tf.keras.Model):
                nested = search(layer.layers)
                if nested is not None:
                    return nested
            if isinstance(layer, conv_types):
                return layer
        return None

    layer = search(backbone.layers)
    if layer is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")
    return layer


def build_head_layers(model: tf.keras.Model, backbone: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    backbone_index = model.layers.index(backbone)
    return model.layers[backbone_index + 1 :]


def make_gradcam_heatmap(
    input_tensor: np.ndarray,
    backbone: tf.keras.Model,
    head_layers: List[tf.keras.layers.Layer],
    last_conv_layer_name: str,
    class_index: int,
) -> np.ndarray:
    conv_and_features_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, features = conv_and_features_model(input_tensor, training=False)
        x = features
        for layer in head_layers:
            x = layer(x, training=False)
        preds = x
        target_score = preds[:, class_index]

    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(gray_image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    heatmap_u8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray_bgr, 1 - alpha, colored_heatmap, alpha, 0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for test samples.")
    parser.add_argument("--processed-data-dir", type=Path, default=Path(PROCESSED_DATA_DIR), help="Directory containing test_labels.csv")
    parser.add_argument("--model-dir", type=Path, default=Path(MODEL_DIR), help="Directory containing trained model")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional explicit output folder")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of random test images to visualize")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Image size for Grad-CAM generation")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay opacity")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for sample selection")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    output_dir = args.output_dir or (args.processed_data_dir / "gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_test_dataframe(args.processed_data_dir)
    model = load_trained_model(args.model_dir)
    backbone = find_backbone_model(model)
    last_conv_layer = find_last_conv_layer(backbone)
    head_layers = build_head_layers(model, backbone)

    LOGGER.info("Using last conv layer: %s", last_conv_layer.name)

    sample_count = min(args.num_samples, len(df))
    sampled_df = df.sample(n=sample_count, random_state=args.seed)

    for idx, row in sampled_df.iterrows():
        image_path = Path(row["image_path"])

        gray_image, rgb_image = preprocess_image(image_path, args.image_size)
        input_tensor = np.expand_dims(rgb_image, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[0]
        top_class_idx = int(np.argmax(prediction))
        top_class_name = DISEASE_CLASSES[top_class_idx]
        top_class_score = float(prediction[top_class_idx])

        heatmap = make_gradcam_heatmap(
            input_tensor=input_tensor,
            backbone=backbone,
            head_layers=head_layers,
            last_conv_layer_name=last_conv_layer.name,
            class_index=top_class_idx,
        )
        heatmap = cv2.resize(heatmap, (args.image_size, args.image_size))
        overlay = overlay_heatmap(gray_image, heatmap, alpha=args.alpha)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gray_image, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Overlay\n{top_class_name}: {top_class_score:.3f}")
        axes[2].axis("off")

        plt.tight_layout()
        file_name = f"gradcam_{idx}_{image_path.stem}.png"
        save_path = output_dir / file_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    LOGGER.info("Grad-CAM complete. Saved outputs to: %s", output_dir)


if __name__ == "__main__":
    main()