"""Grad-CAM generation script for chest disease classifier."""

import ast
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import PROCESSED_DATA_DIR, MODEL_DIR, IMAGE_SIZE, DISEASE_CLASSES


NUM_SAMPLES = 12


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

    df["labels_arr"] = df["labels"].apply(_parse_label_cell)
    df = df[df["image_path"].apply(lambda p: Path(str(p)).exists())]

    if len(df) == 0:
        raise RuntimeError("No valid test images found for Grad-CAM.")

    return df.reset_index(drop=True)


def _load_model():
    model_dir = Path(MODEL_DIR)
    final_path = model_dir / "final_model.keras"
    best_path = model_dir / "best_model.keras"

    if final_path.exists():
        return tf.keras.models.load_model(final_path)
    if best_path.exists():
        return tf.keras.models.load_model(best_path)

    raise FileNotFoundError("No trained model found in saved_model/.")


def _preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    return img_resized, img_rgb


def _find_last_conv_layer(backbone):
    """Find the last convolutional-like layer, including inside nested models."""
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    def _search(layers_list):
        for layer in reversed(layers_list):
            if isinstance(layer, tf.keras.Model):
                nested = _search(layer.layers)
                if nested is not None:
                    return nested
            if isinstance(layer, conv_types):
                return layer
        return None

    last_conv = _search(backbone.layers)
    if last_conv is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")
    return last_conv


def _make_gradcam_heatmap(image_tensor, backbone, head_layers, last_conv_layer_name, class_idx):
    conv_backbone_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, x = conv_backbone_model(image_tensor, training=False)
        for layer in head_layers:
            x = layer(x, training=False)
        preds = x
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap(gray_img, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    gray_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_3ch, 1 - alpha, color_map, alpha, 0)
    return overlay


def main():
    out_dir = Path(PROCESSED_DATA_DIR) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_test_split()
    model = _load_model()
    backbone = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if backbone is None:
        raise RuntimeError("Backbone model not found.")

    last_conv_layer = _find_last_conv_layer(backbone)
    head_layers = model.layers[2:]  # layers after [InputLayer, backbone]
    print(f"Using last conv layer: {last_conv_layer.name}")

    df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)

    for idx, row in df.iterrows():
        image_path = Path(row["image_path"])

        gray_img, img_rgb = _preprocess_image(image_path)
        input_tensor = np.expand_dims(img_rgb, axis=0)

        pred = model.predict(input_tensor, verbose=0)[0]
        top_class = int(np.argmax(pred))
        top_score = float(pred[top_class])
        top_name = DISEASE_CLASSES[top_class]

        heatmap = _make_gradcam_heatmap(
            input_tensor, backbone, head_layers, last_conv_layer.name, top_class
        )
        heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        overlay = _overlay_heatmap(gray_img, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gray_img, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Overlay\n{top_name}: {top_score:.3f}")
        axes[2].axis("off")

        plt.tight_layout()
        out_file = out_dir / f"gradcam_{idx}_{image_path.stem}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Grad-CAM complete. Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
