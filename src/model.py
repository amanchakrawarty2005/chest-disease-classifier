"""Model definition for chest disease multi-label classification."""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    MODEL_NAME,
    BACKBONE_TRAINABLE_LAYERS,
)


def _build_backbone(input_shape):
    if MODEL_NAME != "EfficientNetB0":
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    return backbone


def build_model(freeze_backbone=True):
    """Build EfficientNetB0-based multi-label classifier."""
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    inputs = layers.Input(shape=input_shape, name="image")

    backbone = _build_backbone(input_shape)
    backbone.trainable = not freeze_backbone

    x = backbone(inputs, training=not freeze_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="chest_disease_classifier")
    return model


def unfreeze_last_layers(model):
    """Unfreeze the last N layers of the backbone for fine-tuning."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("Backbone not found in model.")

    for layer in backbone.layers:
        layer.trainable = False

    if BACKBONE_TRAINABLE_LAYERS > 0:
        for layer in backbone.layers[-BACKBONE_TRAINABLE_LAYERS:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    return model
