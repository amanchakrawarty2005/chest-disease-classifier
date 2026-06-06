import math
import tensorflow as tf
layers, models = tf.keras.layers, tf.keras.models

from config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    MODEL_NAME,
    BACKBONE_TRAINABLE_LAYERS,
    FOCAL_GAMMA,
)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_binary_crossentropy(gamma: float = 2.0):
    """
    Per-class focal BCE for multi-label classification.

    Down-weights easy negatives and focuses training on hard/rare positives.
    For a Cardiomegaly sample with predicted p=0.12 (true label=1):
      - focal weight = (1-0.12)^2 = 0.77  -> 77% of full loss signal preserved
    For an easy No-Finding negative with predicted p=0.02 (true label=0):
      - focal weight = (1-0.98)^2 = 0.0004 -> nearly zeroed out
    This forces the model to focus on rare disease patterns.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)

    loss_fn.__name__ = f"focal_bce_gamma{gamma}"
    return loss_fn


# ---------------------------------------------------------------------------
# LR warmup + cosine decay callback
# ---------------------------------------------------------------------------

class WarmupCosineDecay(tf.keras.callbacks.Callback):
    """Linear warmup for warmup_epochs, then cosine decay to min_lr."""

    def __init__(self, base_lr: float, total_epochs: int, warmup_epochs: int = 3, min_lr: float = 1e-7):
        super().__init__()
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch: int, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        # .assign() works across Keras 2 and Keras 3 (tf.keras.backend.set_value removed in K3)
        opt_lr = self.model.optimizer.learning_rate
        if hasattr(opt_lr, "assign"):
            opt_lr.assign(float(lr))
        else:
            tf.keras.backend.set_value(opt_lr, float(lr))


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_backbone(input_shape):
    if MODEL_NAME != "EfficientNetB0":
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
    return tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )


def build_model(freeze_backbone: bool = True) -> tf.keras.Model:
    """
    EfficientNetB0-based multi-label classifier.

    Head improvements over baseline:
    - BatchNorm after GAP stabilises training with imbalanced batches
    - Two dense blocks (512->256) give more capacity for disease-specific patterns
    """
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    inputs = layers.Input(shape=input_shape, name="image")

    backbone = _build_backbone(input_shape)
    backbone.trainable = not freeze_backbone

    x = backbone(inputs, training=not freeze_backbone)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.2, name="drop3")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid", name="predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="chest_disease_classifier")


def unfreeze_last_layers(model: tf.keras.Model) -> tf.keras.Model:
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