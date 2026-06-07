# Model Serialization Guide

## Overview

This project saves trained models in the **Keras format (.keras)** with support for custom objects through registration. The Keras format provides a clean, reliable way to save models with custom loss functions and callbacks.

## File Formats

### Keras Format (`.keras`)

Saves complete model including architecture, weights, and optimizer state:
- ✅ Self-contained (no need to rebuild architecture)
- ✅ Supports custom objects via registration
- ✅ Includes training checkpoints for resuming

**Files:**
- `saved_model/final_model.keras` - Final trained model after all phases
- `saved_model/best_model.keras` - Best checkpoint during training (by validation AUC)

## Usage

### Training

Training automatically saves models in Keras format:

```python
python src/train.py
# Creates: saved_model/best_model.keras, saved_model/final_model.keras
```

### Loading for Inference

All scripts automatically register custom objects when loading:

```python
from model import focal_binary_crossentropy

# Custom objects for deserialization
custom_objects = {
    "focal_bce_gamma2.0": focal_binary_crossentropy(gamma=2.0),
    "focal_bce_gamma2": focal_binary_crossentropy(gamma=2.0),
}

# Load model
model = tf.keras.models.load_model("saved_model/final_model.keras", custom_objects=custom_objects)

# Model is ready for inference
predictions = model.predict(images)
```

### API/Web Service

The inference service in `api/inference.py` automatically handles custom object registration:

```python
service = ChestXrayInferenceService(
    model_dir=Path("saved_model"),
    image_size=224,
    disease_classes=DISEASE_CLASSES
)
service.load_model()  # Automatically registers and loads with custom objects
```

### Evaluation

```bash
python src/evaluate.py
# Automatically registers custom objects when loading model
```

### GradCAM Visualization

```bash
python src/gradcam.py
# Automatically registers custom objects when loading model
```

## Implementation Details

### Model Architecture

The `build_model()` function in `src/model.py` defines the complete architecture:
- EfficientNetB0 backbone (pre-trained on ImageNet)
- Custom classification head with BatchNorm and Dropout
- Multi-label output layer (14 disease classes)

### Custom Loss Function

We use focal loss for handling class imbalance:

```python
from model import focal_binary_crossentropy

loss_fn = focal_binary_crossentropy(gamma=2.0)
```

This custom loss is automatically registered when loading models through the `custom_objects` dictionary.

### Saving with Custom Objects

When saving the model, Keras automatically serializes custom functions by name and definition. To ensure proper deserialization:

1. **Loss function names** are preserved: `focal_bce_gamma{gamma}`
2. **Custom objects dict** registers these functions during loading
3. No additional configuration needed beyond passing `custom_objects` to `load_model()`

## Migration from Legacy Format

If you have existing weights-only files and want to convert to full Keras format:

```python
import tensorflow as tf
from pathlib import Path
import sys
sys.path.insert(0, "src")

from model import build_model

# Rebuild architecture
model = build_model(freeze_backbone=False)

# Load weights
model.load_weights("saved_model/final_model.weights.h5")

# Save as keras
model.save("saved_model/final_model.keras")
```

## Troubleshooting

### Error: "Unknown loss function"

This occurs when custom objects aren't registered. Solution:
- Always pass `custom_objects` when loading:
```python
from model import focal_binary_crossentropy

custom_objects = {
    "focal_bce_gamma2.0": focal_binary_crossentropy(gamma=2.0),
    "focal_bce_gamma2": focal_binary_crossentropy(gamma=2.0),
}
model = tf.keras.models.load_model("model.keras", custom_objects=custom_objects)
```

### Error: "No trained model found"

The loading functions search for these files in order:
- `final_model.keras`
- `best_model.keras`

Ensure at least one exists in your model directory.

### Error: "axes don't match array" during weight loading

This indicates shape incompatibility between model architecture and saved weights. This happens because:
- Model architecture changed between saving and loading
- Checkpoint format mismatch (Keras 2 vs 3)
- Layer reordering or modification

**Solution:** Use full `.keras` files instead of weights-only formats, as they preserve the exact architecture.
