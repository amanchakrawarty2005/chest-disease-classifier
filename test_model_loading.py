#!/usr/bin/env python
"""Test script to verify model loading works correctly."""

import sys
import traceback
from pathlib import Path

import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DISEASE_CLASSES, IMAGE_SIZE

def test_keras_model():
    """Test loading the full keras model."""
    print("\n=== Testing Full Keras Model Loading ===")
    keras_path = Path("saved_model/final_model.keras")
    print(f"Loading from: {keras_path}")
    try:
        model = tf.keras.models.load_model(str(keras_path), compile=False)
        print(f"✓ Full keras model loads successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"✗ Error loading keras model: {e}")
        traceback.print_exc()
        return None

def test_weights_loading():
    """Test loading from weights file."""
    print("\n=== Testing Weights File Loading ===")
    from model import load_model_from_weights
    
    weights_path = Path("saved_model/final_model.weights.h5")
    print(f"Loading from: {weights_path}")
    try:
        model = load_model_from_weights(str(weights_path), freeze_backbone=False)
        print(f"✓ Weights model loads successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"✗ Error loading weights model: {e}")
        traceback.print_exc()
        return None

def test_inference_service():
    """Test inference service loading."""
    print("\n=== Testing Inference Service ===")
    from api.inference import ChestXrayInferenceService
    
    try:
        service = ChestXrayInferenceService(
            model_dir=Path("saved_model"),
            image_size=IMAGE_SIZE,
            disease_classes=DISEASE_CLASSES
        )
        path = service.load_model()
        print(f"✓ Inference service loads model successfully!")
        print(f"  Loaded from: {path}")
        print(f"  Model type: {type(service._model)}")
        return service
    except Exception as e:
        print(f"✗ Error loading inference service: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Testing Model Loading Mechanisms...")
    
    # Test 1: Full keras model
    keras_model = test_keras_model()
    
    # Test 2: Weights loading
    weights_model = test_weights_loading()
    
    # Test 3: Inference service
    service = test_inference_service()
    
    print("\n=== Summary ===")
    print(f"Keras model: {'✓' if keras_model else '✗'}")
    print(f"Weights model: {'✓' if weights_model else '✗'}")
    print(f"Inference service: {'✓' if service else '✗'}")
