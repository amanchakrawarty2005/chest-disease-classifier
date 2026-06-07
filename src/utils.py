"""Shared utility functions used across the project."""
from __future__ import annotations

import ast
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf


def parse_label_cell(cell) -> np.ndarray:
    """
    Parse label cell from CSV (can be string representation or list).
    
    Args:
        cell: Label cell value (string or list)
        
    Returns:
        NumPy array of float32 labels
    """
    if isinstance(cell, str):
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    return np.array(cell, dtype=np.float32)


def parse_finding_labels(raw_label: str) -> List[str]:
    """
    Parse pipe-separated disease labels from raw CSV.
    
    Args:
        raw_label: Raw label string (e.g., "Cardiomegaly|Effusion")
        
    Returns:
        List of disease names
    """
    if pd.isna(raw_label) or str(raw_label).strip() == "No Finding":
        return []
    return [item.strip() for item in str(raw_label).split("|") if item.strip()]


def preprocess_image_tensor(image_bytes: bytes, image_size: int) -> tf.Tensor:
    """
    Decode and preprocess image bytes to model-ready tensor.
    
    This function implements the standard preprocessing pipeline:
    - Decode grayscale image
    - Resize to target size
    - Normalize to [0, 1]
    - Convert grayscale to RGB (repeat channel 3 times)
    
    Args:
        image_bytes: Raw image bytes
        image_size: Target image size (width and height)
        
    Returns:
        Preprocessed image tensor of shape (image_size, image_size, 3)
        
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc
    
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.grayscale_to_rgb(image)
    image.set_shape([image_size, image_size, 3])
    return image


def preprocess_image_for_inference(image_bytes: bytes, image_size: int) -> np.ndarray:
    """
    Preprocess image bytes for inference (adds batch dimension).
    
    Args:
        image_bytes: Raw image bytes
        image_size: Target image size
        
    Returns:
        Batched image array of shape (1, image_size, image_size, 3)
    """
    image = preprocess_image_tensor(image_bytes, image_size)
    return tf.expand_dims(image, axis=0).numpy()


def validate_image_path(path: str) -> bool:
    """
    Validate that image path exists and is not empty.
    
    Args:
        path: Image file path
        
    Returns:
        True if path is valid, False otherwise
    """
    try:
        from pathlib import Path
        p = Path(path)
        return p.exists() and p.is_file() and len(str(path)) > 0
    except Exception:
        return False
