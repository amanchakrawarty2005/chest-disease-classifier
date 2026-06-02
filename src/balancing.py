"""
Data balancing strategies for handling class imbalance in multi-label classification.

Implements:
- Class weight computation (inverse-frequency and focal-loss based)
- Stratified sampling with minority class preservation
- Oversampling for minority classes
- Class balance analysis and reporting
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("balancing")


def compute_balanced_class_weights(
    train_labels: np.ndarray,
    class_names: List[str],
    method: str = "inverse_frequency",
    smoothing: float = 1.0,
) -> Dict[str, float]:
    """
    Compute class weights to handle multi-label class imbalance.
    
    Args:
        train_labels: (n_samples, n_classes) binary label matrix
        class_names: List of disease class names
        method: "inverse_frequency", "effective_num", or "focal_loss"
        smoothing: Smoothing parameter to avoid extreme weights
    
    Returns:
        Dictionary mapping class names to float weights
    """
    if train_labels.ndim != 2:
        raise ValueError("Expected 2D label matrix for class-weight computation.")

    total_samples, num_classes = train_labels.shape
    weights: Dict[str, float] = {}

    if method == "inverse_frequency":
        # Standard inverse-frequency weighting
        for idx, class_name in enumerate(class_names):
            positive_count = int(np.sum(train_labels[:, idx]))
            positive_count = max(positive_count, 1)
            weight = float(total_samples / positive_count)
            weights[class_name] = weight

    elif method == "effective_num":
        # Effective number of samples (handles repeated oversampling)
        beta = 0.99  # Empirical hyperparameter
        for idx, class_name in enumerate(class_names):
            positive_count = int(np.sum(train_labels[:, idx]))
            positive_count = max(positive_count, 1)
            effective_num = 1.0 - np.power(beta, positive_count)
            weight = (1.0 - beta) / max(effective_num, 1e-6)
            weights[class_name] = weight

    elif method == "focal_loss":
        # Focal loss weights (emphasizes hard examples)
        gamma = 2.0  # Focusing parameter
        for idx, class_name in enumerate(class_names):
            positive_count = int(np.sum(train_labels[:, idx]))
            positive_count = max(positive_count, 1)
            positive_ratio = positive_count / total_samples
            weight = (1.0 - positive_ratio) ** gamma / positive_ratio
            weights[class_name] = weight

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights so the average weight is 1.0 and clamp extremes
    weight_values = np.array(list(weights.values()), dtype=np.float32)
    mean_weight = float(np.mean(weight_values)) if weight_values.size else 1.0
    normalized_weights: Dict[str, float] = {}
    for class_name, weight in weights.items():
        normalized = float(weight / max(mean_weight, 1e-6))
        normalized_weights[class_name] = float(np.clip(normalized, 0.1, 10.0))

    return normalized_weights


def compute_sample_weights(
    y_train: np.ndarray,
    method: str = "inverse_frequency",
) -> np.ndarray:
    """
    Compute per-sample weights based on label composition.
    
    Emphasizes samples with rare diseases and downweights common ones.
    
    Args:
        y_train: (n_samples, n_classes) binary label matrix
        method: "inverse_frequency" or "effective_num"
    
    Returns:
        (n_samples,) weight array
    """
    total_samples, num_classes = y_train.shape
    
    # Compute class frequencies
    class_frequencies = np.mean(y_train, axis=0)
    class_frequencies = np.clip(class_frequencies, 1e-6, 1.0 - 1e-6)
    
    if method == "inverse_frequency":
        class_weights = 1.0 / class_frequencies
    elif method == "effective_num":
        # Effective number weighting
        beta = 0.99
        effective_num = 1.0 - np.power(beta, np.sum(y_train, axis=0))
        class_weights = (1.0 - beta) / np.maximum(effective_num, 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize class weights
    class_weights = class_weights / np.mean(class_weights)
    
    # Per-sample weight = average weight of positive classes
    sample_weights = np.sum(y_train * class_weights[None, :], axis=1)
    # For samples with no disease, assign weight of 1.0
    sample_weights = np.where(sample_weights > 0, sample_weights, 1.0)
    
    # Normalize to prevent extreme scaling
    sample_weights = sample_weights / np.mean(sample_weights)
    
    return sample_weights.astype(np.float32)


def get_class_balance_report(
    labels: np.ndarray,
    class_names: List[str],
    split_name: str = "dataset",
) -> Dict[str, object]:
    """
    Generate detailed class balance report.
    
    Args:
        labels: (n_samples, n_classes) binary label matrix
        class_names: List of disease class names
        split_name: Name of the split (train/val/test)
    
    Returns:
        Dictionary with balance statistics
    """
    class_counts = np.sum(labels, axis=0)
    total_samples = labels.shape[0]
    
    # Statistics
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / max(min_count, 1)
    
    class_distribution = {}
    for idx, class_name in enumerate(class_names):
        count = int(class_counts[idx])
        ratio = count / total_samples * 100
        class_distribution[class_name] = {
            "count": count,
            "percentage": round(ratio, 2),
        }
    
    # Healthy samples (no disease)
    healthy_count = np.sum(np.all(labels == 0, axis=1))
    diseased_count = total_samples - healthy_count
    
    report = {
        "split": split_name,
        "total_samples": total_samples,
        "total_classes": len(class_names),
        "healthy_samples": int(healthy_count),
        "diseased_samples": int(diseased_count),
        "healthy_percentage": round(healthy_count / total_samples * 100, 2),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "class_distribution": class_distribution,
    }
    
    return report


def oversample_minority_classes(
    df: pd.DataFrame,
    labels: np.ndarray,
    class_names: List[str],
    target_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Oversample minority classes to improve balance.
    
    Creates copies of samples from rare classes to increase their representation.
    
    Args:
        df: DataFrame with sample metadata (must have index matching labels)
        labels: (n_samples, n_classes) binary label matrix
        class_names: List of disease class names
        target_ratio: Target ratio of minority to majority class
        seed: Random seed
    
    Returns:
        Oversampled (df, labels) tuple
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    
    class_counts = np.sum(labels, axis=0)
    max_count = class_counts.max()
    target_count = int(max_count * target_ratio)
    
    indices_to_add = []
    
    for class_idx, class_name in enumerate(class_names):
        class_indices = np.where(labels[:, class_idx] == 1)[0]
        current_count = len(class_indices)

        if current_count > 0 and current_count < target_count:
            # Calculate how many samples to add
            samples_needed = target_count - current_count

            if samples_needed > 0:
                selected = rng.choice(class_indices, size=samples_needed, replace=True)
                indices_to_add.extend(selected)

    if not indices_to_add:
        LOGGER.info("No oversampling needed - classes are already balanced")
        return df, labels

    LOGGER.info("Oversampling %d duplicate records from minority classes", len(indices_to_add))

    # Append oversampled data and shuffle the resulting training set
    oversampled_df = pd.concat([df, df.iloc[indices_to_add]], ignore_index=True)
    oversampled_labels = np.vstack([labels, labels[indices_to_add]])

    shuffled_indices = np.arange(len(oversampled_df))
    rng.shuffle(shuffled_indices)
    oversampled_df = oversampled_df.iloc[shuffled_indices].reset_index(drop=True)
    oversampled_labels = oversampled_labels[shuffled_indices]

    return oversampled_df, oversampled_labels


def stratified_split_by_disease(
    df: pd.DataFrame,
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    minority_threshold: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data with stratification on minority classes.
    
    Ensures all classes, especially rare ones, are represented in each split.
    
    Args:
        df: DataFrame with samples
        labels: (n_samples, n_classes) binary label matrix
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        minority_threshold: Classes with < this ratio are treated as minority
        seed: Random seed
    
    Returns:
        (train_df, val_df, test_df, train_labels, val_labels, test_labels)
    """
    rng = np.random.RandomState(seed)
    
    # Identify minority classes
    class_ratios = np.mean(labels, axis=0)
    minority_classes = np.where(class_ratios < minority_threshold)[0]
    
    LOGGER.info("Identified %d minority classes", len(minority_classes))
    
    # For each minority class, ensure balanced representation
    train_indices = []
    val_indices = []
    test_indices = []
    
    all_indices = set(range(len(df)))
    
    # First, handle minority classes separately
    for class_idx in minority_classes:
        class_indices = np.where(labels[:, class_idx] == 1)[0]
        rng.shuffle(class_indices)
        
        n_train = max(1, int(len(class_indices) * train_ratio))
        n_val = max(1, int(len(class_indices) * val_ratio))
        
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])
        
        # Remove from all_indices
        for idx in class_indices:
            all_indices.discard(idx)
    
    # For remaining samples, random split
    remaining = np.array(list(all_indices))
    rng.shuffle(remaining)
    
    n_train = int(len(remaining) * train_ratio)
    n_val = int(len(remaining) * val_ratio)
    
    train_indices.extend(remaining[:n_train])
    val_indices.extend(remaining[n_train:n_train + n_val])
    test_indices.extend(remaining[n_train + n_val:])
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[val_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True),
        labels[train_indices],
        labels[val_indices],
        labels[test_indices],
    )


def save_balance_report(
    report: Dict[str, object],
    output_path: Path,
) -> None:
    """Save class balance report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOGGER.info("Saved balance report: %s", output_path)
