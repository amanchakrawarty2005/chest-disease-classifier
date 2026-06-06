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
    Compute class weights to counteract imbalance.
    
    Methods:
    - inverse_frequency: Simple 1/frequency weighting (most effective for high imbalance)
    - effective_num: Per-class effective sample number (less aggressive)
    - focal_loss: (1-p)^gamma / p weighting (good for hard negatives)
    - sqrt_inverse: sqrt(1/frequency) - balanced between aggressive and conservative
    """
    if train_labels.ndim != 2:
        raise ValueError("Expected 2D label matrix for class-weight computation.")

    total_samples, num_classes = train_labels.shape
    weights: Dict[str, float] = {}

    if method == "inverse_frequency":
        for idx, class_name in enumerate(class_names):
            positive_count = int(np.sum(train_labels[:, idx]))
            positive_count = max(positive_count, 1)
            weight = float(total_samples / positive_count)
            weights[class_name] = weight

    elif method == "sqrt_inverse":
        # Middle ground: sqrt of inverse frequency
        # Less aggressive than pure inverse but more balanced than effective_num
        for idx, class_name in enumerate(class_names):
            positive_count = int(np.sum(train_labels[:, idx]))
            positive_count = max(positive_count, 1)
            weight = np.sqrt(float(total_samples / positive_count))
            weights[class_name] = weight

    elif method == "effective_num":
        beta = 0.99
        for idx, class_name in enumerate(class_names):
            positive_count = int(np.sum(train_labels[:, idx]))
            positive_count = max(positive_count, 1)
            effective_num = 1.0 - np.power(beta, positive_count)
            weight = (1.0 - beta) / max(effective_num, 1e-6)
            weights[class_name] = weight

    elif method == "focal_loss":
        gamma = 2.0
        for idx, class_name in enumerate(class_names):
            positive_count = int(np.sum(train_labels[:, idx]))
            positive_count = max(positive_count, 1)
            positive_ratio = positive_count / total_samples
            weight = (1.0 - positive_ratio) ** gamma / positive_ratio
            weights[class_name] = weight

    else:
        raise ValueError(f"Unknown method: {method}")

    weight_values = np.array(list(weights.values()), dtype=np.float32)
    mean_weight = float(np.mean(weight_values)) if weight_values.size else 1.0
    normalized_weights: Dict[str, float] = {}
    for class_name, weight in weights.items():
        normalized = float(weight / max(mean_weight, 1e-6))
        # Allow wider range for rare classes
        normalized_weights[class_name] = float(np.clip(normalized, 0.1, 15.0))

    return normalized_weights


def compute_sample_weights(
    y_train: np.ndarray,
    method: str = "inverse_frequency",
) -> np.ndarray:
    total_samples, num_classes = y_train.shape
    
    class_frequencies = np.mean(y_train, axis=0)
    class_frequencies = np.clip(class_frequencies, 1e-6, 1.0 - 1e-6)
    
    if method == "inverse_frequency":
        class_weights = 1.0 / class_frequencies
    elif method == "effective_num":
        beta = 0.99
        effective_num = 1.0 - np.power(beta, np.sum(y_train, axis=0))
        class_weights = (1.0 - beta) / np.maximum(effective_num, 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    class_weights = class_weights / np.mean(class_weights)
    
    sample_weights = np.sum(y_train * class_weights[None, :], axis=1)
    sample_weights = np.where(sample_weights > 0, sample_weights, 1.0)
    
    sample_weights = sample_weights / np.mean(sample_weights)
    
    return sample_weights.astype(np.float32)


def get_class_balance_report(
    labels: np.ndarray,
    class_names: List[str],
    split_name: str = "dataset",
) -> Dict[str, object]:
    class_counts = np.sum(labels, axis=0)
    total_samples = labels.shape[0]
    
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
    Aggressive oversampling for minority classes to achieve true balance.
    
    Strategy:
    - Calculate target count per class to equalize representation
    - Oversample rare classes (especially Cardiomegaly, Hernia) to match most common
    - Apply stronger oversampling for classes < 5% baseline frequency
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    
    class_counts = np.sum(labels, axis=0)
    max_count = class_counts.max()
    
    # AGGRESSIVE OVERSAMPLING: target ratio should be 1.0 to balance all classes
    target_count = int(max_count * target_ratio)
    
    indices_to_add = []
    class_oversampling_stats = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_indices = np.where(labels[:, class_idx] == 1)[0]
        current_count = len(class_indices)
        class_freq = current_count / len(labels) * 100

        if current_count > 0 and current_count < target_count:
            samples_needed = target_count - current_count

            if samples_needed > 0:
                # For very rare classes, allow repetition with replacement
                selected = rng.choice(class_indices, size=samples_needed, replace=True)
                indices_to_add.extend(selected)
                
                class_oversampling_stats[class_name] = {
                    "original_count": int(current_count),
                    "target_count": int(target_count),
                    "oversampled": int(samples_needed),
                    "frequency_pct": round(class_freq, 2)
                }

    if not indices_to_add:
        LOGGER.info("No oversampling needed - classes are already balanced")
        return df, labels

    LOGGER.info("Aggressive oversampling: %d total duplicates from %d classes", 
                len(indices_to_add), len(class_oversampling_stats))
    for cls_name, stats in class_oversampling_stats.items():
        LOGGER.info("  %s: %d → %d (+%d samples, freq: %.2f%%)", 
                    cls_name, stats["original_count"], stats["target_count"], 
                    stats["oversampled"], stats["frequency_pct"])

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
    rng = np.random.RandomState(seed)
    
    class_ratios = np.mean(labels, axis=0)
    minority_classes = np.where(class_ratios < minority_threshold)[0]
    
    LOGGER.info("Identified %d minority classes", len(minority_classes))
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    all_indices = set(range(len(df)))
    
    for class_idx in minority_classes:
        class_indices = np.where(labels[:, class_idx] == 1)[0]
        rng.shuffle(class_indices)
        
        n_train = max(1, int(len(class_indices) * train_ratio))
        n_val = max(1, int(len(class_indices) * val_ratio))
        
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])
        
        for idx in class_indices:
            all_indices.discard(idx)
    
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOGGER.info("Saved balance report: %s", output_path)