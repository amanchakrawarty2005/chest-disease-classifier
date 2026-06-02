import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from balancing import oversample_minority_classes
from data_preprocessing import (  # noqa: E402
    compute_class_weights,
    parse_finding_labels,
    safe_stratify_target,
    validate_split_ratios,
)


def test_parse_finding_labels_handles_no_finding():
    assert parse_finding_labels("No Finding") == []
    assert parse_finding_labels(np.nan) == []


def test_parse_finding_labels_splits_pipe_separated_values():
    labels = parse_finding_labels("Mass|Nodule| Effusion ")
    assert labels == ["Mass", "Nodule", "Effusion"]


def test_validate_split_ratios_rejects_invalid_total():
    with pytest.raises(ValueError):
        validate_split_ratios(0.7, 0.2, 0.2)


def test_safe_stratify_target_returns_none_for_rare_bucket():
    series = pd.Series([0, 0, 1])
    stratify_target = safe_stratify_target(series, stage_name="test-stage", min_count=2)
    assert stratify_target is None


def test_compute_class_weights_produces_inverse_frequency_behavior():
    # class0: 2 positives, class1: 1 positive, class2: 0 positives
    labels = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    class_names = ["A", "B", "C"]
    weights = compute_class_weights(labels, class_names)

    assert set(weights.keys()) == set(class_names)
    assert weights["B"] > weights["A"]  # rarer positive class gets higher weight
    assert weights["C"] > 0  # zero-positive class still has finite fallback weight


def test_oversample_minority_classes_balances_rare_labels():
    labels = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    df = pd.DataFrame({"Image Index": ["a", "b", "c", "d"]})
    oversampled_df, oversampled_labels = oversample_minority_classes(
        df,
        labels,
        ["A", "B", "C"],
        target_ratio=1.0,
        seed=123,
    )

    counts = np.sum(oversampled_labels, axis=0)
    assert counts[0] == counts[1] == counts[2]
    assert len(oversampled_df) == len(oversampled_labels)
    assert len(oversampled_df) >= len(df)
