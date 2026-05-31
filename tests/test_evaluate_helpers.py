import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import DISEASE_CLASSES, evaluate_predictions  # noqa: E402


def test_evaluate_predictions_returns_expected_shape_and_keys():
    n_classes = len(DISEASE_CLASSES)

    y_true = np.zeros((4, n_classes), dtype=np.float32)
    y_prob = np.zeros((4, n_classes), dtype=np.float32)

    # Add signal to first 2 classes so AUC/AP are computable for those.
    y_true[:, 0] = np.array([1, 0, 1, 0], dtype=np.float32)
    y_true[:, 1] = np.array([0, 1, 0, 1], dtype=np.float32)
    y_prob[:, 0] = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    y_prob[:, 1] = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float32)

    report = evaluate_predictions(y_true, y_prob, threshold=0.5)

    assert "metrics" in report
    assert "per_class_auc" in report
    assert "per_class_average_precision" in report
    assert report["threshold"] == 0.5

    metrics = report["metrics"]
    assert "micro_auc" in metrics
    assert "macro_auc" in metrics
    assert "subset_accuracy" in metrics
    assert "hamming_accuracy" in metrics

    # At least classes with real positives should be evaluated.
    assert report["per_class_auc"][DISEASE_CLASSES[0]] is not None
    assert report["per_class_auc"][DISEASE_CLASSES[1]] is not None
