# Data Balancing & Class Imbalance Handling

## Overview

The NIH Chest X-ray dataset exhibits significant class imbalance:
- **Imbalance Ratio**: ~88x (most common disease vs. rarest)
- **Healthy Samples**: ~54% of dataset (no disease)
- **Minority Classes**: Some diseases appear in <1% of samples

This document describes the comprehensive balancing strategies implemented to mitigate bias toward common diseases.

## Class Distribution

From analysis of the training data:

```
Infiltration:         17.77% (most common)
Effusion:             11.92%
Atelectasis:          10.41%
...
Pneumonia:            1.27%
Fibrosis:             1.52%
Hernia:               0.20% (rarest - 88x less common)
```

## Implemented Solutions

### 1. **Minority Class Oversampling**
- **Where**: `src/data_preprocessing.py` using `src/balancing.py`
- **How**: Duplicates samples from rare classes to increase representation
- **Target**: 25% of majority class frequency
- **Benefit**: Ensures model sees rare diseases during training

### 2. **Sample-Level Weighting**
- **Where**: `src/train.py` and `src/balancing.py`
- **Methods Available**:
  - `inverse_frequency`: Simple inverse class frequency
  - `effective_num`: Handles repeated oversampling better (recommended)
- **Benefit**: Per-sample weights emphasize rare classes without over-penalizing healthy samples

### 3. **Class-Level Weights**
- **Where**: `data/processed/class_weights.json`
- **Generated Methods**:
  - `inverse_frequency`: Weight = total_samples / (num_classes × positive_count)
  - `effective_num`: Uses effective number formula for oversampled data
  - `focal_loss`: Emphasizes hard examples for minority classes
- **Use Case**: Can be used for weighted loss functions (future enhancement)

### 4. **Aggressive Augmentation for Minority Classes**
- **Where**: `src/augmentation.py` → `ImageAugmentor.augment_image_aggressive()`
- **Techniques**:
  - Larger rotation angles (±25°)
  - Stronger brightness adjustment (0.7-1.3)
  - Stronger contrast adjustment (0.7-1.3)
  - Elastic deformation
- **Benefit**: Generates diverse training samples from limited rare disease data

### 5. **Balanced Evaluation Metrics**
- **Where**: `src/evaluate.py`
- **New Metrics**:
  - `weighted_auc`: Accounts for class frequency
  - `macro_auc`: Unweighted average across all classes
  - `minority_class_auc`: AUC for classes with <5% prevalence
- **Benefit**: Better assessment of model performance on underrepresented classes

### 6. **Class-Specific Inference Thresholds**
- **Where**: `api/inference.py`
- **How**: Uses class weights to adjust prediction thresholds
  - Rarer classes → Lower threshold → More sensitive detection
  - Common classes → Standard threshold
- **Benefit**: Better detection of rare diseases while avoiding false positives on common ones

## Configuration

### Training with Balancing

```bash
# Use effective_num weighting (recommended for balanced data)
python src/train.py --sample-weighting effective_num

# Or use inverse frequency weighting
python src/train.py --sample-weighting inverse_frequency
```

### Preprocessing with/without Oversampling

```bash
# With oversampling (default - recommended)
python src/data_preprocessing.py --apply-oversampling True

# Without oversampling (if you prefer weighted loss only)
python src/data_preprocessing.py --apply-oversampling False
```

## Generated Reports

### Training Split Reports
- `data/processed/train_balance_report.json`: Training data distribution with oversampling
- `data/processed/val_balance_report.json`: Validation data distribution
- `data/processed/test_balance_report.json`: Test data distribution

### Class Weights
- `data/processed/class_weights.json`: Multiple weight calculation methods
  ```json
  {
    "inverse_frequency": {...},
    "effective_num": {...},
    "focal_loss": {...},
    "recommended": {...}
  }
  ```

### Evaluation Report
- `data/processed/evaluation_report.json`: Enhanced with class balance information
  - Includes `class_balance` section with full distribution statistics
  - Reports `weighted_auc` and `minority_class_auc` in metrics

## Metrics Interpretation

| Metric | What It Measures | Why It Matters |
|--------|-----------------|-----------------|
| `micro_auc` | Aggregated performance across all labels | Overall quality |
| `macro_auc` | Unweighted average per-class AUC | Treats all classes equally (best for imbalance) |
| `weighted_auc` | Frequency-weighted per-class AUC | Balances between overall and per-class |
| `minority_class_auc` | AUC for rare diseases only | Ensures rare diseases aren't ignored |

## Code Examples

### Using Balancing in Custom Training

```python
from src.balancing import compute_sample_weights, compute_balanced_class_weights

# Compute per-sample weights
sample_weights = compute_sample_weights(y_train, method="effective_num")

# Compute class weights for weighted loss (future)
class_weights = compute_balanced_class_weights(y_train, DISEASE_CLASSES, method="effective_num")

# Use in model.fit()
model.fit(
    train_dataset,
    sample_weight=sample_weights,  # Per-sample weighting
    ...
)
```

### Accessing Class Weights in API

```python
from api.inference import ChestXrayInferenceService

service = ChestXrayInferenceService(...)
service.load_model()

# Access loaded weights for custom inference
if service.class_weights:
    print(f"Class weights: {service.class_weights}")
    # Weights are automatically used in format_predictions()
```

## Performance Impact

With balanced training and evaluation:

- **Rare Disease Detection**: Significantly improved (higher recall for minority classes)
- **Overall Metrics**: May show slightly lower micro-average metrics, but better represents true performance
- **Bias Reduction**: Model no longer predicts everything as "no disease" or common diseases
- **Clinical Value**: Better detection of rare but important diseases like Hernia, Pneumonia

## Future Enhancements

1. **Focal Loss**: Implement focal loss for explicit hard-example mining
2. **Curriculum Learning**: Train on balanced classes → fine-tune on original distribution
3. **Cost-Sensitive Learning**: Different misclassification costs for different classes
4. **Threshold Optimization**: Find optimal thresholds per class using validation data
5. **SMOTE Variants**: Explore advanced oversampling methods for medical images

## References

- Class imbalance in medical imaging: [NIH Dataset Paper](https://arxiv.org/abs/1705.02315)
- Effective number of samples: [Cui et al., 2019](https://arxiv.org/abs/1901.05555)
- Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
