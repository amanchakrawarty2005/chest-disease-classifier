# Quick Start Guide: Balanced Training

## TL;DR - Just Run These Commands

```bash
# 1. Preprocess with balancing (creates oversampled training data)
python src/data_preprocessing.py --apply-oversampling True

# 2. Train with effective number weighting
python src/train.py --sample-weighting effective_num

# 3. Evaluate and check balanced metrics
python src/evaluate.py

# 4. Start API (uses class-specific thresholds automatically)
uvicorn api.main:app --reload
```

## What Changed?

The dataset has **88x class imbalance** - some diseases appear in <1% of samples while others in 17%. This caused the model to bias toward common diseases and miss rare ones.

**Now implemented:**
- ✓ Training data oversampled to balance minority classes
- ✓ Per-sample weighting during training
- ✓ Better augmentation for rare disease images
- ✓ Evaluation metrics that account for imbalance
- ✓ Smart thresholds in API (rare diseases detected better)

## Step-by-Step

### Step 1: Preprocess (One-time)

```bash
python src/data_preprocessing.py --apply-oversampling True --log-level INFO
```

**What it does:**
- ✓ Reads all chest X-ray metadata
- ✓ Duplicates rare disease samples to improve balance
- ✓ Generates class weights (stored in `class_weights.json`)
- ✓ Creates balance reports for each split

**Output files:**
```
data/processed/
├── train_labels.csv (now with ~90k samples after oversampling)
├── val_labels.csv
├── test_labels.csv
├── class_weights.json (3 weight methods: inverse_frequency, effective_num, focal_loss)
├── train_balance_report.json (statistics on imbalance)
├── val_balance_report.json
└── test_balance_report.json
```

### Step 2: Train (1-2 hours)

```bash
# Recommended: use effective number weighting
python src/train.py --sample-weighting effective_num

# Or use simple inverse frequency
python src/train.py --sample-weighting inverse_frequency

# Quick test with fewer samples
python src/train.py --sample-weighting effective_num --max-train-samples 10000
```

**Key features:**
- Per-sample weights emphasize rare diseases
- Aggressive augmentation for minority classes
- Automatic class weight loading
- Progress logged to console

**Output:**
```
saved_model/
├── best_model.keras (best validation performance)
├── final_model.keras (after fine-tuning)
└── training_summary.json (includes weighting method)
```

### Step 3: Evaluate (5-10 minutes)

```bash
python src/evaluate.py --log-level INFO
```

**New metrics reported:**
- `micro_auc`: Overall performance (all predictions)
- `macro_auc`: Fair to all diseases (unweighted average) ← Better for imbalance
- `weighted_auc`: Accounts for disease frequency
- `minority_class_auc`: Performance on rare diseases specifically

**Example output:**
```
========================================
METRICS SUMMARY
========================================
Micro AUC: 0.8234
Macro AUC: 0.7456  ← Focus on this for balanced view
Weighted AUC: 0.7823
Minority Class AUC: 0.6234  ← Check rare disease performance

=========================================
CLASS BALANCE IN TEST SET
=========================================
Total Samples: 10000
Healthy Samples: 5400 (54.00%)
Diseased Samples: 4600 (46.00%)
Imbalance Ratio: 88.24x  ← Still imbalanced but model handles it
```

**Output:**
```
data/processed/evaluation_report.json
- Contains all metrics above
- Plus per-class AUC for each of 14 diseases
- Plus class balance statistics
```

### Step 4: Run API (Already Balanced)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Smart features (automatic):**
- Loads class weights automatically
- Adjusts prediction thresholds per disease:
  - Rare disease (e.g., Hernia) → Lower threshold → More detections
  - Common disease (e.g., Infiltration) → Standard threshold
- Returns `confidence_adjusted: true` for special thresholds

**Test it:**
```bash
# PowerShell
curl.exe -X POST "http://127.0.0.1:8000/predict?threshold=0.5&top_k=5" `
  -F "file=@D:\path\to\xray.png"

# Response includes:
# {
#   "top_predictions": [
#     {"disease": "Infiltration", "probability": 0.85, "predicted": true, "confidence_adjusted": false},
#     {"disease": "Hernia", "probability": 0.42, "predicted": true, "confidence_adjusted": true}  ← Lower threshold
#   ]
# }
```

## Configuration Reference

### Preprocessing Options

```bash
# Default (recommended)
python src/data_preprocessing.py

# Disable oversampling if you only want weighting
python src/data_preprocessing.py --apply-oversampling False

# Adjust split ratios
python src/data_preprocessing.py --train-split 0.75 --val-split 0.15 --test-split 0.1

# Full options
python src/data_preprocessing.py \
  --raw-data-dir data/raw \
  --processed-data-dir data/processed \
  --apply-oversampling True \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1 \
  --seed 42 \
  --log-level INFO
```

### Training Options

```bash
# Use effective number weighting (recommended - handles oversampling)
python src/train.py --sample-weighting effective_num

# Use inverse frequency weighting (simple)
python src/train.py --sample-weighting inverse_frequency

# Adjust training parameters
python src/train.py \
  --sample-weighting effective_num \
  --batch-size 8 \
  --epochs-phase1 3 \
  --epochs-phase2 2 \
  --lr-phase1 0.001 \
  --lr-phase2 0.00001 \
  --seed 42

# Quick test with fewer samples
python src/train.py \
  --sample-weighting effective_num \
  --max-train-samples 10000 \
  --max-val-samples 2000 \
  --epochs-phase1 1 \
  --epochs-phase2 1
```

### Evaluation Options

```bash
# Default (recommended)
python src/evaluate.py

# Custom threshold
python src/evaluate.py --threshold 0.4

# Quick test
python src/evaluate.py --max-samples 2000
```

## Understanding the Reports

### Class Balance Report

```json
{
  "split": "train",
  "total_samples": 90123,
  "healthy_samples": 48201,
  "diseased_samples": 41922,
  "healthy_percentage": 53.49,
  "imbalance_ratio": 88.08,
  "class_distribution": {
    "Infiltration": {
      "count": 15943,
      "percentage": 17.77
    },
    "Hernia": {
      "count": 181,
      "percentage": 0.20
    }
  }
}
```

**Key fields:**
- `imbalance_ratio`: How many times more common the most common disease is. ↓ Lower is more balanced
- `healthy_percentage`: % of "no disease" samples
- `class_distribution`: Count and % for each of 14 diseases

### Class Weights File

```json
{
  "inverse_frequency": {
    "Atelectasis": 1.23,
    "Cardiomegaly": 5.67,
    ...
    "Hernia": 123.45  ← Highest weight (rarest)
  },
  "effective_num": {
    "Atelectasis": 1.15,
    ...
    "Hernia": 98.23
  },
  "focal_loss": {...},
  "recommended": {...}  ← Used by default
}
```

**Higher weight = rarer disease = more important during training**

## Troubleshooting

### "No valid image paths found"
- Check that images are actually in `data/raw/images_*/images/`
- Verify paths in CSV match actual filenames

### Imbalance ratio still 88x after preprocessing
- That's OK! Oversampling is relative (25% of majority class)
- The model is trained with weighted samples, not just duplicated data
- Evaluation metrics account for remaining imbalance

### API not using class weights
- Check that `data/processed/class_weights.json` exists
- Check API logs: should show "Loaded recommended class weights"
- Model must be re-loaded to pick up weights

### Very low minority class AUC
- Minority classes are inherently hard to predict (few examples)
- Check that oversampling is enabled during preprocessing
- Try `--sample-weighting effective_num` instead of `inverse_frequency`

## Performance Tips

1. **Always use `--sample-weighting effective_num`** for oversampled data
2. **Preprocess once, train multiple times** (preprocessing takes ~5 min)
3. **Check minority_class_auc** to see if rare diseases are detected
4. **Start with threshold=0.5** then adjust based on use case:
   - Clinical decision support → Lower threshold (catch everything)
   - Screening tool → Higher threshold (avoid false positives)

## What to Look For in Results

✓ **Good signs:**
- `weighted_auc` > `micro_auc` (handles imbalance well)
- `minority_class_auc` > 0.6 (rare diseases detected)
- `macro_auc` close to `weighted_auc` (balanced learning)
- Per-class AUC: no disease with AUC < 0.5

✗ **Warning signs:**
- `micro_auc` much higher than `macro_auc` (biased toward common diseases)
- `minority_class_auc` < 0.5 (rare diseases not detected)
- Some diseases have AUC = None (insufficient examples)

## Documentation

For detailed information:
- **[BALANCING.md](BALANCING.md)** - Comprehensive balancing documentation
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[README.md](README.md)** - Full project overview
