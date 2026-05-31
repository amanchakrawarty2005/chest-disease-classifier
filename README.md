# Chest Disease Classifier

End-to-end multi-label chest X-ray classification project built on the NIH Chest X-ray dataset.

## What Is Implemented (Phase 1 to 3)

- Phase 1A: Exploratory Data Analysis (EDA)
- Phase 1B: Metadata preprocessing and train/val/test split generation
- Phase 2: EfficientNetB0 training with two-stage fine-tuning
- Phase 3: Evaluation metrics + Grad-CAM explainability

## Dataset

- Source: NIH Chest X-rays (Kaggle mirror)
- Expected CSV: `data/raw/Data_Entry_2017.csv`
- Expected images: nested folders under `data/raw/`

## Tech Stack

- Python 3.10
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn
- OpenCV (currently used by augmentation + Grad-CAM)

## Folder and File Responsibilities

- `notebooks/01_EDA.py`
- Performs EDA and creates visual/statistical summaries.

- `src/data_preprocessing.py`
- Parses labels, creates multi-hot targets, handles robust splitting, saves split metadata files.

- `src/model.py`
- Defines EfficientNetB0 classifier architecture and fine-tuning layer-unfreeze logic.

- `src/train.py`
- Streams images from disk with `tf.data`, runs two training stages, saves best/final model.

- `src/evaluate.py`
- Runs inference on test split and writes evaluation metrics report.

- `src/gradcam.py`
- Generates Grad-CAM overlays for sampled test images.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Runbook

### Phase 1A: EDA

```bash
python notebooks/01_EDA.py
```

Outputs:
- `data/processed/01_disease_distribution.png`
- `data/processed/02_multilabel_distribution.png`
- `data/processed/03_class_imbalance.png`
- `data/processed/EDA_REPORT.txt`

### Phase 1B: Preprocessing

```bash
python src/data_preprocessing.py
```

Optional flags:

```bash
python src/data_preprocessing.py --log-level INFO --train-split 0.8 --val-split 0.1 --test-split 0.1
```

Outputs:
- `data/processed/data_splits.json`
- `data/processed/train_labels.csv`
- `data/processed/val_labels.csv`
- `data/processed/test_labels.csv`
- `data/processed/class_weights.json`

### Phase 2: Training

```bash
python src/train.py
```

Optional quick smoke run:

```bash
python src/train.py --max-train-samples 5000 --max-val-samples 1000
```

Outputs:
- `saved_model/best_model.keras`
- `saved_model/final_model.keras`
- `saved_model/training_summary.json`

### Phase 3A: Evaluation

```bash
python src/evaluate.py
```

Output:
- `data/processed/evaluation_report.json`

### Phase 3B: Explainability (Grad-CAM)

```bash
python src/gradcam.py
```

Output folder:
- `data/processed/gradcam/`

## Current Laptop-Friendly Defaults

Configured in `src/config.py` for lower-memory environments:
- `BATCH_SIZE = 8`
- `EPOCHS_PHASE1 = 3`
- `EPOCHS_PHASE2 = 2`
- `BACKBONE_TRAINABLE_LAYERS = 10`

## Notes

- On native Windows with TensorFlow >= 2.11, training usually runs on CPU unless using WSL2/DirectML setup.
- Some warnings from oneDNN and CPU feature logs are expected and non-blocking.
