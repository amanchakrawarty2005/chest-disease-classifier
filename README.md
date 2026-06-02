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

- `src/balancing.py`
- Comprehensive class imbalance handling: weighted sampling, oversampling, and balanced metrics.

- `src/balance_visualization.py`
- Utilities for analyzing and visualizing class distribution and imbalance statistics.

## Data Balancing & Class Imbalance

This project implements comprehensive strategies to handle the **88x class imbalance** in the NIH Chest X-ray dataset:

✓ **Minority class oversampling** - Increases representation of rare diseases  
✓ **Sample-level weighting** - Effective Number of Samples method  
✓ **Aggressive augmentation** - For minority class samples  
✓ **Weighted evaluation metrics** - Including minority class AUC  
✓ **Class-specific inference thresholds** - Better detection of rare diseases  

**See [BALANCING.md](BALANCING.md) for detailed documentation.**

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
python src/data_preprocessing.py --log-level INFO --train-split 0.8 --val-split 0.1 --test-split 0.1 --apply-oversampling True
```

Outputs:
- `data/processed/data_splits.json`
- `data/processed/train_labels.csv` (with oversampled minority classes)
- `data/processed/val_labels.csv`
- `data/processed/test_labels.csv`
- `data/processed/class_weights.json` (multiple weighting methods)
- `data/processed/train_balance_report.json`
- `data/processed/val_balance_report.json`
- `data/processed/test_balance_report.json`

### Phase 2: Training

```bash
python src/train.py
```

Optional flags:

```bash
# Use effective number weighting for balanced training
python src/train.py --sample-weighting effective_num

# Quick smoke run
python src/train.py --max-train-samples 5000 --max-val-samples 1000
```

Outputs:
- `saved_model/best_model.keras`
- `saved_model/final_model.keras`
- `saved_model/training_summary.json` (includes weighting method used)

### Phase 3A: Evaluation

```bash
python src/evaluate.py
```

Output:
- `data/processed/evaluation_report.json` (includes class balance statistics and weighted metrics)

Key metrics reported:
- `micro_auc`: Overall performance across all predictions
- `macro_auc`: Unweighted average (better for imbalanced data)
- `weighted_auc`: Frequency-weighted AUC (accounts for class imbalance)
- `minority_class_auc`: Performance on rare diseases specifically

### Phase 3B: Explainability (Grad-CAM)

```bash
python src/gradcam.py
```

Output folder:
- `data/processed/gradcam/`

### Phase 4: FastAPI Inference Service

Run API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:
- `GET /`
- `GET /health`
- `POST /predict` (multipart form-data, field name: `file`)

Example test (PowerShell):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?threshold=0.5&top_k=5" `
  -F "file=@D:\path\to\xray.png"
```

### Phase 5: Streamlit Dashboard

Run dashboard:

```bash
streamlit run dashboard/app.py
```

What it provides:
- Upload chest X-ray image
- Set prediction threshold and top-k
- Call FastAPI `/predict` endpoint
- View top predictions and full probability chart
- Show existing Grad-CAM match (if available in `data/processed/gradcam/`)

### Phase 6: Docker + CI/CD

Build and run with Docker Compose:

```bash
docker compose up --build
```

Services:
- API: `http://127.0.0.1:8000`
- Dashboard: `http://127.0.0.1:8501`

Docker files:
- `Dockerfile.api`
- `Dockerfile.dashboard`
- `docker-compose.yml`
- `.dockerignore`

CI workflow:
- `.github/workflows/ci.yml`
- Runs on push/PR to `main`
- Performs syntax checks and unit tests

## Current Laptop-Friendly Defaults

Configured in `src/config.py` for lower-memory environments:
- `BATCH_SIZE = 8`
- `EPOCHS_PHASE1 = 2`
- `EPOCHS_PHASE2 = 1`
- `BACKBONE_TRAINABLE_LAYERS = 10`

## Notes

- On native Windows with TensorFlow >= 2.11, training usually runs on CPU unless using WSL2/DirectML setup.
- Some warnings from oneDNN and CPU feature logs are expected and non-blocking.
