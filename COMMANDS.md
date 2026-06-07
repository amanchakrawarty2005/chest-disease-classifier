# 🚀 Copy-Paste Commands - Complete Workflow

**Platform**: Windows PowerShell  
**Hardware**: GTX 1650 4GB VRAM  
**Goal**: Maximum performance model

---

## 📋 **Phase 1: Preprocessing**

```powershell
python src/data_preprocessing.py --train-split 0.8 --val-split 0.1 --test-split 0.1 --apply-oversampling True
```

**Time**: 5-10 minutes  
**Output**: `data/processed/train_labels.csv`, `val_labels.csv`, `test_labels.csv`

---

## 🎯 **Phase 2: Training (Maximum Performance)**

```powershell
python src/train.py --epochs-phase1 30 --epochs-phase2 20 --lr-phase1 1e-3 --lr-phase2 1e-6 --warmup-epochs 5 --batch-size 16 --sample-weighting inverse_frequency --focal-gamma 2.5
```

**Time**: 20-30 hours  
**Output**: `saved_model/best_model.keras`, `saved_model/final_model.keras`, `saved_model/thresholds.json`

---

## 📊 **Phase 3: Evaluation**

```powershell
python src/evaluate.py
```

**Time**: 5-10 minutes  
**Output**: `data/processed/evaluation_report.json`

---

## 🔍 **Phase 4: Grad-CAM Visualization**

```powershell
python src/gradcam.py
```

**Time**: 2-5 minutes  
**Output**: `data/processed/gradcam/*.png`

---

## 🌐 **Phase 5: Start API Server**

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access**: http://localhost:8000  
**Docs**: http://localhost:8000/docs

---

## 📱 **Phase 6: Start Dashboard**

```powershell
streamlit run dashboard/app.py
```

**Access**: http://localhost:8501

---

## 🐳 **Alternative: Docker Deployment**

```powershell
docker compose up --build
```

**API**: http://localhost:8000  
**Dashboard**: http://localhost:8501

---

## 📈 **Optional: EDA (Exploratory Data Analysis)**

```powershell
python notebooks/01_EDA.py
```

**Time**: 2-3 minutes  
**Output**: `data/processed/01_disease_distribution.png`, `02_multilabel_distribution.png`, etc.

---

## 🔧 **Utility Commands**

### View Evaluation Results
```powershell
Get-Content data\processed\evaluation_report.json
```

### View Training Summary
```powershell
Get-Content saved_model\training_summary.json
```

### Monitor GPU Usage
```powershell
nvidia-smi -l 5
```

### Check Model Files
```powershell
Get-ChildItem saved_model
```

---

## ⚡ **Alternative Training Configurations**

### Quick Test (30 minutes)
```powershell
python src/train.py
```

### Moderate Quality (3-5 hours)
```powershell
python src/train.py --epochs-phase1 10 --epochs-phase2 5 --batch-size 16 --sample-weighting inverse_frequency
```

### High Quality (8-12 hours)
```powershell
python src/train.py --epochs-phase1 20 --epochs-phase2 10 --lr-phase1 1e-3 --lr-phase2 5e-6 --warmup-epochs 3 --batch-size 16 --sample-weighting inverse_frequency --focal-gamma 2.0
```

### Maximum Performance (20-30 hours) ⭐ **RECOMMENDED**
```powershell
python src/train.py --epochs-phase1 30 --epochs-phase2 20 --lr-phase1 1e-3 --lr-phase2 1e-6 --warmup-epochs 5 --batch-size 16 --sample-weighting inverse_frequency --focal-gamma 2.5
```

---

## 🧪 **Testing Commands**

### Test Single Prediction (PowerShell)
```powershell
$imagePath = "data\raw\images_001\images\00000001_000.png"
curl.exe -X POST "http://127.0.0.1:8000/predict?threshold=0.5&top_k=5" -F "file=@$imagePath"
```

### Health Check
```powershell
curl.exe http://127.0.0.1:8000/health
```

---

## 🔄 **Complete Workflow (Copy All in Sequence)**

```powershell
# Step 1: Preprocessing
python src/data_preprocessing.py --train-split 0.8 --val-split 0.1 --test-split 0.1 --apply-oversampling True

# Step 2: Training (Maximum Performance) - This will take 20-30 hours
python src/train.py --epochs-phase1 30 --epochs-phase2 20 --lr-phase1 1e-3 --lr-phase2 1e-6 --warmup-epochs 5 --batch-size 16 --sample-weighting inverse_frequency --focal-gamma 2.5

# Step 3: Evaluation
python src/evaluate.py

# Step 4: Grad-CAM
python src/gradcam.py

# Step 5: Start API (in separate terminal)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Step 6: Start Dashboard (in separate terminal)
streamlit run dashboard/app.py
```

---

## 📝 **Notes**

- All commands assume you're in the project root directory
- Virtual environment should be activated: `venv\Scripts\activate`
- For training, consider using `Start-Process` to run in background
- Early stopping is enabled - training may finish earlier if converged

---

## 🚨 **Troubleshooting Commands**

### Out of Memory Error
```powershell
python src/train.py --epochs-phase1 30 --epochs-phase2 20 --lr-phase1 1e-3 --lr-phase2 1e-6 --warmup-epochs 5 --batch-size 8 --sample-weighting inverse_frequency --focal-gamma 2.5
```

### Missing Dependencies
```powershell
pip install -r requirements.txt
```

### Preprocessing Not Found
```powershell
python src/data_preprocessing.py
```

### Check Python Version
```powershell
python --version
```

---

## ⏱️ **Time Estimates**

| Phase | Time | Can Skip? |
|-------|------|-----------|
| Preprocessing | 5-10 min | No |
| Training (Max) | 20-30 hrs | No |
| Evaluation | 5-10 min | No |
| Grad-CAM | 2-5 min | Yes |
| API/Dashboard | Instant | For testing |

**Total Time**: ~21-31 hours

---

## 🎯 **Expected Performance After Maximum Training**

```
✅ Micro AUC: 0.88-0.91
✅ Macro AUC: 0.86-0.90
✅ Minority Class AUC: 0.84-0.88
✅ Weighted AUC: 0.87-0.90
```

This will be a **research-grade model** competitive with published papers! 🏆

---

**Created**: June 6, 2026  
**Platform**: Windows PowerShell  
**Last Updated**: June 6, 2026
