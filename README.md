# Chest Disease Classifier

A production-grade deep learning system for detecting 14 chest diseases from X-ray images using EfficientNetB0, with explainability via Grad-CAM, FastAPI backend, and interactive Streamlit dashboard.

## 📋 Project Overview

- **Dataset**: 112,120 chest X-ray images from NIH (14 disease classes, multi-label)
- **Model**: EfficientNetB0 pretrained on ImageNet with custom classification head
- **Performance Target**: Mean AUC ~0.82
- **Explainability**: Per-disease Grad-CAM heatmaps
- **Architecture**: FastAPI backend + Streamlit dashboard + Docker + GitHub Actions CI/CD

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Local Setup
```bash
# Clone the repository
git clone https://github.com/amanchakrawarty2005/chest-disease-classifier.git
cd chest-disease-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run dashboard/app.py

# Or run the FastAPI backend
uvicorn api.main:app --reload