"""
Global configuration for the Chest Disease Classifier project.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "saved_model"
NOTEBOOK_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Disease classes (14 chest diseases)
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

NUM_CLASSES = len(DISEASE_CLASSES)

# Model settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5

# Training
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Chest Disease Classifier API"
API_VERSION = "1.0.0"

# Streamlit
STREAMLIT_PAGE_TITLE = "Chest Disease Classifier"
STREAMLIT_LAYOUT = "wide"