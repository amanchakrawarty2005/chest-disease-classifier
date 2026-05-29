"""
Global configuration for the Chest Disease Classifier project.
Centralized settings for all phases of the project.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "saved_model"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
API_DIR = BASE_DIR / "api"
DASHBOARD_DIR = BASE_DIR / "dashboard"
TESTS_DIR = BASE_DIR / "tests"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, NOTEBOOK_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# 14 Chest Disease Classes (Multi-label classification)
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

# Dataset splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

RANDOM_SEED = 42

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

IMAGE_SIZE = 224  # EfficientNetB0 standard input size
IMAGE_CHANNEL = 1  # Grayscale X-rays

# Normalization (ImageNet statistics adapted for medical images)
IMAGE_MEAN = 0.485
IMAGE_STD = 0.229

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Architecture
MODEL_NAME = "EfficientNetB0"
BACKBONE_TRAINABLE_LAYERS = 30  # Unfreeze last 30 layers for fine-tuning

# Training - Phase 1: Frozen backbone, train custom head
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
LEARNING_RATE_PHASE1 = 1e-3
OPTIMIZER_PHASE1 = "adam"

# Training - Phase 2: Fine-tune entire model
EPOCHS_PHASE2 = 10
LEARNING_RATE_PHASE2 = 1e-5
OPTIMIZER_PHASE2 = "adam"

# Loss & Metrics
LOSS_FUNCTION = "binary_crossentropy"  # Multi-label classification
METRICS = ["AUC", "Precision", "Recall"]
TARGET_AUC = 0.82

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

AUGMENTATION_ENABLED = True
ROTATION_RANGE = 15
BRIGHTNESS_RANGE = (0.8, 1.2)
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Chest Disease Classifier API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Multi-label chest disease classification from X-ray images"

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

STREAMLIT_PAGE_TITLE = "Chest Disease Classifier Dashboard"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_THEME = "light"

# ============================================================================
# MLFLOW EXPERIMENT TRACKING
# ============================================================================

MLFLOW_EXPERIMENT_NAME = "chest-disease-classifier"
MLFLOW_RUN_NAME_PREFIX = "chest-xray"

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

LOG_LEVEL = "INFO"
DEBUG_MODE = False

# Print configuration on import
if DEBUG_MODE:
    print(f"✅ Configuration loaded from {__file__}")
    print(f"   Base directory: {BASE_DIR}")
    print(f"   Number of disease classes: {NUM_CLASSES}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")