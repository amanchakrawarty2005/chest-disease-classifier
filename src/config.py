import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "saved_model"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
API_DIR = BASE_DIR / "api"
DASHBOARD_DIR = BASE_DIR / "dashboard"
TESTS_DIR = BASE_DIR / "tests"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, NOTEBOOK_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

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

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

RANDOM_SEED = 42

IMAGE_SIZE = 224
IMAGE_CHANNEL = 1

IMAGE_MEAN = 0.485
IMAGE_STD = 0.229

MODEL_NAME = "EfficientNetB0"
BACKBONE_TRAINABLE_LAYERS = 50  # Unfreeze more layers for medical image adaptation

BATCH_SIZE = 16          # Reduced for 4GB VRAM GTX 1650
EPOCHS_PHASE1 = 3        # Minimal training for Phase 1 (frozen backbone)
LEARNING_RATE_PHASE1 = 1e-3
LR_WARMUP_EPOCHS = 1     # Reduced warmup epochs
OPTIMIZER_PHASE1 = "adam"

EPOCHS_PHASE2 = 2        # Minimal fine-tuning for Phase 2
LEARNING_RATE_PHASE2 = 1e-5
OPTIMIZER_PHASE2 = "adam"

# Focal loss focuses training on hard/rare positives (Cardiomegaly, Hernia, Pneumonia)
# and down-weights the easy No-Finding negatives (53.8% of dataset).
LOSS_FUNCTION = "focal_bce"
FOCAL_GAMMA = 2.0
METRICS = ["AUC", "Precision", "Recall"]
TARGET_AUC = 0.82

AUGMENTATION_ENABLED = True
ROTATION_RANGE = 15
BRIGHTNESS_RANGE = (0.8, 1.2)
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Chest Disease Classifier API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Chest X-ray multi-label classification"

STREAMLIT_PAGE_TITLE = "Chest Disease Classifier Dashboard"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_THEME = "light"

MLFLOW_EXPERIMENT_NAME = "chest-disease-classifier"
MLFLOW_RUN_NAME_PREFIX = "chest-xray"

LOG_LEVEL = "INFO"
DEBUG_MODE = False

if DEBUG_MODE:
    print(f"config: {__file__}")
    print(f"base: {BASE_DIR}, classes: {NUM_CLASSES}, size: {IMAGE_SIZE}")