"""
Spine Foundation Model Configuration
====================================

Central configuration file for the project.

All hyperparameters, paths, and training settings should be
defined here so that the rest of the codebase never contains
hard-coded values.
"""

import os
from pathlib import Path
import torch


# ==========================================================
# Project Paths
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "dataset" / "data"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"

for _d in (CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)


# Annotation / label files (all optional except the disc landmark CSV)
DISC_LANDMARK_CSV = ROOT_DIR / "dataset" / "coords_pretrain.csv"
VERTEBRA_LANDMARK_CSV = ROOT_DIR / "dataset" / "coords_vertebrae.csv"
DDD_LABELS_CSV = ROOT_DIR / "dataset" / "ddd_labels.csv"
SPONDY_LABELS_CSV = ROOT_DIR / "dataset" / "spondy_labels.csv"
LONGITUDINAL_RECORDS_CSV = ROOT_DIR / "dataset" / "longitudinal_records.csv"


# ==========================================================
# Anatomy Layout
# ==========================================================

# Fixed order used everywhere (dataset -> model heads -> UI).
VERTEBRAE = ["L1", "L2", "L3", "L4", "L5"]
DISC_LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

NUM_VERTEBRAE = len(VERTEBRAE)      # 5 vertebral body centres
NUM_DISCS = len(DISC_LEVELS)        # 5 disc centres
NUM_KEYPOINTS = NUM_VERTEBRAE + NUM_DISCS   # 10 points total
NUM_OUTPUTS = NUM_KEYPOINTS * 2             # x,y per point

# When no explicit vertebra-centre annotations exist (coords_vertebrae.csv),
# derive weak supervision from the annotated disc centroids:
#   L2..L5  = midpoint of adjacent disc centres
#   L1      = extrapolated one inter-discal spacing above L1/L2
DERIVE_VERTEBRA_CENTERS = True

# Geometric indicator thresholds (relative units, no diagnosis implied)
NARROWED_DISC_THRESHOLD = 0.85     # relative disc space below mean -> flag
LISTHESIS_OFFSET_THRESHOLD = 0.25  # horizontal offset ratio -> flag


# ==========================================================
# Training Configuration
# ==========================================================

IMAGE_SIZE = 256          # set 512 on GPU/Colab for maximum accuracy
BATCH_SIZE = 4            # increase to 16-32 on Colab GPU

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0           # Windows compatibility; use 2-4 on Linux/Colab
PIN_MEMORY = True

CPU_NUM_THREADS = min(8, os.cpu_count() or 8)

GRAD_CLIP_NORM = 1.0

EARLY_STOPPING_PATIENCE = 10


# Multi-task loss weights
COORD_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 0.1
DDD_LOSS_WEIGHT = 1.0       # auto-disabled when no DDD labels exist
SPONDY_LOSS_WEIGHT = 1.0    # auto-disabled when no spondylolisthesis labels

# Self-supervised confidence supervision:
# target confidence = exp(-error / TEMPERATURE) using the model's own error.
CONFIDENCE_SUPERVISION = True
CONFIDENCE_TEMPERATURE = 0.05   # normalized coordinate units
GRADE_TEMPERATURE = 0.75        # grade units (DDD 0-4 scale)


# ==========================================================
# Clinical / Longitudinal Model Configuration
# ==========================================================

CLINICAL_FEATURE_DIM = 128
LONGITUDINAL_HIDDEN_DIM = 256
IMAGE_FEATURE_DIM = 512          # shared embedding dim of SpineFoundationModel

# Normalisation ranges for raw clinical inputs
AGE_RANGE = (1, 100)
VAS_RANGE = (0, 10)
PAIN_YEARS_RANGE = (0, 50)
START_YEAR_RANGE = (1950, 2026)
HORIZON_RANGE = (0, 10)     # years ahead the model predicts

SEX_CATEGORIES = ["male", "female"]
MODALITY_CATEGORIES = ["xray", "mri"]

LONGITUDINAL_MODEL_NAME = CHECKPOINT_DIR / "longitudinal_model.pth"


# ==========================================================
# Model Configuration
# ==========================================================

MODEL_NAME = "convnext_tiny"
PRETRAINED = True   # ImageNet-pretrained backbone via timm


# ==========================================================
# Device / Seed / Split
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


# ==========================================================
# Checkpoint Names
# ==========================================================

BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL = CHECKPOINT_DIR / "last_model.pth"

LOG_FILE = LOG_DIR / "training_log.csv"

PRINT_EVERY = 10


# ==========================================================
# Inference
# ==========================================================

CONFIDENCE_THRESHOLD = 0.50


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".dcm",
)
