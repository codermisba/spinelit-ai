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

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# Dataset
DATA_DIR = ROOT_DIR / "dataset" / "data"

# Output directories
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Create output folders automatically
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# Training Configuration
# ==========================================================

IMAGE_SIZE = 256          # 512 is more accurate but ~4-6x slower / hotter on CPU.
                          # 256 trains much faster with far less laptop heating.
BATCH_SIZE = 4            # Practical for an 8 GB RAM machine
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0           # Windows compatibility
PIN_MEMORY = True

# Limit CPU threads used by PyTorch during training.
# Reduces overheating on laptops (oversubscribing all logical cores
# causes thermal throttling, which is slower anyway).
CPU_NUM_THREADS = min(8, os.cpu_count() or 8)

# Gradient clipping
GRAD_CLIP_NORM = 1.0

# Early stopping (patience in epochs)
EARLY_STOPPING_PATIENCE = 10

# Confidence supervision
# ----------------------
# The dataset labels every sample with confidence = 1, so a plain BCE head
# just learns to always output 1.0. Instead, we train the confidence head to
# predict the model's OWN per-level localization error (self-supervised):
#   confidence_target = exp(-euclidean_error / CONFIDENCE_TEMPERATURE)
# This yields a real, per-image confidence at inference time.
CONFIDENCE_SUPERVISION = True
CONFIDENCE_TEMPERATURE = 0.05   # normalized 0-1 coordinate units


# ==========================================================
# Model Configuration
# ==========================================================

MODEL_NAME = "convnext_tiny"

PRETRAINED = True

NUM_KEYPOINTS = 5         # L1-L5

NUM_OUTPUTS = NUM_KEYPOINTS * 2   # x,y for each vertebra


# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# Random Seed
# ==========================================================

SEED = 42


# ==========================================================
# Dataset Split
# ==========================================================

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


# ==========================================================
# Checkpoint Names
# ==========================================================

BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL = CHECKPOINT_DIR / "last_model.pth"

# Small inference-only export (fp16, model-only) committed to Git so a
# fresh clone works without retraining. Full checkpoints stay gitignored.
RELEASE_MODEL = CHECKPOINT_DIR / "release_model.pth"


# ==========================================================
# Logging
# ==========================================================

PRINT_EVERY = 10


# ==========================================================
# Inference
# ==========================================================

CONFIDENCE_THRESHOLD = 0.50


# ==========================================================
# Supported Image Formats
# ==========================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".dcm",
)