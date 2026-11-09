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


# ----------------------------------------------------------
# Load optional local secrets (`.env`) — never committed to Git.
# Supported so `GEMINI_API_KEY` works without editing this file.
# ----------------------------------------------------------
def _load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


# ==========================================================
# Project Paths
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "dataset" / "data"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"

for _d in (CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR):
    try:
        _d.mkdir(exist_ok=True, parents=True)
    except OSError:
        # Read-only / restricted hosts (e.g. some container images) must not
        # crash at import time; callers handle missing dirs gracefully.
        pass


# Annotation / label files (all optional except the disc landmark CSV)
DISC_LANDMARK_CSV = ROOT_DIR / "dataset" / "coords_pretrain.csv"
VERTEBRA_LANDMARK_CSV = ROOT_DIR / "dataset" / "coords_vertebrae.csv"
# Per-level Pfirrmann grade (1-5) labels for Disc Degenerative Disease (DDD).
# Format: filename,level,pfirrmann_grade  (grade in 1..5)
DDD_LABELS_CSV = ROOT_DIR / "dataset" / "ddd_labels.csv"
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

# Pfirrmann grading for DDD: 5 grades (I..V) per disc level.
PFRRMANN_GRADES = ["I", "II", "III", "IV", "V"]
NUM_PFRRMANN_CLASSES = len(PFRRMANN_GRADES)

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

# Checkpoint selection score = val_loss + w * (keypoint_error_px / image_size).
# Localisation error is the product's primary output ("does it plot the right
# blocks?"), so a model with much better landmarks must not be discarded just
# because its confidence-calibration loss is temporarily higher.
LOCALIZATION_SELECTION_WEIGHT = 5.0


# ==========================================================
# Image normalisation (MUST match training and inference)
# ==========================================================
# torchvision `ToTensor()` alone leaves pixels in 0..1.  ImageNet-pretrained
# backbones (ConvNeXt) expect the standard ImageNet statistics; using them
# consistently is one of the biggest accuracy wins in this pipeline.
USE_IMAGENET_NORM = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==========================================================
# Vision architecture (spatial heatmap localizer + FPN)
# ==========================================================
# The earlier design pooled the whole feature map with global average
# pooling and then regressed coordinates from that vector.  GAP destroys
# the spatial layout, so the head could only learn the *average* spine
# position.  The rebuilt model keeps a dense feature map (FPN at stride 4)
# and predicts one heatmap per keypoint -> soft-argmax coordinates.  That
# is what makes per-image, per-vertebra localisation possible.
FPN_DIM = 128                # channels in the fused dense feature map
HEATMAP_STRIDE = 4           # output stride of the keypoint heatmaps
HEATMAP_SIGMA = 2.0          # gaussian target radius (feature-map pixels)
# Soft-argmax temperature.  Logits are not probabilities: background logits
# sit near 0 while a trained gaussian peak is only ~1-2.  A plain softmax is
# therefore almost uniform and the coordinate expectation collapses to the
# image centre even when the heatmap peak is correct.  Sharpening the logits
# before the spatial softmax makes the integral-soft-argmax read the peak
# (validated: 5.5 px vs 32 px mean error at the same checkpoint).
SOFTARGMAX_TEMPERATURE = 10.0
# Gaussian pooling radius for the per-disc DDD classifier, as a fraction of
# the fused feature map's size.  Small enough to pool around ONE disc, large
# enough to include the whole endplate; ~12% ≈ 5 px at 40x40 (20 px at 160px).
DISC_POOL_SIGMA_FRACTION = 0.12
NUM_SEG_CLASSES = 4          # background, vertebra, disc, spinal canal
SEG_CLASS_NAMES = ["background", "vertebra", "disc", "spinal_canal"]
SEG_CLASS_COLOURS = {
    "vertebra": (80, 200, 255),
    "disc": (255, 200, 60),
    "spinal_canal": (140, 255, 140),
}

# Multi-task loss weights
COORD_LOSS_WEIGHT = 1.0       # direct soft-argmax coordinate regression
HEATMAP_LOSS_WEIGHT = 2.0     # dense gaussian heatmap supervision (main signal)
CONF_LOSS_WEIGHT = 0.1
DDD_LOSS_WEIGHT = 1.0         # auto-disabled when no Pfirrmann labels exist
SEG_LOSS_WEIGHT = 1.0         # auto-disabled when no masks are available
# Label smoothing for the Pfirrmann cross-entropy (grading is ordinal and
# labels can be noisy; smoothing stabilizes training).
DDD_LABEL_SMOOTHING = 0.1

# Self-supervised confidence supervision:
# target confidence = exp(-error / TEMPERATURE) using the model's own error.
CONFIDENCE_SUPERVISION = True
CONFIDENCE_TEMPERATURE = 0.05   # normalized coordinate units


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

# Which ConvNeXt stage (features_only index, stride 4/8/16/32) is pooled for
# disc-localized DDD classification. Stage 2 = stride 16 (good resolution).
DISC_FEATURE_STAGE = 2


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

# Small fp16 inference-only artifact committed to Git so a fresh clone
# works without retraining (created via: python export_release.py)
RELEASE_MODEL = CHECKPOINT_DIR / "release_model.pth"

# Optional YOLO detector (boxes for vertebrae + discs) and U-Net
# segmentation model.  Both are optional: the API falls back to
# landmark-derived blocks when they are absent.
YOLO_CHECKPOINT = CHECKPOINT_DIR / "yolo_best.pt"
YOLO_DATA_DIR = ROOT_DIR / "dataset" / "yolo"
SEG_CHECKPOINT = CHECKPOINT_DIR / "segmentation.pth"

LOG_FILE = LOG_DIR / "training_log.csv"

PRINT_EVERY = 10


# ==========================================================
# Inference
# ==========================================================

CONFIDENCE_THRESHOLD = 0.50

# Detection / overlay defaults
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
KEYPOINT_CONF_THRESHOLD = 0.20
SEG_OVERLAY_ALPHA = 0.38


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".dcm",
)


# ==========================================================
# Agentic AI Pipeline Configuration
# ==========================================================
#
# The end-to-end agentic pipeline orchestrates several specialist
# LLM "agents" around the calibrated vision engine to produce an
# auditable, radiologist-style report for two diseases:
#   - Disc Degenerative Disease (DDD)
#   - Spondylolisthesis
#
# Every agent returns structured JSON (LLM) or numeric findings
# (vision engine). A verification/critique agent double-checks the
# clinical interpretation for accuracy, and calibration converts raw
# confidence into well-calibrated probabilities (0-1).
# ==========================================================

# --- LLM backend -------------------------------------------------
# "gemini"  : Google Gemini free API (needs GEMINI_API_KEY) - recommended
# "ollama"  : Local Ollama server (needs a running `ollama serve`)
LLM_PROVIDER = "gemini"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.6-flash"   # current free-tier multimodal model
GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = "llama3.2:3b"          # local fallback (no API key needed)

LLM_TIMEOUT = 90                     # seconds for a single agent call
LLM_TEMPERATURE = 0.2                # low temperature -> reproducible agents

# --- Two supported diseases --------------------------------------
DISEASES = ["disc_degenerative_disease"]

# --- Calibration --------------------------------------------------
# Calibration artifacts (fitted on Colab with labelled data). When
# absent a deterministic fallback calibration is used so the pipeline
# still runs end-to-end on a fresh clone.
CALIB_ARTIFACT = CHECKPOINT_DIR / "calibration.pkl"

# Per-level / per-disease reporting thresholds
REPORT_CONFIDENCE_THRESHOLD = 0.5    # below -> flagged "low confidence"
VERIFY_ITERATIONS = 1                 # critique passes in the verification loop

# Longitudinal / progression horizon used by the longitudinal agent
PROGRESSION_HORIZON_YEARS = 5.0
