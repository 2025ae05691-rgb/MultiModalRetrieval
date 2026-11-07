
from pathlib import Path

# ==========================================
# PROJECT DIRECTORIES
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
ANNOTATION_DIR = DATA_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"


# ==========================================
# DATASET CONFIGURATION
# ==========================================

NUM_VIDEOS = 100
NUM_CATEGORIES = 20
VIDEOS_PER_CATEGORY = 5
NUM_CAPTIONS_PER_VIDEO = 2


# ==========================================
# VIDEO CONFIGURATION
# ==========================================

# Number of frames sampled uniformly from each video
NUM_FRAMES = 8

# Frame sampling strategy
FRAME_SAMPLING = "uniform"


# ==========================================
# CLIP MODEL CONFIGURATION
# ==========================================

# Pretrained image-text model
CLIP_MODEL_NAME = "ViT-B-32"

CLIP_PRETRAINED = "laion2b_s34b_b79k"

# ViT-B/32 CLIP embedding dimension
EMBEDDING_DIM = 512


# ==========================================
# TEMPORAL MODEL CONFIGURATION
# ==========================================

TEMPORAL_MODEL = "GRU"

# GRU hidden layer dimension
TEMPORAL_HIDDEN_DIM = 512

# Training configuration
TEMPORAL_LEARNING_RATE = 0.0002

TEMPORAL_EPOCHS = 100

TEMPORAL_WEIGHT_DECAY = 0.0001

# Contrastive learning temperature
TEMPERATURE = 0.07


# ==========================================
# GENERAL CONFIGURATION
# ==========================================

BATCH_SIZE = 16
SEED = 42
TOP_K_DEFAULT = 5


# ==========================================
# CREATE REQUIRED DIRECTORIES
# ==========================================

for folder in [
    VIDEO_DIR,
    ANNOTATION_DIR,
    PROCESSED_DIR,
    ARTIFACT_DIR,
    RESULTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)
