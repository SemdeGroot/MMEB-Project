import torch

# ---- Device ----
# Prefer CUDA, then Apple MPS, then CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ---- Paths ----
DATA_DIR        = "data"
OCCURRENCE_FILE = f"{DATA_DIR}/occurrence.txt"
MULTIMEDIA_FILE = f"{DATA_DIR}/multimedia.txt"
IMAGE_DIR       = "data/images"

# ---- Dataset ----
NUM_SAMPLES           = None    # Set an integer for quick experiments.
MIN_SAMPLES_PER_CLASS = 20      # Species below this threshold are removed.
LIFE_STAGE            = "Adult" # Keep one life stage for a cleaner classification task.

# ---- Image ----
IMAGE_SIZE  = 224
NUM_WORKERS = 8

# ---- Training ----
BATCH_SIZE      = 256
EPOCHS        = 100
LEARNING_RATE = 3e-5
LABEL_SMOOTHING = 0.05         # Stored here for experiments, the current train loop uses focal loss.
TRAIN_SPLIT   = 0.7
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42
DETERMINISTIC_TRAINING = True
