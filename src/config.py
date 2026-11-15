"""Central configuration for the façade parsing prototype."""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "cfp_raw" / "translated_data"
DATA_PROCESSED = PROJECT_ROOT / "data" / "cfp_processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Classes ──────────────────────────────────────────────────────────────
NUM_CLASSES = 10  # including background

CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "window",
    3: "sky",
    4: "roof",
    5: "door",
    6: "tree",
    7: "people",
    8: "car",
    9: "sign",
}

CLASS_COLORS_RGB = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (250, 170, 30),
    3: (70, 130, 180),
    4: (0, 60, 100),
    5: (153, 153, 153),
    6: (107, 142, 35),
    7: (255, 0, 0),
    8: (0, 0, 142),
    9: (220, 220, 0),
}

# ── Data split ───────────────────────────────────────────────────────────
SPLIT_SEED = 1          # matches RTFP code
TRAIN_RATIO = 0.8       # 481 images
VAL_TEST_RATIO = 0.5    # remaining 121 → 60 val + 61 test

# ── Preprocessing ────────────────────────────────────────────────────────
RESIZE_LONGEST = 1024   # resize longest side, keep aspect ratio

# ── Training (used in train.py / Kaggle notebook) ────────────────────────
CROP_SIZE = 512
BATCH_SIZE_B0 = 16
BATCH_SIZE_B2 = 8
LR = 6e-5
EPOCHS = 50
EARLY_STOP_PATIENCE = 10

