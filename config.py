# ─────────────────────────────────────────────────────────────────────────────
#  config.py  –  Single source of truth for every script in the project
# ─────────────────────────────────────────────────────────────────────────────
import os

# ── Project folder layout ─────────────────────────────────────────────────────
#
#   lip_reader/
#   ├── data/                <- extracted landmarks (.pkl files)
#   │   └── landmarks.pkl
#   ├── checkpoints/         <- saved Keras model weights (.h5 files)
#   │   └── lip_reader_best.h5
#   ├── plots/               <- training curves and evaluation figures
#   │   └── training_history.png
#   ├── logs/                <- TensorBoard event files
#   └── miraclvc1/dataset/   <- raw dataset (place here after unzipping Kaggle zip)
#
DATA_DIR        = "data"
CHECKPOINTS_DIR = "checkpoints"
PLOTS_DIR       = "plots"
LOG_DIR         = "logs"

# Create all managed directories on first import.
# Every script that imports config can assume these dirs already exist —
# no scattered os.makedirs() calls needed anywhere else.
for _dir in (DATA_DIR, CHECKPOINTS_DIR, PLOTS_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT  = "miraclvc1/dataset"
LANDMARKS_PKL = os.path.join(DATA_DIR,        "landmarks.pkl")
MODEL_PATH    = os.path.join(CHECKPOINTS_DIR, "lip_reader_best.h5")
HISTORY_PNG   = os.path.join(PLOTS_DIR,       "training_history.png")

# ── Sequence ──────────────────────────────────────────────────────────────────
NUM_FRAMES    = 75     # every sequence is resampled to this many frames
NUM_LANDMARKS = 40     # 20 outer-contour + 20 inner-contour lip points
NUM_FEATURES  = NUM_LANDMARKS * 2   # x + y per landmark  ->  80

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 250
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.10
RANDOM_SEED   = 42

# ── MediaPipe full lip landmark indices ───────────────────────────────────────
#
#  MediaPipe Face Mesh defines the mouth via two closed contours:
#
#   Outer contour – the visible edge of both lips (20 points)
#   Inner contour – the mouth opening/closing aperture (20 points)
#
#  The original 20 indices were a single lower-lip ring, giving the model
#  no signal from upper-lip shape or mouth aperture — both critical cues
#  for distinguishing words like "hello" vs "well done" or "start" vs "stop".
#
#  Outer contour (counter-clockwise from right corner):
#    right corner -> upper lip -> left corner -> lower lip -> back
#  Inner contour (counter-clockwise from right corner):
#    right corner -> upper inner -> left corner -> lower inner -> back
#
LIP_INDICES = [
    # ── Outer lip contour (20 points) ─────────────────────────────────────────
     61, 185,  40,  39,  37,   0, 267, 269, 270, 409,   # upper outer
    291, 375, 321, 405, 314,  17,  84, 181,  91, 146,   # lower outer
    # ── Inner lip contour (20 points) ─────────────────────────────────────────
     78, 191,  80,  81,  82,  13, 312, 311, 310, 415,   # upper inner
    308, 324, 318, 402, 317,  14,  87, 178,  88,  95,   # lower inner
]

# ── Class labels ──────────────────────────────────────────────────────────────
#   words   -> class  0-9
#   phrases -> class 10-19
WORD_LABELS = {
    "01": "begin",      "02": "choose",     "03": "connection",
    "04": "navigation", "05": "next",       "06": "previous",
    "07": "start",      "08": "stop",       "09": "hello",
    "10": "well done",
}
PHRASE_LABELS = {
    "01": "stop navigation", "02": "excuse me",        "03": "i am sorry",
    "04": "thank you",       "05": "good bye",         "06": "i love this game",
    "07": "nice to meet you","08": "you are welcome",  "09": "how are you",
    "10": "have a good time",
}

# Flat ordered list — index == class integer fed to the model
CLASSES     = list(WORD_LABELS.values()) + list(PHRASE_LABELS.values())
NUM_CLASSES = len(CLASSES)   # 20

# ── Prediction ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45   # below this -> shown as "uncertain"
