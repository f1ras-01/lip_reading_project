# ─────────────────────────────────────────────────────────────────────────────
#  config.py  –  Single source of truth for every script in the project
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT  = "miraclvc1/dataset"   # top-level folder inside the Kaggle zip
LANDMARKS_PKL = "landmarks.pkl"       # extracted landmarks cache
MODEL_PATH    = "lip_reader.h5"       # saved Keras model
LOG_DIR       = "logs"                # TensorBoard logs

# ── Sequence ─────────────────────────────────────────────────────────────────
NUM_FRAMES   = 75     # every sequence is resampled to this many frames
NUM_LANDMARKS = 20    # MediaPipe lip landmark indices (see below)
NUM_FEATURES  = NUM_LANDMARKS * 2   # x + y per landmark  →  40

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 150
LEARNING_RATE = 1e-3
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.10
RANDOM_SEED  = 42

# ── MediaPipe lip contour indices (inner + outer ring) ────────────────────────
LIP_INDICES = [
    61, 146,  91, 181,  84,  17, 314, 405, 321, 375,
   291, 308, 324, 318, 402, 317,  14,  87, 178,  88,
]

# ── Class labels ─────────────────────────────────────────────────────────────
#   words  → class  0-9
#   phrases → class 10-19
WORD_LABELS = {
    "01": "begin",      "02": "choose",     "03": "connection",
    "04": "navigation", "05": "next",       "06": "previous",
    "07": "start",      "08": "stop",       "09": "hello",
    "10": "well done",
}
PHRASE_LABELS = {
    "01": "stop navigation", "02": "excuse me",       "03": "i am sorry",
    "04": "thank you",       "05": "good bye",        "06": "i love this game",
    "07": "nice to meet you","08": "you are welcome", "09": "how are you",
    "10": "have a good time",
}

# Flat ordered list – index == class integer fed to the model
CLASSES = list(WORD_LABELS.values()) + list(PHRASE_LABELS.values())
NUM_CLASSES = len(CLASSES)   # 20

# ── Prediction ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45   # below this → label shown as "uncertain"
