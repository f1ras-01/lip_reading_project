# ─────────────────────────────────────────────────────────────────────────────
#  extract_landmarks.py  –  Walk the entire MIRACL-VC1 tree, extract
#  MediaPipe lip landmarks from every color frame, normalise them, resample
#  to a fixed length and save a single .pkl ready for training.
#
#  Run once:
#      python extract_landmarks.py
#  Expected output (≈19 speakers × 20 classes × 10 instances):
#      Dataset shape: (3800, 75, 40)   [some frames may be skipped on failure]
# ─────────────────────────────────────────────────────────────────────────────

import os, glob, pickle
import cv2
import numpy as np
from tqdm import tqdm

from config import (
    DATASET_ROOT, LANDMARKS_PKL,
    WORD_LABELS, PHRASE_LABELS, CLASSES,
    NUM_FRAMES, NUM_FEATURES,
)
from mediapipe_compat import setup_static

# ── MediaPipe: initialise once for the whole extraction run ──────────────────
#  setup_static() handles the 0.10 Tasks API and auto-downloads the model file.
#  It returns a plain function: extract(bgr_frame) → (40,) array | None
_extract_frame = setup_static()


# ─────────────────────────────────────────────────────────────────────────────
#  Sequence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample(seq: np.ndarray, target: int = NUM_FRAMES) -> np.ndarray:
    n = len(seq)
    if n == 0:
        raise ValueError("Empty sequence passed to _resample")
    if n < target:
        pad = np.repeat(seq[-1:], target - n, axis=0)
        return np.vstack([seq, pad])
    if n > target:
        idx = np.linspace(0, n - 1, target, dtype=int)
        return seq[idx]
    return seq


# ─────────────────────────────────────────────────────────────────────────────
#  Per-instance processing
# ─────────────────────────────────────────────────────────────────────────────

def process_instance(folder: str):
    jpgs = sorted(glob.glob(os.path.join(folder, "color_*.jpg")))
    if not jpgs:
        return None

    frames = []
    for path in jpgs:
        img = cv2.imread(path)
        if img is None:
            continue
        feat = _extract_frame(img)   # normalised (40,) or None
        if feat is not None:
            frames.append(feat)

    if len(frames) < 5:
        return None

    seq = np.array(frames, dtype=np.float32)
    return _resample(seq)


# ─────────────────────────────────────────────────────────────────────────────
#  Full dataset traversal
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    X, y, skipped = [], [], 0

    speakers = sorted(
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    )
    print(f"Found {len(speakers)} speakers: {speakers}")

    for speaker in tqdm(speakers, desc="Speakers"):
        speaker_path = os.path.join(DATASET_ROOT, speaker)

        for category, label_map, offset in [
            ("words",   WORD_LABELS,   0),
            ("phrases", PHRASE_LABELS, 10),
        ]:
            cat_path = os.path.join(speaker_path, category)
            if not os.path.isdir(cat_path):
                continue

            for word_id in sorted(os.listdir(cat_path)):
                word_path = os.path.join(cat_path, word_id)
                if not os.path.isdir(word_path):
                    continue
                if word_id not in label_map:
                    continue

                class_idx = offset + (int(word_id) - 1)

                for instance in sorted(os.listdir(word_path)):
                    inst_path = os.path.join(word_path, instance)
                    if not os.path.isdir(inst_path):
                        continue

                    feats = process_instance(inst_path)
                    if feats is not None:
                        X.append(feats)
                        y.append(class_idx)
                    else:
                        skipped += 1

    print(f"\nExtraction complete — {len(X)} instances kept, {skipped} skipped.")
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Output file  : {LANDMARKS_PKL}")
    print(f"Sequence len : {NUM_FRAMES} frames  x {NUM_FEATURES} features\n")

    X, y = build_dataset()

    print(f"\nFinal shapes  : X={X.shape}  y={y.shape}")
    print(f"Class distribution (should be ~balanced):")
    for i, cls in enumerate(CLASSES):
        count = int((y == i).sum())
        bar = "X" * (count // 2)
        print(f"  {i:2d}  {cls:<22s}  {count:4d}  {bar}")

    with open(LANDMARKS_PKL, "wb") as f:
        pickle.dump({"X": X, "y": y, "classes": CLASSES}, f)

    print(f"\nSaved -> {LANDMARKS_PKL}  ({os.path.getsize(LANDMARKS_PKL) / 1e6:.1f} MB)")
