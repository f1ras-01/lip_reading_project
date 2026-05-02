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
import mediapipe as mp
from tqdm import tqdm
from config import (
    DATASET_ROOT, LANDMARKS_PKL, LIP_INDICES,
    WORD_LABELS, PHRASE_LABELS, CLASSES,
    NUM_FRAMES, NUM_FEATURES,
)

# ── MediaPipe: static-image mode for per-frame extraction ────────────────────
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _raw_landmarks(image_path: str):
    """Return raw (40,) landmark array for one JPEG, or None on failure."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    xs = np.array([lm[i].x for i in LIP_INDICES], dtype=np.float32)
    ys = np.array([lm[i].y for i in LIP_INDICES], dtype=np.float32)
    return np.concatenate([xs, ys])   # shape (40,)


def _normalise(raw: np.ndarray) -> np.ndarray:
    """
    Make landmarks invariant to face position and scale.

    Strategy
    --------
    1. Centre  : subtract the centroid of all lip points.
    2. Scale   : divide by the RMS distance of points from the centroid.
                 This preserves the mouth *shape* regardless of how close
                 the speaker sits to the camera.
    """
    n = len(LIP_INDICES)
    xs = raw[:n].copy()
    ys = raw[n:].copy()

    cx, cy = xs.mean(), ys.mean()
    xs -= cx
    ys -= cy

    scale = float(np.sqrt((xs**2 + ys**2).mean())) + 1e-8
    xs /= scale
    ys /= scale

    return np.concatenate([xs, ys])   # shape (40,)


def _resample(seq: np.ndarray, target: int = NUM_FRAMES) -> np.ndarray:
    """
    Bring a variable-length sequence to exactly `target` frames.

    * Shorter: pad by repeating the last frame.
    * Longer : uniformly subsample.
    """
    n = len(seq)
    if n == 0:
        raise ValueError("Empty sequence")
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
    """
    Process one leaf folder (e.g. …/F01/words/01/03/).

    Returns
    -------
    np.ndarray of shape (NUM_FRAMES, 40), or None if too few frames detected.
    """
    jpgs = sorted(glob.glob(os.path.join(folder, "color_*.jpg")))
    if not jpgs:
        return None

    frames = []
    for path in jpgs:
        raw = _raw_landmarks(path)
        if raw is not None:
            frames.append(_normalise(raw))

    # Require at least 5 valid frames (some instances have detection failures)
    if len(frames) < 5:
        return None

    seq = np.array(frames, dtype=np.float32)   # (T, 40)
    return _resample(seq)                        # (NUM_FRAMES, 40)


# ─────────────────────────────────────────────────────────────────────────────
#  Full dataset traversal
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    """
    Walk DATASET_ROOT and return (X, y).

    X : np.ndarray  (N, NUM_FRAMES, NUM_FEATURES)
    y : np.ndarray  (N,)  – integer class indices matching CLASSES list
    """
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

                class_idx = offset + (int(word_id) - 1)   # 0-based

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
    print(f"Sequence len : {NUM_FRAMES} frames  × {NUM_FEATURES} features\n")

    X, y = build_dataset()

    print(f"\nFinal shapes : X={X.shape}  y={y.shape}")
    print(f"Class distribution (should be balanced):")
    for i, cls in enumerate(CLASSES):
        count = int((y == i).sum())
        bar = "█" * (count // 2)
        print(f"  {i:2d}  {cls:<22s}  {count:4d}  {bar}")

    with open(LANDMARKS_PKL, "wb") as f:
        pickle.dump({"X": X, "y": y, "classes": CLASSES}, f)

    print(f"\nSaved → {LANDMARKS_PKL}  ({os.path.getsize(LANDMARKS_PKL) / 1e6:.1f} MB)")
