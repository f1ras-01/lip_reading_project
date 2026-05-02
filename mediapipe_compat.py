# =============================================================================
#  mediapipe_compat.py  –  Unified lip-landmark extractor for MediaPipe 0.10+
#
#  MediaPipe 0.10 removed mp.solutions entirely and replaced it with a
#  Tasks API that requires a downloaded .task model file.
#
#  This module hides all of that complexity behind two functions:
#
#    setup_static()   – call once in extract_landmarks.py  (image mode)
#    setup_video()    – call once in predict.py            (video/webcam mode)
#
#  Both return an extract(frame_bgr) function that accepts a BGR numpy array
#  and returns a normalised (40,) float32 landmark vector, or None on failure.
# =============================================================================

import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

from config import LIP_INDICES

# ── Model file ────────────────────────────────────────────────────────────────
#  MediaPipe 0.10 no longer bundles the model inside the pip package.
#  We download it once into the project root and reuse it from there.

_MODEL_PATH = "face_landmarker.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model():
    """Download face_landmarker.task if it is not already present."""
    if os.path.isfile(_MODEL_PATH):
        return
    print(f"Downloading MediaPipe face landmarker model → {_MODEL_PATH}")
    print("(~29 MB, one-time download)")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"  Saved → {_MODEL_PATH}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the MediaPipe model.\n"
            f"URL : {_MODEL_URL}\n"
            f"Error: {exc}\n\n"
            f"Manual fix: download the file from the URL above and place it "
            f"in the project root as '{_MODEL_PATH}'."
        ) from exc


# ── Normalisation (same maths as the rest of the pipeline) ───────────────────

def _normalise(raw: np.ndarray) -> np.ndarray:
    """
    Remove position and scale from raw (40,) landmark vector.

    1. Centre  : subtract centroid of all 20 lip points.
    2. Scale   : divide by RMS distance from centroid.
    """
    n = len(LIP_INDICES)
    xs = raw[:n].copy()
    ys = raw[n:].copy()
    cx, cy = xs.mean(), ys.mean()
    xs -= cx;  ys -= cy
    scale = float(np.sqrt((xs ** 2 + ys ** 2).mean())) + 1e-8
    xs /= scale;  ys /= scale
    return np.concatenate([xs, ys]).astype(np.float32)


# ── Landmarker factory ────────────────────────────────────────────────────────

def _make_landmarker(running_mode):
    """Build a FaceLandmarker for the given RunningMode."""
    _ensure_model()
    base_opts = _mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    opts = _mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return _mp_vision.FaceLandmarker.create_from_options(opts)


# ── Public setup functions ────────────────────────────────────────────────────

def setup_static():
    """
    Return an extract(bgr_frame) function suitable for still images.
    Use this in extract_landmarks.py.
    """
    landmarker = _make_landmarker(_mp_vision.RunningMode.IMAGE)

    def extract(bgr_frame: np.ndarray):
        """
        Accept a BGR uint8 numpy array.
        Return normalised (40,) float32 array, or None if no face detected.
        """
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])   # BGR → RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        raw = np.array(
            [lm[i].x for i in LIP_INDICES] + [lm[i].y for i in LIP_INDICES],
            dtype=np.float32,
        )
        return _normalise(raw)

    return extract


def setup_video():
    """
    Return an extract(bgr_frame, timestamp_ms) function for live video.
    The caller must pass a monotonically increasing timestamp in milliseconds.
    Use this in predict.py.
    """
    landmarker = _make_landmarker(_mp_vision.RunningMode.VIDEO)

    def extract(bgr_frame: np.ndarray, timestamp_ms: int):
        """
        Accept a BGR uint8 numpy array and a millisecond timestamp.
        Return normalised (40,) float32 array, or None if no face detected.
        """
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])   # BGR → RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        raw = np.array(
            [lm[i].x for i in LIP_INDICES] + [lm[i].y for i in LIP_INDICES],
            dtype=np.float32,
        )
        return _normalise(raw)

    return extract


def setup_video_landmark_overlay():
    """
    Return an overlay(bgr_frame, timestamp_ms) function that draws the 20 lip
    landmark dots on the frame in-place and returns the modified frame.
    Used by the webcam UI in predict.py.
    """
    landmarker = _make_landmarker(_mp_vision.RunningMode.VIDEO)

    def overlay(bgr_frame: np.ndarray, timestamp_ms: int):
        import cv2
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            h, w = bgr_frame.shape[:2]
            lm = result.face_landmarks[0]
            for idx in LIP_INDICES:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                cv2.circle(bgr_frame, (x, y), 2, (0, 255, 180), -1)
        return bgr_frame, (result.face_landmarks is not None and len(result.face_landmarks) > 0)

    return overlay
