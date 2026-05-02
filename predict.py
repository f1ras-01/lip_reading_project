# ─────────────────────────────────────────────────────────────────────────────
#  predict.py  –  Run inference on:
#    • a saved video file      →  python predict.py path/to/video.mp4
#    • live webcam             →  python predict.py
#
#  Controls (webcam mode)
#  ──────────────────────
#  SPACE  – start / stop recording
#  R      – instant 3-second auto-capture  (just say the word / phrase)
#  C      – clear last result
#  Q      – quit
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import models

from config import (
    MODEL_PATH, LIP_INDICES, NUM_FRAMES, NUM_FEATURES,
    CLASSES, NUM_CLASSES, CONFIDENCE_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe  –  video (tracking) mode for smoother real-time detection
# ─────────────────────────────────────────────────────────────────────────────
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utilities (same maths as extract_landmarks.py)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(raw: np.ndarray) -> np.ndarray:
    n = len(LIP_INDICES)
    xs = raw[:n].copy()
    ys = raw[n:].copy()
    cx, cy = xs.mean(), ys.mean()
    xs -= cx;  ys -= cy
    scale = float(np.sqrt((xs**2 + ys**2).mean())) + 1e-8
    xs /= scale;  ys /= scale
    return np.concatenate([xs, ys]).astype(np.float32)


def _extract_landmarks(frame: np.ndarray):
    """Return normalised (40,) landmark array from a BGR frame, or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    raw = np.array(
        [lm[i].x for i in LIP_INDICES] + [lm[i].y for i in LIP_INDICES],
        dtype=np.float32,
    )
    return _normalise(raw)


def _resample(seq: np.ndarray, target: int = NUM_FRAMES) -> np.ndarray:
    n = len(seq)
    if n < target:
        pad = np.repeat(seq[-1:], target - n, axis=0)
        return np.vstack([seq, pad])
    if n > target:
        idx = np.linspace(0, n - 1, target, dtype=int)
        return seq[idx]
    return seq


def _draw_lip_mesh(frame: np.ndarray) -> np.ndarray:
    """Overlay MediaPipe lip landmarks on the frame (purely visual)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return frame
    h, w = frame.shape[:2]
    lm = result.multi_face_landmarks[0].landmark
    for idx in LIP_INDICES:
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(frame, (x, y), 2, (0, 255, 180), -1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def _predict(model, landmark_frames: list):
    """
    Given a list of (40,) landmark arrays, predict the spoken word / phrase.

    Returns
    -------
    label      : str   – predicted class or "uncertain"
    confidence : float – softmax probability of top class (0-1)
    top5       : list  – [(label, prob), …] top-5 predictions
    """
    if not landmark_frames:
        return "NO FACE DETECTED", 0.0, []

    seq = _resample(np.array(landmark_frames, dtype=np.float32))  # (75, 40)
    probs = model.predict(
        seq.reshape(1, NUM_FRAMES, NUM_FEATURES), verbose=0
    )[0]

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [(CLASSES[i], float(probs[i])) for i in top5_idx]

    best_idx  = int(np.argmax(probs))
    best_prob = float(probs[best_idx])
    label     = CLASSES[best_idx] if best_prob >= CONFIDENCE_THRESHOLD else "uncertain"

    return label, best_prob, top5


# ─────────────────────────────────────────────────────────────────────────────
#  Video-file mode
# ─────────────────────────────────────────────────────────────────────────────

def predict_video(video_path: str):
    """Run inference on every frame of a saved video and print the result."""
    print(f"Predicting from: {video_path}")
    model = models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = _extract_landmarks(frame)
        if feat is not None:
            frames.append(feat)
    cap.release()

    label, conf, top5 = _predict(model, frames)
    print(f"\n{'─'*40}")
    print(f"  Prediction : {label.upper()}")
    print(f"  Confidence : {conf*100:.1f}%")
    print(f"\n  Top-5:")
    for rank, (cls, p) in enumerate(top5, 1):
        bar = "█" * int(p * 30)
        print(f"    {rank}. {cls:<22s}  {p*100:5.1f}%  {bar}")
    print(f"{'─'*40}")
    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
#  Webcam demo
# ─────────────────────────────────────────────────────────────────────────────

# UI colours
_GREEN  = (100, 220,  80)
_RED    = (  0,  60, 220)
_YELLOW = ( 30, 210, 230)
_WHITE  = (240, 240, 240)
_DARK   = ( 30,  30,  30)


def _overlay_text(frame, text, pos, scale=0.7, color=_WHITE, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, _DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_ui(frame, state: dict):
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ─────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), _DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Mode label
    mode_text, mode_color = (
        ("● RECORDING", _RED) if state["recording"]
        else ("● READY",  _GREEN)
    )
    _overlay_text(frame, mode_text, (10, 30), scale=0.8, color=mode_color)

    # Controls hint
    hints = "SPACE: start/stop  |  R: 3-sec auto  |  C: clear  |  Q: quit"
    _overlay_text(frame, hints, (10, 60), scale=0.45, color=(180, 180, 180))

    # ── Recording progress bar ───────────────────────────────────────────────
    if state["recording"]:
        progress = min(len(state["frames"]), NUM_FRAMES) / NUM_FRAMES
        bar_w    = w - 20
        cv2.rectangle(frame, (10, 70), (10 + bar_w, 85), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 70), (10 + int(bar_w * progress), 85),
                      _RED, -1)

    # ── Countdown for auto-capture ───────────────────────────────────────────
    if state.get("countdown") is not None:
        remaining = state["countdown"]
        txt = f"Auto-capture in {remaining:.1f}s …"
        _overlay_text(frame, txt, (10, 115), scale=0.65, color=_YELLOW)

    # ── Result panel ─────────────────────────────────────────────────────────
    if state["label"]:
        # Background box
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 175), (w, h), _DARK, -1)
        cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

        label_color = _YELLOW if state["conf"] >= CONFIDENCE_THRESHOLD else (120, 120, 220)
        _overlay_text(frame, f"→ {state['label'].upper()}", (10, h - 140),
                      scale=1.1, color=label_color, thickness=3)
        _overlay_text(frame, f"confidence: {state['conf']*100:.1f}%",
                      (10, h - 108), scale=0.65)

        # Top-5 mini-list
        for rank, (cls, prob) in enumerate(state["top5"][:5], 1):
            bar_len = int(prob * 120)
            y_pos   = h - 90 + (rank - 1) * 18
            cv2.rectangle(frame, (10, y_pos - 10),
                          (10 + bar_len, y_pos), (70, 130, 70), -1)
            _overlay_text(frame, f"{rank}. {cls} ({prob*100:.0f}%)",
                          (12, y_pos - 1), scale=0.38, color=_WHITE, thickness=1)

    # ── No-face warning ──────────────────────────────────────────────────────
    if state.get("no_face"):
        _overlay_text(frame, "⚠  No face detected", (w // 2 - 130, h // 2),
                      scale=0.8, color=_RED)

    return frame


def webcam_demo():
    print("Loading model …")
    model = models.load_model(MODEL_PATH)
    print("Model ready.\n")
    print("Controls:")
    print("  SPACE  – manual start / stop recording")
    print("  R      – auto-record for 3 seconds")
    print("  C      – clear result")
    print("  Q      – quit\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    state = {
        "recording": False,
        "frames":    [],
        "label":     "",
        "conf":      0.0,
        "top5":      [],
        "no_face":   False,
        "countdown": None,
        "auto_end":  None,
    }

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # mirror for natural feel

        # ── Landmark extraction ──────────────────────────────────────────────
        feat = _extract_landmarks(frame.copy())
        state["no_face"] = (feat is None)

        if state["recording"] and feat is not None:
            state["frames"].append(feat)

        # Draw lip mesh on frame
        frame = _draw_lip_mesh(frame)

        # ── Auto-capture countdown ───────────────────────────────────────────
        if state["auto_end"] is not None:
            remaining = state["auto_end"] - time.time()
            if remaining <= 0:
                # Stop and predict
                state["recording"] = False
                state["auto_end"]  = None
                state["countdown"] = None
                label, conf, top5  = _predict(model, state["frames"])
                state.update(label=label, conf=conf, top5=top5)
                print(f"Prediction: {label}  ({conf*100:.1f}%)")
            else:
                state["countdown"] = remaining
        else:
            state["countdown"] = None

        # ── Draw UI ──────────────────────────────────────────────────────────
        frame = _draw_ui(frame, state)

        # FPS counter
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_time
            fps = 30 / elapsed
            fps_time = time.time()
            _overlay_text(frame, f"FPS: {fps:.0f}", (10, frame.shape[0] - 5),
                          scale=0.4, color=(150, 150, 150), thickness=1)

        cv2.imshow("Lip Reader", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            if state["recording"]:
                # Stop manual recording and predict
                state["recording"] = False
                state["auto_end"]  = None
                label, conf, top5  = _predict(model, state["frames"])
                state.update(label=label, conf=conf, top5=top5)
                print(f"Prediction: {label}  ({conf*100:.1f}%)")
            else:
                # Start manual recording
                state["recording"] = True
                state["frames"]    = []
                state["label"]     = ""

        elif key == ord("r"):
            # Auto: record for 3 seconds then predict
            state["recording"] = True
            state["frames"]    = []
            state["label"]     = ""
            state["auto_end"]  = time.time() + 3.0

        elif key == ord("c"):
            state.update(label="", conf=0.0, top5=[])

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_video(sys.argv[1])
    else:
        webcam_demo()
