# Lip Reader — MIRACL-VC1 + MediaPipe + TCN

A complete pipeline that reads lips in real time from a webcam, trained on the
MIRACL-VC1 dataset (10 words + 10 phrases = 20 classes, 19 speakers).

---

## Project layout

```
lip_reader/
├── config.py              ← single source of truth (paths, hyper-params, labels)
├── extract_landmarks.py   ← step 1: walk MIRACL-VC1, extract & normalise landmarks
├── train.py               ← step 2: augment, train TCN, evaluate
├── predict.py             ← step 3: webcam demo or video-file inference
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9.x |
| TensorFlow | 2.10.1 |
| CUDA Toolkit | 11.2 |
| cuDNN | 8.1 |

> TF 2.10 is the **last** version with native Windows GPU support.
> CUDA 11.2 + cuDNN 8.1 are the correct pair.  
> Download cuDNN from https://developer.nvidia.com/cudnn

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Step 1 — Download and extract the dataset

1. Download from Kaggle: https://www.kaggle.com/datasets/apoorvwatsky/miraclvc1/data  
2. Unzip so the layout is:

```
miraclvc1/
└── dataset/
    ├── F01/
    │   ├── words/
    │   │   ├── 01/
    │   │   │   ├── 01/
    │   │   │   │   ├── color_001.jpg
    │   │   │   │   └── ...
    │   │   │   └── 02/ … 10/
    │   │   └── 02/ … 10/
    │   └── phrases/
    │       └── (same layout)
    └── F02/ … M08/
```

3. Place the `miraclvc1/` folder next to `config.py`.

---

## Step 2 — Extract landmarks  (~15-30 min, CPU-bound)

```bash
python extract_landmarks.py
```

Produces `landmarks.pkl` (~50 MB).  
Expected output: roughly 3 500–3 800 instances across 20 classes.

---

## Step 3 — Train  (~30-60 min on NVIDIA GPU)

```bash
python train.py
```

Key outputs:
- `lip_reader.h5`  — best checkpoint (saved whenever val_accuracy improves)
- `logs/`          — TensorBoard logs  
- `training_history.png`

Monitor training live:
```bash
tensorboard --logdir logs
```

Typical results on MIRACL-VC1:
- Speaker-dependent split: **~85-92% test accuracy**
- The model improves throughout training; early stopping prevents over-fitting.

---

## Step 4 — Real-time webcam demo

```bash
python predict.py
```

### Controls

| Key | Action |
|---|---|
| `SPACE` | Start / stop manual recording — say the word then press again |
| `R` | Auto-record for exactly 3 seconds — cleanest for single words |
| `C` | Clear the last result |
| `Q` | Quit |

### Tips for best accuracy on yourself

1. **Lighting** — face a window or lamp; avoid backlight.  
2. **Distance** — keep your face 40–70 cm from the camera so it fills ~1/4 of the frame.  
3. **Pace** — speak at the same pace as the dataset (natural, not slow).  
4. **`R` mode** — the 3-second auto-capture is most reliable because it gives the model a consistent window length.
5. If accuracy is low, try recording a short video of yourself saying each word and run:  
   ```bash
   python predict.py my_video.mp4
   ```

---

## Predict from a video file

```bash
python predict.py path/to/video.mp4
```

Prints prediction + full top-5 ranking.

---

## Configuration

All tuneable values live in `config.py`.  
The most useful ones to adjust:

| Parameter | Default | Effect |
|---|---|---|
| `NUM_FRAMES` | 75 | Sequence length (must match extraction) |
| `EPOCHS` | 150 | Max training epochs (early stopping applies) |
| `BATCH_SIZE` | 32 | Lower if GPU OOM |
| `CONFIDENCE_THRESHOLD` | 0.45 | Below this → shown as "uncertain" |
| `DATASET_ROOT` | `miraclvc1/dataset` | Path to unzipped dataset |

---

## How the normalisation works

Raw MediaPipe landmarks (x, y in image-space 0-1) shift with head position
and scale with face size.  Two steps remove this dependency before feeding
the model:

1. **Centre** — subtract the centroid of the 20 lip points per frame.  
2. **Scale** — divide by the RMS distance of all points from the centroid.

This means the model sees only mouth *shape*, not position or size — which is
why it generalises to your face even though it was trained on other people.

---

## Architecture — Temporal Convolutional Network (TCN)

```
Input (75, 40)
  └─ Conv1D 1×1 projection → 64 channels
       └─ TCN Stack A: 5 residual blocks, dilations [1,2,4,8,16], 64 filters
            └─ TCN Stack B: 3 residual blocks, dilations [1,2,4],   128 filters
                 └─ GlobalAveragePooling1D
                      └─ Dense 256 → Dropout 0.4
                           └─ Dense 128 → Dropout 0.3
                                └─ Dense 20  (softmax)
```

Causal dilated convolutions capture temporal context without future leakage,
making the same architecture usable for streaming (real-time) inference.
