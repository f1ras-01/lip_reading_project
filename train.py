# ─────────────────────────────────────────────────────────────────────────────
#  train.py  –  Train the TCN lip-reading model on MIRACL-VC1 landmarks.
#
#  Run after extract_landmarks.py:
#      python train.py
#
#  Output: lip_reader.h5  +  logs/  (TensorBoard)
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 0: GPU wiring (must happen before any TF import) ────────────────────
#
#  On a laptop with Intel integrated + NVIDIA discrete, TensorFlow 2.10
#  on Windows uses CUDA, so it will only see the NVIDIA card.
#  CUDA_VISIBLE_DEVICES=0 makes sure TF picks the first (and only) CUDA GPU.
#  We also cap memory growth so the GPU doesn't pre-allocate everything.
#
import os
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"       # use NVIDIA, ignore Intel iGPU
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"        # suppress verbose TF startup logs

import pickle, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib; matplotlib.use("Agg")          # headless plotting
import matplotlib.pyplot as plt

from config import (
    LANDMARKS_PKL, MODEL_PATH, LOG_DIR, HISTORY_PNG,
    NUM_FRAMES, NUM_FEATURES, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, CLASSES,
)

# ── GPU setup ────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU detected: {[g.name for g in gpus]}")
else:
    print("⚠  No GPU found – training on CPU (will be slow)")


# ─────────────────────────────────────────────────────────────────────────────
#  Data augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment(seq: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a single sequence (NUM_FRAMES, NUM_FEATURES).

    Techniques
    ----------
    1. Gaussian noise  – simulate landmark jitter / detection imprecision.
    2. Temporal shift  – randomly roll the sequence up to ±6 frames
                          (mimics variation in recording start time).
    3. Speed warp      – randomly stretch/compress time by ±15 %
                          (mimics different speaking rates).
    """
    seq = seq.copy()

    # 1. Gaussian noise
    seq += np.random.normal(0, 0.012, seq.shape).astype(np.float32)

    # 2. Temporal shift
    if np.random.rand() > 0.4:
        shift = np.random.randint(-6, 7)
        seq = np.roll(seq, shift, axis=0)

    # 3. Speed warp  (resample to a slightly different length, then back)
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.85, 1.15)
        warped_len = max(10, int(NUM_FRAMES * factor))
        idx_orig  = np.linspace(0, NUM_FRAMES - 1, warped_len)
        idx_back  = np.linspace(0, warped_len - 1, NUM_FRAMES).astype(int)
        seq_warp  = np.array(
            [np.interp(idx_orig, np.arange(NUM_FRAMES), seq[:, d])
             for d in range(NUM_FEATURES)], dtype=np.float32
        ).T                    # (warped_len, NUM_FEATURES)
        seq = seq_warp[idx_back]

    return seq.astype(np.float32)


class AugmentedSequence(tf.keras.utils.Sequence):
    """Keras Sequence that optionally augments each batch on-the-fly."""

    def __init__(self, X, y, batch_size=BATCH_SIZE, augment_data=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment_data = augment_data
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        X_batch = np.array(
            [augment(self.X[i]) if self.augment_data else self.X[i]
             for i in batch_idx],
            dtype=np.float32,
        )
        return X_batch, self.y[batch_idx]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ─────────────────────────────────────────────────────────────────────────────
#  Model: Temporal Convolutional Network (TCN)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Why TCN for lip reading?
#  ------------------------
#  * Causal convolutions respect temporal ordering (no future leakage).
#  * Dilated convolutions capture long-range dependencies without
#    the vanishing-gradient problems of RNNs.
#  * Residual connections stabilise deep networks on small datasets.

def _residual_block(x, dilation: int, filters: int, dropout: float = 0.1):
    """One TCN residual block with two causal dilated convolutions."""
    shortcut = x

    # First causal conv
    h = layers.Conv1D(filters, kernel_size=3, padding="causal",
                      dilation_rate=dilation, use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.SpatialDropout1D(dropout)(h)

    # Second causal conv
    h = layers.Conv1D(filters, kernel_size=3, padding="causal",
                      dilation_rate=dilation, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.SpatialDropout1D(dropout)(h)

    # 1×1 projection when channel width changes
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, use_bias=False)(shortcut)

    return layers.Add()([shortcut, h])


def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    TCN architecture:

        Input (75, 40)
        → Conv1D projection to 64 channels
        → TCN stack A: 5 blocks, dilations [1,2,4,8,16], 64 filters
        → TCN stack B: 3 blocks, dilations [1,2,4],      128 filters
        → GlobalAveragePooling
        → Dense 256 → Dropout 0.4
        → Dense 128 → Dropout 0.3
        → Dense 20  softmax
    """
    inp = layers.Input(shape=(NUM_FRAMES, NUM_FEATURES), name="landmarks")

    # Projection
    x = layers.Conv1D(64, 1, use_bias=False, name="input_proj")(inp)

    # Stack A – fine-grained temporal patterns
    for i in range(5):
        x = _residual_block(x, dilation=2**i, filters=64, dropout=0.05)

    # Stack B – longer-range context, wider channels
    for i in range(3):
        x = _residual_block(x, dilation=2**i, filters=128, dropout=0.10)

    # Classifier head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inp, out, name="LipReader_TCN")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history, save_path=HISTORY_PNG):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[0].plot(history.history["accuracy"],     label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val acc")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Training curves saved → {save_path}")


def print_per_class_report(model, X_test, y_test):
    preds      = model.predict(X_test, batch_size=64, verbose=0)
    pred_cls   = np.argmax(preds, axis=1)

    print("\n── Per-class results on TEST set ─────────────────────────────────")
    print(classification_report(y_test, pred_cls,
                                 target_names=CLASSES, zero_division=0))

    # Confusion matrix (print only non-zero cells to keep it readable)
    cm = confusion_matrix(y_test, pred_cls)
    print("Confusion matrix (rows=true, cols=pred) – non-zero entries only:")
    for i, row in enumerate(cm):
        errors = [(j, v) for j, v in enumerate(row) if v > 0 and j != i]
        if errors:
            wrong = ", ".join(f"{CLASSES[j]}({v})" for j, v in errors)
            print(f"  {CLASSES[i]:22s} correct={row[i]:3d}  confused with → {wrong}")
        else:
            print(f"  {CLASSES[i]:22s} correct={row[i]:3d}  (perfect)")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # ── Load landmarks ───────────────────────────────────────────────────────
    print(f"\nLoading {LANDMARKS_PKL} …")
    with open(LANDMARKS_PKL, "rb") as f:
        data = pickle.load(f)
    X: np.ndarray = data["X"]   # (N, 75, 40)
    y: np.ndarray = data["y"]   # (N,)
    print(f"  Loaded X={X.shape}  y={y.shape}  classes={NUM_CLASSES}")

    # ── Split ────────────────────────────────────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=y,
        random_state=RANDOM_SEED,
    )
    # Split the held-out portion into val / test equally
    val_frac = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1.0 - val_frac,
        stratify=y_tmp,
        random_state=RANDOM_SEED,
    )
    print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    train_gen = AugmentedSequence(X_train, y_train, augment_data=True)
    val_gen   = AugmentedSequence(X_val,   y_val,   augment_data=False)

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    cbs = [
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=12,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training …  (TensorBoard: tensorboard --logdir logs)\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1,
    )

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\nLoading best checkpoint for evaluation …")
    best_model = models.load_model(MODEL_PATH)

    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test accuracy: {acc*100:.2f}%   loss: {loss:.4f}")

    print_per_class_report(best_model, X_test, y_test)
    plot_history(history)

    print(f"\nAll done.  Model saved → {MODEL_PATH}")
