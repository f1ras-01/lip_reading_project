# =============================================================================
#  verify_gpu.py  –  Confirm TensorFlow sees your NVIDIA GPU and nothing else.
#
#  Run standalone at any time:
#      python verify_gpu.py
#
#  Called automatically by setup_env.bat after installation.
# =============================================================================

import os
import sys

# ── Force NVIDIA GPU only before importing TensorFlow ────────────────────────
#
#  CUDA_VISIBLE_DEVICES controls which CUDA-capable cards TF can see.
#  Setting it to "0" means "use only the first CUDA device" — which on a
#  dual-GPU laptop (Intel iGPU + NVIDIA dGPU) will always be the NVIDIA card,
#  because Intel's integrated GPU has no CUDA support and is never enumerated
#  by the CUDA runtime.
#
#  The two environment variables below suppress verbose TF startup logs so
#  the output of this script stays readable.
#
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress INFO / WARNING / ERROR logs

import tensorflow as tf

SEP  = "=" * 60
SEP2 = "-" * 60

def _ok(msg):  print(f"  [OK]   {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")


def check_python():
    print("\n── Python ──────────────────────────────────────────────────")
    v = sys.version_info
    print(f"  Version : {sys.version}")
    if v.major == 3 and v.minor == 9:
        _ok("Python 3.9 ✓")
    else:
        _warn(f"Expected Python 3.9, got {v.major}.{v.minor}. "
              "TF 2.10 works best on 3.9.")


def check_tensorflow():
    print("\n── TensorFlow ──────────────────────────────────────────────")
    print(f"  Version : {tf.__version__}")
    if tf.__version__.startswith("2.10"):
        _ok("TensorFlow 2.10 ✓")
    else:
        _warn(f"Expected 2.10.x, got {tf.__version__}")


def check_gpu():
    print("\n── GPU Detection ───────────────────────────────────────────")

    physical = tf.config.list_physical_devices("GPU")
    cpu_only = tf.config.list_physical_devices("CPU")

    print(f"  Physical CPUs : {len(cpu_only)}")
    print(f"  Physical GPUs : {len(physical)}")

    if not physical:
        _fail("No GPU detected by TensorFlow.")
        print()
        print("  Possible causes:")
        print("  • CUDA 11.2 or cuDNN 8.1 not installed / not on PATH")
        print("  • NVIDIA driver outdated (need ≥ 452.39)")
        print("  • Environment was not activated before running this script")
        print()
        print("  Quick checks:")
        print("    nvidia-smi                  ← driver check")
        print("    conda activate lip_reader   ← activate env first")
        return False

    for i, gpu in enumerate(physical):
        # Enable memory growth so TF does not pre-allocate all VRAM
        tf.config.experimental.set_memory_growth(gpu, True)
        _ok(f"GPU {i}: {gpu.name}")

    # Show logical devices (after memory-growth config)
    logical = tf.config.list_logical_devices("GPU")
    print(f"\n  Logical GPU(s) visible to TF: {len(logical)}")
    for g in logical:
        print(f"    {g.name}")

    return True


def check_cuda_ops():
    print("\n── CUDA Compute Test ───────────────────────────────────────")
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
        _ok(f"Matrix multiply on GPU succeeded → result shape {c.shape}")
        return True
    except Exception as e:
        _fail(f"GPU compute failed: {e}")
        return False


def check_model_build():
    """Build the actual project model and confirm it places ops on GPU."""
    print("\n── Model Build Test ────────────────────────────────────────")
    try:
        from tensorflow.keras import layers, models

        # Minimal replica of the project's TCN
        inp = layers.Input(shape=(75, 40))
        x   = layers.Conv1D(64, 1)(inp)
        x   = layers.GlobalAveragePooling1D()(x)
        out = layers.Dense(20, activation="softmax")(x)
        m   = models.Model(inp, out)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        import numpy as np
        dummy_x = np.random.randn(4, 75, 40).astype("float32")
        dummy_y = np.array([0, 1, 2, 3], dtype="int32")
        m.fit(dummy_x, dummy_y, epochs=1, verbose=0)

        _ok("Model built and ran a dummy training step successfully.")
        return True
    except Exception as e:
        _fail(f"Model build/run failed: {e}")
        return False


def print_gpu_memory():
    """Print VRAM usage after all tests."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print("\n── GPU Memory (MB) ─────────────────────────────────────────")
            for line in result.stdout.strip().splitlines():
                name, used, free, total = [s.strip() for s in line.split(",")]
                bar_len = 30
                used_bar = int(int(used) / int(total) * bar_len)
                bar = "█" * used_bar + "░" * (bar_len - used_bar)
                print(f"  {name}")
                print(f"  [{bar}] {used}/{total} MB used  ({free} MB free)")
    except Exception:
        pass   # nvidia-smi unavailable — skip silently


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(SEP)
    print("  verify_gpu.py — TensorFlow GPU Verification")
    print(SEP)

    check_python()
    check_tensorflow()
    gpu_ok   = check_gpu()
    cuda_ok  = check_cuda_ops() if gpu_ok else False
    model_ok = check_model_build()
    print_gpu_memory()

    print()
    print(SEP)
    if gpu_ok and cuda_ok and model_ok:
        print("  ✓  All checks passed — GPU is ready for training.")
    elif model_ok:
        print("  ⚠  GPU not available, but CPU training will work.")
        print("     See GPU detection errors above to fix.")
    else:
        print("  ✗  Something is broken. Review the errors above.")
    print(SEP)
    print()
