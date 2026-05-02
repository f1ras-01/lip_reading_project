@echo off
:: =============================================================================
::  setup_env.bat  –  One-shot environment setup for the lip-reader project
::
::  Requirements
::  ────────────
::  • Anaconda or Miniconda installed  (https://docs.conda.io/en/latest/miniconda.html)
::  • NVIDIA driver ≥ 452.39           (check: nvidia-smi)
::  • Internet access
::
::  Run once from the project root:
::      setup_env.bat
:: =============================================================================
setlocal EnableDelayedExpansion

set ENV_NAME=lip_reader
set PYTHON_VER=3.9.13

echo.
echo ============================================================
echo   Lip Reader — Environment Setup
echo ============================================================
echo.

:: ── 0. Verify conda is available ─────────────────────────────────────────────
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found.
    echo         Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
    echo         Then re-run this script.
    pause
    exit /b 1
)
echo [OK] conda found.

:: ── 1. Verify NVIDIA driver is present ───────────────────────────────────────
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARN] nvidia-smi not found — your NVIDIA driver may not be installed.
    echo        Download from: https://www.nvidia.com/drivers
    echo        Continuing anyway; training will fall back to CPU.
    echo.
) else (
    echo [OK] NVIDIA driver detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
)

:: ── 2. Create conda environment ───────────────────────────────────────────────
echo [1/5] Creating conda environment "%ENV_NAME%" with Python %PYTHON_VER% ...
conda create -n %ENV_NAME% python=%PYTHON_VER% -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment.
    pause
    exit /b 1
)
echo [OK] Environment created.
echo.

:: ── 3. Install CUDA toolkit + cuDNN via conda-forge ──────────────────────────
::
::  TensorFlow 2.10.1 requires:
::    CUDA  11.2.x
::    cuDNN  8.1.x
::
::  We install them inside the conda env so they are completely isolated from
::  any system-level CUDA installation and do not conflict with other projects.
::
echo [2/5] Installing CUDA 11.2 + cuDNN 8.1 (this may take a few minutes) ...
call conda run -n %ENV_NAME% conda install -c conda-forge ^
    cudatoolkit=11.2 ^
    cudnn=8.1.0 ^
    -y
if errorlevel 1 (
    echo [ERROR] CUDA/cuDNN installation failed.
    pause
    exit /b 1
)
echo [OK] CUDA toolkit installed inside conda env.
echo.

:: ── 4. Install Python packages ────────────────────────────────────────────────
echo [3/5] Installing Python packages from requirements.txt ...
call conda run -n %ENV_NAME% pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo [OK] Python packages installed.
echo.

:: ── 5. Run GPU verification ───────────────────────────────────────────────────
echo [4/5] Verifying TensorFlow GPU detection ...
call conda run -n %ENV_NAME% python verify_gpu.py
echo.

:: ── 6. Done ───────────────────────────────────────────────────────────────────
echo [5/5] Setup complete.
echo.
echo ============================================================
echo   Next steps:
echo ============================================================
echo   1. Activate the environment in every new terminal:
echo         conda activate %ENV_NAME%
echo.
echo   2. Extract landmarks (run once):
echo         python extract_landmarks.py
echo.
echo   3. Train:
echo         python train.py
echo.
echo   4. Webcam demo:
echo         python predict.py
echo ============================================================
echo.
pause
