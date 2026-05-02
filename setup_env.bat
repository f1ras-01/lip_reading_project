@echo off
:: =============================================================================
::  setup_env.bat  --  Environment setup for the lip-reader project
::
::  Works with your existing setup:
::    - Python 3.9.13 already installed system-wide (used directly via venv)
::    - CUDA + cuDNN already installed system-wide (used as-is, nothing reinstalled)
::    - No Miniconda required
::
::  Run once from the project root folder:
::      setup_env.bat
:: =============================================================================
setlocal EnableDelayedExpansion

set VENV_DIR=venv

echo.
echo ============================================================
echo   Lip Reader -- Environment Setup
echo ============================================================
echo.

:: ── 0. Verify NVIDIA driver ───────────────────────────────────────────────────
echo [CHECK] NVIDIA driver ...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [WARN] nvidia-smi not found. Make sure NVIDIA drivers are installed.
) else (
    echo   [OK] NVIDIA driver:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)
echo.

:: ── 1. Verify system CUDA ─────────────────────────────────────────────────────
echo [CHECK] CUDA toolkit ...
where nvcc >nul 2>&1
if errorlevel 1 (
    echo   [WARN] nvcc not found in PATH.
    echo          TF 2.10 needs CUDA 11.2. Check that your CUDA bin folder is in PATH:
    echo          e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
    echo          If it is there and nvcc still fails, restart this terminal.
) else (
    echo   [OK] CUDA detected:
    nvcc --version | findstr /C:"release"
    nvcc --version | findstr /C:"release" | findstr /C:"11.2" >nul 2>&1
    if errorlevel 1 (
        echo   [WARN] TF 2.10 requires CUDA 11.2 specifically.
        echo          A different version was found -- GPU may not work with TF 2.10.
        echo          See INSTALL.md ^> "CUDA version mismatch" for how to handle this.
    ) else (
        echo   [OK] CUDA 11.2 confirmed -- correct version for TF 2.10.
    )
)
echo.

:: ── 2. Find Python 3.9 ────────────────────────────────────────────────────────
echo [1/4] Locating Python 3.9 ...

:: Try the Python Launcher first (py -3.9). It reads all installed versions
:: from the Windows registry and picks the right one regardless of what is
:: currently first on your PATH (avoids accidentally using 3.13 or 2.7).
py -3.9 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.9
    echo   [OK] Found via Python Launcher: & py -3.9 --version
    goto :create_venv
)

:: Fallback: look for python3.9.exe directly in common install locations
for %%P in (
    "C:\Python39\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe"
    "C:\Program Files\Python39\python.exe"
) do (
    if exist %%P (
        set PYTHON_CMD=%%P
        echo   [OK] Found at %%P
        goto :create_venv
    )
)

echo   [FAIL] Python 3.9 not found via 'py -3.9' or common paths.
echo          Make sure Python 3.9.13 is installed and the Python Launcher is enabled.
echo          See INSTALL.md for details.
pause
exit /b 1

:create_venv
:: ── 3. Create virtual environment ─────────────────────────────────────────────
echo.
echo [2/4] Creating virtual environment in .\%VENV_DIR%\ ...

if exist %VENV_DIR%\ (
    echo   [INFO] %VENV_DIR%\ already exists -- skipping creation.
    echo          Delete it and re-run if you want a clean rebuild.
    goto :install_packages
)

%PYTHON_CMD% -m venv %VENV_DIR%
if errorlevel 1 (
    echo   [FAIL] venv creation failed.
    pause
    exit /b 1
)
echo   [OK] Virtual environment created.

:install_packages
:: ── 4. Install packages ────────────────────────────────────────────────────────
echo.
echo [3/4] Installing packages from requirements.txt ...
echo       (This may take 5-10 minutes on the first run)

call %VENV_DIR%\Scripts\activate.bat

:: Upgrade pip first to avoid resolver warnings
python -m pip install --upgrade pip --quiet

pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo   [FAIL] pip install failed. See errors above.
    pause
    exit /b 1
)
echo   [OK] All packages installed.

:: ── 5. Verify GPU ─────────────────────────────────────────────────────────────
echo.
echo [4/4] Running GPU verification ...
echo.
python verify_gpu.py

:: ── 5. Initialise git repository ─────────────────────────────────────────────
echo.
echo [5/5] Setting up git ...

where git >nul 2>&1
if errorlevel 1 (
    echo   [WARN] git not found in PATH. Install from https://git-scm.com
    echo          Skipping git init -- do it manually later.
    goto :done
)

if exist .git\ (
    echo   [INFO] git repository already exists -- skipping init.
) else (
    git init
    echo   [OK] git repository initialised.
)

:: Create .gitkeep files so the managed output folders are tracked by git
:: (their contents are ignored by .gitignore, but the folders themselves
::  must exist for the project scripts to run without errors.)
for %%D in (data checkpoints plots logs) do (
    if not exist %%D\ mkdir %%D
    if not exist %%D\.gitkeep type nul > %%D\.gitkeep
)
echo   [OK] Output folders created with .gitkeep sentinels:
echo          data\          <- landmarks.pkl will be saved here
echo          checkpoints\   <- lip_reader_best.h5 will be saved here
echo          plots\         <- training_history.png will be saved here
echo          logs\          <- TensorBoard logs will be saved here

:: Stage everything for a clean first commit
git add .
echo   [OK] All project files staged.
echo.
echo   Make your first commit when ready:
echo       git commit -m "initial project structure"

:done
:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   Activate the environment in every new terminal BEFORE running
echo   any project script:
echo.
echo       venv\Scripts\activate
echo.
echo   Then run in order:
echo       python extract_landmarks.py    (once, ~20 min)
echo       python train.py                (~30-60 min on GPU)
echo       python predict.py              (webcam demo)
echo.
echo   To monitor training live:
echo       tensorboard --logdir logs
echo.
echo   Project folder structure after running:
echo       data\landmarks.pkl
echo       checkpoints\lip_reader_best.h5
echo       plots\training_history.png
echo       logs\  (TensorBoard events)
echo ============================================================
echo.
pause
