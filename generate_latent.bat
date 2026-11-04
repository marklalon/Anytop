@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
set "RUN_NAME=tiny_latent_diffusion_fly_shake_v2"
set "MODEL_FILE=latent_diffusion.pt"
set "NUM_SAMPLES=8"

REM Usage: generate_latent.bat --object_type Horse [additional sampling arguments]
REM Examples:
REM   generate_latent.bat --object_type Horse --action_label Run
REM   generate_latent.bat --object_type Horse --fullbody_ik --stretch_factor 0
REM   generate_latent.bat --object_type Horse --reference_motion path\Flamingo_Walk.npy
REM --object_type is required, including when using a reference motion.
REM --reference_motion and --action_label are mutually exclusive.
REM The checkpoint loads its bound frozen AE; use --ae_checkpoint to relocate it.
REM Extra arguments are forwarded to sample/generate_latent.py via %*.

if "%~1"=="" (
    echo Usage: generate_latent.bat --object_type Horse [additional sampling arguments]
    exit /b 1
)
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: "%PYTHON_EXE%"
    exit /b 1
)

pushd "%SCRIPT_DIR%"
if errorlevel 1 exit /b 1

set "MODEL_PATH=save\%RUN_NAME%\%MODEL_FILE%"
set "OUTPUT_DIR=outputs\%RUN_NAME%\generate_latent"
if not exist "%MODEL_PATH%" (
    echo ERROR: Model file not found: "%MODEL_PATH%"
    popd
    exit /b 1
)

if exist "%OUTPUT_DIR%" (
    echo Cleaning output directory: %OUTPUT_DIR%
    rmdir /s /q "%OUTPUT_DIR%"
    if exist "%OUTPUT_DIR%" (
        echo ERROR: Could not clean output directory: "%OUTPUT_DIR%"
        popd
        exit /b 1
    )
)
mkdir "%OUTPUT_DIR%"
if errorlevel 1 (
    popd
    exit /b 1
)

echo Output dir: %OUTPUT_DIR%

"%PYTHON_EXE%" sample/generate_latent.py ^
    --model_path "%MODEL_PATH%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --num_samples %NUM_SAMPLES% %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
