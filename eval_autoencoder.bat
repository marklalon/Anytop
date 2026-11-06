@echo off
setlocal EnableExtensions
REM Usage: eval_autoencoder.bat --npy "dataset\...\Bear_WetShake.npy" [options]
REM Transfer: eval_autoencoder.bat --npy "dataset\...\Bear_WetShake.npy" --object_type Horse
REM Source skeleton and motion come from --npy; --object_type selects the target skeleton.
REM Without --object_type, each clip is reconstructed on its own skeleton.
REM IK BVH export: append --fullbody_ik --stretch_factor 0.1 (0 = rigid bones).
REM IK applies to both original and reconstruction BVH files; NPY features stay unchanged.
REM Set RUN_NAME before calling to select another run, or pass --model_path explicitly.
REM Outputs original/reconstruction .npy and .bvh pairs; VAE uses posterior mean.
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
if not defined RUN_NAME set "RUN_NAME=all_autoencoder_v2"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: "%PYTHON_EXE%"
    exit /b 1
)
pushd "%SCRIPT_DIR%"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" sample/eval_autoencoder.py --model_path "save/%RUN_NAME%/autoencoder.pt" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
