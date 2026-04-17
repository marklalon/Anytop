@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=stage1_tiny_overfit_horse_locomotion_pose_v1
set SAVE_DIR=%SCRIPT_DIR%save\%RUN_NAME%\stage1_pretrain
set OUTPUT_DIR=%SCRIPT_DIR%outputs\%RUN_NAME%
rem set MODEL_PATH="D:\AI\pcvg-skeleton-animation\Anytop\save\stage1_tiny_overfit_horse_resetless_v1\stage1_pretrain\model000040000.pt"

if not defined MODEL_PATH (
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-ChildItem -Path '%SAVE_DIR%' -Filter 'model*.pt' | Where-Object { $_.BaseName -match '^model\d+$' } | Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName"`) do set MODEL_PATH=%%I
)

if not defined MODEL_PATH (
  echo No checkpoint found under "%SAVE_DIR%".
  exit /b 1
)

%PYTHON_EXE% .\tools\stage1_pretrain_sampling_debug.py ^
  --model-path "%MODEL_PATH%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --num-frames 60 ^
  --num-eval-samples 32 ^
  --num-threads 8 ^
  --sampling-method p ^
  --sampling-steps 100 ^
  --no-ema %*

endlocal