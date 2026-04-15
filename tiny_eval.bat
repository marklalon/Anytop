@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=stage1_tiny_overfit_horse_resetless_v1
set SAVE_DIR=%SCRIPT_DIR%save\%RUN_NAME%\stage1_pretrain
set OUTPUT_DIR=%SCRIPT_DIR%outputs\%RUN_NAME%
set MODEL_PATH=

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
  --eval-split all ^
  --sample-limit -1 ^
  --num-frames 30 ^
  --num-eval-samples 4 ^
  --num-trials 1 ^
  --eval-batch-size 4 ^
  --sampling-method ddim ^
  --sampling-steps 100 ^
  --no-ema %*

endlocal