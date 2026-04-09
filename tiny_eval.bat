@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% .\tools\stage1_pretrain_sampling_debug.py ^
  --model-path D:\AI\pcvg-skeleton-animation\Anytop\save\stage1_tiny_overfit_all_move_s10000\stage1_pretrain\model000004000.pt ^
  --output-dir D:\AI\pcvg-skeleton-animation\Anytop\outputs\stage1_tiny_overfit_all_move_s4000 ^
  --eval-split train ^
  --num-eval-samples 8 ^
  --num-trials 2 ^
  --eval-batch-size 8 ^
  --sampling-method ddim ^
  --sampling-steps 50 ^
  --export-samples 8 ^
  --selection-seed 10