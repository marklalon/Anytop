@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% .\tools\stage1_pretrain_sampling_debug.py ^
  --model-path D:\AI\pcvg-skeleton-animation\Anytop\save\stage1_tiny_overfit_all_move_clean_s100000\stage1_pretrain\model000100000.pt ^
  --output-dir D:\AI\pcvg-skeleton-animation\Anytop\outputs\stage1_tiny_overfit_all_move_clean_s100000 ^
  --action-tags locomotion ^
  --eval-split train ^
  --num-eval-samples 32 ^
  --num-trials 1 ^
  --eval-batch-size 16 ^
  --sampling-method ddim ^
  --sampling-steps 50 %*