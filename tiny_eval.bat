@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% .\tools\stage1_pretrain_sampling_debug.py ^
  --model-path D:\AI\pcvg-skeleton-animation\Anytop\save\stage1_tiny_overfit_horse_runloop28_fixed_v1\stage1_pretrain\model000010000.pt ^
  --output-dir D:\AI\pcvg-skeleton-animation\Anytop\outputs\stage1_tiny_overfit_horse_runloop28_fixed_v1 ^
  --eval-split all ^
  --sample-limit -1 ^
  --num-frames 30 ^
  --num-eval-samples 1 ^
  --num-trials 1 ^
  --eval-batch-size 1 ^
  --sampling-method ddim ^
  --sampling-steps 100 ^
  --no-ema %*