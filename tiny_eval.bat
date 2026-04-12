@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% .\tools\stage1_pretrain_sampling_debug.py ^
  --model-path save\stage1_tiny_overfit_all_move_fm_s100000\stage1_pretrain\model000005000.pt ^
  --output-dir outputs\stage1_tiny_overfit_all_move_fm_s5000 ^
  --eval-split train ^
  --num-eval-samples 8 ^
  --num-trials 2 ^
  --eval-batch-size 8 ^
  --fm-solver midpoint ^
  --fm-num-steps 50 ^
  --export-samples 8 