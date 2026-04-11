@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% train/train_anytop.py ^
	--objects_subset quadropeds ^
	--batch_size 32 --lambda_geo 1.0 --overwrite --balanced --use_ema ^
	--num_steps 30000 --save_interval 5000 --num_workers 4 ^
	--amp_dtype bf16 --cudnn_benchmark --allow_tf32 ^
	--ml_platform_type TensorboardPlatform