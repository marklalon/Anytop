@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--experiment_root save/stage1_tiny_overfit_all_move_s50000 ^
	--overwrite ^
	--objects_subset all ^
	--motion_name_keywords walk,run,sprint ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--stage1_batch_size 32 ^
	--stage1_sample_limit -1 ^
	--stage1_lr 5e-5 ^
	--stage1_num_steps 50000 ^
	--save_interval 5000 ^
	--log_interval 100 ^
	--num_workers 4 ^
	--ml_platform_type TensorboardPlatform ^
	--use_ema