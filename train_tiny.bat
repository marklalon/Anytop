@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--experiment_root save/stage1_tiny_overfit_all_move_fm_s100000 ^
	--motion_name_keywords walk,run,sprint ^
	--overwrite ^
	--objects_subset all ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--stage1_batch_size 32 ^
	--stage1_sample_limit -1 ^
	--stage1_lr 1e-4 ^
	--stage1_num_steps 100000 ^
	--lambda_geo 1.0 ^
	--save_interval 5000 ^
	--log_interval 100 ^
	--num_workers 8 ^
	--amp_dtype bf16 ^
	--cudnn_benchmark ^
	--allow_tf32 ^
	--ml_platform_type TensorboardPlatform ^
	--use_ema