@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=stage1_tiny_overfit_horse_resetless_v2

pushd "%SCRIPT_DIR%"

%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--output-dir save/%RUN_NAME% ^
	--fixed_motion dataset\truebones\zoo\truebones_processed\bvhs\Horse___Restless2_25.bvh ^
	--fixed_window_start -1 ^
	--auto_resume ^
	--objects_subset all ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--stage1_batch_size 4 ^
	--stage1_lr 2e-5 ^
	--stage1_num_steps 50000 ^
	--dropout_prob 0.0 ^
	--lambda_geo 0.0 ^
	--save_interval 2000 ^
	--log_interval 100 ^
	--num_workers 0 ^
	--motion_cache_size 1 ^
	--main_process_prefetch_batches 1 ^
	--amp_dtype fp32 ^
	--cudnn_benchmark ^
	--no-allow_tf32 ^
	--ml_platform_type TensorboardPlatform

popd