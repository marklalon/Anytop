@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe

pushd "%SCRIPT_DIR%"

%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--output-dir save/stage1_tiny_overfit_horse_runloop28_fixed_v1 ^
	--fixed_motion dataset\truebones\zoo\truebones_processed\bvhs\Horse___RunLoop_28.bvh ^
	--fixed_window_start 0 ^
	--auto_resume ^
	--objects_subset all ^
	--diffusion_steps 100 ^
	--num_frames 30 ^
	--stage1_batch_size 1 ^
	--stage1_lr 5e-5 ^
	--stage1_num_steps 10000 ^
	--dropout_prob 0 ^
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