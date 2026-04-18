@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=stage1_tiny_overfit_horse_locomotion_pose_v5

pushd "%SCRIPT_DIR%"

REM 使用 --objects_subset Horse 指定单个物种的所有动作作为训练集
REM 支持任何在数据集中的物种名称，如 Dragon, Bird, Camel 等
%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--output-dir save/%RUN_NAME% ^
	--objects_subset Horse ^
	--action_tags locomotion,pose ^
	--use_action_cond ^
	--latent_dim 256 ^
	--layers 8 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--stage1_batch_size 4 ^
	--stage1_lr 5e-5 ^
	--stage1_num_steps 30000 ^
	--dropout_prob 0.1 ^
	--lambda_geo 0.0 ^
	--save_interval 2000 ^
	--log_interval 100 ^
	--num_workers 0 ^
	--motion_cache_size 512 ^
	--main_process_prefetch_batches 4 ^
	--amp_dtype fp32 ^
	--cudnn_benchmark ^
	--no-allow_tf32 ^
	--auto_resume ^
	--ml_platform_type TensorboardPlatform
	
popd