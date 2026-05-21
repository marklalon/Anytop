@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=quadropeds_locomotion_controlnet_v1

pushd "%SCRIPT_DIR%"

REM 使用 --objects_subset Horse 指定单个物种的所有动作作为训练集
REM 支持任何在数据集中的物种名称，如 Dragon, Bird, Camel 等
%PYTHON_EXE% train/train_anytop.py ^
	--save_dir save/%RUN_NAME% ^
	--save_interval 2000 ^
	--log_interval 100 ^
	--auto_resume ^
	--ml_platform_type TensorboardPlatform ^
	--objects_subset quadropeds ^
	--train_split train ^
	--action_tags locomotion ^
	--latent_dim 256 ^
	--layers 8 ^
	--reference_cond ^
	--cross_limb_dim 128 ^
	--temporal_window 41 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--batch_size 16 ^
	--lr 5e-5 ^
	--num_steps 100000 ^
	--lr_scheduler_step_size 5000 ^
	--lr_scheduler_gamma 0.95 ^
	--eval_during_training ^
	--eval_interval 500 ^
	--eval_num_samples 32 ^
	--dropout_prob 0.1 ^
	--aug_speed_range 0.2 ^
	--aug_mirror_prob 0.5 ^
	--joint_mask_prob 0.15 ^
	--lambda_vel 0.5 ^
	--lambda_geo 0.5 ^
	--motion_cache_size 512 ^
	--amp_dtype bf16 ^
	--main_process_prefetch_batches 64
	
popd