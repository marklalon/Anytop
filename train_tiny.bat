@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=stage1_tiny_overfit_horse_locomotion_pose_v17

pushd "%SCRIPT_DIR%"

REM 使用 --objects_subset Horse 指定单个物种的所有动作作为训练集
REM 支持任何在数据集中的物种名称，如 Dragon, Bird, Camel 等
%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--output-dir save/%RUN_NAME% ^
	--objects_subset quadropeds_test ^
	--action_tags locomotion,jump,attack,pose ^
	--use_action_cond ^
	--latent_dim 256 ^
	--layers 8 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--stage1_batch_size 4 ^
	--stage1_lr 5e-5 ^
	--stage1_num_steps 100000 ^
	--lr_anneal_steps 150000 ^
	--dropout_prob 0.1 ^
	--cond_mask_prob 0.2 ^
	--aug_speed_range 0.2 ^
	--aug_mirror_prob 0.5 ^
	--joint_mask_prob 0.2 ^
	--joint_mask_max_frac 0.3 ^
	--lambda_vel 0.5 ^
	--lambda_geo 0.0 ^
	--save_interval 2000 ^
	--log_interval 100 ^
	--motion_cache_size 512 ^
	--main_process_prefetch_batches 4 ^
	--auto_resume ^
	--ml_platform_type TensorboardPlatform
	
popd