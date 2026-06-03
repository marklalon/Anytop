@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=quadropeds_locomotion_retarget_v4

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
	--global_energy_cond ^
	--loop_cond_prob 0.5 ^
	--cross_limb_dim 128 ^
	--cross_limb_last_n 4 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--batch_size 16 ^
	--lr 1e-4 ^
	--lr_scheduler_gamma 0.95 ^
	--use_ema ^
	--num_steps 100000 ^
	--eval_during_training ^
	--eval_interval 500 ^
	--dropout_prob 0.1 ^
	--joint_mask_prob 0.5 ^
	--joint_mask_budget 0.15 ^
	--temporal_window 41 ^
	--temporal_span_mask_prob 0.3 ^
	--temporal_span_seam_loss_weight 0.5 ^
	--lambda_loop_wrap 0.2 ^
	--lambda_loop_root_xz 0.2 ^
	--lambda_vel 0.5 ^
	--lambda_geo 0.5 ^
	--lambda_energy 0.1 ^
	--motion_cache_size 512 ^
	--amp_dtype bf16 ^
	--main_process_prefetch_batches 64
	
popd