@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=quadropeds_final_v2

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
	--latent_dim 256 ^
	--layers 8 ^
	--global_energy_cond ^
	--cross_limb_dim 128 ^
	--cross_limb_last_n 4 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--batch_size 16 ^
	--lr 1e-4 ^
	--use_ema ^
	--ema_rate 0.995 ^
	--num_steps 100000 ^
	--dropout_prob 0.1 ^
	--joint_mask_prob 0.5 ^
	--joint_mask_budget 0.15 ^
	--temporal_window 41 ^
	--temporal_span_mask_prob 0.3 ^
	--temporal_span_seam_loss_weight 0.5 ^
	--lambda_loop_wrap 0.3 ^
	--lambda_loop_root_xz 0.3 ^
	--lambda_vel 0.5 ^
	--lambda_geo 0.5 ^
	--motion_cache_size 512 ^
	--amp_dtype bf16 ^
	--main_process_prefetch_batches 64
	
popd