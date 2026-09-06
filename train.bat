@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=merged_locomotion_v5_pwp
set TORCH_LOGS=recompiles,graph_breaks

REM --compile builds Triton kernel launchers with MSVC cl.exe. Initialize
REM the VS 2022 x64 dev env pinned to 14.41.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.41

pushd "%SCRIPT_DIR%"

REM --objects_subset selects the training set. Use a species name (Horse,
REM Dragon, Bird, Camel, ...) to train on that species' actions only.
%PYTHON_EXE% train/train_anytop.py ^
	--cond_path dataset/merged/cond.npy ^
	--save_dir save/%RUN_NAME% ^
	--save_interval 5000 ^
	--log_interval 100 ^
	--auto_resume ^
	--ml_platform_type TensorboardPlatform ^
	--objects_subset all ^
	--action_group locomotion ^
	--train_split train ^
	--balanced ^
	--latent_dim 256 ^
	--ff_size 2048 ^
	--layers 8 ^
	--action_label_cond ^
	--species_cond ^
	--species_joint_cond ^
	--loop_cond_prob 0.5 ^
	--cross_limb_dim 128 ^
	--cross_limb_last_n 4 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--batch_size 16 ^
	--lr 1e-4 ^
	--weight_decay 0.01 ^
	--use_ema ^
	--ema_rate 0.995 ^
	--num_steps 200000 ^
	--dropout_prob 0.1 ^
	--joint_mask_prob 0.3 ^
	--joint_mask_budget 0.15 ^
	--temporal_span_mask_prob 0.3 ^
	--temporal_span_seam_loss_weight 0.2 ^
	--lambda_loop_wrap 0.04 ^
	--lambda_vel 0.2 ^
	--lambda_geo 0.1 ^
	--motion_cache_size 32768 ^
	--amp_dtype bf16 ^
	--main_process_prefetch_batches 64 ^
	--compile default

popd