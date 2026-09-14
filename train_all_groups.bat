@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=merged_all_v1
set TORCH_LOGS=recompiles,graph_breaks

REM --compile builds Triton kernel launchers with MSVC cl.exe. Initialize
REM the VS 2022 x64 dev env pinned to 14.41.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.41

pushd "%SCRIPT_DIR%"

REM One model over every action group (docs/unified_action_group_training.md).
REM Compared against the single-group runs at the same per-group sample count:
REM with equal --action_group_weights each group gets 1/3 of the draws, so step
REM 3S here matches step S of a single-group run (600k <-> 200k). The LR decays
REM every 30k steps instead of 10k for the same reason.
REM
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
	--action_group all ^
	--action_group_weights 1,1,1 ^
	--train_split train ^
	--latent_dim 256 ^
	--ff_size 2048 ^
	--layers 8 ^
	--action_label_cond ^
	--action_group_cond ^
	--species_cond ^
	--species_joint_cond ^
	--loop_cond_prob 0.7 ^
	--motion_speed_aug 1.3 ^
	--cross_limb_dim 128 ^
	--cross_limb_last_n 4 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--batch_size 16 ^
	--lr 1e-4 ^
	--weight_decay 0.01 ^
	--use_ema ^
	--ema_rate 0.995 ^
	--num_steps 600000 ^
	--lr_scheduler_step_size 30000 ^
	--dropout_prob 0.1 ^
	--action_label_cfg_drop_prob 0.3 ^
	--action_group_cfg_drop_prob 0.15 ^
	--joint_mask_prob 0.3 ^
	--joint_mask_budget 0.15 ^
	--unreliable_mask_drop_prob 0.2 ^
	--renoise_same_level_prob 0.8 ^
	--joint_name_drop_prob 0.15 ^
	--temporal_span_mask_prob 0.3 ^
	--temporal_span_seam_loss_weight 0.2 ^
	--lambda_loop_wrap 0.04 ^
	--lambda_loop_root_closure 0.05 ^
	--lambda_vel 0.2 ^
	--lambda_geo 0.1 ^
	--motion_cache_size 32768 ^
	--amp_dtype fp16 ^
	--main_process_prefetch_batches 64 ^
	--compile default

popd
