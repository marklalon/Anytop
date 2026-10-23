@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=merged_all_v55
set TORCH_LOGS=recompiles,graph_breaks

REM train_anytop requires bundled TinyCC for --compile, overriding inherited CC.
REM Inductor C++ codegen is disabled; no Visual Studio environment is needed.

pushd "%SCRIPT_DIR%"

"%PYTHON_EXE%" train/train_anytop.py ^
	--cond_path dataset/cond_all.npy ^
	--save_dir save/%RUN_NAME% ^
	--save_interval 10000 ^
	--log_interval 100 ^
	--auto_resume ^
	--ml_platform_type TensorboardPlatform ^
	--objects_subset all ^
	--action_group all ^
	--rare_head_word_floor 20 ^
	--rare_head_word_max_boost 4 ^
	--head_word_weights attack,idle=0.6 ^
	--train_split train ^
    --noise_schedule_shift 2 ^
	--latent_dim 384 ^
	--ff_size 2048 ^
	--last_layer_ff 256 ^
	--layers 8 ^
	--action_label_cond ^
	--action_adaln_bottleneck 64 ^
	--species_film_bottleneck 64 ^
	--species_cond ^
	--ref_encoder ^
	--ref_slots 4 ^
	--ref_dim 8 ^
	--ref_width 128 ^
	--ref_layers 2 ^
	--ref_cfg_drop_prob 0.6 ^
	--ref_leaf_drop_prob 0.3 ^
	--ref_bone_length_aug_prob 0.3 ^
	--ref_bone_length_aug 0.2 ^
	--ref_joint_name_drop_prob 0.2 ^
	--leaf_drop_prob 0.3 ^
	--bone_length_aug_prob 0.3 ^
	--bone_length_aug 0.2 ^
	--motion_speed_aug 1.3 ^
	--loop_cond_prob 0.8 ^
	--cross_limb_dim 128 ^
	--cross_limb_last_n 4 ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--batch_size 16 ^
	--lr 1e-4 ^
	--lr_final 3e-5 ^
	--lr_decay_start 0 ^
	--weight_decay 0.03 ^
	--use_ema ^
	--ema_rate 0.9995 ^
	--num_steps 300000 ^
	--dropout_prob 0.1 ^
	--action_label_cfg_drop_prob 0.2 ^
	--species_cfg_drop_prob 0.2 ^
	--tempo_cfg_drop_prob 0.1 ^
	--joint_mask_prob 0.3 ^
	--joint_mask_budget 0.15 ^
	--unreliable_mask_drop_prob 0.2 ^
	--renoise_same_level_prob 0.8 ^
	--joint_name_drop_prob 0.2 ^
	--direction_slot_drop_prob 0.15 ^
	--modifier_slot_drop_prob 0.15 ^
	--temporal_span_mask_prob 0.3 ^
	--temporal_span_seam_loss_weight 0.2 ^
	--lambda_loop_wrap 0.04 ^
	--lambda_loop_root_closure 0.04 ^
	--lambda_root_vel 0.1 ^
	--lambda_dpos 0.08 ^
	--lambda_geo 0.1 ^
	--lambda_bone 0.06 ^
	--motion_cache_size 16384 ^
	--amp_dtype fp16 ^
	--main_process_prefetch_batches 64 ^
	--num_workers 2 ^
	--compile default

set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
