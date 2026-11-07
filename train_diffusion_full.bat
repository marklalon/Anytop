@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=motion_diffusion_alpha_v2
set TORCH_LOGS=recompiles,graph_breaks
set "TORCHINDUCTOR_CUDAGRAPHS=1"

REM train_motion_diffusion requires bundled TinyCC for --compile, overriding inherited CC.
REM Inductor C++ codegen is disabled; no Visual Studio environment is needed.
REM reduce-overhead uses CUDA Graphs for eligible model forward/backward regions.

pushd "%SCRIPT_DIR%"

"%PYTHON_EXE%" train/train_motion_diffusion.py ^
	--cond_path dataset/cond_all.npy ^
	--save_dir save/%RUN_NAME% ^
	--save_interval 10000 ^
	--log_interval 100 ^
	--eval_interval 5000 ^
	--auto_resume ^
	--ml_platform_type TensorboardPlatform ^
	--rare_head_word_floor 20 ^
	--rare_head_word_max_boost 4 ^
	--head_word_weights attack,hurt=0.8,idle=0.6 ^
	--train_split train ^
    --noise_schedule_shift 2 ^
	--latent_dim 512 ^
	--ff_size 2048 ^
	--last_layer_ff 256 ^
	--layers 8 ^
    --rest_repose_prob 0.1 ^
    --rest_variant_prob 0.5 ^
	--topology_aug_prob 0.2 ^
	--bone_length_aug_prob 0.2 ^
	--motion_speed_aug 1.3 ^
    --loop_cond_prob_free 0.8 ^
    --loop_cond_prob_anchored 0.2 ^
    --drot_joint_boost 1 ^
	--diffusion_steps 100 ^
	--num_frames 90 ^
	--batch_size 16 ^
	--lr 1e-4 ^
	--lr_final 3e-5 ^
	--lr_decay_start 0 ^
	--use_ema ^
	--ema_rate 0.9995 ^
	--num_steps 200000 ^
	--dropout_prob 0.05 ^
    --action_label_cfg_drop_prob 0.3 ^
	--joint_name_drop_prob 0.15 ^
	--lambda_loop_wrap 0.04 ^
	--lambda_dpos 0.1 ^
	--lambda_drot 0.1 ^
	--lambda_acc 0.05 ^
	--lambda_root_pos 0.1 ^
	--lambda_root_dpos 0.1 ^
	--lambda_root_acc 0.1 ^
	--lambda_limb_corr 0.1 ^
	--lambda_geo 0.1 ^
	--lambda_bone 0.06 ^
	--motion_cache_size 16384 ^
	--amp_dtype fp16 ^
	--main_process_prefetch_batches 64 ^
	--num_workers 2 ^
	--compile reduce-overhead

set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
