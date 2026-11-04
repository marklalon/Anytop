@echo off
setlocal EnableExtensions
REM Online-augmented tiny.json training, fixed evaluation windows, Gaussian VAE, beta=2e-4 per latent scalar.
REM Usage: train_tiny_autoencoder.bat [run_name]
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
set "TORCHINDUCTOR_CUDAGRAPHS=1"
set "RUN_NAME=tiny_autoencoder_fly_shake_v2"
if not "%~1"=="" set "RUN_NAME=%~1"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: "%PYTHON_EXE%"
    exit /b 1
)
pushd "%SCRIPT_DIR%"
if errorlevel 1 exit /b 1
if not exist "dataset\cond_all.npy" (
    echo ERROR: Run preprocess_all_datasets.bat first.
    popd
    exit /b 1
)
"%PYTHON_EXE%" train/train_motion_autoencoder.py ^
    --cond_path dataset/cond_all.npy ^
    --sample_filter configs/sample_filters/tiny.json ^
    --save_dir "save/%RUN_NAME%" ^
    --seed 10 ^
    --motion_speed_aug 1.3 ^
    --loop_tile_single_prob 0.5 ^
    --loop_cond_prob_free 0.8 ^
    --loop_cond_prob_anchored 0.2 ^
    --joint_name_drop_prob 0.15 ^
    --rest_geom_drop_prob 0.0 ^
    --rest_geom_joint_drop_prob 0.1 ^
    --mirror_twin_drop_prob 0.1 ^
    --lateral_drop_prob 0.1 ^
    --dropout_prob 0.05 ^
    --topology_aug_prob 0.2 ^
    --bone_length_aug_prob 0.2 ^
    --rest_repose_prob 0.1 ^
    --rest_variant_prob 0.5 ^
    --auto_resume ^
    --encoder_width 256 ^
    --decoder_width 256 ^
    --encoder_layers 4 ^
    --decoder_layers 8 ^
    --encoder_ff_size 1024 ^
    --decoder_ff_size 1024 ^
    --cross_limb_dim 0 ^
    --cross_limb_last_n 0 ^
    --temporal_rope ^
    --layerwise_latent ^
    --compile reduce-overhead ^
    --amp_dtype fp16 ^
    --heads 4 ^
    --slots 8 ^
    --variational ^
    --kl_weight 1e-4 ^
    --latent_dim 32 ^
    --num_frames 60 ^
    --batch_size 16 ^
    --motion_cache_size 16384 ^
    --main_process_prefetch_batches 64 ^
    --num_workers 2 ^
    --lr 1e-4 ^
    --lr_final 5e-5 ^
    --lr_decay_start 0 ^
    --use_ema ^
    --ema_rate 0.995 ^
    --motion_step_warmup_steps 500 ^
    --drot_joint_boost 1 ^
    --lambda_dpos 0.1 ^
    --lambda_drot 0.1 ^
    --lambda_acc 0.05 ^
    --lambda_root_dpos 0.1 ^
    --lambda_root_acc 0.1 ^
    --lambda_root_pos 0.1 ^
    --lambda_geo 0.1 ^
    --lambda_bone 0.2 ^
    --lambda_limb_corr 0.1 ^
    --lambda_loop_wrap 0.1 ^
    --num_steps 15000 ^
    --log_interval 100 ^
    --save_interval 1000 ^
    --eval_interval 1000 ^
    --export_samples 6 ^
    --export_motion_names Bear_WetShake PP_Hen_Idle3 SC2_Infestor_Idle3 MON_Crab_Roar LH_Hero_ShakeHead Tukan_Shake
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
