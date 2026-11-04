@echo off
setlocal EnableExtensions
REM Fixed 60-frame latent diffusion, frozen EMA AE, posterior-mean normalization.
REM Usage: train_tiny_latent_diffusion.bat [run_name]
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
set "RUN_NAME=tiny_latent_diffusion_fly_shake_v2"
if not "%~1"=="" set "RUN_NAME=%~1"
REM Frozen AE checkpoint directory (under save\). Change only here to swap ckpts.
set "AE_CKPT_DIR=tiny_autoencoder_fly_shake_v2"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: "%PYTHON_EXE%"
    exit /b 1
)
pushd "%SCRIPT_DIR%"
if errorlevel 1 exit /b 1
if not exist "save\%AE_CKPT_DIR%\autoencoder.pt" (
    echo ERROR: Frozen AE checkpoint not found.
    popd
    exit /b 1
)
"%PYTHON_EXE%" train/train_latent_diffusion.py ^
    --ae_checkpoint save/%AE_CKPT_DIR% ^
    --cond_path dataset/cond_all.npy ^
    --sample_filter configs/sample_filters/tiny.json ^
    --save_dir "save/%RUN_NAME%" ^
    --auto_resume ^
    --seed 10 ^
    --num_frames 60 ^
    --latent_mode mean ^
    --stats_batches 64 ^
    --width 256 ^
    --layers 8 ^
    --heads 4 ^
    --ff_size 1024 ^
    --skeleton_layers 2 ^
    --action_label_cond ^
    --species_cond ^
    --ref_encoder ^
    --ref_slots 2 ^
    --ref_dim 8 ^
    --ref_width 128 ^
    --ref_layers 2 ^
    --ref_cfg_drop_prob 0.6 ^
    --temporal_rope ^
    --action_label_cfg_drop_prob 0.2 ^
    --tempo_cfg_drop_prob 0.0 ^
    --denoiser_joint_name_drop_prob 0.15 ^
    --denoiser_rest_geom_joint_drop_prob 0.1 ^
    --motion_speed_aug 1.3 ^
    --loop_tile_single_prob 0.5 ^
    --loop_cond_prob_free 0.8 ^
    --loop_cond_prob_anchored 0.2 ^
    --topology_aug_prob 0.2 ^
    --bone_length_aug_prob 0.2 ^
    --rest_repose_prob 0.1 ^
    --rest_variant_prob 0.5 ^
    --amp_dtype fp16 ^
    --compile reduce-overhead ^
    --batch_size 16 ^
    --joint_buckets 62 100 ^
    --motion_cache_size 16384 ^
    --main_process_prefetch_batches 64 ^
    --num_workers 2 ^
    --lr 1e-4 ^
    --lr_final 5e-5 ^
    --lr_decay_start 0 ^
    --use_ema ^
    --ema_rate 0.995 ^
    --diffusion_steps 100 ^
    --noise_schedule cosine ^
    --num_steps 15000 ^
    --log_interval 100 ^
    --save_interval 1000 ^
    --eval_interval 1000
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
