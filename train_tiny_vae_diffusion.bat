@echo off
setlocal EnableExtensions
REM Train the tiny VAE first, then latent diffusion using its checkpoint.
REM Usage: train_tiny_vae_diffusion.bat [--autoencoder-only ^| --latent-diffusion-only] [vae_run_name] [diffusion_run_name]
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
set "TORCHINDUCTOR_CUDAGRAPHS=1"
set "AE_RUN_NAME=tiny_autoencoder_reference_pairs_fly_shake_v1"
set "DIFFUSION_RUN_NAME=tiny_latent_diffusion_reference_pairs_fly_shake_v1"
set "TRAIN_STAGE=both"
set "RUN_NAME_COUNT=0"
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--autoencoder-only" (
    if "%TRAIN_STAGE%"=="diffusion" goto :conflicting_stages
    set "TRAIN_STAGE=autoencoder"
    shift
    goto :parse_args
)
if /i "%~1"=="--latent-diffusion-only" (
    if "%TRAIN_STAGE%"=="autoencoder" goto :conflicting_stages
    set "TRAIN_STAGE=diffusion"
    shift
    goto :parse_args
)
set "ARG=%~1"
if "%ARG:~0,1%"=="-" (
    echo ERROR: Unknown option: %~1
    exit /b 2
)
if "%RUN_NAME_COUNT%"=="2" (
    echo ERROR: Expected at most two run names.
    exit /b 2
)
if "%RUN_NAME_COUNT%"=="0" set "AE_RUN_NAME=%~1"
if "%RUN_NAME_COUNT%"=="1" set "DIFFUSION_RUN_NAME=%~1"
set /a RUN_NAME_COUNT+=1 >nul
shift
goto :parse_args
:conflicting_stages
echo ERROR: --autoencoder-only and --latent-diffusion-only are mutually exclusive.
exit /b 2
:usage
echo Usage: %~nx0 [--autoencoder-only ^| --latent-diffusion-only] [vae_run_name] [diffusion_run_name]
echo.
echo Default: train the VAE, then latent diffusion.
echo   --autoencoder-only       Train only the VAE.
echo   --latent-diffusion-only  Train only latent diffusion using save/vae_run_name/autoencoder.pt.
echo Options can appear before or after the run names.
echo Omitted run names use the script defaults.
exit /b 0
:args_done
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
set "EXIT_CODE=0"
if "%TRAIN_STAGE%"=="diffusion" goto :latent_diffusion
if "%TRAIN_STAGE%"=="autoencoder" (
    echo Training tiny VAE: %AE_RUN_NAME%
) else (
    echo [1/2] Training tiny VAE: %AE_RUN_NAME%
)
"%PYTHON_EXE%" train/train_autoencoder.py ^
    --cond_path dataset/cond_all.npy ^
    --sample_filter configs/sample_filters/tiny.json ^
    --save_dir "save/%AE_RUN_NAME%" ^
    --seed 10 ^
    --dual_view ^
    --reference_pairs ^
    --motion_speed_aug 1.3 ^
    --loop_tile_single_prob 0.5 ^
    --loop_cond_prob_free 0.8 ^
    --loop_cond_prob_anchored 0.2 ^
    --joint_name_drop_prob 0.15 ^
    --rest_geom_joint_drop_prob 0.1 ^
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
    --warmup_steps 500 ^
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
    --eval_interval 1000
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto :finish
if "%TRAIN_STAGE%"=="autoencoder" goto :finish
:latent_diffusion
if not exist "save\%AE_RUN_NAME%\autoencoder.pt" (
    echo ERROR: VAE checkpoint not found: "save\%AE_RUN_NAME%\autoencoder.pt"
    echo Train the VAE first or provide an existing vae_run_name.
    set "EXIT_CODE=1"
    goto :finish
)
if "%TRAIN_STAGE%"=="diffusion" (
    echo Training tiny latent diffusion: %DIFFUSION_RUN_NAME%
) else (
    echo [2/2] Training tiny latent diffusion: %DIFFUSION_RUN_NAME%
)
"%PYTHON_EXE%" train/train_latent_diffusion.py ^
    --ae_checkpoint "save/%AE_RUN_NAME%" ^
    --cond_path dataset/cond_all.npy ^
    --sample_filter configs/sample_filters/tiny.json ^
    --save_dir "save/%DIFFUSION_RUN_NAME%" ^
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
:finish
popd
endlocal & exit /b %EXIT_CODE%
