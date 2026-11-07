@echo off
setlocal EnableExtensions
REM Train the full-dataset autoencoder. Usage: %~nx0 [run_name]
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
set "TORCHINDUCTOR_CUDAGRAPHS=1"
set "AE_RUN_NAME=autoencoder_alpha_v1"
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
if not "%~2"=="" goto :invalid_args
set "ARG=%~1"
if "%ARG:~0,1%"=="-" goto :invalid_args
if not "%~1"=="" set "AE_RUN_NAME=%~1"
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
"%PYTHON_EXE%" train/train_autoencoder.py ^
    --cond_path dataset/cond_all.npy ^
    --save_dir "save/%AE_RUN_NAME%" ^
    --seed 10 ^
    --motion_speed_aug 1.3 ^
    --loop_tile_single_prob 0.5 ^
    --loop_cond_prob_free 0.8 ^
    --loop_cond_prob_anchored 0.2 ^
    --joint_name_drop_prob 0.15 ^
    --rest_geom_joint_drop_prob 0.1 ^
    --dropout_prob 0.05 ^
    --topology_aug_prob 0.3 ^
    --bone_length_aug_prob 0.3 ^
    --rest_repose_prob 0.1 ^
    --rest_variant_prob 0.5 ^
    --auto_resume ^
    --encoder_width 512 ^
    --decoder_width 512 ^
    --encoder_layers 4 ^
    --decoder_layers 8 ^
    --encoder_ff_size 2048 ^
    --decoder_ff_size 2048 ^
    --compile reduce-overhead ^
    --amp_dtype fp16 ^
    --heads 4 ^
    --slots 8 ^
    --latent_dim 32 ^
    --variational ^
    --kl_weight 3e-4 ^
    --num_frames 90 ^
    --batch_size 16 ^
    --motion_cache_size 16384 ^
    --main_process_prefetch_batches 64 ^
    --num_workers 2 ^
    --lr 1e-4 ^
    --lr_final 5e-5 ^
    --lr_decay_start 0 ^
    --use_ema ^
    --ema_rate 0.999 ^
    --warmup_steps 2000 ^
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
    --num_steps 40000 ^
    --log_interval 100 ^
    --save_interval 1000 ^
    --eval_interval 1000
set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
:invalid_args
echo ERROR: Expected at most one run name.
echo Usage: %~nx0 [run_name]
exit /b 2
:usage
echo Usage: %~nx0 [run_name]
exit /b 0
