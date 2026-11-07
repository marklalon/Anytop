@echo off
setlocal EnableExtensions
REM All-dataset locomotion motion-diffusion training. Optional first argument: a NEW run name.
REM Usage: train_diffusion_tiny.bat [run_name]
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
set "TORCHINDUCTOR_CUDAGRAPHS=1"
set "RUN_NAME=tiny_diffusion_v3"
if not "%~1"=="" set "RUN_NAME=%~1"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: "%PYTHON_EXE%"
    exit /b 1
)
pushd "%SCRIPT_DIR%"
if errorlevel 1 exit /b 1
if not exist "dataset\cond_all.npy" (
    echo ERROR: Run preprocess_all_datasets.bat to build dataset\cond_all.npy before running this experiment.
    popd
    exit /b 1
)

REM A new run trains motion diffusion from scratch; auto_resume continues this run.
REM Topology and bone-length augmentation apply to the training motion.
REM Keep the full locomotion pool; no ordered sample_limit truncation.
REM train_motion_diffusion forces bundled TinyCC for --compile and disables C++ codegen.
REM reduce-overhead uses CUDA Graphs for eligible model forward/backward regions.
REM Edit configs/sample_filters/tiny.json to combine or exclude training clips.
"%PYTHON_EXE%" train/train_motion_diffusion.py ^
    --cond_path dataset/cond_all.npy ^
    --save_dir "save/%RUN_NAME%" ^
    --seed 10 ^
    --save_interval 5000 ^
    --log_interval 100 ^
    --auto_resume ^
    --ml_platform_type TensorboardPlatform ^
    --sample_filter configs/sample_filters/tiny.json ^
    --rare_head_word_floor 10 ^
    --rare_head_word_max_boost 2 ^
    --train_split train ^
    --noise_schedule_shift 2 ^
    --latent_dim 256 ^
    --ff_size 1024 ^
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
    --num_frames 60 ^
    --batch_size 16 ^
    --lr 1e-4 ^
    --lr_final 5e-5 ^
    --lr_decay_start 0 ^
    --use_ema ^
    --ema_rate 0.995 ^
    --num_steps 10000 ^
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
