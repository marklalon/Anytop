@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe
set SAVE_DIR=save\motion_scorer_v10_semantic_only

%PYTHON_EXE% "%~dp0train\train_motion_scorer.py" ^
    --objects_subset all ^
    --num_frames 60 ^
    --batch_size 32 ^
    --latent_dim 128 ^
    --d_model 256 ^
    --num_conv_layers 4 ^
    --kernel_size 5 ^
    --num_steps 10000 ^
    --lr 1.5e-4 ^
    --lr_step_size 6000 ^
    --lr_gamma 0.97 ^
    --save_dir %SAVE_DIR% ^
    --save_interval 2500 ^
    --log_interval 100 ^
    --timing_log_interval 100 ^
    --num_workers 0 ^
    --motion_cache_size 512 ^
    --main_process_prefetch_batches 6 ^
    --amp_dtype bf16 ^
    --use_ema ^
    --balanced ^
    --auto_resume
