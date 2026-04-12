@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% "%~dp0train\train_motion_scorer.py" ^
    --objects_subset all ^
    --motion_name_keywords walk,run,sprint ^
    --num_frames 60 ^
    --batch_size 32 ^
    --latent_dim 512 ^
    --d_model 256 ^
    --num_conv_layers 4 ^
    --kernel_size 5 ^
    --num_steps 50000 ^
    --lr 5e-5 ^
    --save_dir save\motion_scorer ^
    --save_interval 5000 ^
    --log_interval 100 ^
    --num_workers 8 ^
    --amp_dtype bf16 ^
    --ml_platform_type TensorboardPlatform ^
    --overwrite