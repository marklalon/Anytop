@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe
set CHECKPOINT_DIR=save\motion_scorer_perceptual_v2

%PYTHON_EXE% "%~dp0tools\recompute_motion_scorer_stats.py" ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --device cuda
if errorlevel 1 exit /b %errorlevel%

%PYTHON_EXE% "%~dp0tools\debug_motion_quality_scorer.py" ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --device cuda ^
    --split train ^
    --batch_size 16 ^
    --sample_limit 64 ^
    --noise_sigma 0.10 ^
    --random_sigma 1.0 ^
    --output_json %CHECKPOINT_DIR%\debug_score_report_train.json ^
    --fail_on_unexpected_order
if errorlevel 1 exit /b %errorlevel%

%PYTHON_EXE% "%~dp0tools\debug_motion_quality_scorer.py" ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --device cuda ^
    --batch_size 16 ^
    --sample_limit 64 ^
    --noise_sigma 0.10 ^
    --random_sigma 1.0 ^
    --split val ^
    --output_json %CHECKPOINT_DIR%\debug_score_report_val.json ^
    --fail_on_unexpected_order