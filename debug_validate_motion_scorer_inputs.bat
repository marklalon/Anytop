@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe
set OUTPUT_DIR=%~dp0tmp\motion_scorer_input_validation

%PYTHON_EXE% "%~dp0tools\validate_motion_scorer_inputs.py" ^
    --split train ^
    --objects_subset all ^
    --action_tags locomotion ^
    --num_frames 60 ^
    --num_samples 12 ^
    --sample_pool_size 48 ^
    --batch_size 8 ^
    --num_workers 0 ^
    --motion_cache_size 128 ^
    --output_dir "%OUTPUT_DIR%" ^
    %*

if errorlevel 1 exit /b %errorlevel%

echo Validation finished. Use the printed markdown_report and json_report paths above.