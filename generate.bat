@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=merged_all_v70

rem set MODEL_FILE=model000020000.pt

REM Generate with --object_type and optional --action_label.
REM --reference_motion requires --skip_timesteps.
REM Editing uses the target's own rig; cross-skeleton transfer uses the motion autoencoder.
REM ----------------------------------------------------------------------
set BATCH_SIZE=8

pushd "%SCRIPT_DIR%"

if not defined MODEL_FILE (
    for /f "delims=" %%i in ('dir /b /o-d "save\%RUN_NAME%\model*.pt" 2^>nul') do (
        set MODEL_FILE=%%i
        goto :found_model
    )
)

:found_model
if not defined MODEL_FILE (
    echo Error: No model file found in save\%RUN_NAME%\
    popd
    exit /b 1
)

set STEP_NUM=%MODEL_FILE:.pt=%
set STEP_NUM=%STEP_NUM:model=%

set MODEL_PATH=save\%RUN_NAME%\%MODEL_FILE%
set OUTPUT_DIR=outputs\%RUN_NAME%\generate_step%STEP_NUM%

if exist %OUTPUT_DIR% (
    echo Cleaning output directory: %OUTPUT_DIR%
    rmdir /s /q %OUTPUT_DIR%
)
mkdir %OUTPUT_DIR% 2>nul

echo Output dir: %OUTPUT_DIR%

%PYTHON_EXE% sample/generate.py ^
    --model_path "%MODEL_PATH%" ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size %BATCH_SIZE% ^
    --amp_dtype fp32 %*

popd

