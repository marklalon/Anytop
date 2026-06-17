@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=truebones_zoo_locomotion_v1
REM ----------------------------------------------------------------------
REM Combination rules for target skeleton + reference motion (passed via %* to sample/generate.py):
REM   1) --object_type Horse only                          : random generation for Horse
REM   2) --reference_motion path\Horse___Run.npy only      : infer "Horse" from filename, reference-guided generation
REM   3) --reference_motion path\Flamingo___Walk.npy --object_type Horse
REM      : inferred "Flamingo" differs from target "Horse" -> auto retarget Flamingo motion to Horse skeleton,
REM        then use as reference in diffusion process (intermediate _retargeted_*.npy / .bvh written to OUTPUT_DIR)
REM   4) neither provided -> error and exit
REM Supports zero-shot inference on object types not seen during training.
REM ----------------------------------------------------------------------
set BATCH_SIZE=8

rem set MODEL_FILE=model000010000.pt

pushd "%SCRIPT_DIR%"

if not defined MODEL_FILE (
    REM 自动查找最新的 model 文件
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

REM 通用方法：先去掉扩展名，再去掉 "model" 前缀
set STEP_NUM=%MODEL_FILE:.pt=%
set STEP_NUM=%STEP_NUM:model=%

set MODEL_PATH=save\%RUN_NAME%\%MODEL_FILE%
set OUTPUT_DIR=outputs\%RUN_NAME%\generate_step%STEP_NUM%

REM 清空 output_dir 目录
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
    --amp_dtype bf16 %*

popd

