@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=quadropeds_locomotion_v4
REM 仅支持单个类型。支持指定训练时模型没见过的Object Type来实现0样本泛化推理
REM set OBJECT_TYPE=Horse
set BATCH_SIZE=8

rem set MODEL_FILE=model000002000.pt

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
if not "%*"=="" echo Extra args: %*

%PYTHON_EXE% sample/generate.py ^
    --model_path "%MODEL_PATH%" ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size %BATCH_SIZE% ^
    --motion_length 2.0 ^
    --sampling_method ddim ^
    --sampling_steps 100 %*

popd

