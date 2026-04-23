@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=stage1_quadropeds_v2_9
REM 多个类型用空格分隔（直接传给 --object_type nargs='+'）。支持指定训练时模型没见过的Object Type来实现0样本泛化推理
set OBJECT_TYPE=Buffalo
set BATCH_SIZE=32
set ACTION_CATEGORY=locomotion

rem set MODEL_FILE=model000080000.pt

pushd "%SCRIPT_DIR%"

if not defined MODEL_FILE (
    REM 自动查找最新的 model 文件
    for /f "delims=" %%i in ('dir /b /o-d "save\%RUN_NAME%\stage1_pretrain\model*.pt" 2^>nul') do (
        set MODEL_FILE=%%i
        goto :found_model
    )
)

:found_model
if not defined MODEL_FILE (
    echo Error: No model file found in save\%RUN_NAME%\stage1_pretrain\
    popd
    exit /b 1
)

REM 通用方法：先去掉扩展名，再去掉 "model" 前缀
set STEP_NUM=%MODEL_FILE:.pt=%
set STEP_NUM=%STEP_NUM:model=%

set MODEL_PATH=save\%RUN_NAME%\stage1_pretrain\%MODEL_FILE%
set OUTPUT_DIR=outputs\%RUN_NAME%\generate_step%STEP_NUM%

REM 清空 output_dir 目录
if exist %OUTPUT_DIR% (
    echo Cleaning output directory: %OUTPUT_DIR%
    rmdir /s /q %OUTPUT_DIR%
)
mkdir %OUTPUT_DIR% 2>nul

echo Object types: %OBJECT_TYPE%
echo Output dir: %OUTPUT_DIR%
if not "%*"=="" echo Extra args: %*

REM 一次调用 generate.py；脚本会按 object_type 逐个生成，每个 object_type 内一次性生成 BATCH_SIZE 个样本
%PYTHON_EXE% sample/generate.py ^
    --model_path "%MODEL_PATH%" ^
    --output_dir %OUTPUT_DIR% ^
    --object_type %OBJECT_TYPE% ^
    --batch_size %BATCH_SIZE% ^
    --action_category %ACTION_CATEGORY% ^
    --motion_length 2.0 ^
    --sampling_method ddim ^
    --sampling_steps 50 %*

popd

