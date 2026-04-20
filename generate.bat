@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=stage1_tiny_overfit_horse_locomotion_pose_v16
REM 多个类型用空格分隔（直接传给 --object_type nargs='+'）。支持指定训练时模型没见过的Object Type来实现0样本泛化推理
set OBJECT_TYPE=Buffalo Deer Crocodile
set NUM_REPETITIONS=8
set ACTION_CATEGORY=locomotion
set GUIDANCE_SCALE=1

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

echo Generating with model: %MODEL_PATH%
echo Object types: %OBJECT_TYPE%
echo Output dir: %OUTPUT_DIR%

REM 一次调用 generate.py，所有类型作为 batch 并行生成
%PYTHON_EXE% sample/generate.py ^
    --model_path "%MODEL_PATH%" ^
    --output_dir %OUTPUT_DIR% ^
    --object_type %OBJECT_TYPE% ^
    --num_repetitions %NUM_REPETITIONS% ^
    --action_category %ACTION_CATEGORY% ^
    --guidance_scale %GUIDANCE_SCALE% ^
    --sampling_method ddim ^
    --sampling_steps 100

if %errorlevel% neq 0 (
    echo Error: Generation failed.
    popd
    exit /b 1
)

REM 一次调用 test_generation_diversity.py，支持多个 object_type
echo Running diversity test for all object types...
%PYTHON_EXE% tests\test_generation_diversity.py --gen_dir %OUTPUT_DIR% --object_type "%OBJECT_TYPE%" --action_tags %ACTION_CATEGORY%

popd

