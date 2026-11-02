@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=zoo_locomotion_tiny_v71

rem set MODEL_FILE=model000020000.pt

rem ---- Optional motion detail refiner (see docs/motion_detail_refiner.md) ----
rem Leave REFINER_PATH empty to disable refinement (plain backbone output).
rem A bare save-dir name (e.g. detail_v67_v3_200) resolves to save\<name>\refiner_latest.pt;
rem a full path to the .pt is used as-is.
rem The backbone checkpoint must match the refiner's contract (zoo_locomotion_tiny_v67).
rem set REFINER_PATH=detail_v67_v3_200
set REFINER_PATH=
set REFINER_ALPHA=1.0
set REFINER_SEED=20001

REM ----------------------------------------------------------------------
REM Combination rules for target skeleton + reference motion (passed via %* to sample/generate.py):
REM   1) --object_type Horse only                          : random generation for Horse
REM   2) --reference_motion path\Horse_Run.npy only      : infer "Horse" from filename; the reference is
REM      encoded by the motion encoder (needs a --ref_encoder checkpoint), sampling starts from noise
REM   3) --reference_motion path\Flamingo_Walk.npy --object_type Horse
REM      : the Flamingo reference drives the Horse skeleton through the encoder, no retargeting
REM   4) + --skip_timesteps / --inpaint_* / --outpaint : edit the reference on the target skeleton;
REM      the reference must be the SAME rig as --object_type: a preprocessed .npy, or a .glb/.fbx whose own
REM      bind pose is used as the rest pose.
REM      --inpaint_frames also encodes the reference itself as z (inpainted frames hidden from the encoder),
REM      so it refuses --action_label and ignores the --species_tags restyle; --inpaint_joints alone,
REM      --skip_timesteps alone and --outpaint take both (with --outpaint + --inpaint_frames, they steer
REM      only the pass that appends frames)
REM   5) neither provided -> error and exit
REM Supports zero-shot inference on object types not seen during training.
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
set DIR_SUFFIX=
if not "%REFINER_PATH%"=="" set DIR_SUFFIX=_refiner
set OUTPUT_DIR=outputs\%RUN_NAME%\generate_step%STEP_NUM%%DIR_SUFFIX%

if exist %OUTPUT_DIR% (
    echo Cleaning output directory: %OUTPUT_DIR%
    rmdir /s /q %OUTPUT_DIR%
)
mkdir %OUTPUT_DIR% 2>nul

echo Output dir: %OUTPUT_DIR%

set REFINER_FILE=%REFINER_PATH%
if exist "save\%REFINER_PATH%\refiner_latest.pt" set REFINER_FILE=save\%REFINER_PATH%\refiner_latest.pt
if not "%REFINER_PATH%"=="" if not exist "%REFINER_FILE%" (
    echo Error: Refiner checkpoint not found: %REFINER_FILE%
    popd
    exit /b 1
)
set REFINER_ARGS=
if not "%REFINER_PATH%"=="" set REFINER_ARGS=--refiner_path "%REFINER_FILE%" --refiner_alpha %REFINER_ALPHA% --refiner_seed %REFINER_SEED%

%PYTHON_EXE% sample/generate.py ^
    --model_path "%MODEL_PATH%" ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size %BATCH_SIZE% ^
    --amp_dtype fp32 %REFINER_ARGS% %*

popd

