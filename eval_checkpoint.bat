@echo off
REM ----------------------------------------------------------------------
REM Run the checkpoint evaluation battery + HTML report.
REM
REM Usage:
REM   eval_checkpoint.bat --model_path save\<RUN_NAME>\<model>.pt [extra args]
REM
REM   - Defaults to the latest model*.pt in save\%RUN_NAME% if no --model_path
REM     is given (same auto-discovery as generate.bat).
REM   - Default: incremental (reuses existing outputs, generates new tasks).
REM   - --force: wipes output root and regenerates everything.
REM   - Forwards all args to eval/eval_checkpoint.py.
REM ----------------------------------------------------------------------
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set RUN_NAME=quadropeds_action_tags_v1

pushd "%SCRIPT_DIR%"

REM If the caller passed --model_path, just forward everything through.
echo %* | findstr /C:"--model_path" >nul
if %errorlevel%==0 (
    "%PYTHON_EXE%" eval\eval_checkpoint.py %*
    popd
    exit /b %errorlevel%
)

REM Otherwise auto-discover the latest model in save\%RUN_NAME%.
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

"%PYTHON_EXE%" eval\eval_checkpoint.py --model_path "save\%RUN_NAME%\%MODEL_FILE%" %*

popd
