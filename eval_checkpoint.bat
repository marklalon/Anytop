@echo off
setlocal EnableExtensions
REM ----------------------------------------------------------------------
REM Run the checkpoint evaluation battery + HTML report.
REM
REM Usage:
REM   eval_checkpoint.bat <task_config.json> [extra eval args]
REM   eval_checkpoint.bat eval\eval_tasks.json --overwrite
REM
REM The task config must define checkpoint.RUN_NAME. checkpoint.MODEL_FILE is
REM optional; when omitted, eval_checkpoint.py selects the newest model*.pt
REM under save\<RUN_NAME>. checkpoint.OUTPUT_DIR is optional; it names the run
REM folder the results are written under (default: RUN_NAME), which lets several
REM batteries of one checkpoint keep separate outputs.
REM ----------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"

if "%~1"=="" (
    echo Usage: eval_checkpoint.bat ^<task_config.json^> [extra eval args]
    exit /b 2
)

set "TASK_CONFIG=%~1"
shift

REM Preserve all remaining arguments while making the task config explicit to
REM the Python entry point. This also keeps quoted paths with spaces intact.
set "EXTRA_ARGS="
:collect_args
if "%~1"=="" goto :run
set "EXTRA_ARGS=%EXTRA_ARGS% %1"
shift
goto :collect_args

:run
pushd "%SCRIPT_DIR%"
"%PYTHON_EXE%" eval\eval_checkpoint.py --task_config "%TASK_CONFIG%" %EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
