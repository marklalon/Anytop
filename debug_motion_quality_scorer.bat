@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe
set CHECKPOINT_TARGET=save\motion_scorer_v7
set REPORT_DIR=%CHECKPOINT_TARGET%
for %%I in ("%CHECKPOINT_TARGET%") do (
    if /I "%%~xI"==".pt" set REPORT_DIR=%%~dpI
)
if "%REPORT_DIR:~-1%"=="\" set REPORT_DIR=%REPORT_DIR:~0,-1%

rem %PYTHON_EXE% "%~dp0tools\recompute_motion_scorer_stats.py" --checkpoint_dir "%CHECKPOINT_TARGET%" --device cuda
rem if errorlevel 1 exit /b %errorlevel%

%PYTHON_EXE% "%~dp0tools\debug_motion_quality_scorer.py" ^
    --checkpoint_dir %CHECKPOINT_TARGET% ^
    --device cuda ^
    --split train ^
    --batch_size 16 ^
    --sample_limit 64 ^
    --noise_sigma 0.10 ^
    --random_sigma 1.0 ^
    --output_json %REPORT_DIR%\debug_score_report_train.json ^
    --fail_on_unexpected_order ^
    %*
if errorlevel 1 exit /b %errorlevel%

%PYTHON_EXE% "%~dp0tools\debug_motion_quality_scorer.py" ^
    --checkpoint_dir %CHECKPOINT_TARGET% ^
    --device cuda ^
    --batch_size 16 ^
    --sample_limit 64 ^
    --noise_sigma 0.10 ^
    --random_sigma 1.0 ^
    --split val ^
    --output_json %REPORT_DIR%\debug_score_report_val.json ^
    --fail_on_unexpected_order ^
    %*
