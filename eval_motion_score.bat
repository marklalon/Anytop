@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe
set DEFAULT_CHECKPOINT_DIR=save\motion_scorer_v3

%PYTHON_EXE% "%~dp0tools\eval_motion_score.py" --checkpoint_dir %DEFAULT_CHECKPOINT_DIR% %*
exit /b %errorlevel%