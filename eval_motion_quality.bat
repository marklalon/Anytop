@echo off
setlocal

set SCRIPT_DIR=%~dp0
set RUN_NAME=stage1_tiny_overfit_horse_locomotion_pose_v1
set OUTPUT_DIR=%SCRIPT_DIR%outputs\%RUN_NAME%
set CLEAN_DIR=%OUTPUT_DIR%\stage1_sampling_eval
set GENERATED_DIR=%OUTPUT_DIR%\stage1_sampling_eval
set OUTPUT_JSON=%OUTPUT_DIR%\motion_quality_report.json

python "%SCRIPT_DIR%eval\evaluate_motion_quality.py" ^
  --clean     "%CLEAN_DIR%\sample_*/clean_target.npy" ^
  --generated "%GENERATED_DIR%\sample_*/generated_prediction.npy" ^
  --output-json "%OUTPUT_JSON%"

endlocal
