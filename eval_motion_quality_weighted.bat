@echo off
REM Evaluation with weighted macro score (pos and rot: 1.5x, limbs: 1.5x)
REM Weights are hardcoded in scorer.py

python eval/evaluate_motion_quality.py ^
  --clean     "outputs/stage1_tiny_overfit_horse_resetless_v2/stage1_sampling_eval/trials/trial_00/*/clean_target.npy" ^
  --generated "outputs/stage1_tiny_overfit_horse_resetless_v2/stage1_sampling_eval/trials/trial_00/*/generated_prediction.npy" ^
  --output-json outputs/stage1_tiny_overfit_horse_resetless_v2/motion_quality_report_weighted.json
