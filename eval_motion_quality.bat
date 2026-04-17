python eval/evaluate_motion_quality.py ^
  --clean     "outputs/stage1_tiny_overfit_horse_slowwalk_v1/stage1_sampling_eval/trials/trial_00/*/clean_target.npy" ^
  --generated "outputs/stage1_tiny_overfit_horse_slowwalk_v1/stage1_sampling_eval/trials/trial_00/*/generated_prediction.npy" ^
  --output-json outputs/stage1_tiny_overfit_horse_slowwalk_v1/motion_quality_report.json
