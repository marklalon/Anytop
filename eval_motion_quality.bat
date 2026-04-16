python eval/evaluate_motion_quality.py ^
  "outputs/stage1_tiny_overfit_all_move_clean_s100000/stage1_sampling_eval/trials/trial_00/*/clean_target.npy" ^
  "outputs/stage1_tiny_overfit_all_move_clean_s100000/stage1_sampling_eval/trials/trial_00/*/generated_prediction.npy" ^
  --dataset-dir dataset/truebones/zoo/truebones_processed ^
  --quiet --cond-file dataset/truebones/zoo/truebones_processed/cond.npy
