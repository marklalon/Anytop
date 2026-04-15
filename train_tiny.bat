@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe

pushd "%SCRIPT_DIR%"

%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--output-dir save/stage1_tiny_overfit_locomotion_teacher_v5 ^
	--motion_scorer_checkpoint_dir save/motion_scorer_v8 ^
	--physics_teacher_weight 0.05 ^
	--physics_teacher_feature_weight 1.0 ^
	--physics_teacher_margin_weight 0.25 ^
	--physics_teacher_start_step 0 ^
	--physics_teacher_ramp_steps 0 ^
	--physics_teacher_max_t 30 ^
	--physics_features_device cpu ^
	--semantic_teacher_weight 0.05 ^
	--semantic_teacher_species_weight 1.0 ^
	--semantic_teacher_action_weight 1.0 ^
	--semantic_teacher_kl_weight 0.25 ^
	--semantic_teacher_start_step 0 ^
	--semantic_teacher_ramp_steps 0 ^
	--semantic_teacher_max_t 30 ^
	--semantic_teacher_temperature 1.0 ^
	--action_tags locomotion ^
	--auto_resume ^
	--objects_subset all ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--stage1_batch_size 32 ^
	--stage1_sample_limit -1 ^
	--stage1_lr 5e-5 ^
	--stage1_num_steps 100000 ^
	--lambda_geo 1.0 ^
	--save_interval 1000 ^
	--log_interval 100 ^
	--num_workers 0 ^
	--motion_cache_size 512 ^
	--main_process_prefetch_batches 6 ^
	--amp_dtype bf16 ^
	--cudnn_benchmark ^
	--allow_tf32 ^
	--ml_platform_type TensorboardPlatform ^
	--use_ema

popd