@echo off
set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe

%PYTHON_EXE% train/train_anytop_two_stage.py ^
	--run_stage stage1 ^
	--output-dir save/stage1_tiny_overfit_locomotion_proxy_s20000 ^
	--enable_quality_proxy ^
	--motion_scorer_checkpoint_dir save/motion_scorer_v8 ^
	--quality_proxy_layer 1 ^
	--quality_proxy_hidden_dim 128 ^
	--quality_proxy_guidance_weight 0.05 ^
	--quality_proxy_supervision_weight 1.0 ^
	--quality_proxy_guidance_start_step 1000 ^
	--quality_proxy_teacher_interval 10 ^
	--quality_proxy_teacher_microbatch 4 ^
	--quality_proxy_teacher_low_noise_max_t 20 ^
	--quality_proxy_score_floor 0.05 ^
	--quality_proxy_score_ceiling 0.95 ^
	--quality_proxy_agreement_interval 1000 ^
	--action_tags locomotion ^
	--auto_resume ^
	--objects_subset all ^
	--diffusion_steps 100 ^
	--num_frames 60 ^
	--stage1_batch_size 32 ^
	--stage1_sample_limit -1 ^
	--stage1_lr 5e-5 ^
	--stage1_num_steps 20000 ^
	--lambda_geo 1.0 ^
	--save_interval 2500 ^
	--log_interval 100 ^
	--num_workers 0 ^
	--amp_dtype bf16 ^
	--cudnn_benchmark ^
	--allow_tf32 ^
	--ml_platform_type TensorboardPlatform ^
	--use_ema