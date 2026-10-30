@echo off
setlocal EnableExtensions
REM All-dataset locomotion reference-encoder experiment. Optional first argument: a NEW run name.
REM Usage: train_tiny.bat [run_name]
REM See docs/tiny_locomotion_baseline.md for the fixed ablation protocol.
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
set "RUN_NAME=zoo_locomotion_tiny_v56"
if not "%~1"=="" set "RUN_NAME=%~1"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: "%PYTHON_EXE%"
    exit /b 1
)
pushd "%SCRIPT_DIR%"
if errorlevel 1 exit /b 1
if not exist "dataset\cond_all.npy" (
    echo ERROR: Run preprocess_all_datasets.bat to build dataset\cond_all.npy before running this experiment.
    popd
    exit /b 1
)

REM A new run trains the reference encoder from scratch; auto_resume continues this run.
REM Target and reference input draw topology and bone-length augmentation independently.
REM Keep the full locomotion pool; no ordered sample_limit truncation.
REM train_anytop forces bundled TinyCC for --compile and disables C++ codegen.
REM --dataset_namespaces: ';'-separated namespaces from dataset\datasets_all.jsonl, e.g.
REM "truebones/zoo;unitybundles"; a parent such as "truebones" selects all its children; "" = every dataset.
"%PYTHON_EXE%" train/train_anytop.py ^
    --cond_path dataset/cond_all.npy ^
    --save_dir "save/%RUN_NAME%" ^
    --seed 10 ^
    --save_interval 5000 ^
    --log_interval 100 ^
    --auto_resume ^
    --ml_platform_type TensorboardPlatform ^
    --dataset_namespaces "truebones/zoo;integrate_20260924" ^
    --objects_subset Winged ^
    --action_group locomotion ^
    --rare_head_word_floor 10 ^
    --rare_head_word_max_boost 2 ^
    --train_split train ^
    --noise_schedule_shift 2 ^
    --latent_dim 256 ^
    --ff_size 1024 ^
    --last_layer_ff 256 ^
    --layers 8 ^
    --action_label_cond ^
    --action_adaln_bottleneck 64 ^
    --species_film_bottleneck 64 ^
    --species_cond ^
    --ref_encoder ^
    --ref_slots 2 ^
    --ref_dim 8 ^
    --ref_width 128 ^
    --ref_layers 2 ^
    --ref_cfg_drop_prob 0.6 ^
    --rest_repose_prob 0.1 ^
    --rest_variant_prob 0.5 ^
    --topology_aug_prob 0.2 ^
    --bone_length_aug_prob 0.2 ^
    --motion_speed_aug 1.3 ^
    --loop_cond_prob 0.8 ^
    --cross_limb_dim 128 ^
	--cross_limb_last_n 4 ^
    --temporal_rope ^
    --temporal_rope_ref_drop_prob 0.8 ^
    --temporal_rope_drop_prob 0.2 ^
    --diffusion_steps 100 ^
    --num_frames 60 ^
    --batch_size 16 ^
    --lr 1e-4 ^
    --lr_final 5e-5 ^
    --lr_decay_start 0 ^
    --weight_decay 0.03 ^
    --use_ema ^
    --ema_rate 0.995 ^
    --num_steps 10000 ^
    --dropout_prob 0.1 ^
    --action_label_cfg_drop_prob 0.2 ^
    --joint_mask_prob 0.15 ^
    --unreliable_mask_drop_prob 0.15 ^
    --renoise_same_level_prob 0.6 ^
    --joint_name_drop_prob 0.15 ^
    --direction_slot_drop_prob 0.15 ^
    --modifier_slot_drop_prob 0.15 ^
    --temporal_span_mask_prob 0.15 ^
    --temporal_span_seam_loss_weight 0.2 ^
    --lambda_loop_wrap 0.04 ^
    --lambda_dpos 0.1 ^
    --lambda_root_dpos 0.1 ^
    --lambda_root_acc 0.1 ^
    --lambda_limb_corr 0.1 ^
    --lambda_geo 0.1 ^
    --lambda_bone 0.06 ^
    --motion_cache_size 16384 ^
    --amp_dtype fp16 ^
    --main_process_prefetch_batches 64 ^
    --num_workers 2 ^
	--compile default

set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
