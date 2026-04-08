@echo off
setlocal

set "PYTHON=d:\AI\pcvg-skeleton-animation\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" .\tools\non_deterministic_restoration_debug.py ^
	--output-dir .\outputs\one_shot_sampler_scratch32_10600_lr5e5_bs2 ^
	--objects-subset quadropeds_clean ^
	--sample-limit 32 ^
	--num-frames 120 ^
	--batch-size 2 ^
	--num-workers 0 ^
	--num-eval-samples 32 ^
	--num-steps 10600 ^
	--log-interval 100 ^
	--save-interval 500 ^
	--lr 5e-5 ^
	--schedule-sampler uniform ^
	--sampling-method ddim ^
	--sampling-steps 25 ^
	--num-trials 2 ^
	--eval-batch-size 2 ^
	--lambda-confidence-recon 4.0 ^
	--lambda-repair-recon 2.0 ^
	--lambda-root 1.0 ^
	--lambda-velocity 0.25 ^
	--skip-video-export

endlocal