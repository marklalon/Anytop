@echo off
setlocal

set "PYTHON=d:\AI\pcvg-skeleton-animation\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" .\tools\non_deterministic_restoration_debug.py ^
	--output-dir .\outputs\one_shot_sampler_scratch100 ^
	--objects-subset quadropeds_clean ^
	--sample-limit 100 ^
	--num-frames 120 ^
	--batch-size 8 ^
	--num-workers 4 ^
	--num-eval-samples 32 ^
	--num-steps 100000 ^
	--log-interval 1000 ^
	--save-interval 10000 ^
	--lr 2e-5 ^
	--schedule-sampler uniform ^
	--sampling-method ddim ^
	--sampling-steps 100 ^
	--num-trials 2 ^
	--eval-batch-size 4 ^
	--lambda-confidence-recon 5.0 ^
	--lambda-repair-recon 2.0 ^
	--lambda-root 1.0 ^
	--lambda-velocity 0.25 ^
	--skip-video-export

endlocal