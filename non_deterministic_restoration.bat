@echo off
setlocal

set "PYTHON=d:\AI\pcvg-skeleton-animation\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" .\tools\non_deterministic_restoration_debug.py ^
	--output-dir .\outputs\non_deterministic_restoration_fullschedule_10k ^
	--objects-subset quadropeds_clean ^
	--sample-limit 100 ^
	--num-frames 120 ^
	--batch-size 1 ^
	--num-workers 4 ^
	--num-steps 10000 ^
	--log-interval 100 ^
	--save-interval 1000 ^
	--lr 5e-5 ^
	--schedule-sampler uniform ^
	--sampling-method ddim ^
	--sampling-steps 25 ^
	--num-trials 2 ^
	--lambda-confidence-recon 4.0 ^
	--lambda-repair-recon 2.0 ^
	--lambda-root 1.0 ^
	--lambda-velocity 0.25 ^
	--skip-video-export

endlocal