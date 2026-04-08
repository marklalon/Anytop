@echo off
setlocal

set "PYTHON=d:\AI\pcvg-skeleton-animation\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" .\tools\non_deterministic_restoration_debug.py ^
	--output-dir .\outputs\non_deterministic_restoration_staged_repro ^
	--objects-subset quadropeds_clean ^
	--sample-limit 32 ^
	--num-frames 120 ^
	--batch-size 1 ^
	--num-workers 0 ^
	--num-eval-samples 8 ^
	--num-steps 3000 ^
	--log-interval 100 ^
	--save-interval 500 ^
	--lr 1e-4 ^
	--train-timestep-mode staged ^
	--bootstrap-steps 1000 ^
	--bootstrap-timestep-ranges 8:12,5:25 ^
	--bootstrap-end-fractions 0.5,1.0 ^
	--stage-timestep-ranges 8:12,5:25,0:99 ^
	--stage-end-fractions 0.3,0.7,1.0 ^
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