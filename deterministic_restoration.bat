@echo off
setlocal

set "PYTHON=d:\AI\pcvg-skeleton-animation\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" .\tools\deterministic_restoration_debug.py ^
	--output-dir .\outputs\deterministic_restoration_v4 ^
	--objects-subset quadropeds_clean ^
	--sample-limit 32 ^
	--num-frames 60 ^
	--num-steps 10000 ^
	--log-interval 100 ^
	--save-interval 1000 ^
	--fixed-timestep 10 ^
	--noise-mode zero ^
	--lambda-confidence-recon 4.0 ^
	--lambda-repair-recon 2.0 ^
	--lambda-root 1.0 ^
	--lambda-velocity 0.25 ^
	--skip-video-export

endlocal