@echo off
REM ----------------------------------------------------------------------
REM Preprocess the three AnyTop datasets (incremental by default) and merge
REM them into a single training cond.
REM
REM   1) truebones/zoo         -> dataset\truebones\zoo\truebones_processed   (default dataset)
REM   2) truebones/zoo_upgrade -> dataset\truebones\zoo_upgrade\clean_processed
REM   3) unitybundles          -> dataset\unitybundles\processed
REM   4) merge                 -> dataset\datasets.jsonl -> dataset\merged\cond.npy
REM
REM Any failing step aborts the run before the merge, so the merged cond is
REM never built from incomplete data.
REM
REM Any CLI args passed to this bat are forwarded to EVERY preprocess step
REM (the merge step is unaffected). E.g. `preprocess_all_datasets.bat --overwrite --yes`
REM forces a full rebuild of all three datasets with no interactive confirmation;
REM --filter / --object-workers work the same way.
REM ----------------------------------------------------------------------
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe

REM Source raw BVH dir for the UnityBundles dataset (absolute).
set UNITYBUNDLES_RAW=E:\Dataset\UnityBundles_anytop\raw

REM Capture the CLI args to forward to every preprocess step (e.g. --overwrite).
set BATCH_ARGS=%*

pushd "%SCRIPT_DIR%"

REM ---- 1) default dataset (truebones/zoo) --------------------------------
%PYTHON_EXE% .\preprocess_and_validate.py %BATCH_ARGS%
if errorlevel 1 goto :fail

REM ---- 2) truebones/zoo_upgrade (clean) -----------------------------------
%PYTHON_EXE% preprocess_and_validate.py ^
	--raw-data-dir dataset\truebones\zoo_upgrade\clean ^
	--dataset-dir dataset\truebones\zoo_upgrade\clean_processed %BATCH_ARGS%
if errorlevel 1 goto :fail

REM ---- 3) UnityBundles ----------------------------------------------------
%PYTHON_EXE% preprocess_and_validate.py ^
	--raw-data-dir %UNITYBUNDLES_RAW% ^
	--dataset-dir dataset\unitybundles\processed %BATCH_ARGS%
if errorlevel 1 goto :fail

REM ---- 4) merge the three datasets into one training cond -----------------
%PYTHON_EXE% tools\merge_dataset_cond.py ^
	--datasets dataset\datasets.jsonl ^
	--out dataset\merged\cond.npy
if errorlevel 1 goto :fail

echo.
echo [OK] All three datasets preprocessed and merged into dataset\merged\cond.npy
popd
exit /b 0

:fail
echo.
echo [FAIL] A step above failed -- aborting before the merge.
popd
exit /b 1
