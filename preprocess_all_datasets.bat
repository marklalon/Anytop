@echo off
setlocal
REM ----------------------------------------------------------------------
REM Preprocess every dataset listed in dataset\datasets.jsonl (incremental by
REM default) and merge them into a single training cond.
REM
REM   1) each row of dataset\datasets.jsonl, in file order:
REM        preprocess_and_validate.py --raw-data-dir <raw> --dataset-dir <path>
REM   2) merge -> dataset\merged\cond.npy
REM
REM A row without a "raw" field is only merged, not preprocessed. Adding a
REM dataset to the manifest is all it takes to preprocess and merge it here.
REM
REM Any failing step aborts the run before the merge, so the merged cond is
REM never built from incomplete data.
REM
REM Any CLI args passed to this bat are forwarded to EVERY preprocess step
REM (the merge step is unaffected). E.g. `preprocess_all_datasets.bat --overwrite --yes`
REM forces a full rebuild of every dataset with no interactive confirmation;
REM --filter / --object-workers work the same way.
REM ----------------------------------------------------------------------
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe
set DATASETS_FILE=dataset\datasets.jsonl

REM Capture the CLI args to forward to every preprocess step (e.g. --overwrite).
set BATCH_ARGS=%*

pushd "%SCRIPT_DIR%"

REM ---- 1) read the manifest into namespace|path|raw lines ------------------
set DATASET_LIST=%TEMP%\anytop_datasets_%RANDOM%.txt
"%PYTHON_EXE%" -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]; [print(r['namespace']+'|'+r['path']+'|'+r.get('raw','')) for r in rows]" "%DATASETS_FILE%" > "%DATASET_LIST%"
if errorlevel 1 goto :fail

REM ---- 2) preprocess each dataset -------------------------------------------
for /f "usebackq tokens=1-3 delims=|" %%A in ("%DATASET_LIST%") do (
	call :preprocess "%%A" "%%B" "%%C"
	if errorlevel 1 goto :fail
)

REM ---- 3) merge every dataset into one training cond ----------------------
"%PYTHON_EXE%" tools\merge_dataset_cond.py ^
	--datasets "%DATASETS_FILE%" ^
	--out dataset\merged\cond.npy
if errorlevel 1 goto :fail

del "%DATASET_LIST%" 2>nul
echo.
echo [OK] Every dataset in %DATASETS_FILE% preprocessed and merged into dataset\merged\cond.npy
popd
exit /b 0

:preprocess
REM %1 namespace, %2 processed dir, %3 raw dir (empty = merge only)
if "%~3"=="" (
	echo.
	echo [SKIP] %~1: no "raw" in %DATASETS_FILE%, merged as is
	exit /b 0
)
echo.
echo ==== %~1 ====
"%PYTHON_EXE%" preprocess_and_validate.py ^
	--raw-data-dir "%~3" ^
	--dataset-dir "%~2" %BATCH_ARGS%
exit /b %errorlevel%

:fail
del "%DATASET_LIST%" 2>nul
echo.
echo [FAIL] A step above failed -- aborting before the merge.
popd
exit /b 1
