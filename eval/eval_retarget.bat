@echo off
REM ===========================================================================
REM  eval_retarget.bat  —  End-to-end retarget CLI smoke tests
REM ===========================================================================
REM  Tests the ``Anytop.utils.retarget`` CLI across all supported source
REM  formats (.npy / .fbx / .glb), same-species and cross-species cases,
REM  and with/without --cond_path merging.
REM ===========================================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..\
set ANYTOP_ROOT=%SCRIPT_DIR%..\
set PYTHON_EXE=%REPO_ROOT%.venv\Scripts\python.exe
set OUTPUT_DIR=%ANYTOP_ROOT%outputs\eval_retarget_output

if NOT EXIST "%PYTHON_EXE%" (
    echo [ERROR] Python venv not found at %PYTHON_EXE%
    exit /b 1
)

echo ============================================
echo  Retarget CLI smoke tests
echo  Output root: %OUTPUT_DIR%
echo ============================================
echo.

REM Clean output directory
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%" 2>nul

set PASS=0
set FAIL=0

REM ==================================================================
REM  T1: Buffalo .fbx -> Buffalo (same-species, raw animation)
REM ==================================================================
echo --- T1: Buffalo .fbx -^> Buffalo ^(same-species^) ---
echo.
"%PYTHON_EXE%" -m Anytop.utils.retarget --source "%ANYTOP_ROOT%dataset\truebones\zoo\Truebone_Z-OO\Buffalo\Buffalo-RunLoop.fbx" --object_type Buffalo --output_dir "%OUTPUT_DIR%"
if !errorlevel! equ 0 (set /a PASS+=1 & echo [PASS] T1) else (set /a FAIL+=1 & echo [FAIL] T1  ^(exit !errorlevel!^))
echo.
echo.

REM ==================================================================
REM  T2: Buffalo .fbx -> Horse (cross-species, raw animation)
REM ==================================================================
echo --- T2: Buffalo .fbx -^> Horse ^(cross-species^) ---
echo.
"%PYTHON_EXE%" -m Anytop.utils.retarget --source "%ANYTOP_ROOT%dataset\truebones\zoo\Truebone_Z-OO\Buffalo\Buffalo-RunLoop.fbx" --object_type Horse --output_dir "%OUTPUT_DIR%"
if !errorlevel! equ 0 (set /a PASS+=1 & echo [PASS] T2) else (set /a FAIL+=1 & echo [FAIL] T2  ^(exit !errorlevel!^))
echo.
echo.

REM ==================================================================
REM  T3: Buffalo .npy -> Horse (cross-species, cond-free source)
REM ==================================================================
echo --- T3: Buffalo .npy -^> Horse ^(cross-species, cond-free^) ---
echo.
"%PYTHON_EXE%" -m Anytop.utils.retarget --source "%ANYTOP_ROOT%dataset\truebones\zoo\truebones_processed\motions\Buffalo_RunLoop_115.npy" --object_type Horse --output_dir "%OUTPUT_DIR%"
if !errorlevel! equ 0 (set /a PASS+=1 & echo [PASS] T3) else (set /a FAIL+=1 & echo [FAIL] T3  ^(exit !errorlevel!^))
echo.
echo.

REM ==================================================================
REM  T4: Buffalo .glb -> Buffalo (debug_phase2, same-species)
REM ==================================================================
echo --- T4: Buffalo .glb -^> Buffalo ^(debug_phase2^) ---
echo.
"%PYTHON_EXE%" -m Anytop.utils.retarget --source "%REPO_ROOT%outputs\debug_phase2\Buffalo\debug_phase2_animation.glb" --object_type Buffalo --output_dir "%OUTPUT_DIR%"
if !errorlevel! equ 0 (set /a PASS+=1 & echo [PASS] T4) else (set /a FAIL+=1 & echo [FAIL] T4  ^(exit !errorlevel!^))
echo.
echo.

REM ==================================================================
REM  T5: Buffalo .glb -> Horse (debug_phase2, cross-species)
REM ==================================================================
echo --- T5: Buffalo .glb -^> Horse ^(debug_phase2, cross-species^) ---
echo.
"%PYTHON_EXE%" -m Anytop.utils.retarget --source "%REPO_ROOT%outputs\debug_phase2\Buffalo\debug_phase2_animation.glb" --object_type Horse --output_dir "%OUTPUT_DIR%"
if !errorlevel! equ 0 (set /a PASS+=1 & echo [PASS] T5) else (set /a FAIL+=1 & echo [FAIL] T5  ^(exit !errorlevel!^))
echo.
echo.

REM ==================================================================
REM  T6: Horse .npy -> dragon (with --cond_path)
REM ==================================================================
echo --- T6: Horse .npy -^> dragon ^(with --cond_path^) ---
echo.
"%PYTHON_EXE%" -m Anytop.utils.retarget --source "%ANYTOP_ROOT%dataset\truebones\zoo\truebones_processed\motions\Horse_RunLoop_28.npy" --object_type dragon --cond_path "%ANYTOP_ROOT%outputs\new_skeleton\cond.npy" --output_dir "%OUTPUT_DIR%"
if !errorlevel! equ 0 (set /a PASS+=1 & echo [PASS] T6) else (set /a FAIL+=1 & echo [FAIL] T6  ^(exit !errorlevel!^))
echo.
echo.

REM ==================================================================
REM  T7: Monkey .fbx -> dragon (cond-free source + --cond_path)
REM ==================================================================
echo --- T7: Monkey .fbx -^> dragon ^(cond-free + --cond_path^) ---
echo.
"%PYTHON_EXE%" -m Anytop.utils.retarget --source "%ANYTOP_ROOT%dataset\truebones\zoo\Truebone_Z-OO\Monkey\walk.fbx" --object_type dragon --cond_path "%ANYTOP_ROOT%outputs\new_skeleton\cond.npy" --output_dir "%OUTPUT_DIR%"
if !errorlevel! equ 0 (set /a PASS+=1 & echo [PASS] T7) else (set /a FAIL+=1 & echo [FAIL] T7  ^(exit !errorlevel!^))
echo.
echo.

REM ==================================================================
REM  Summary
REM ==================================================================
echo ============================================
echo  Results: !PASS! passed, !FAIL! failed
echo ============================================

endlocal
exit /b !FAIL!
