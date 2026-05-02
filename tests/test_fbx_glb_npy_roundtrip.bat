@echo off
set SCRIPT_DIR=%~dp0
set ANYTOP_ROOT=%SCRIPT_DIR%..

python "%SCRIPT_DIR%test_fbx_glb_npy_roundtrip.py" ^
    --fbx "D:\AI\pcvg-skeleton-animation\Anytop\dataset\truebones\zoo\Truebone_Z-OO\Horse\HorseALL-RunToStop.fbx" ^
    --output-dir "D:\AI\pcvg-skeleton-animation\Anytop\outputs\fbx_npy_roundtrip"