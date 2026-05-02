@echo off
set SCRIPT_DIR=%~dp0
set ANYTOP_ROOT=%SCRIPT_DIR%..

python "%SCRIPT_DIR%test_glb_npy_roundtrip.py" ^
    --output-dir "D:\AI\pcvg-skeleton-animation\Anytop\outputs\glb_npy_roundtrip" ^
    --anim-glb "D:\AI\pcvg-skeleton-animation\Anytop\outputs\glb_npy_roundtrip\AlligatorALL-Bite1.glb" ^
    --object-type Horse