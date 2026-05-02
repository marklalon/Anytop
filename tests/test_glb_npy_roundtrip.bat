@echo off
set SCRIPT_DIR=%~dp0
set ANYTOP_ROOT=%SCRIPT_DIR%..

python "%SCRIPT_DIR%test_glb_npy_roundtrip.py" ^
    --output-dir "%ANYTOP_ROOT%\outputs\glb_npy_roundtrip" ^
    --tpose-glb "%ANYTOP_ROOT%\outputs\glb_npy_roundtrip\tpose.glb" ^
    --anim-glb "%ANYTOP_ROOT%\outputs\glb_npy_roundtrip\original.glb" ^
    --object-type Horse