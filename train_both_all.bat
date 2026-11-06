@echo off
setlocal EnableExtensions
REM Train the full-dataset autoencoder, then full-dataset diffusion.
set "SCRIPT_DIR=%~dp0"

call "%SCRIPT_DIR%train_autoencoder_all.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

call "%SCRIPT_DIR%train_diffusion_full.bat"
exit /b %ERRORLEVEL%
