@echo off
setlocal
set SCRIPT_DIR=%~dp0

call "%SCRIPT_DIR%train_locomotion.bat"
if errorlevel 1 exit /b %errorlevel%

call "%SCRIPT_DIR%train_stationary.bat"
if errorlevel 1 exit /b %errorlevel%

call "%SCRIPT_DIR%train_transition.bat"
if errorlevel 1 exit /b %errorlevel%

endlocal
