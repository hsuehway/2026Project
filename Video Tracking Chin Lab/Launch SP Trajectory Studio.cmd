@echo off
setlocal
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launch_sp_trajectory_studio.py
) else (
  python launch_sp_trajectory_studio.py
)
if errorlevel 1 pause
popd
endlocal
