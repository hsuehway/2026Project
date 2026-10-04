@echo off
setlocal
pushd "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m video_analysis_studio
) else (
  start "" pythonw -m video_analysis_studio
)
popd
endlocal

