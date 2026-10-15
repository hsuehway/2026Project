#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ -x ".venv-macos/bin/python" ]]; then
  exec ".venv-macos/bin/python" -m video_analysis_studio
fi
exec python3 -m video_analysis_studio
