"""Compatibility entry point for the maintained particle-mode analyzer."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / "orbit-detection"
        / "code"
        / "analyze_particle_modes.py"
    )
    runpy.run_path(str(script), run_name="__main__")
