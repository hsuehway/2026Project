"""Open Physics Model v1 in the shared SP Trajectory Studio."""
from dataclasses import replace
from pathlib import Path

from trajectory_ml.config import load_settings
from trajectory_ml.studio import launch


if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    settings = load_settings(project/"trajectory_ml"/"sp_physics_v1.yaml",project)
    session = project/".sp_trajectory_studio_session.json"
    if session.exists():
        try:
            settings = replace(load_settings(session,project),experiment_family="physics_v1",validation_protocol="behavior_v1")
        except (ValueError,TypeError):
            pass
    launch(project,settings=settings)
