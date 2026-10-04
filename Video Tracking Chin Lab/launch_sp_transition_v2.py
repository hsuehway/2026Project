"""Open the direct-position SP experiment in Trajectory Studio."""
from pathlib import Path

from trajectory_ml.config import load_settings
from trajectory_ml.studio import launch


if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    settings = load_settings(project/"trajectory_ml"/"sp_transition_v2.yaml", project)
    session = project/".sp_trajectory_studio_session.json"
    if session.exists():
        try:
            saved = load_settings(session, project)
            if saved.experiment_family == "sp_transition_v2":
                settings = saved
        except (ValueError, TypeError):
            pass
    launch(project, settings=settings)
