"""Launch the SP Trajectory Studio, or run its reproducible CLI pipeline."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="SP-only YAML experiment configuration")
    parser.add_argument("--qc", action="store_true", help="Write a QC report without training")
    parser.add_argument("--run", action="store_true", help="Run nested SP validation from config")
    parser.add_argument("--no-browser", action="store_true", help="Serve without opening a browser")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--family", choices=["acceleration_v1", "sp_transition_v2", "physics_v1"], help="Select an experiment family")
    parser.add_argument("--resume",type=Path,help="Resume an interrupted v2 output folder using its saved configuration")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    from trajectory_ml.config import load_settings
    from trajectory_ml.pipeline import run_pipeline
    from trajectory_ml.studio import launch
    settings = load_settings(args.resume/"resolved_config.yaml" if args.resume else args.config, project)
    if args.resume:
        import yaml
        old = yaml.safe_load((args.resume/"resolved_config.yaml").read_text(encoding="utf-8"))
        settings.validation_protocol = old.get("validation_protocol", "legacy")
    if args.family:
        settings.experiment_family = args.family
    if args.qc or args.run or args.resume:
        if settings.experiment_family == "sp_transition_v2":
            from trajectory_ml.transition_pipeline import run_transition_pipeline
            run_pipeline = run_transition_pipeline
        elif settings.experiment_family == "physics_v1":
            from trajectory_ml.behavior_pipeline import run_behavior_pipeline
            run_pipeline = run_behavior_pipeline
        if args.resume:
            if settings.experiment_family != "sp_transition_v2":
                parser.error("Resume is available for v2 training only.")
            result = run_pipeline(settings,"train",lambda message: print(message,flush=True),resume_output=args.resume)
        else:
            result = run_pipeline(settings, "qc" if args.qc else "train", lambda message: print(message, flush=True))
        print(f"Report: {Path(result['output_dir']) / 'report.html'}", flush=True)
        return 0
    return launch(project, args.port, not args.no_browser, settings if args.config or args.family else None)


if __name__ == "__main__":
    raise SystemExit(main())
