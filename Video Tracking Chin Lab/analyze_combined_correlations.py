"""Run motion correlations and distance-volatility correlations together."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import analyze_distance_volatility as distance_volatility
import analyze_motion_correlations as motion_correlations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run analyze_motion_correlations.py and analyze_distance_volatility.py "
            "from one velocities_accelerations.csv selection."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="velocities_accelerations.csv. If omitted, File Explorer opens once.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Fallback FPS if the CSV time_s column is missing or unusable.",
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=60,
        help="Maximum +/- lag for both correlation analyses.",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=5,
        help="Minimum shared frames required to report a correlation.",
    )
    parser.add_argument(
        "--volatility-window-frames",
        type=int,
        default=15,
        help=(
            "Rolling window used for distance-volatility local volatility. "
            "Default matches analyze_distance_volatility.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "correlations",
        help="Parent folder used by both individual analyses.",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Optional output folder name passed to each analysis. If both would "
            "use the same folder, the second gets the usual numeric suffix."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plots for both analyses.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    csv_path = args.csv or motion_correlations.choose_csv_file()
    source = pd.read_csv(csv_path)

    motion_args = SimpleNamespace(
        csv=csv_path,
        fps=args.fps,
        max_lag_frames=args.max_lag_frames,
        min_overlap=args.min_overlap,
        output_dir=args.output_dir,
        output_name=args.output_name,
        no_plots=args.no_plots,
    )
    distance_args = SimpleNamespace(
        csv=csv_path,
        fps=args.fps,
        volatility_window_frames=args.volatility_window_frames,
        max_lag_frames=args.max_lag_frames,
        min_overlap=args.min_overlap,
        output_dir=args.output_dir,
        output_name=args.output_name,
        no_plots=args.no_plots,
    )

    motion_output_dir = run_motion_correlations(source, csv_path, motion_args)
    distance_output_dir = run_distance_volatility(source, csv_path, distance_args)

    print("\nCombined correlations complete.")
    print(f"Source CSV:                 {csv_path.resolve()}")
    print(f"Motion correlations:        {motion_output_dir.resolve()}")
    print(f"Distance-volatility output: {distance_output_dir.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    motion_correlations.validate_args(
        SimpleNamespace(
            fps=args.fps,
            max_lag_frames=args.max_lag_frames,
            min_overlap=args.min_overlap,
        )
    )
    distance_volatility.validate_args(
        SimpleNamespace(
            fps=args.fps,
            volatility_window_frames=args.volatility_window_frames,
            max_lag_frames=args.max_lag_frames,
            min_overlap=args.min_overlap,
        )
    )


def run_motion_correlations(
    source: pd.DataFrame,
    csv_path: Path,
    args: SimpleNamespace,
) -> Path:
    motion = motion_correlations.load_motion_csv(source.copy(), csv_path, fps=args.fps)
    correlations = motion_correlations.calculate_lagged_correlations(
        motion,
        max_lag_frames=args.max_lag_frames,
        min_overlap=args.min_overlap,
    )
    lag_summary = motion_correlations.summarize_lagged_correlations(correlations)
    vector_alignment = motion_correlations.calculate_vector_alignment(motion)
    alignment_summary = motion_correlations.summarize_vector_alignment(vector_alignment)

    output_dir = motion_correlations.make_output_dir(
        csv_path,
        args.output_dir,
        args.output_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    motion.to_csv(output_dir / "motion_used_for_correlations.csv", index=False)
    correlations.to_csv(output_dir / "pairwise_correlations.csv", index=False)
    lag_summary.to_csv(output_dir / "best_lag_summary.csv", index=False)
    vector_alignment.to_csv(output_dir / "vector_alignment_by_frame.csv", index=False)
    alignment_summary.to_csv(output_dir / "vector_alignment_summary.csv", index=False)
    motion_correlations.write_metadata(
        output_dir / "metadata.json",
        csv_path,
        args,
        motion,
        correlations,
    )
    if not args.no_plots:
        motion_correlations.plot_correlation_outputs(correlations, output_dir)

    print("\nDone.")
    print(f"Source CSV:    {csv_path.resolve()}")
    print("Input type:    motion")
    print(f"Particles:     {motion['particle'].nunique()}")
    print(f"Rows:          {len(motion)}")
    print(f"Output folder: {output_dir.resolve()}")
    return output_dir


def run_distance_volatility(
    source: pd.DataFrame,
    csv_path: Path,
    args: SimpleNamespace,
) -> Path:
    motion = distance_volatility.load_motion_csv(source.copy(), csv_path, fps=args.fps)
    pair_metrics = distance_volatility.calculate_distance_volatility_metrics(
        motion,
        window_frames=args.volatility_window_frames,
    )
    correlations = distance_volatility.calculate_distance_volatility_correlations(
        pair_metrics,
        max_lag_frames=args.max_lag_frames,
        min_overlap=args.min_overlap,
    )
    best_lags = distance_volatility.summarize_best_lags(correlations)
    zero_lag = correlations.loc[correlations["lag_frames"] == 0].copy()

    output_dir = distance_volatility.make_output_dir(
        csv_path,
        args.output_dir,
        args.output_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_metrics.to_csv(output_dir / "distance_volatility_by_frame.csv", index=False)
    correlations.to_csv(output_dir / "distance_volatility_correlations.csv", index=False)
    zero_lag.to_csv(output_dir / "distance_volatility_zero_lag.csv", index=False)
    best_lags.to_csv(output_dir / "distance_volatility_best_lags.csv", index=False)
    distance_volatility.write_metadata(
        output_dir / "metadata.json",
        csv_path,
        args,
        motion,
        pair_metrics,
        correlations,
    )
    if not args.no_plots:
        distance_volatility.plot_outputs(correlations, output_dir)

    print("\nDone.")
    print(f"Source CSV:    {csv_path.resolve()}")
    print(f"Pairs:         {pair_metrics[['particle_a', 'particle_b']].drop_duplicates().shape[0]}")
    print(f"Rows:          {len(pair_metrics)}")
    print(f"Correlations:  {len(correlations)}")
    print(f"Output folder: {output_dir.resolve()}")
    return output_dir


if __name__ == "__main__":
    raise SystemExit(main())
