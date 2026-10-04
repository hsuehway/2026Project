"""Correlate particle separation distance with motion volatility metrics."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from background_subtraction_tracker import sanitize_output_name

REQUIRED_COLUMNS = (
    "particle",
    "frame",
    "time_s",
    "x_mm",
    "y_mm",
    "z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "ax_mm_s2",
    "ay_mm_s2",
    "az_mm_s2",
)

DISTANCE_COLUMNS = {
    "distance_3d": "distance_3d_mm",
    "distance_xy": "distance_xy_mm",
    "distance_xz": "distance_xz_mm",
    "distance_yz": "distance_yz_mm",
}

VOLATILITY_COLUMNS = {
    "separation_velocity_volatility": (
        "separation_velocity_volatility_mm_s",
        "mm/s",
    ),
    "relative_acceleration_volatility": (
        "relative_acceleration_volatility_mm_s2",
        "mm/s^2",
    ),
    "relative_jerk_volatility": (
        "relative_jerk_volatility_mm_s3",
        "mm/s^3",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute lagged and zero-lag correlations between two-particle "
            "distance and three volatility metrics."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="velocities_accelerations.csv. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Fallback FPS if the CSV time_s column is missing or unusable.",
    )
    parser.add_argument(
        "--volatility-window-frames",
        type=int,
        default=15,
        help="Rolling window used for local volatility. Default: 15 frames.",
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=60,
        help="Maximum +/- lag for distance-vs-volatility correlations.",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=5,
        help="Minimum overlapping frames required to report Pearson correlation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "correlations",
        help="Parent folder for timestamped distance-volatility output folders.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output folder name.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    csv_path = args.csv or choose_csv_file()
    motion = load_motion_csv(pd.read_csv(csv_path), csv_path, fps=args.fps)
    pair_metrics = calculate_distance_volatility_metrics(
        motion,
        window_frames=args.volatility_window_frames,
    )
    correlations = calculate_distance_volatility_correlations(
        pair_metrics,
        max_lag_frames=args.max_lag_frames,
        min_overlap=args.min_overlap,
    )
    best_lags = summarize_best_lags(correlations)
    zero_lag = correlations.loc[correlations["lag_frames"] == 0].copy()

    output_dir = make_output_dir(csv_path, args.output_dir, args.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_metrics.to_csv(output_dir / "distance_volatility_by_frame.csv", index=False)
    correlations.to_csv(output_dir / "distance_volatility_correlations.csv", index=False)
    zero_lag.to_csv(output_dir / "distance_volatility_zero_lag.csv", index=False)
    best_lags.to_csv(output_dir / "distance_volatility_best_lags.csv", index=False)
    write_metadata(
        output_dir / "metadata.json",
        csv_path,
        args,
        motion,
        pair_metrics,
        correlations,
    )

    if not args.no_plots:
        plot_outputs(correlations, output_dir)

    print("\nDone.")
    print(f"Source CSV:    {csv_path.resolve()}")
    print(f"Pairs:         {pair_metrics[['particle_a', 'particle_b']].drop_duplicates().shape[0]}")
    print(f"Rows:          {len(pair_metrics)}")
    print(f"Correlations:  {len(correlations)}")
    print(f"Output folder: {output_dir.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.volatility_window_frames < 3:
        raise SystemExit("--volatility-window-frames must be at least 3.")
    if args.max_lag_frames < 0:
        raise SystemExit("--max-lag-frames must be zero or positive.")
    if args.min_overlap < 2:
        raise SystemExit("--min-overlap must be at least 2.")


def choose_csv_file() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(f"Pass --csv because File Explorer is unavailable ({exc}).") from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose velocities_accelerations.csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No CSV file selected.")
    return Path(path)


def load_motion_csv(source: pd.DataFrame, csv_path: Path, fps: float) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing required column(s): {', '.join(missing)}. "
            "Run calculate_motion.py first and choose velocities_accelerations.csv."
        )
    if source.empty:
        raise SystemExit(f"{csv_path} is empty.")

    motion = source.copy()
    motion["particle"] = motion["particle"].astype(int)
    motion["frame"] = motion["frame"].astype(int)
    if "time_s" not in motion.columns or motion["time_s"].isna().all():
        motion["time_s"] = motion["frame"].astype(float) / fps
    motion["time_s"] = motion["time_s"].astype(float)
    return motion.sort_values(["particle", "frame"]).reset_index(drop=True)


def calculate_distance_volatility_metrics(
    motion: pd.DataFrame,
    window_frames: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    particles = sorted(motion["particle"].dropna().unique())
    for particle_a, particle_b in combinations(particles, 2):
        pair = merge_pair_motion(motion, particle_a, particle_b)
        if pair.empty:
            continue
        metrics = calculate_pair_metrics(pair, particle_a, particle_b, window_frames)
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def merge_pair_motion(
    motion: pd.DataFrame,
    particle_a: int,
    particle_b: int,
) -> pd.DataFrame:
    columns = [
        "frame",
        "time_s",
        "x_mm",
        "y_mm",
        "z_mm",
        "vx_mm_s",
        "vy_mm_s",
        "vz_mm_s",
        "ax_mm_s2",
        "ay_mm_s2",
        "az_mm_s2",
    ]
    a = motion.loc[motion["particle"] == particle_a, columns].copy()
    b = motion.loc[motion["particle"] == particle_b, columns].copy()
    return a.merge(b, on="frame", suffixes=("_a", "_b")).sort_values("frame")


def calculate_pair_metrics(
    pair: pd.DataFrame,
    particle_a: int,
    particle_b: int,
    window_frames: int,
) -> pd.DataFrame:
    dx = pair["x_mm_b"].to_numpy(dtype=float) - pair["x_mm_a"].to_numpy(dtype=float)
    dy = pair["y_mm_b"].to_numpy(dtype=float) - pair["y_mm_a"].to_numpy(dtype=float)
    dz = pair["z_mm_b"].to_numpy(dtype=float) - pair["z_mm_a"].to_numpy(dtype=float)
    time_s = pair["time_s_a"].to_numpy(dtype=float)

    distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    distance_xy = np.sqrt(dx**2 + dy**2)
    distance_xz = np.sqrt(dx**2 + dz**2)
    distance_yz = np.sqrt(dy**2 + dz**2)
    separation_velocity = derivative_1d(distance_3d, time_s)

    accel_a = pair[["ax_mm_s2_a", "ay_mm_s2_a", "az_mm_s2_a"]].to_numpy(dtype=float)
    accel_b = pair[["ax_mm_s2_b", "ay_mm_s2_b", "az_mm_s2_b"]].to_numpy(dtype=float)
    relative_accel = accel_b - accel_a
    relative_accel_magnitude = np.linalg.norm(relative_accel, axis=1)
    relative_jerk = derivative_nd(relative_accel, time_s)
    relative_jerk_magnitude = np.linalg.norm(relative_jerk, axis=1)

    metrics = pd.DataFrame(
        {
            "particle_a": int(particle_a),
            "particle_b": int(particle_b),
            "frame": pair["frame"].astype(int).to_numpy(),
            "time_s": time_s,
            "distance_3d_mm": distance_3d,
            "distance_xy_mm": distance_xy,
            "distance_xz_mm": distance_xz,
            "distance_yz_mm": distance_yz,
            "separation_velocity_mm_s": separation_velocity,
            "relative_acceleration_magnitude_mm_s2": relative_accel_magnitude,
            "relative_jerk_magnitude_mm_s3": relative_jerk_magnitude,
        }
    )
    metrics["separation_velocity_volatility_mm_s"] = rolling_std(
        metrics["separation_velocity_mm_s"],
        window_frames,
    )
    metrics["relative_acceleration_volatility_mm_s2"] = rolling_std(
        metrics["relative_acceleration_magnitude_mm_s2"],
        window_frames,
    )
    metrics["relative_jerk_volatility_mm_s3"] = rolling_std(
        metrics["relative_jerk_magnitude_mm_s3"],
        window_frames,
    )
    return metrics


def derivative_1d(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.full_like(values, np.nan, dtype=float)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, time_s, edge_order=edge_order)


def derivative_nd(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.full_like(values, np.nan, dtype=float)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, time_s, axis=0, edge_order=edge_order)


def rolling_std(values: pd.Series, window_frames: int) -> pd.Series:
    min_periods = max(3, min(window_frames, window_frames // 2 + 1))
    return values.replace([np.inf, -np.inf], np.nan).rolling(
        window=window_frames,
        center=True,
        min_periods=min_periods,
    ).std(ddof=0)


def calculate_distance_volatility_correlations(
    pair_metrics: pd.DataFrame,
    max_lag_frames: int,
    min_overlap: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if pair_metrics.empty:
        return pd.DataFrame()
    frame_dt = estimate_frame_dt_s(pair_metrics)
    lag_values = range(-max_lag_frames, max_lag_frames + 1)
    for (particle_a, particle_b), group in pair_metrics.groupby(
        ["particle_a", "particle_b"],
        sort=True,
    ):
        group = group.sort_values("frame").copy()
        for distance_name, distance_column in DISTANCE_COLUMNS.items():
            for volatility_name, (volatility_column, volatility_units) in VOLATILITY_COLUMNS.items():
                for lag_frames in lag_values:
                    merged = lagged_distance_volatility_frame(
                        group,
                        distance_column,
                        volatility_column,
                        lag_frames,
                    )
                    r, p_value, n = pearson_result(
                        merged["distance"],
                        merged["volatility"],
                        min_overlap,
                    )
                    rows.append(
                        {
                            "particle_a": int(particle_a),
                            "particle_b": int(particle_b),
                            "distance_quantity": distance_name,
                            "distance_units": "mm",
                            "volatility_quantity": volatility_name,
                            "volatility_units": volatility_units,
                            "lag_frames": int(lag_frames),
                            "lag_s": float(lag_frames * frame_dt),
                            "lag_definition": (
                                "positive lag means volatility occurs after distance"
                            ),
                            "n_overlap": int(n),
                            "pearson_r": r,
                            "p_value": p_value,
                        }
                    )
    return pd.DataFrame(rows)


def lagged_distance_volatility_frame(
    group: pd.DataFrame,
    distance_column: str,
    volatility_column: str,
    lag_frames: int,
) -> pd.DataFrame:
    distance = group[["frame", distance_column]].rename(
        columns={distance_column: "distance"}
    )
    volatility = group[["frame", volatility_column]].rename(
        columns={volatility_column: "volatility"}
    )
    volatility["frame"] = volatility["frame"].astype(int) - int(lag_frames)
    return distance.merge(volatility, on="frame")


def pearson_result(
    values_a: pd.Series,
    values_b: pd.Series,
    min_overlap: int,
) -> tuple[float, float, int]:
    pair = pd.DataFrame({"a": values_a, "b": values_b}).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    n = len(pair)
    if n < min_overlap or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return float("nan"), float("nan"), n
    try:
        from scipy.stats import pearsonr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pearsonr(
                pair["a"].to_numpy(dtype=float),
                pair["b"].to_numpy(dtype=float),
            )
        return float(result.statistic), float(result.pvalue), n
    except Exception:
        return float(pair["a"].corr(pair["b"])), float("nan"), n


def summarize_best_lags(correlations: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return pd.DataFrame()
    valid = correlations.dropna(subset=["pearson_r"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["abs_r"] = valid["pearson_r"].abs()
    rows = []
    for _, group in valid.groupby(
        ["particle_a", "particle_b", "distance_quantity", "volatility_quantity"],
        sort=True,
    ):
        rows.append(group.sort_values(["abs_r", "n_overlap"], ascending=False).iloc[0])
    return pd.DataFrame(rows).drop(columns=["abs_r"], errors="ignore").reset_index(drop=True)


def estimate_frame_dt_s(data: pd.DataFrame) -> float:
    values = data[["frame", "time_s"]].dropna().drop_duplicates("frame")
    if len(values) < 2:
        return float("nan")
    values = values.sort_values("frame")
    frame_delta = values["frame"].diff().dropna()
    time_delta = values["time_s"].diff().dropna()
    valid = frame_delta != 0
    if not valid.any():
        return float("nan")
    return float((time_delta[valid] / frame_delta[valid]).median())


def make_output_dir(
    csv_path: Path,
    output_parent: Path,
    output_name: str | None,
) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_name:
        base_name = sanitize_output_name(output_name)
    else:
        source_name = sanitize_output_name(csv_path.parent.name or csv_path.stem)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"distance_volatility_{source_name}_{timestamp}"
    output_dir = output_parent / base_name
    if not output_dir.exists():
        return output_dir
    suffix = 2
    while True:
        candidate = output_parent / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def write_metadata(
    output_path: Path,
    source_csv: Path,
    args: argparse.Namespace,
    motion: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_csv": str(source_csv.resolve()),
                "input_type": "motion",
                "fps_fallback": float(args.fps),
                "volatility_window_frames": int(args.volatility_window_frames),
                "max_lag_frames": int(args.max_lag_frames),
                "min_overlap": int(args.min_overlap),
                "lag_definition": "positive lag means volatility occurs after distance",
                "distance_metrics": DISTANCE_COLUMNS,
                "volatility_metrics": {
                    key: {"column": column, "units": units}
                    for key, (column, units) in VOLATILITY_COLUMNS.items()
                },
                "motion_rows": int(len(motion)),
                "pair_metric_rows": int(len(pair_metrics)),
                "correlation_rows": int(len(correlations)),
                "zero_lag_rows": int((correlations["lag_frames"] == 0).sum())
                if "lag_frames" in correlations.columns
                else 0,
            },
            f,
            indent=2,
        )
        f.write("\n")


def plot_outputs(correlations: pd.DataFrame, output_dir: Path) -> None:
    if correlations.empty:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_zero_lag(correlations, output_dir / "distance_volatility_zero_lag.png", plt)
    plot_lag_curves(correlations, output_dir / "distance_volatility_lag_curves.png", plt)


def plot_zero_lag(correlations: pd.DataFrame, output_path: Path, plt) -> None:
    zero = correlations.loc[correlations["lag_frames"] == 0].dropna(
        subset=["pearson_r"]
    )
    if zero.empty:
        return
    zero = zero.loc[zero["distance_quantity"] == "distance_3d"].copy()
    if zero.empty:
        return
    labels = (
        zero["particle_a"].astype(str)
        + "-"
        + zero["particle_b"].astype(str)
        + " "
        + zero["volatility_quantity"]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(zero)), zero["pearson_r"])
    ax.set_xticks(range(len(zero)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lag_curves(correlations: pd.DataFrame, output_path: Path, plt) -> None:
    data = correlations.loc[
        correlations["distance_quantity"] == "distance_3d"
    ].dropna(subset=["pearson_r"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for (particle_a, particle_b, volatility), group in data.groupby(
        ["particle_a", "particle_b", "volatility_quantity"],
        sort=True,
    ):
        label = f"{particle_a}-{particle_b} {volatility}"
        ax.plot(group["lag_frames"], group["pearson_r"], label=label)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("lag_frames")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
