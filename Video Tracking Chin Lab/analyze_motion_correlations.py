"""Analyze particle motion and acceleration correlations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from background_subtraction_tracker import sanitize_output_name

MOTION_COLUMNS = (
    "particle",
    "frame",
    "time_s",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "speed_mm_s",
    "ax_mm_s2",
    "ay_mm_s2",
    "az_mm_s2",
    "acceleration_mm_s2",
)
CORRELATION_SERIES = {
    "vx": "vx_mm_s",
    "vy": "vy_mm_s",
    "vz": "vz_mm_s",
    "speed": "speed_mm_s",
    "ax": "ax_mm_s2",
    "ay": "ay_mm_s2",
    "az": "az_mm_s2",
    "acceleration": "acceleration_mm_s2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute particle-to-particle correlations from velocities_accelerations.csv."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Input CSV. If omitted, File Explorer will open.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Fallback FPS only if the motion CSV has no usable time_s column.",
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=60,
        help="Maximum +/- lag in frames for lagged correlations. Use 0 for zero-lag only.",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=5,
        help="Minimum shared frames required to report a correlation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "correlations",
        help="Parent folder for timestamped correlation output folders.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output folder name. Defaults to input folder/name plus timestamp.",
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

    correlations = calculate_lagged_correlations(
        motion,
        max_lag_frames=args.max_lag_frames,
        min_overlap=args.min_overlap,
    )
    lag_summary = summarize_lagged_correlations(correlations)
    vector_alignment = calculate_vector_alignment(motion)
    alignment_summary = summarize_vector_alignment(vector_alignment)

    output_dir = make_output_dir(csv_path, args.output_dir, args.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion.to_csv(output_dir / "motion_used_for_correlations.csv", index=False)
    correlations.to_csv(output_dir / "pairwise_correlations.csv", index=False)
    lag_summary.to_csv(output_dir / "best_lag_summary.csv", index=False)
    vector_alignment.to_csv(output_dir / "vector_alignment_by_frame.csv", index=False)
    alignment_summary.to_csv(output_dir / "vector_alignment_summary.csv", index=False)
    write_metadata(
        output_dir / "metadata.json",
        csv_path,
        args,
        motion,
        correlations,
    )

    if not args.no_plots:
        plot_correlation_outputs(correlations, output_dir)

    print("\nDone.")
    print(f"Source CSV:    {csv_path.resolve()}")
    print("Input type:    motion")
    print(f"Particles:     {motion['particle'].nunique()}")
    print(f"Rows:          {len(motion)}")
    print(f"Output folder: {output_dir.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
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


def load_motion_csv(
    source: pd.DataFrame,
    csv_path: Path,
    fps: float,
) -> pd.DataFrame:
    missing = [column for column in MOTION_COLUMNS if column not in source.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing motion column(s): {', '.join(missing)}. "
            "Run calculate_motion.py first and choose velocities_accelerations.csv."
        )
    if source.empty:
        raise SystemExit(f"{csv_path} is empty.")

    motion = source.copy()
    motion["frame"] = motion["frame"].astype(int)
    if "time_s" not in motion.columns or motion["time_s"].isna().all():
        motion["time_s"] = motion["frame"].astype(float) / fps
    motion["time_s"] = motion["time_s"].astype(float)
    return motion


def calculate_lagged_correlations(
    motion: pd.DataFrame,
    max_lag_frames: int,
    min_overlap: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    particles = sorted(motion["particle"].dropna().unique())
    lag_values = range(-max_lag_frames, max_lag_frames + 1)
    frame_dt = estimate_frame_dt_s(motion)
    for particle_a, particle_b in combinations(particles, 2):
        for lag_frames in lag_values:
            merged = merge_particle_frames(
                motion,
                particle_a,
                particle_b,
                lag_frames=lag_frames,
            )
            for quantity, column in CORRELATION_SERIES.items():
                r, p_value, n = pearson_result(
                    merged[f"{column}_a"],
                    merged[f"{column}_b"],
                    min_overlap,
                )
                rows.append(
                    {
                        "particle_a": int(particle_a),
                        "particle_b": int(particle_b),
                        "quantity": quantity,
                        "lag_frames": int(lag_frames),
                        "lag_s": float(lag_frames * frame_dt),
                        "n_overlap": int(n),
                        "pearson_r": r,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def merge_particle_frames(
    motion: pd.DataFrame,
    particle_a: int,
    particle_b: int,
    lag_frames: int,
) -> pd.DataFrame:
    columns = ["frame", "time_s", *CORRELATION_SERIES.values()]
    a = motion.loc[motion["particle"] == particle_a, columns].copy()
    b = motion.loc[motion["particle"] == particle_b, columns].copy()
    b["frame"] = b["frame"].astype(int) - int(lag_frames)
    return a.merge(b, on="frame", suffixes=("_a", "_b"))


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

        result = pearsonr(pair["a"].to_numpy(dtype=float), pair["b"].to_numpy(dtype=float))
        return float(result.statistic), float(result.pvalue), n
    except Exception:
        return float(pair["a"].corr(pair["b"])), float("nan"), n


def summarize_lagged_correlations(lagged: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    if lagged.empty:
        return pd.DataFrame()
    valid = lagged.dropna(subset=["pearson_r"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["abs_r"] = valid["pearson_r"].abs()
    for _, group in valid.groupby(["particle_a", "particle_b", "quantity"], sort=True):
        rows.append(group.sort_values(["abs_r", "n_overlap"], ascending=False).iloc[0])
    summary = pd.DataFrame(rows).drop(columns=["abs_r"], errors="ignore")
    return summary.reset_index(drop=True)


def calculate_vector_alignment(motion: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    particles = sorted(motion["particle"].dropna().unique())
    for particle_a, particle_b in combinations(particles, 2):
        columns = [
            "frame",
            "time_s",
            "vx_mm_s",
            "vy_mm_s",
            "vz_mm_s",
            "ax_mm_s2",
            "ay_mm_s2",
            "az_mm_s2",
        ]
        a = motion.loc[motion["particle"] == particle_a, columns].copy()
        b = motion.loc[motion["particle"] == particle_b, columns].copy()
        merged = a.merge(b, on="frame", suffixes=("_a", "_b"))
        for _, row in merged.iterrows():
            velocity_a = np.array([row["vx_mm_s_a"], row["vy_mm_s_a"], row["vz_mm_s_a"]])
            velocity_b = np.array([row["vx_mm_s_b"], row["vy_mm_s_b"], row["vz_mm_s_b"]])
            accel_a = np.array([row["ax_mm_s2_a"], row["ay_mm_s2_a"], row["az_mm_s2_a"]])
            accel_b = np.array([row["ax_mm_s2_b"], row["ay_mm_s2_b"], row["az_mm_s2_b"]])
            rows.append(
                {
                    "particle_a": int(particle_a),
                    "particle_b": int(particle_b),
                    "frame": int(row["frame"]),
                    "time_s": float(row["time_s_a"]),
                    "velocity_cosine": cosine_similarity(velocity_a, velocity_b),
                    "acceleration_cosine": cosine_similarity(accel_a, accel_b),
                }
            )
    return pd.DataFrame(rows)


def summarize_vector_alignment(alignment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    if alignment.empty:
        return pd.DataFrame()
    for (particle_a, particle_b), group in alignment.groupby(
        ["particle_a", "particle_b"],
        sort=True,
    ):
        row: dict[str, float | int] = {
            "particle_a": int(particle_a),
            "particle_b": int(particle_b),
            "frames": int(len(group)),
        }
        for column in ("velocity_cosine", "acceleration_cosine"):
            values = group[column].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{column}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{column}_median"] = (
                float(values.median()) if not values.empty else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    if not np.all(np.isfinite(vector_a)) or not np.all(np.isfinite(vector_b)):
        return float("nan")
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def estimate_frame_dt_s(motion: pd.DataFrame) -> float:
    values = motion[["frame", "time_s"]].dropna().drop_duplicates("frame")
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
    base_name = (
        sanitize_output_name(output_name)
        if output_name
        else f"{sanitize_output_name(csv_path.parent.name or csv_path.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
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
    correlations: pd.DataFrame,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_csv": str(source_csv.resolve()),
                "input_type": "motion",
                "fps": args.fps,
                "max_lag_frames": args.max_lag_frames,
                "min_overlap": args.min_overlap,
                "motion_rows": int(len(motion)),
                "particles": int(motion["particle"].nunique()),
                "correlation_rows": int(len(correlations)),
                "zero_lag_rows": int((correlations["lag_frames"] == 0).sum())
                if "lag_frames" in correlations.columns
                else 0,
            },
            f,
            indent=2,
        )
        f.write("\n")


def plot_correlation_outputs(
    correlations: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    zero_lag = correlations.loc[correlations["lag_frames"] == 0].copy()
    plot_zero_lag_bars(zero_lag, output_dir / "zero_lag_correlations.png", plt)
    plot_lag_curves(correlations, output_dir / "lagged_correlations.png", plt)


def plot_zero_lag_bars(
    zero_lag: pd.DataFrame,
    output_path: Path,
    plt,
) -> None:
    if zero_lag.empty:
        return
    plot_data = zero_lag.dropna(subset=["pearson_r"]).copy()
    if plot_data.empty:
        return
    plot_data["pair"] = (
        plot_data["particle_a"].astype(str) + "-" + plot_data["particle_b"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = plot_data["pair"] + " " + plot_data["quantity"]
    ax.bar(range(len(plot_data)), plot_data["pearson_r"])
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lag_curves(
    lagged: pd.DataFrame,
    output_path: Path,
    plt,
) -> None:
    if lagged.empty:
        return
    plot_data = lagged[
        lagged["quantity"].isin(["speed", "acceleration"])
    ].dropna(subset=["pearson_r"])
    if plot_data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for (particle_a, particle_b, quantity), group in plot_data.groupby(
        ["particle_a", "particle_b", "quantity"],
        sort=True,
    ):
        label = f"{particle_a}-{particle_b} {quantity}"
        ax.plot(group["lag_frames"], group["pearson_r"], label=label)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("lag_frames")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
