"""Calculate velocity and acceleration from 3D particle tracks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from background_subtraction_tracker import sanitize_output_name
from trajectory_smoothing import (
    RAW_POSITION_ALIAS_COLUMNS,
    RAW_POSITION_COLUMNS,
    SMOOTH_POLYORDER_COLUMN,
    SMOOTH_POSITION_COLUMNS,
    SMOOTH_WINDOW_COLUMN,
    continuous_segment_slices,
    effective_savgol_window,
    smooth_positions,
    validate_smoothing_parameters,
)

REQUIRED_COLUMNS = ("particle", "frame", "x_mm", "y_mm", "z_mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate particle velocities and accelerations from tracks_3d.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="tracks_3d.csv file. If omitted, File Explorer will open.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Video frame rate used to convert frame number to seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "motion",
        help=(
            "Parent folder for timestamped motion output folders."
        ),
    )
    parser.add_argument(
        "--output-name",
        help="Optional output folder name. Defaults to CSV name plus timestamp.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help=(
            "Fallback Savitzky-Golay window for legacy CSVs without tracker-"
            "generated smooth positions. Tracker settings take precedence."
        ),
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Fallback polynomial order for legacy, unsmoothed track CSVs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip position-comparison and speed/acceleration PNG plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    csv_path = args.csv or choose_csv_file()
    tracks = pd.read_csv(csv_path)
    validate_tracks(tracks, csv_path)

    motion = calculate_motion(
        tracks,
        fps=args.fps,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )
    summary = summarize_motion(motion, args.fps, args.smooth_window, args.smooth_polyorder)

    output_dir = make_motion_output_dir(csv_path, args.output_dir, args.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_path = output_dir / "velocities_accelerations.csv"
    summary_path = output_dir / "motion_summary.csv"
    metadata_path = output_dir / "metadata.json"
    motion.to_csv(motion_path, index=False)
    summary.to_csv(summary_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_csv": str(csv_path.resolve()),
                "fps": args.fps,
                "smooth_window": args.smooth_window,
                "smooth_polyorder": args.smooth_polyorder,
                "position_smoothing_source": sorted(
                    motion["position_smoothing_source"].dropna().astype(str).unique()
                ),
                "effective_smooth_windows": sorted(
                    int(value)
                    for value in motion[SMOOTH_WINDOW_COLUMN].dropna().unique()
                ),
                "effective_smooth_polyorders": sorted(
                    int(value)
                    for value in motion[SMOOTH_POLYORDER_COLUMN].dropna().unique()
                ),
                "raw_derivative_method": "finite differences within continuous frame segments",
                "smoothed_derivative_method": (
                    "Savitzky-Golay local-polynomial derivatives with finite-difference fallback"
                ),
                "split_at_frame_gaps": True,
                "rows": int(len(motion)),
                "particles": int(motion["particle"].nunique()),
            },
            f,
            indent=2,
        )
        f.write("\n")

    if not args.no_plots:
        plot_position_coordinates(motion, output_dir / "position_coordinates.png")
        plot_velocity_comparison(motion, output_dir / "velocity_comparison.png")
        plot_acceleration_comparison(
            motion,
            output_dir / "acceleration_comparison.png",
        )
        plot_motion(motion, output_dir / "motion_plots.png")

    print("\nDone.")
    print(f"Source CSV:     {csv_path.resolve()}")
    print(f"FPS:            {args.fps:g}")
    print(f"Rows:           {len(motion)}")
    print(f"Particles:      {motion['particle'].nunique()}")
    print(f"Output folder:  {output_dir.resolve()}")
    print(f"Motion CSV:     {motion_path.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.smooth_window < 0:
        raise SystemExit("--smooth-window must be zero or positive.")
    if args.smooth_polyorder < 0:
        raise SystemExit("--smooth-polyorder must be zero or positive.")
    if args.smooth_window not in (0, 1) and args.smooth_window % 2 == 0:
        raise SystemExit("--smooth-window must be odd, or use 0/1 to disable.")
    if args.smooth_window > 1 and args.smooth_polyorder >= args.smooth_window:
        raise SystemExit("--smooth-polyorder must be smaller than --smooth-window.")


def validate_tracks(tracks: pd.DataFrame, csv_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in tracks.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {missing}")
    if tracks.empty:
        raise SystemExit(f"{csv_path} is empty.")


def choose_csv_file() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(f"Pass --csv because File Explorer is unavailable ({exc}).") from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose tracks_3d.csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No CSV file selected.")
    return Path(path)


def calculate_motion(
    tracks: pd.DataFrame,
    fps: float,
    smooth_window: int,
    smooth_polyorder: int,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for particle, group in tracks.groupby("particle", sort=True):
        group = group.sort_values("frame").drop_duplicates("frame").copy()
        frame = group["frame"].to_numpy(dtype=float)
        time_s = frame / fps
        position = group.loc[:, RAW_POSITION_COLUMNS].to_numpy(dtype=float)
        group.loc[:, RAW_POSITION_ALIAS_COLUMNS] = position
        tracker_positions_available = all(
            column in group.columns for column in SMOOTH_POSITION_COLUMNS
        )
        if tracker_positions_available:
            provided_smooth_position = group.loc[
                :, SMOOTH_POSITION_COLUMNS
            ].to_numpy(dtype=float)
            group_smooth_window = unique_integer_setting(
                group,
                SMOOTH_WINDOW_COLUMN,
                smooth_window,
            )
            group_smooth_polyorder = unique_integer_setting(
                group,
                SMOOTH_POLYORDER_COLUMN,
                smooth_polyorder,
            )
            smoothing_source = "tracker"
        else:
            provided_smooth_position = None
            group_smooth_window = smooth_window
            group_smooth_polyorder = smooth_polyorder
            smoothing_source = "motion_legacy_fallback"
        validate_smoothing_parameters(
            group_smooth_window,
            group_smooth_polyorder,
        )
        smooth_position = np.full_like(position, np.nan, dtype=float)
        raw_velocity = np.full_like(position, np.nan, dtype=float)
        raw_acceleration = np.full_like(position, np.nan, dtype=float)
        velocity = np.full_like(position, np.nan, dtype=float)
        acceleration = np.full_like(position, np.nan, dtype=float)

        for start, stop in continuous_segment_slices(frame):
            segment_position = position[start:stop]
            segment_time = time_s[start:stop]
            segment_raw_velocity = derivative(segment_position, segment_time)
            segment_raw_acceleration = derivative(
                segment_raw_velocity,
                segment_time,
            )
            (
                segment_smooth_position,
                segment_velocity,
                segment_acceleration,
            ) = smooth_kinematics(
                segment_position,
                segment_time,
                group_smooth_window,
                group_smooth_polyorder,
                provided_smooth_position=(
                    None
                    if provided_smooth_position is None
                    else provided_smooth_position[start:stop]
                ),
            )
            smooth_position[start:stop] = segment_smooth_position
            raw_velocity[start:stop] = segment_raw_velocity
            raw_acceleration[start:stop] = segment_raw_acceleration
            velocity[start:stop] = segment_velocity
            acceleration[start:stop] = segment_acceleration

        group["time_s"] = time_s
        group["x_smooth_mm"] = smooth_position[:, 0]
        group["y_smooth_mm"] = smooth_position[:, 1]
        group["z_smooth_mm"] = smooth_position[:, 2]
        group[SMOOTH_WINDOW_COLUMN] = int(group_smooth_window)
        group[SMOOTH_POLYORDER_COLUMN] = int(group_smooth_polyorder)
        group["position_smoothing_source"] = smoothing_source
        group["vx_raw_mm_s"] = raw_velocity[:, 0]
        group["vy_raw_mm_s"] = raw_velocity[:, 1]
        group["vz_raw_mm_s"] = raw_velocity[:, 2]
        group["speed_raw_mm_s"] = np.linalg.norm(raw_velocity, axis=1)
        group["vx_mm_s"] = velocity[:, 0]
        group["vy_mm_s"] = velocity[:, 1]
        group["vz_mm_s"] = velocity[:, 2]
        group["speed_mm_s"] = np.linalg.norm(velocity, axis=1)
        group["ax_raw_mm_s2"] = raw_acceleration[:, 0]
        group["ay_raw_mm_s2"] = raw_acceleration[:, 1]
        group["az_raw_mm_s2"] = raw_acceleration[:, 2]
        group["acceleration_raw_mm_s2"] = np.linalg.norm(
            raw_acceleration,
            axis=1,
        )
        group["ax_mm_s2"] = acceleration[:, 0]
        group["ay_mm_s2"] = acceleration[:, 1]
        group["az_mm_s2"] = acceleration[:, 2]
        group["acceleration_mm_s2"] = np.linalg.norm(acceleration, axis=1)
        group["dt_s"] = np.gradient(time_s) if len(time_s) > 1 else np.nan
        group["particle"] = particle
        groups.append(group)

    return pd.concat(groups, ignore_index=True).sort_values(
        ["particle", "frame"]
    ).reset_index(drop=True)


def smooth_kinematics(
    position: np.ndarray,
    time_s: np.ndarray,
    smooth_window: int,
    smooth_polyorder: int,
    provided_smooth_position: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth_position = (
        smooth_positions(position, smooth_window, smooth_polyorder)
        if provided_smooth_position is None
        else np.asarray(provided_smooth_position, dtype=float).copy()
    )
    velocity = derivative(smooth_position, time_s)
    acceleration = derivative(velocity, time_s)
    window = effective_savgol_window(
        len(position),
        smooth_window,
        smooth_polyorder,
    )
    if window is None or len(time_s) < 2:
        return smooth_position, velocity, acceleration

    time_steps = np.diff(time_s)
    delta_s = float(np.median(time_steps))
    if (
        not np.isfinite(delta_s)
        or delta_s <= 0
        or not np.allclose(time_steps, delta_s, rtol=1e-6, atol=1e-12)
    ):
        return smooth_position, velocity, acceleration

    try:
        from scipy.signal import savgol_filter
    except Exception:
        return smooth_position, velocity, acceleration

    if smooth_polyorder >= 1:
        velocity = savgol_filter(
            position,
            window_length=window,
            polyorder=smooth_polyorder,
            deriv=1,
            delta=delta_s,
            axis=0,
        )
    if smooth_polyorder >= 2:
        acceleration = savgol_filter(
            position,
            window_length=window,
            polyorder=smooth_polyorder,
            deriv=2,
            delta=delta_s,
            axis=0,
        )
    else:
        acceleration = derivative(velocity, time_s)
    return smooth_position, velocity, acceleration


def unique_integer_setting(
    group: pd.DataFrame,
    column: str,
    fallback: int,
) -> int:
    if column not in group.columns:
        return int(fallback)
    values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
    if len(values) == 0:
        return int(fallback)
    if len(values) > 1:
        raise ValueError(
            f"Particle {int(group['particle'].iloc[0])} has multiple {column} values: "
            f"{sorted(float(value) for value in values)}"
        )
    return int(values[0])


def derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.full_like(values, np.nan, dtype=float)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, time_s, axis=0, edge_order=edge_order)


def summarize_motion(
    motion: pd.DataFrame,
    fps: float,
    smooth_window: int,
    smooth_polyorder: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for particle, group in motion.groupby("particle", sort=True):
        duration_s = float(group["time_s"].max() - group["time_s"].min())
        row: dict[str, float | int] = {
            "particle": int(particle),
            "points": int(len(group)),
            "start_frame": int(group["frame"].min()),
            "end_frame": int(group["frame"].max()),
            "duration_s": duration_s,
            "fps": float(fps),
            "smooth_window": unique_integer_setting(
                group,
                SMOOTH_WINDOW_COLUMN,
                smooth_window,
            ),
            "smooth_polyorder": unique_integer_setting(
                group,
                SMOOTH_POLYORDER_COLUMN,
                smooth_polyorder,
            ),
        }
        for column in ("speed_mm_s", "acceleration_mm_s2", "vx_mm_s", "vy_mm_s", "vz_mm_s"):
            values = group[column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_max"] = float("nan")
            else:
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def make_motion_output_dir(
    csv_path: Path,
    output_parent: Path | None,
    output_name: str | None,
) -> Path:
    parent = output_parent or (Path("outputs_background") / "motion")
    parent.mkdir(parents=True, exist_ok=True)
    base_name = (
        sanitize_output_name(output_name)
        if output_name
        else f"{sanitize_output_name(csv_path.parent.name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = parent / base_name
    if not output_dir.exists():
        return output_dir

    suffix = 2
    while True:
        candidate = parent / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def plot_motion(motion: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for particle, group in motion.groupby("particle", sort=True):
        label = f"particle {particle}"
        axes[0].plot(group["time_s"], group["speed_mm_s"], label=label)
        axes[1].plot(group["time_s"], group["acceleration_mm_s2"], label=label)

    axes[0].set_ylabel("speed_mm_s")
    axes[1].set_ylabel("acceleration_mm_s2")
    axes[1].set_xlabel("time_s")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_position_coordinates(motion: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    coordinate_columns = (
        ("x_mm", "x_smooth_mm", "x position (mm)"),
        ("y_mm", "y_smooth_mm", "y position (mm)"),
        ("z_mm", "z_smooth_mm", "z position (mm)"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for particle_index, (particle, group) in enumerate(
        motion.groupby("particle", sort=True)
    ):
        color = f"C{particle_index % 10}"
        for axis_index, (raw_column, smooth_column, ylabel) in enumerate(
            coordinate_columns
        ):
            ax = axes[axis_index]
            ax.plot(
                group["time_s"],
                group[raw_column],
                color=color,
                linewidth=0.8,
                alpha=0.4,
                label=f"particle {particle} raw",
            )
            ax.plot(
                group["time_s"],
                group[smooth_column],
                color=color,
                linewidth=1.6,
                label=f"particle {particle} smooth",
            )
            ax.set_ylabel(ylabel)

    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Raw and smoothed particle positions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_velocity_comparison(motion: pd.DataFrame, output_path: Path) -> None:
    plot_kinematic_comparison(
        motion=motion,
        output_path=output_path,
        columns=(
            ("vx_raw_mm_s", "vx_mm_s", "x velocity (mm/s)"),
            ("vy_raw_mm_s", "vy_mm_s", "y velocity (mm/s)"),
            ("vz_raw_mm_s", "vz_mm_s", "z velocity (mm/s)"),
            ("speed_raw_mm_s", "speed_mm_s", "speed (mm/s)"),
        ),
        title="Frame-difference and smooth-curve velocities",
    )


def plot_acceleration_comparison(motion: pd.DataFrame, output_path: Path) -> None:
    plot_kinematic_comparison(
        motion=motion,
        output_path=output_path,
        columns=(
            ("ax_raw_mm_s2", "ax_mm_s2", "x acceleration (mm/s²)"),
            ("ay_raw_mm_s2", "ay_mm_s2", "y acceleration (mm/s²)"),
            ("az_raw_mm_s2", "az_mm_s2", "z acceleration (mm/s²)"),
            (
                "acceleration_raw_mm_s2",
                "acceleration_mm_s2",
                "acceleration magnitude (mm/s²)",
            ),
        ),
        title="Frame-difference and smooth-curve accelerations",
    )


def plot_kinematic_comparison(
    motion: pd.DataFrame,
    output_path: Path,
    columns: tuple[tuple[str, str, str], ...],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 11), sharex=True)
    for particle_index, (particle, group) in enumerate(
        motion.groupby("particle", sort=True)
    ):
        color = f"C{particle_index % 10}"
        for axis_index, (raw_column, smooth_column, ylabel) in enumerate(columns):
            ax = axes[axis_index]
            ax.plot(
                group["time_s"],
                group[raw_column],
                color=color,
                linewidth=0.8,
                alpha=0.35,
                label=f"particle {particle} frame difference",
            )
            ax.plot(
                group["time_s"],
                group[smooth_column],
                color=color,
                linewidth=1.6,
                label=f"particle {particle} smooth curve",
            )
            ax.set_ylabel(ylabel)

    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
