"""Measure circular-motion episodes and frequencies from one 3-D track.

The input is a ``tracks_3d.csv`` produced by the video analyzer. Raw positions
are preserved. Short, single-frame gaps may be interpolated only for filtering
and spectral analysis; interpolated points are never added to the exported
measurement rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage, signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="particle")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--temperature-csv", type=Path)
    parser.add_argument("--band-low-hz", type=float, default=1.0)
    parser.add_argument("--band-high-hz", type=float, default=4.0)
    parser.add_argument("--orbit-threshold-mm", type=float, default=0.10)
    parser.add_argument("--minimum-episode-s", type=float, default=2.0)
    parser.add_argument("--merge-gap-s", type=float, default=0.75)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not 0 < args.band_low_hz < args.band_high_hz < args.fps / 2:
        raise SystemExit("Mode band must satisfy 0 < low < high < Nyquist")
    if args.orbit_threshold_mm <= 0:
        raise SystemExit("--orbit-threshold-mm must be positive")


def contiguous_intervals(
    mask: np.ndarray,
    fps: float,
    minimum_s: float,
    merge_gap_s: float,
) -> list[tuple[int, int]]:
    closing = max(1, int(round(merge_gap_s * fps)))
    closed = ndimage.binary_closing(mask, structure=np.ones(closing, dtype=bool))
    labels, count = ndimage.label(closed)
    minimum = max(1, int(round(minimum_s * fps)))
    intervals: list[tuple[int, int]] = []
    for label in range(1, count + 1):
        indexes = np.flatnonzero(labels == label)
        if len(indexes) >= minimum:
            intervals.append((int(indexes[0]), int(indexes[-1])))
    return intervals


def interval_metrics(
    episode: int,
    start: int,
    end: int,
    time_s: np.ndarray,
    positions: np.ndarray,
    oscillatory: np.ndarray,
    orbit_rms: np.ndarray,
    temperature_time: np.ndarray | None,
    temperature_c: np.ndarray | None,
) -> dict[str, float | int | str]:
    indexes = slice(start, end + 1)
    xy = oscillatory[indexes, :2]
    covariance = np.cov(xy, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    projected = xy @ vectors
    if np.linalg.det(vectors) < 0:
        projected[:, 1] *= -1

    safe_values = np.maximum(values, np.finfo(float).eps)
    normalized_angle = np.unwrap(
        np.arctan2(
            projected[:, 1] / np.sqrt(safe_values[1]),
            projected[:, 0] / np.sqrt(safe_values[0]),
        )
    )
    local_time = time_s[indexes]
    slope, intercept = np.polyfit(local_time, normalized_angle, 1)
    frequency_hz = abs(float(slope)) / (2 * np.pi)
    period_s = 1.0 / frequency_hz if frequency_hz > 0 else math.nan
    fitted_phase = slope * local_time + intercept
    phase_residual_std = float(np.std(normalized_angle - fitted_phase))
    increments = np.diff(normalized_angle)
    winding_sign = 1.0 if slope >= 0 else -1.0
    direction_consistency = float(np.mean(np.sign(increments) == winding_sign))
    circularity = float(np.sqrt(safe_values[1] / safe_values[0]))
    cycles = abs(float(normalized_angle[-1] - normalized_angle[0])) / (2 * np.pi)

    result: dict[str, float | int | str] = {
        "episode": episode,
        "start_s": float(local_time[0]),
        "end_s": float(local_time[-1]),
        "duration_s": float(local_time[-1] - local_time[0]),
        "frequency_hz": frequency_hz,
        "period_s": period_s,
        "cycles": cycles,
        "winding_xy": "clockwise" if slope < 0 else "counterclockwise",
        "direction_consistency": direction_consistency,
        "xy_circularity": circularity,
        "orbit_rms_median_mm": float(np.median(orbit_rms[indexes])),
        "orbit_rms_peak_mm": float(np.max(orbit_rms[indexes])),
        "center_x_mm": float(np.mean(positions[indexes, 0])),
        "center_y_mm": float(np.mean(positions[indexes, 1])),
        "center_z_mm": float(np.mean(positions[indexes, 2])),
        "phase_fit_residual_rad": phase_residual_std,
    }
    if temperature_time is not None and temperature_c is not None:
        temp = np.interp(local_time, temperature_time, temperature_c)
        temp_slope = float(np.polyfit(local_time, temp, 1)[0])
        result.update(
            {
                "temperature_start_c": float(temp[0]),
                "temperature_end_c": float(temp[-1]),
                "temperature_mean_c": float(np.mean(temp)),
                "temperature_min_c": float(np.min(temp)),
                "temperature_max_c": float(np.max(temp)),
                "temperature_slope_c_per_s": temp_slope,
            }
        )
    return result


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measured = pd.read_csv(args.tracks).sort_values("frame").reset_index(drop=True)
    required = {"frame", "x_mm", "y_mm", "z_mm"}
    missing = required - set(measured.columns)
    if missing:
        raise SystemExit(f"Tracks CSV is missing columns: {sorted(missing)}")

    smooth_columns = ["x_smooth_mm", "y_smooth_mm", "z_smooth_mm"]
    if not set(smooth_columns).issubset(measured.columns):
        smooth_columns = ["x_mm", "y_mm", "z_mm"]

    first_frame = int(measured["frame"].min())
    last_frame = int(measured["frame"].max())
    frames = np.arange(first_frame, last_frame + 1)
    analysis = measured.set_index("frame").reindex(frames)
    for column in smooth_columns:
        analysis[column] = analysis[column].interpolate(limit=2, limit_direction="both")
    if analysis[smooth_columns].isna().any().any():
        raise SystemExit("Track contains gaps longer than the two-frame analysis limit")

    time_s = frames.astype(float) / args.fps
    positions = analysis[smooth_columns].to_numpy(dtype=float)
    positions = signal.savgol_filter(positions, 13, 2, axis=0)
    sos = signal.butter(
        4,
        (args.band_low_hz, args.band_high_hz),
        btype="bandpass",
        fs=args.fps,
        output="sos",
    )
    oscillatory = signal.sosfiltfilt(sos, positions, axis=0)
    rms_samples = max(3, int(round(0.5 * args.fps)))
    orbit_power = ndimage.uniform_filter1d(
        oscillatory[:, 0] ** 2 + oscillatory[:, 1] ** 2,
        size=rms_samples,
        mode="nearest",
    )
    orbit_rms = np.sqrt(np.maximum(orbit_power, 0.0))

    intervals = contiguous_intervals(
        orbit_rms >= args.orbit_threshold_mm,
        fps=args.fps,
        minimum_s=args.minimum_episode_s,
        merge_gap_s=args.merge_gap_s,
    )
    sensitivity: dict[str, list[tuple[int, int]]] = {}
    for threshold in (0.08, args.orbit_threshold_mm, 0.15):
        sensitivity[f"{threshold:.2f}"] = contiguous_intervals(
            orbit_rms >= threshold,
            fps=args.fps,
            minimum_s=args.minimum_episode_s,
            merge_gap_s=args.merge_gap_s,
        )

    temperature_time = None
    temperature_c = None
    if args.temperature_csv:
        temperature = pd.read_csv(args.temperature_csv)
        temperature_time = temperature["time_s"].to_numpy(dtype=float)
        temperature_c = temperature["bottom_temperature_c"].to_numpy(dtype=float)

    episode_rows = [
        interval_metrics(
            episode=index,
            start=start,
            end=end,
            time_s=time_s,
            positions=positions,
            oscillatory=oscillatory,
            orbit_rms=orbit_rms,
            temperature_time=temperature_time,
            temperature_c=temperature_c,
        )
        for index, (start, end) in enumerate(intervals, start=1)
    ]
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(args.output_dir / f"{args.label}_circular_intervals.csv", index=False)

    in_episode = np.zeros(len(analysis), dtype=bool)
    episode_number = np.zeros(len(analysis), dtype=int)
    for index, (start, end) in enumerate(intervals, start=1):
        in_episode[start : end + 1] = True
        episode_number[start : end + 1] = index

    export_columns = [
        column
        for column in (
            "particle",
            "frame",
            "time_s",
            "x_mm",
            "y_mm",
            "z_mm",
            "x_smooth_mm",
            "y_smooth_mm",
            "z_smooth_mm",
            "z_error_mm",
            "front_confidence",
            "side_confidence",
        )
        if column in measured.columns
    ]
    positions_export = measured[export_columns].copy()
    if "time_s" not in positions_export:
        positions_export.insert(
            positions_export.columns.get_loc("frame") + 1,
            "time_s",
            positions_export["frame"].to_numpy(dtype=float) / args.fps,
        )
    frame_indexes = positions_export["frame"].to_numpy(dtype=int) - first_frame
    positions_export["orbit_band_x_mm"] = oscillatory[frame_indexes, 0]
    positions_export["orbit_band_y_mm"] = oscillatory[frame_indexes, 1]
    positions_export["orbit_band_z_mm"] = oscillatory[frame_indexes, 2]
    positions_export["orbit_rms_mm"] = orbit_rms[frame_indexes]
    positions_export["circular_episode"] = episode_number[frame_indexes]
    positions_export["in_circular_episode"] = in_episode[frame_indexes]
    positions_export["frame_gap_before"] = positions_export["frame"].diff().fillna(1).astype(int)
    if temperature_time is not None and temperature_c is not None:
        positions_export["bottom_temperature_c"] = np.interp(
            positions_export["time_s"].to_numpy(dtype=float),
            temperature_time,
            temperature_c,
        )
    positions_export.to_csv(args.output_dir / f"{args.label}_position_time.csv", index=False)

    frequencies = episodes["frequency_hz"].to_numpy(dtype=float)
    durations = episodes["duration_s"].to_numpy(dtype=float)
    weighted_frequency = float(np.average(frequencies, weights=durations))
    weighted_period = 1.0 / weighted_frequency
    centers = (episodes["start_s"] + episodes["end_s"]) / 2
    recurrence = np.diff(centers.to_numpy(dtype=float))

    summary = {
        "label": args.label,
        "source_tracks": str(args.tracks.resolve()),
        "fps": args.fps,
        "measured_rows": int(len(measured)),
        "measured_start_s": float(measured["frame"].min() / args.fps),
        "measured_end_s": float(measured["frame"].max() / args.fps),
        "analysis_interpolates_only_gaps_up_to_frames": 2,
        "mode_band_hz": [args.band_low_hz, args.band_high_hz],
        "primary_orbit_threshold_mm": args.orbit_threshold_mm,
        "episode_count": int(len(episodes)),
        "duration_weighted_orbit_frequency_hz": weighted_frequency,
        "duration_weighted_orbit_period_s": weighted_period,
        "episode_frequency_mean_hz": float(np.mean(frequencies)),
        "episode_frequency_std_hz": float(np.std(frequencies, ddof=1)),
        "episode_period_mean_s": float(np.mean(episodes["period_s"])),
        "recurrence_period_mean_s": float(np.mean(recurrence)),
        "recurrence_period_std_s": float(np.std(recurrence, ddof=1)),
        "recurrence_frequency_hz": float(1.0 / np.mean(recurrence)),
        "boundary_sensitivity": {
            threshold: [
                {
                    "start_s": float(time_s[start]),
                    "end_s": float(time_s[end]),
                }
                for start, end in threshold_intervals
            ]
            for threshold, threshold_intervals in sensitivity.items()
        },
        "notes": [
            "Circular episodes are defined from 1-4 Hz horizontal band-pass RMS.",
            "Frequency is the fitted winding rate after ellipse normalization in the reconstructed x-y plane.",
            "Episode boundaries depend on the reported amplitude threshold and are not exact physical transition times.",
        ],
    }
    (args.output_dir / f"{args.label}_mode_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for column, label in zip(range(3), ("x", "y", "z")):
        axes[0].plot(time_s, positions[:, column], lw=0.55, label=label)
    axes[0].set_ylabel("position [mm]")
    axes[0].legend(ncol=3)
    axes[1].plot(time_s, orbit_rms, color="tab:purple", lw=0.8)
    axes[1].axhline(args.orbit_threshold_mm, color="black", ls="--", lw=0.8)
    axes[1].set_ylabel("1–4 Hz XY RMS [mm]")
    if temperature_time is not None and temperature_c is not None:
        axes[2].plot(temperature_time, temperature_c, color="tab:red", lw=1.0)
        axes[2].set_ylabel("bottom temperature [°C]")
    else:
        axes[2].axis("off")
    for axis in axes:
        for row in episode_rows:
            axis.axvspan(row["start_s"], row["end_s"], color="tab:green", alpha=0.12)
    axes[-1].set_xlabel("video time [s]")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.label}_timeline.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(intervals))))
    for color, row, (start, end) in zip(colors, episode_rows, intervals):
        axis.plot(
            positions[start : end + 1, 0],
            positions[start : end + 1, 1],
            lw=0.65,
            color=color,
            label=f"{row['episode']}: {row['start_s']:.1f}–{row['end_s']:.1f} s",
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_title(f"{args.label} circular-motion episodes")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.label}_xy_circular_episodes.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
