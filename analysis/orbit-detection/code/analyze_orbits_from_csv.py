#!/usr/bin/env python3
"""Detect circular-motion episodes in a position-versus-time CSV.

The input file is never modified. The script writes a derived time series,
one row per detected orbit, a JSON summary, diagnostic figures, and a compact
HTML report to a separate output directory.  A band-pass RMS envelope finds
candidate motion; an explicit rolling moving-center fit then validates geometry
and estimates local center, drift, radius, frequency, and residual error.

At minimum the input needs X and Y position plus either time in seconds or a
frame number with an explicitly supplied frame rate. Common column names are
detected automatically; use the column arguments for other schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage, signal

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from orbit_statistics import (  # noqa: E402
    analyze_orbit_statistics,
    summary_stats,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


XY_COLUMN_PAIRS = (
    ("x_mm", "y_mm"),
    ("x_smooth_mm", "y_smooth_mm"),
    ("front_image_x_raw_px", "side_image_x_raw_px"),
    ("front_image_x_px", "side_image_x_px"),
    ("front_image_x_direct_smooth_px", "side_image_x_direct_smooth_px"),
    ("front_image_x_smooth_px", "side_image_x_smooth_px"),
    ("x_smooth_px", "y_smooth_px"),
    ("x_px", "y_px"),
    ("x_smooth", "y_smooth"),
    ("x", "y"),
)
TIME_COLUMNS = ("time_s", "timestamp_s", "time", "timestamp")
FRAME_COLUMNS = ("frame", "frame_number", "frame_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Position-versus-time CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for derived outputs (default: <CSV stem>_orbit_results)",
    )
    parser.add_argument("--label", help="Dataset label used in the report")
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional matching source video for the synchronized HTML report",
    )
    parser.add_argument(
        "--particle",
        help="Optional particle identifier when the CSV contains multiple tracks",
    )
    parser.add_argument("--time-col", help="Time column; values must be seconds")
    parser.add_argument("--frame-col", help="Frame-number column")
    parser.add_argument(
        "--fps",
        type=float,
        help="Required when the input has frames but no time column",
    )
    parser.add_argument("--x-col", help="Horizontal X-position column")
    parser.add_argument("--y-col", help="Horizontal Y-position column")
    parser.add_argument("--z-col", help="Optional vertical Z-position column")
    parser.add_argument(
        "--position-unit",
        help="Position unit, e.g. mm or px; inferred only from recognized suffixes",
    )
    parser.add_argument("--band-low-hz", type=float, default=1.0)
    parser.add_argument("--band-high-hz", type=float, default=4.0)
    parser.add_argument(
        "--threshold",
        default="auto",
        help="Oscillation-amplitude threshold in position units, or 'auto'",
    )
    parser.add_argument("--minimum-episode-s", type=float, default=2.0)
    parser.add_argument("--merge-gap-s", type=float, default=0.75)
    parser.add_argument("--smooth-window-s", type=float, default=0.10)
    parser.add_argument("--radius-window-s", type=float, default=0.50)
    parser.add_argument(
        "--fit-window-s",
        type=float,
        default=2.0,
        help="Duration of each explicit moving-center geometry fit",
    )
    parser.add_argument(
        "--fit-step-s",
        type=float,
        default=0.25,
        help="Spacing between fitted window centers",
    )
    parser.add_argument("--fit-minimum-turns", type=float, default=1.5)
    parser.add_argument("--fit-minimum-axis-ratio", type=float, default=0.70)
    parser.add_argument(
        "--fit-maximum-normalized-radial-rmse", type=float, default=0.25
    )
    parser.add_argument(
        "--fit-minimum-direction-consistency", type=float, default=0.80
    )
    parser.add_argument(
        "--fit-minimum-accepted-fraction",
        type=float,
        default=0.60,
        help="Fraction of local fit windows that must pass within an RMS candidate",
    )
    parser.add_argument(
        "--rms-only",
        action="store_true",
        help="Keep RMS candidate intervals without geometric validation",
    )
    parser.add_argument("--max-gap-samples", type=int, default=2)
    parser.add_argument(
        "--no-html", action="store_true", help="Do not create orbit_report.html"
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def choose_column(columns: set[str], requested: str | None, candidates: tuple[str, ...], role: str) -> str | None:
    if requested:
        if requested not in columns:
            fail(f"{role} column {requested!r} was not found")
        return requested
    return next((name for name in candidates if name in columns), None)


def choose_position_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str, str | None]:
    columns = set(df.columns)
    if bool(args.x_col) != bool(args.y_col):
        fail("Supply both --x-col and --y-col, or neither")
    if args.x_col and args.y_col:
        if args.x_col not in columns or args.y_col not in columns:
            fail("The requested X or Y position column was not found")
        x_col, y_col = args.x_col, args.y_col
    else:
        pair = next((pair for pair in XY_COLUMN_PAIRS if set(pair) <= columns), None)
        if pair is None:
            fail("Could not identify X/Y columns; supply --x-col and --y-col")
        x_col, y_col = pair

    if args.z_col:
        if args.z_col not in columns:
            fail(f"Z column {args.z_col!r} was not found")
        z_col = args.z_col
    else:
        z_candidates = []
        if x_col.startswith("x_"):
            z_candidates.append("z_" + x_col[2:])
        if x_col.endswith("_mm") and y_col.endswith("_mm"):
            z_candidates.extend(("z_smooth_mm", "z_mm"))
        elif x_col.endswith("_px") and y_col.endswith("_px"):
            z_candidates.extend(("z_smooth_px", "z_px"))
        else:
            z_candidates.extend(("z_smooth", "z"))
        z_col = next((name for name in z_candidates if name in columns), None)
    return x_col, y_col, z_col


def infer_position_unit(columns: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    suffixes = (("_mm", "mm"), ("_cm", "cm"), ("_um", "um"), ("_px", "px"), ("_m", "m"))
    inferred = []
    for column in columns:
        unit = next((unit for suffix, unit in suffixes if column.endswith(suffix)), None)
        if unit is None:
            fail("Position unit is not encoded in the selected columns; supply --position-unit")
        inferred.append(unit)
    if len(set(inferred)) != 1:
        fail(f"Selected position columns imply inconsistent units: {inferred}")
    return inferred[0]


def numeric_column(df: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        fail(f"Column {name!r} has no finite numeric values")
    return values


def load_measurements(
    csv_path: Path,
) -> tuple[pd.DataFrame, str, list[dict[str, str]]]:
    """Load the requested CSV, using paired camera pixels only as a fallback."""
    measured = pd.read_csv(csv_path)
    sources = [{"path": str(csv_path.resolve()), "sha256": file_sha256(csv_path)}]
    if csv_path.name != "tracks_3d.csv":
        return measured, "input_csv", sources

    physical_pairs = (("x_mm", "y_mm"), ("x_smooth_mm", "y_smooth_mm"))
    if any(set(pair) <= set(measured.columns) for pair in physical_pairs):
        return measured, "input_csv", sources

    front_path = csv_path.with_name("front_background_tracks.csv")
    side_path = csv_path.with_name("side_background_tracks.csv")
    if not front_path.is_file() or not side_path.is_file():
        return measured, "input_csv", sources

    front = pd.read_csv(front_path)
    side = pd.read_csv(side_path)
    keys = [name for name in ("seed_id", "particle", "frame") if name in front and name in side]
    if "frame" not in keys:
        return measured, "input_csv", sources
    front_required = ("front_image_x_px", "front_image_y_px")
    side_required = ("side_image_x_px", "side_image_y_px")
    if not set(front_required) <= set(front) or not set(side_required) <= set(side):
        return measured, "input_csv", sources

    front_columns = keys + [
        name for name in (
            "time_s", *front_required, "front_confidence", "front_peak_difference"
        ) if name in front and name not in keys
    ]
    side_columns = keys + [
        name for name in (
            "time_s", *side_required, "side_confidence", "side_peak_difference"
        ) if name in side and name not in keys
    ]
    paired = front.loc[:, front_columns].merge(
        side.loc[:, side_columns], on=keys, how="inner", suffixes=("_front", "_side")
    )
    if "time_s_front" in paired and "time_s_side" in paired:
        front_time = pd.to_numeric(paired.pop("time_s_front"), errors="coerce")
        side_time = pd.to_numeric(paired.pop("time_s_side"), errors="coerce")
        if not np.allclose(front_time, side_time, rtol=0, atol=1e-9, equal_nan=False):
            fail("Front and side camera track timestamps do not agree")
        paired.insert(paired.columns.get_loc("frame") + 1, "time_s", front_time)
    elif "time_s" not in paired:
        time_source = front if "time_s" in front else side if "time_s" in side else None
        if time_source is not None:
            time_lookup = time_source.set_index(keys)["time_s"]
            paired["time_s"] = [time_lookup.loc[tuple(row)] for row in paired[keys].to_numpy()]

    for camera in ("front", "side"):
        for axis in ("x", "y"):
            source = f"{camera}_image_{axis}_px"
            paired[f"{camera}_image_{axis}_raw_px"] = paired[source].to_numpy(dtype=float)
    paired = paired.sort_values(keys).reset_index(drop=True)
    sources.extend(
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in (front_path, side_path)
    )
    return paired, "paired_camera_tracks", sources


def missing_run_lengths(mask: np.ndarray) -> list[int]:
    labels, count = ndimage.label(mask.astype(bool))
    return [int(np.sum(labels == index)) for index in range(1, count + 1)]


def make_uniform_grid(
    df: pd.DataFrame,
    time_s: np.ndarray,
    coordinate_columns: list[str],
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    order = np.argsort(time_s)
    sorted_time = time_s[order]
    if not np.isfinite(sorted_time).all():
        fail("Time/frame data contains missing or non-finite values")
    if np.any(np.diff(sorted_time) <= 0):
        fail("Time values must be unique and strictly increasing")

    dt_candidates = np.diff(sorted_time)
    dt = float(np.median(dt_candidates))
    if dt <= 0:
        fail("Could not determine a positive sample interval")
    sample_rate_hz = 1.0 / dt
    indexes = np.rint((sorted_time - sorted_time[0]) / dt).astype(int)
    timing_error = np.abs(sorted_time - (sorted_time[0] + indexes * dt))
    if float(np.max(timing_error)) > 0.20 * dt:
        fail(
            "Timestamps are not close to a uniform sample grid. Resample the data "
            "or provide frame numbers plus --fps."
        )
    if len(np.unique(indexes)) != len(indexes):
        fail("Multiple rows map to the same sample; check timestamps or frame numbers")

    count = int(indexes[-1]) + 1
    grid_time = sorted_time[0] + np.arange(count, dtype=float) * dt
    coordinates = np.full((count, len(coordinate_columns)), np.nan, dtype=float)
    source_rows = np.full(count, -1, dtype=int)
    source_rows[indexes] = order
    for column_index, column in enumerate(coordinate_columns):
        coordinates[indexes, column_index] = numeric_column(df, column)[order]

    missing = ~np.isfinite(coordinates)
    longest_gap = 0
    for column_index, column in enumerate(coordinate_columns):
        runs = missing_run_lengths(missing[:, column_index])
        longest_gap = max([longest_gap, *runs])
        if missing[0, column_index] or missing[-1, column_index]:
            fail(f"Column {column!r} is missing a value at the start or end")
        if runs and max(runs) > max_gap_samples:
            fail(
                f"Column {column!r} has a gap of {max(runs)} samples; "
                f"the configured limit is {max_gap_samples}"
            )
        if runs:
            valid = np.isfinite(coordinates[:, column_index])
            coordinates[:, column_index] = np.interp(
                np.arange(count), np.flatnonzero(valid), coordinates[valid, column_index]
            )

    audit = {
        "input_rows": int(len(df)),
        "analysis_samples": count,
        "inserted_analysis_samples": int(np.sum(source_rows < 0)),
        "longest_interpolated_gap_samples": longest_gap,
        "maximum_timestamp_grid_error_s": float(np.max(timing_error)),
    }
    return grid_time, coordinates, source_rows, indexes, sample_rate_hz, audit


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_DIR.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def prepare_browser_video(source: Path | None, output_dir: Path) -> Path | None:
    """Return a browser-friendly video without modifying or re-encoding the source."""
    if source is None:
        return None
    source = source.expanduser().resolve()
    if not source.is_file():
        fail(f"Companion video does not exist: {source}")
    if source.suffix.lower() in {".mp4", ".m4v"}:
        return source

    destination = output_dir / "orbit_source_video.mp4"
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        command = [
            get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True)
    except (ImportError, OSError, subprocess.CalledProcessError) as exc:
        fail(f"Could not create browser-compatible MP4 from {source}: {exc}")
    return destination


def safe_unit(unit: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", unit.strip()).strip("_")
    return cleaned or "unit"


def make_plots(
    output_dir: Path,
    label: str,
    time_s: np.ndarray,
    positions: np.ndarray,
    oscillatory: np.ndarray,
    radius_envelope: np.ndarray,
    threshold: float,
    candidate_intervals: list[tuple[int, int]],
    intervals: list[tuple[int, int]],
    sample_rate_hz: float,
    band: tuple[float, float],
    unit: str,
    coordinate_names: tuple[str, ...] | None = None,
    path_title: str | None = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    names = coordinate_names or ("x", "y", "z")[: positions.shape[1]]
    for axis_index, name in enumerate(names):
        axes[0].plot(time_s, positions[:, axis_index], lw=0.65, label=name)
    axes[0].set_ylabel(f"position [{unit}]")
    axes[0].legend(ncol=3)
    axes[0].set_title(f"{label}: RMS candidates and geometrically validated motion")
    axes[1].plot(time_s, radius_envelope, color="#4b62a1", lw=0.8)
    axes[1].axhline(threshold, color="black", ls="--", lw=0.9, label="detection threshold")
    axes[1].set_ylabel(f"RMS candidate amplitude [{unit}]")
    axes[1].legend()
    frequencies, psd_x = signal.welch(oscillatory[:, 0], fs=sample_rate_hz, nperseg=min(len(time_s), 4096))
    _, psd_y = signal.welch(oscillatory[:, 1], fs=sample_rate_hz, nperseg=min(len(time_s), 4096))
    maximum_frequency = min(sample_rate_hz / 2, band[1] * 3)
    keep = frequencies <= maximum_frequency
    axes[2].semilogy(frequencies[keep], (psd_x + psd_y)[keep], color="#7d4ca1", lw=0.9)
    axes[2].axvspan(band[0], band[1], color="#4b62a1", alpha=0.12)
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylabel(f"coordinate power [{unit}²/Hz]")
    axes[2].grid(alpha=0.2)
    for axis in axes[:2]:
        for start, end in candidate_intervals:
            axis.axvspan(time_s[start], time_s[end], color="#d7a534", alpha=0.08)
        for start, end in intervals:
            axis.axvspan(time_s[start], time_s[end], color="#36a269", alpha=0.14)
    fig.savefig(output_dir / "orbit_diagnostic.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    axis.plot(positions[:, 0], positions[:, 1], color="#b8c0c8", lw=0.6,
              alpha=0.65, label="full tracked path")
    if intervals:
        colors = plt.cm.viridis(np.linspace(0, 1, len(intervals)))
        for episode_index, (color, (start, end)) in enumerate(
            zip(colors, intervals), start=1
        ):
            episode_positions = positions[start : end + 1]
            axis.plot(
                episode_positions[:, 0],
                episode_positions[:, 1],
                color=color,
                lw=1.2,
                alpha=0.9,
                label="detected orbit" if episode_index == 1 else None,
            )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"{names[0]} [{unit}]")
    axis.set_ylabel(f"{names[1]} [{unit}]")
    axis.set_title(path_title or f"{label}: XY trajectory and detected intervals")
    axis.margins(x=0.25, y=0.25)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.savefig(output_dir / "orbit_xy.png", dpi=170)
    plt.close(fig)


def make_html_report(
    output_dir: Path,
    label: str,
    summary: dict[str, Any],
    episodes: pd.DataFrame,
    derived: pd.DataFrame,
    unit: str,
    source_video: Path | None = None,
) -> None:
    amplitude_column = f"radius_median_{safe_unit(unit)}"
    circle_radius_column = f"circle_fit_radius_{safe_unit(unit)}"
    rows = []
    for row in episodes.to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{int(row['episode'])}</td><td>{row['start_s']:.3f}</td><td>{row['end_s']:.3f}</td>"
            f"<td>{row['period_s']:.4f}</td><td>{row['frequency_hz']:.4f}</td>"
            f"<td>{row[amplitude_column]:.4f}</td><td>{row[circle_radius_column]:.4f}</td>"
            f"<td>{row['circle_fit_normalized_radial_rmse']:.3f}</td>"
            f"<td>{row['circle_fit_axis_ratio']:.3f}</td>"
            f"<td>{html.escape(str(row['direction_xy']))}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="10">No orbit episodes met the configured criteria.</td></tr>')
    frequency = summary["frequency_hz"]
    period = summary["period_s"]
    radius = summary[f"circle_fit_radius_{safe_unit(unit)}"]
    amplitude = summary[f"radius_{safe_unit(unit)}"]
    frequency_mean = "n.a." if frequency["mean"] is None else f"{frequency['mean']:.4f}"
    frequency_sd = "n.a." if frequency["standard_deviation"] is None else f"{frequency['standard_deviation']:.4f}"
    period_mean = "n.a." if period["mean"] is None else f"{period['mean']:.4f}"
    period_sd = "n.a." if period["standard_deviation"] is None else f"{period['standard_deviation']:.4f}"
    radius_mean = "n.a." if radius["mean"] is None else f"{radius['mean']:.4f}"
    radius_sd = "n.a." if radius["standard_deviation"] is None else f"{radius['standard_deviation']:.4f}"
    amplitude_mean = "n.a." if amplitude["mean"] is None else f"{amplitude['mean']:.4f}"
    unit_key = safe_unit(unit)
    x_column = f"position_x_smoothed_{unit_key}"
    y_column = f"position_y_smoothed_{unit_key}"
    z_column = f"position_z_smoothed_{unit_key}"
    x_raw_column = f"position_x_{unit_key}"
    y_raw_column = f"position_y_{unit_key}"
    z_raw_column = f"position_z_{unit_key}"
    radius_envelope_column = f"orbit_radius_envelope_{unit_key}"
    rolling_radius_column = f"rolling_circle_radius_{unit_key}"
    sample_stride = max(1, int(round(summary["sample_rate_hz"] / 30.0)))
    compact = derived.iloc[::sample_stride].copy()
    if len(derived) and compact.index[-1] != derived.index[-1]:
        compact = pd.concat([compact, derived.iloc[[-1]]])
    samples = []
    for row in compact.to_dict(orient="records"):
        samples.append(
            [
                round(float(row["time_s"]), 6),
                round(float(row[x_raw_column]), 7),
                round(float(row[y_raw_column]), 7),
                round(float(row[z_raw_column]), 7) if z_raw_column in row else None,
                round(float(row[x_column]), 7),
                round(float(row[y_column]), 7),
                round(float(row[z_column]), 7) if z_column in row else None,
                round(float(row[radius_envelope_column]), 7),
                int(row["orbit_episode"]),
                int(row["source_row_index"]),
                finite_or_none(row[rolling_radius_column]),
                finite_or_none(row["rolling_circle_normalized_radial_rmse"]),
                bool(row["rolling_circle_fit_accepted"]),
                str(row["rolling_circle_fit_status"]),
            ]
        )
    camera_columns = (
        "front_image_x_raw_px",
        "front_image_y_raw_px",
        "side_image_x_raw_px",
        "side_image_y_raw_px",
        "front_image_x_direct_smooth_px",
        "front_image_y_direct_smooth_px",
        "side_image_x_direct_smooth_px",
        "side_image_y_direct_smooth_px",
    )
    has_camera_pixels = (
        summary.get("trajectory_mode") == "camera_pixel_views"
        and set(camera_columns) <= set(compact.columns)
    )
    camera_samples = (
        compact.loc[:, camera_columns].round(5).to_numpy(dtype=float).tolist()
        if has_camera_pixels
        else []
    )
    episode_records = json.loads(
        episodes.to_json(orient="records") if len(episodes) else "[]"
    )
    report_data = {
        "summary": summary,
        "episodes": episode_records,
        "samples": samples,
        "sampleFields": [
            "time", "xRaw", "yRaw", "zRaw", "x", "y", "z",
            "rmsAmplitude", "episode", "sourceRow", "circleRadius",
            "normalizedRadialRmse", "fitAccepted", "fitStatus"
        ],
        "cameraFields": list(camera_columns) if has_camera_pixels else [],
        "cameraSamples": camera_samples,
    }
    embedded_data = json.dumps(report_data, separators=(",", ":")).replace("</", "<\\/")
    video_attribute = ""
    if source_video is not None:
        source_video = source_video.resolve()
        try:
            relative_video = source_video.relative_to(output_dir.resolve())
            video_attribute = f' src="{html.escape(relative_video.as_posix(), quote=True)}"'
        except ValueError:
            video_attribute = (
                f' data-media-path="{html.escape(str(source_video), quote=True)}"'
            )
    recurrence = summary.get("recurrence_s", {})
    recurrence_mean = recurrence.get("mean")
    recurrence_text = "n.a." if recurrence_mean is None else f"{recurrence_mean:.4f} s"
    path_title = "Tracked path in camera pixels" if has_camera_pixels else "Reconstructed path"
    calibration_warning = (
        '<p class="warning"><strong>Pixel-space display:</strong> the front and side camera measurements are smoothed directly in pixel space. '
        "The horizontal proxy is useful for timing and shape review, but it is not a calibrated 3-D trajectory.</p>"
        if has_camera_pixels
        else ""
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(label)} orbit report</title>
<style>
:root{{--bg:#07111b;--panel:#0d1b28;--line:#294154;--text:#e8f1f7;--muted:#91a6b5;--orbit:#45e09d;--accent:#65a9ff;--quiet:#7f8fa0}}
*{{box-sizing:border-box}} body{{font:14px/1.45 system-ui,sans-serif;max-width:1440px;margin:0 auto;padding:24px;color:var(--text);background:var(--bg)}}
h1,h2,h3{{line-height:1.15;margin-top:0}} .lede,.note{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:18px 0}}
.card,.panel,section{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:15px}} .card span,.card small{{display:block;color:var(--muted)}} .card strong{{display:block;font-size:1.45rem;margin:3px 0}}
.viewer-grid{{display:grid;grid-template-columns:minmax(0,1.08fr) minmax(300px,.92fr);gap:12px;margin:12px 0}} .panel h2{{font-size:18px}}
.video-shell{{position:relative;aspect-ratio:16/9;background:#000;border-radius:6px;overflow:hidden}} video{{display:block;width:100%;height:100%;object-fit:contain}}
canvas.chart{{display:block;width:100%;background:#071421;border-radius:6px;margin-top:10px}} .toolbar,.view-controls{{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}}
button{{border:1px solid #365268;border-radius:5px;padding:7px 10px;color:var(--text);background:#12283a;cursor:pointer}} button:hover,button[aria-pressed=true]{{border-color:var(--orbit);color:var(--orbit)}}
.live-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:7px;margin-top:10px}} .live{{padding:8px;background:#0a1621;border:1px solid #203849;border-radius:5px}} .live span,.live small{{display:block;color:var(--muted);font-size:11px}} .live strong{{font:12px ui-monospace,monospace}}
.mode{{display:inline-block;border-radius:999px;padding:4px 8px;background:#263746;color:var(--muted)}} .mode.orbit{{background:var(--orbit);color:#052b1c}}
section{{margin:14px 0}} img{{max-width:100%;height:auto;border-radius:6px}} .support-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{padding:7px;border-bottom:1px solid #203849;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} tr.current{{background:#15372d}}
.table-wrap{{overflow:auto;max-height:360px}} code{{background:#142535;padding:2px 4px;border-radius:3px}} a{{color:#8fc3ff}}
.warning{{padding:10px 12px;border-left:3px solid #f0bd68;background:#1c2430;color:#d7e2ea}}
details.supporting{{margin:14px 0;background:var(--panel);border:1px solid var(--line);border-radius:8px}} details.supporting>summary{{cursor:pointer;padding:13px 15px;font-weight:700}} details.supporting>div{{padding:0 15px 15px}}
@media(max-width:680px){{.viewer-grid,.support-grid{{grid-template-columns:1fr}} body{{padding:12px}}}}
</style></head><body>
<h1>{html.escape(label)} orbit report</h1>
<p class="lede">Play the unobstructed recording while the geometrically validated mode, tracked path, RMS candidate amplitude, moving-center circle fit, and nearest analysis row update together.</p>
{calibration_warning}
<div class="viewer-grid">
  <div class="panel"><h2>Original video</h2><div class="video-shell"><video id="video" controls preload="metadata"{video_attribute}></video></div><div class="toolbar"><button id="previousEpisode">Previous orbit</button><button id="nextEpisode">Next orbit</button><span id="mode" class="mode">waiting for video</span></div><canvas id="timeline" class="chart" height="92"></canvas></div>
  <div class="panel"><h2>{path_title}</h2><div class="view-controls" id="pathViews"></div><div class="view-controls" id="pathScales"><span class="note">Scale:</span><button data-scale="local" aria-pressed="true">Zoomed-out local view</button><button data-scale="overview" aria-pressed="false">Whole run</button></div><p id="pathWindow" class="note"></p><canvas id="path" class="chart" height="300"></canvas><div class="live-grid"><div class="live"><span>Video / data time</span><strong id="timeValue">—</strong></div><div class="live"><span>Nearest source row</span><strong id="rowValue">—</strong></div><div class="live"><span id="rawPositionLabel">Raw XYZ</span><strong id="rawPositionValue">—</strong></div><div class="live"><span id="smoothPositionLabel">Smoothed XYZ</span><strong id="positionValue">—</strong></div><div class="live"><span>RMS candidate amplitude</span><strong id="radiusValue">—</strong></div><div class="live"><span>Moving-center circle radius</span><strong id="circleRadiusValue">—</strong></div><div class="live"><span>Normalized radial error</span><strong id="radialErrorValue">—</strong></div><div class="live"><span>Geometry fit</span><strong id="fitStatusValue">—</strong></div><div class="live"><span>Episode period</span><strong id="periodValue">—</strong></div><div class="live"><span>Classification</span><strong id="classValue">—</strong></div></div></div>
</div>
<div class="cards">
<div class="card">Validated orbits<strong>{summary['episode_count']}</strong><span>{summary['candidate_episode_count']} RMS candidates</span></div>
<div class="card">Frequency<strong>{frequency_mean} Hz</strong><span>episode SD: {frequency_sd}</span></div>
<div class="card">Period<strong>{period_mean} s</strong><span>episode SD: {period_sd}</span></div>
<div class="card">Circle-fit radius<strong>{radius_mean} {html.escape(unit)}</strong><span>episode SD: {radius_sd}</span></div>
<div class="card">RMS amplitude<strong>{amplitude_mean} {html.escape(unit)}</strong><span>candidate signal, not geometric radius</span></div>
<div class="card">Recurrence<strong>{recurrence_text}</strong><span>episode-center spacing</span></div>
<div class="card">Orbit duty cycle<strong>{100 * summary['orbit_duty_cycle']:.1f}%</strong><span>threshold classification</span></div>
</div>
<section><h2>RMS candidate amplitude and validated intervals</h2><canvas id="signal" class="chart" height="270"></canvas></section>
<section><h2>Validated orbit episodes</h2><div style="overflow:auto"><table><thead><tr><th>Episode</th><th>Start [s]</th><th>End [s]</th><th>Period [s]</th><th>Frequency [Hz]</th><th>RMS amplitude [{html.escape(unit)}]</th><th>Circle radius [{html.escape(unit)}]</th><th>Radial error / radius</th><th>Axis ratio</th><th>Direction</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>Nearest analysis rows</h2><div class="table-wrap"><table><thead><tr><th>Time [s]</th><th>Selected raw coordinates [{html.escape(unit)}]</th><th>Selected smoothed coordinates [{html.escape(unit)}]</th><th>RMS amplitude [{html.escape(unit)}]</th><th>Episode</th></tr></thead><tbody id="nearbyRows"></tbody></table></div></section>
<details class="supporting"><summary>Supporting diagnostic images</summary><div class="support-grid"><img src="orbit_diagnostic.png" alt="Position, orbit amplitude, and frequency diagnostic"><img src="orbit_xy.png" alt="Detected XY orbit paths"></div></details>
<section><h2>Files and method</h2><p><a href="orbit_intervals.csv">Orbit intervals CSV</a> · <a href="orbit_timeseries.csv">Derived time series CSV</a> · <a href="orbit_summary.json">JSON summary</a></p>
<p class="note">The {summary['parameters']['band_low_hz']}–{summary['parameters']['band_high_hz']} Hz XY band-pass RMS envelope finds candidate motion only. Accepted windows must also pass the explicit moving-center harmonic fit, radial-error, axis-ratio, angular-coverage, and winding checks. Boundaries remain detector-dependent; inspect the settings and threshold sensitivity in the JSON summary.</p></section>
<script id="report-data" type="application/json">{embedded_data}</script>
<script>
(() => {{
  const D=JSON.parse(document.getElementById('report-data').textContent),S=D.samples,C=D.cameraSamples||[],E=D.episodes,summary=D.summary,unit={json.dumps(unit)},cameraPixels=C.length===S.length&&C.length>0;
  const video=document.getElementById('video'), timeline=document.getElementById('timeline'), path=document.getElementById('path'), signal=document.getElementById('signal'); let view=cameraPixels?'horizontal':'xy',scaleMode='local';
  document.getElementById('pathViews').innerHTML=(cameraPixels?[['horizontal','Horizontal proxy'],['front','Front camera'],['side','Side camera']]:[['xy','XY'],['xz','XZ'],['yz','YZ']]).map((item,index)=>`<button data-view="${{item[0]}}" aria-pressed="${{index===0}}">${{item[1]}}</button>`).join('');
  const fmt=(v,n=3)=>Number.isFinite(Number(v))?Number(v).toFixed(n):'—';
  function bounds(rows,axes){{let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const r of rows){{const x=r[axes[0]],y=r[axes[1]];if(!Number.isFinite(x)||!Number.isFinite(y))continue;minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y)}}return{{minX,maxX,minY,maxY}}}}
  function nearest(t){{let lo=0,hi=S.length-1;while(lo<hi){{const m=(lo+hi)>>1;if(S[m][0]<t)lo=m+1;else hi=m}}if(lo&&Math.abs(S[lo-1][0]-t)<Math.abs(S[lo][0]-t))lo--;return lo}}
  function episodeAt(t){{return E.find(e=>t>=e.start_s&&t<=e.end_s)}}
  function canvas(c){{const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=Math.max(320,Math.round(r.width*d));c.height=Math.round(Number(c.getAttribute('height'))*d);const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,Number(c.getAttribute('height'))]}}
  function drawTimeline(){{const [c,w,h]=canvas(timeline),a=0,b=Number.isFinite(video.duration)?video.duration:S[S.length-1][0],x=t=>36+(t-a)/(b-a)*(w-48);c.fillStyle='#12283a';c.fillRect(36,18,w-48,28);E.forEach(e=>{{c.fillStyle='#45e09d';c.fillRect(x(e.start_s),22,Math.max(2,x(e.end_s)-x(e.start_s)),20)}});const now=video.currentTime||S[0][0];c.strokeStyle='#fff';c.beginPath();c.moveTo(x(now),10);c.lineTo(x(now),57);c.stroke();c.fillStyle='#91a6b5';c.font='11px system-ui';c.fillText('0 s',36,75);c.textAlign='right';c.fillText(fmt(b,1)+' s',w-12,75)}}
  function drawPath(i,requestedTime,hasTracking){{
    const [c,w,h]=canvas(path),m={{l:58,r:18,t:34,b:46}};c.fillStyle='#071421';c.fillRect(0,0,w,h);if(!hasTracking){{c.fillStyle='#91a6b5';c.font='14px system-ui';c.textAlign='center';c.fillText(`No tracking data at ${{fmt(requestedTime,2)}} s`,w/2,h/2-8);c.font='12px system-ui';c.fillText(`Tracked interval: ${{fmt(S[0][0],2)}}–${{fmt(S[S.length-1][0],2)}} s`,w/2,h/2+16);document.getElementById('pathWindow').textContent='The video is outside the tracked interval.';return}}
    const now=S[i][0],ep=episodeAt(now),pastSeconds=1.5,futureSeconds=.25,start=nearest(Math.max(S[0][0],now-pastSeconds)),end=nearest(Math.min(S[S.length-1][0],now+futureSeconds)),source=cameraPixels?C:S,axes=cameraPixels?(view==='horizontal'?[4,6]:view==='front'?[4,5]:[6,7]):(view==='xy'?[4,5]:view==='xz'?[4,6]:[5,6]),indexed=[];
    for(let j=start;j<=end;j++){{const r=source[j];if(Number.isFinite(r[axes[0]])&&Number.isFinite(r[axes[1]]))indexed.push([j,r])}}if(!indexed.length){{c.fillStyle='#91a6b5';c.font='14px system-ui';c.textAlign='center';c.fillText('No valid position samples in this window',w/2,h/2);document.getElementById('pathWindow').textContent=`No valid tracking samples near ${{fmt(now,2)}} s.`;return}}
    const current=source[i];if(!Number.isFinite(current[axes[0]])||!Number.isFinite(current[axes[1]])){{c.fillStyle='#91a6b5';c.textAlign='center';c.fillText('Current position is unavailable',w/2,h/2);return}}const availableWidth=w-m.l-m.r,availableHeight=h-m.t-m.b,global=bounds(source,axes),localField=cameraPixels?60:(unit==='mm'?1:Math.max(summary.threshold.value*25,1e-6)),observedX=Math.max(global.maxX-global.minX,1e-6),observedY=Math.max(global.maxY-global.minY,1e-6),xmid=scaleMode==='overview'?(global.minX+global.maxX)/2:current[axes[0]],ymid=scaleMode==='overview'?(global.minY+global.maxY)/2:current[axes[1]],scale=scaleMode==='overview'?Math.min(availableWidth/(observedX*1.15),availableHeight/(observedY*1.15)):Math.min(availableWidth/localField,availableHeight/localField),halfX=availableWidth/(2*scale),halfY=availableHeight/(2*scale),sx=v=>(m.l+w-m.r)/2+(v-xmid)*scale,sy=v=>(m.t+h-m.b)/2-(v-ymid)*scale;
    document.getElementById('pathWindow').textContent=`Local trail ${{fmt(S[start][0],2)}}–${{fmt(S[end][0],2)}} s · ${{scaleMode==='overview'?'equal-axis scale for the whole tracked run':`fixed ${{fmt(localField,2)}} ${{cameraPixels?'px':unit}} field`}} · ${{ep?'orbit '+ep.episode:'not classified as orbit'}}`;
    c.font='11px system-ui';c.lineWidth=1;c.strokeStyle='rgba(145,166,181,.18)';c.fillStyle='#91a6b5';for(let tick=0;tick<5;tick++){{const f=tick/4,x=xmid-halfX+2*halfX*f,y=ymid-halfY+2*halfY*f,px=sx(x),py=sy(y);c.beginPath();c.moveTo(px,m.t);c.lineTo(px,h-m.b);c.stroke();c.beginPath();c.moveTo(m.l,py);c.lineTo(w-m.r,py);c.stroke();c.textAlign='center';c.fillText(fmt(x,1),px,h-25);c.textAlign='right';c.fillText(fmt(y,1),m.l-7,py+4)}}
    const axisLabel=cameraPixels?(view==='horizontal'?['front horizontal','side horizontal']:view==='front'?['front horizontal','front vertical']:['side horizontal','side vertical']):(view==='xy'?['x','y']:view==='xz'?['x','z']:['y','z']);c.textAlign='center';c.fillText(`${{axisLabel[0]}} [${{cameraPixels?'px':unit}}]`,(m.l+w-m.r)/2,h-7);c.save();c.translate(13,(m.t+h-m.b)/2);c.rotate(-Math.PI/2);c.fillText(`${{axisLabel[1]}} [${{cameraPixels?'px':unit}}]`,0,0);c.restore();
    c.save();c.beginPath();c.rect(m.l,m.t,availableWidth,availableHeight);c.clip();for(let k=1;k<indexed.length;k++){{const a=indexed[k-1][1],b=indexed[k][1],fraction=k/(indexed.length-1);c.strokeStyle=`rgba(101,169,255,${{(.16+.72*fraction).toFixed(3)}})`;c.lineWidth=1.2+1.5*fraction;c.beginPath();c.moveTo(sx(a[axes[0]]),sy(a[axes[1]]));c.lineTo(sx(b[axes[0]]),sy(b[axes[1]]));c.stroke()}}const first=indexed[0][1];c.strokeStyle='#91a6b5';c.lineWidth=1.2;c.beginPath();c.arc(sx(first[axes[0]]),sy(first[axes[1]]),3.5,0,Math.PI*2);c.stroke();const px=sx(current[axes[0]]),py=sy(current[axes[1]]);c.fillStyle='rgba(69,224,157,.2)';c.beginPath();c.arc(px,py,10,0,Math.PI*2);c.fill();c.fillStyle='#45e09d';c.beginPath();c.arc(px,py,5,0,Math.PI*2);c.fill();c.strokeStyle='#e8f1f7';c.lineWidth=1;c.stroke();c.restore();
  }}
  function drawSignal(i){{const [c,w,h]=canvas(signal),m={{l:54,r:18,t:16,b:35}},a=S[0][0],b=S[S.length-1][0],max=S.reduce((value,row)=>Math.max(value,row[7]),summary.threshold.value*1.7),x=t=>m.l+(t-a)/(b-a)*(w-m.l-m.r),y=v=>m.t+(1-v/max)*(h-m.t-m.b);c.fillStyle='#071421';c.fillRect(0,0,w,h);E.forEach(e=>{{c.fillStyle='rgba(69,224,157,.12)';c.fillRect(x(e.start_s),m.t,x(e.end_s)-x(e.start_s),h-m.t-m.b)}});c.strokeStyle='#65a9ff';c.beginPath();S.forEach((r,j)=>{{j?c.lineTo(x(r[0]),y(r[7])):c.moveTo(x(r[0]),y(r[7]))}});c.stroke();c.setLineDash([5,4]);c.strokeStyle='#f0bd68';c.beginPath();c.moveTo(m.l,y(summary.threshold.value));c.lineTo(w-m.r,y(summary.threshold.value));c.stroke();c.setLineDash([]);c.strokeStyle='#fff';c.beginPath();c.moveTo(x(S[i][0]),m.t);c.lineTo(x(S[i][0]),h-m.b);c.stroke();c.fillStyle='#91a6b5';c.fillText(`RMS candidate amplitude [${{unit}}]`,8,12)}}
  function update(){{
    if(!S.length)return;
    const rawTime=Number.isFinite(video.currentTime)?video.currentTime:S[0][0],inTrackedRange=rawTime>=S[0][0]&&rawTime<=S[S.length-1][0],t=Math.max(S[0][0],Math.min(S[S.length-1][0],rawTime)),i=nearest(t),r=S[i],ep=inTrackedRange?episodeAt(t):null,camera=cameraPixels?C[i]:null;
    document.getElementById('timeValue').textContent=`${{fmt(rawTime,3)}} / ${{fmt(r[0],3)}} s`;
    document.getElementById('rowValue').textContent=r[9]>=0?String(r[9]):'interpolated grid point';
    document.getElementById('rawPositionLabel').textContent=cameraPixels?'Raw front / side pixels':'Raw XYZ';
    document.getElementById('smoothPositionLabel').textContent=cameraPixels?'Smoothed front / side pixels':'Smoothed XYZ';
    document.getElementById('rawPositionValue').textContent=!inTrackedRange?'no tracking data':cameraPixels?`F(${{fmt(camera[0],2)}}, ${{fmt(camera[1],2)}}) · S(${{fmt(camera[2],2)}}, ${{fmt(camera[3],2)}}) px`:`${{fmt(r[1],4)}}, ${{fmt(r[2],4)}}, ${{fmt(r[3],4)}} ${{unit}}`;
    document.getElementById('positionValue').textContent=!inTrackedRange?'no tracking data':cameraPixels?`F(${{fmt(camera[4],2)}}, ${{fmt(camera[5],2)}}) · S(${{fmt(camera[6],2)}}, ${{fmt(camera[7],2)}}) px`:`${{fmt(r[4],4)}}, ${{fmt(r[5],4)}}, ${{fmt(r[6],4)}} ${{unit}}`;
    document.getElementById('radiusValue').textContent=inTrackedRange?`${{fmt(r[7],5)}} ${{unit}}`:'no tracking data';
    document.getElementById('circleRadiusValue').textContent=inTrackedRange&&r[10]!=null?`${{fmt(r[10],5)}} ${{unit}}`:'not fitted';
    document.getElementById('radialErrorValue').textContent=inTrackedRange&&r[11]!=null?fmt(r[11],4):'not fitted';
    document.getElementById('fitStatusValue').textContent=inTrackedRange?r[13]:'no tracking data';
    document.getElementById('periodValue').textContent=!inTrackedRange?'no tracking data':ep?`${{fmt(ep.period_s,4)}} s (${{fmt(ep.frequency_hz,3)}} Hz)`:'not in detected orbit';
    document.getElementById('classValue').textContent=!inTrackedRange?'outside tracked interval':ep?`orbit ${{ep.episode}}`:'not orbit';
    const mode=document.getElementById('mode');mode.textContent=ep?`Circular orbit ${{ep.episode}}`:(rawTime<S[0][0]||rawTime>S[S.length-1][0]?'Outside tracked interval':'Not classified as orbit');mode.className='mode'+(ep?' orbit':'');
    const rows=[];for(let j=Math.max(0,i-3);j<=Math.min(S.length-1,i+3);j++){{const q=S[j],cam=cameraPixels?C[j]:null,raw=cameraPixels?`F(${{fmt(cam[0],2)}}, ${{fmt(cam[1],2)}}) · S(${{fmt(cam[2],2)}}, ${{fmt(cam[3],2)}}) px`:`${{fmt(q[1],5)}}, ${{fmt(q[2],5)}}, ${{fmt(q[3],5)}}`,smooth=cameraPixels?`F(${{fmt(cam[4],2)}}, ${{fmt(cam[5],2)}}) · S(${{fmt(cam[6],2)}}, ${{fmt(cam[7],2)}}) px`:`${{fmt(q[4],5)}}, ${{fmt(q[5],5)}}, ${{fmt(q[6],5)}}`;rows.push(`<tr class="${{j===i?'current':''}}" data-time="${{q[0]}}"><td>${{fmt(q[0],4)}}</td><td>${{raw}}</td><td>${{smooth}}</td><td>${{fmt(q[7],5)}}</td><td>${{q[8]||'—'}}</td></tr>`)}}
    document.getElementById('nearbyRows').innerHTML=inTrackedRange?rows.join(''):'<tr><td colspan="5">No tracking rows at this video time.</td></tr>';document.querySelectorAll('#nearbyRows tr[data-time]').forEach(tr=>tr.onclick=()=>video.currentTime=Number(tr.dataset.time));drawTimeline();drawPath(i,rawTime,inTrackedRange);drawSignal(i)
  }}
  function safeUpdate(){{try{{update()}}catch(error){{console.error(error);document.getElementById('pathWindow').textContent=`Could not render trajectory: ${{error.message}}`;const c=path.getContext('2d');c.fillStyle='#071421';c.fillRect(0,0,path.width,path.height);c.fillStyle='#f0bd68';c.font='14px system-ui';c.fillText('Trajectory rendering error',20,35)}}}}
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{{view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));safeUpdate()}});
  document.querySelectorAll('[data-scale]').forEach(b=>b.onclick=()=>{{scaleMode=b.dataset.scale;document.querySelectorAll('[data-scale]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));safeUpdate()}});
  function seekEpisode(delta){{const t=video.currentTime||S[0][0],idx=E.findIndex(e=>t<=e.end_s),next=Math.max(0,Math.min(E.length-1,(idx<0?E.length-1:idx)+delta));if(E[next])video.currentTime=E[next].start_s}}
  document.getElementById('previousEpisode').onclick=()=>seekEpisode(-1);document.getElementById('nextEpisode').onclick=()=>seekEpisode(1);timeline.onclick=e=>{{const r=timeline.getBoundingClientRect(),f=Math.max(0,Math.min(1,(e.clientX-r.left-36)/(r.width-48))),duration=Number.isFinite(video.duration)?video.duration:S[S.length-1][0];video.currentTime=f*duration}};
  video.addEventListener('loadedmetadata',()=>{{if(video.currentTime<S[0][0])video.currentTime=S[0][0];safeUpdate()}});['timeupdate','seeked','play','pause'].forEach(n=>video.addEventListener(n,safeUpdate));window.addEventListener('resize',safeUpdate);if(!video.getAttribute('src'))document.querySelector('.video-shell').insertAdjacentHTML('beforeend','<p style="position:absolute;inset:40% 10%;text-align:center;color:#91a6b5">Open this report through Video Analysis Studio to load the matching video.</p>');safeUpdate();
}})();
</script>
</body></html>"""
    (output_dir / "orbit_report.html").write_text(report, encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if not args.csv.is_file():
        fail(f"Input CSV does not exist: {args.csv}")
    if args.fps is not None and args.fps <= 0:
        fail("--fps must be positive")
    if args.max_gap_samples < 0:
        fail("--max-gap-samples cannot be negative")

    fit_window_s = float(getattr(args, "fit_window_s", 2.0))
    fit_step_s = float(getattr(args, "fit_step_s", 0.25))
    fit_minimum_turns = float(getattr(args, "fit_minimum_turns", 1.5))
    fit_minimum_axis_ratio = float(getattr(args, "fit_minimum_axis_ratio", 0.70))
    fit_maximum_normalized_radial_rmse = float(
        getattr(args, "fit_maximum_normalized_radial_rmse", 0.25)
    )
    fit_minimum_direction_consistency = float(
        getattr(args, "fit_minimum_direction_consistency", 0.80)
    )
    fit_minimum_accepted_fraction = float(
        getattr(args, "fit_minimum_accepted_fraction", 0.60)
    )
    rms_only = bool(getattr(args, "rms_only", False))

    output_dir = args.output_dir or args.csv.with_name(args.csv.stem + "_orbit_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_video = getattr(args, "video", None)
    browser_video = prepare_browser_video(source_video, output_dir)
    label = args.label or args.csv.stem
    measured, measurement_source_mode, analysis_sources = load_measurements(args.csv)
    selected_particle: Any = None
    particle_argument = getattr(args, "particle", None)
    if particle_argument is not None and str(particle_argument).strip():
        if "particle" not in measured.columns:
            fail("--particle was supplied, but the CSV has no 'particle' column")
        requested = str(particle_argument).strip()
        particle_values = measured["particle"]
        mask = particle_values.astype(str) == requested
        if not mask.any():
            try:
                requested_number = float(requested)
                numeric_particles = pd.to_numeric(particle_values, errors="coerce")
                mask = np.isclose(numeric_particles, requested_number, equal_nan=False)
            except ValueError:
                pass
        if not mask.any():
            available = sorted({str(value) for value in particle_values.dropna().unique()})
            fail(f"Particle {requested!r} was not found; available values: {available}")
        measured = measured.loc[mask].copy()
        selected_particle = finite_or_none(measured["particle"].iloc[0])
    if len(measured) < 20:
        fail("The CSV needs at least 20 rows for filtering and orbit detection")
    x_col, y_col, z_col = choose_position_columns(measured, args)
    coordinate_columns = [x_col, y_col] + ([z_col] if z_col else [])
    unit = infer_position_unit(tuple(coordinate_columns), args.position_unit)
    unit_key = safe_unit(unit)

    columns = set(measured.columns)
    time_col = choose_column(columns, args.time_col, TIME_COLUMNS, "Time")
    frame_col = choose_column(columns, args.frame_col, FRAME_COLUMNS, "Frame")
    if time_col:
        time_s = numeric_column(measured, time_col)
    elif frame_col:
        if args.fps is None:
            fail("The CSV has frame numbers but no recognized time column; supply --fps")
        time_s = numeric_column(measured, frame_col) / args.fps
    else:
        fail("Supply --time-col, or provide a frame column and --fps")

    grid_time, coordinates, source_rows, _, sample_rate_hz, audit = make_uniform_grid(
        measured, time_s, coordinate_columns, args.max_gap_samples
    )
    if args.fps is not None and time_col:
        relative_error = abs(sample_rate_hz - args.fps) / args.fps
        if relative_error > 0.01:
            fail(
                f"Time column implies {sample_rate_hz:.6g} Hz, inconsistent with --fps {args.fps:.6g}"
            )
    try:
        statistics = analyze_orbit_statistics(
            grid_time,
            coordinates,
            sample_rate_hz,
            unit_key,
            band_low_hz=args.band_low_hz,
            band_high_hz=args.band_high_hz,
            threshold_setting=args.threshold,
            minimum_episode_s=args.minimum_episode_s,
            merge_gap_s=args.merge_gap_s,
            smooth_window_s=args.smooth_window_s,
            radius_window_s=args.radius_window_s,
            fit_window_s=fit_window_s,
            fit_step_s=fit_step_s,
            fit_minimum_turns=fit_minimum_turns,
            fit_minimum_axis_ratio=fit_minimum_axis_ratio,
            fit_maximum_normalized_radial_rmse=(
                fit_maximum_normalized_radial_rmse
            ),
            fit_minimum_direction_consistency=fit_minimum_direction_consistency,
            fit_minimum_accepted_fraction=fit_minimum_accepted_fraction,
            validate_geometry=not rms_only,
        )
    except ValueError as exc:
        fail(str(exc))
    smoothed = statistics.smoothed
    oscillatory = statistics.oscillatory
    radius_envelope = statistics.radius_envelope
    smooth_samples = statistics.smooth_samples
    threshold = statistics.threshold
    threshold_method = statistics.threshold_method
    candidate_intervals = statistics.candidate_intervals
    intervals = statistics.intervals
    sensitivity = statistics.boundary_sensitivity
    episode_rows = statistics.episode_rows
    rolling_fit = statistics.rolling_fit
    if episode_rows:
        episodes = pd.DataFrame(episode_rows)
    else:
        episode_columns = [
            "episode",
            "start_s",
            "end_s",
            "duration_s",
            "cycles",
            "frequency_hz",
            "period_s",
            "direction_xy",
            "direction_consistency",
            "xy_circularity",
            f"ellipse_major_radius_{unit_key}",
            f"ellipse_minor_radius_{unit_key}",
            f"radius_median_{unit_key}",
            f"radius_robust_sigma_{unit_key}",
            "radial_coefficient_of_variation",
            f"center_x_{unit_key}",
            f"center_y_{unit_key}",
            "phase_fit_residual_rad",
            f"median_orbit_speed_{unit_key}_per_s",
            f"center_drift_speed_{unit_key}_per_s",
            f"circle_fit_radius_{unit_key}",
            f"circle_fit_major_radius_{unit_key}",
            f"circle_fit_minor_radius_{unit_key}",
            "circle_fit_axis_ratio",
            f"circle_fit_radial_rmse_{unit_key}",
            "circle_fit_normalized_radial_rmse",
            f"circle_fit_position_rmse_{unit_key}",
            "circle_fit_frequency_hz",
            "circle_fit_turns",
            "circle_fit_angular_coverage_deg",
            "circle_fit_direction_consistency",
            f"circle_fit_center_x_{unit_key}",
            f"circle_fit_center_y_{unit_key}",
            f"circle_fit_center_vx_{unit_key}_per_s",
            f"circle_fit_center_vy_{unit_key}_per_s",
            "circle_fit_acceptance_fraction",
        ]
        if coordinates.shape[1] > 2:
            episode_columns.append(f"center_z_{unit_key}")
        episodes = pd.DataFrame(columns=episode_columns)
    episodes.to_csv(output_dir / "orbit_intervals.csv", index=False)

    episode_number = np.zeros(len(grid_time), dtype=int)
    for number, (start, end) in enumerate(intervals, start=1):
        episode_number[start : end + 1] = number
    candidate_episode_number = np.zeros(len(grid_time), dtype=int)
    for number, (start, end) in enumerate(candidate_intervals, start=1):
        candidate_episode_number[start : end + 1] = number
    derived = pd.DataFrame(
        {
            "time_s": grid_time,
            "source_row_index": source_rows,
            "source_row_present": source_rows >= 0,
            f"position_x_{unit_key}": coordinates[:, 0],
            f"position_y_{unit_key}": coordinates[:, 1],
            f"position_x_smoothed_{unit_key}": smoothed[:, 0],
            f"position_y_smoothed_{unit_key}": smoothed[:, 1],
            f"orbit_band_x_{unit_key}": oscillatory[:, 0],
            f"orbit_band_y_{unit_key}": oscillatory[:, 1],
            f"orbit_radius_envelope_{unit_key}": radius_envelope,
            "rms_candidate_episode": candidate_episode_number,
            "is_rms_candidate": candidate_episode_number > 0,
            "rolling_circle_fit_available": rolling_fit.available,
            "rolling_circle_fit_accepted": rolling_fit.accepted,
            "rolling_circle_fit_status": rolling_fit.status,
            f"rolling_center_x_{unit_key}": rolling_fit.center_x,
            f"rolling_center_y_{unit_key}": rolling_fit.center_y,
            f"rolling_center_vx_{unit_key}_per_s": rolling_fit.center_vx,
            f"rolling_center_vy_{unit_key}_per_s": rolling_fit.center_vy,
            f"rolling_circle_radius_{unit_key}": rolling_fit.radius,
            f"rolling_ellipse_major_radius_{unit_key}": rolling_fit.major_radius,
            f"rolling_ellipse_minor_radius_{unit_key}": rolling_fit.minor_radius,
            "rolling_circle_axis_ratio": rolling_fit.axis_ratio,
            f"rolling_circle_radial_rmse_{unit_key}": rolling_fit.radial_rmse,
            "rolling_circle_normalized_radial_rmse": (
                rolling_fit.normalized_radial_rmse
            ),
            f"rolling_harmonic_position_rmse_{unit_key}": rolling_fit.position_rmse,
            "rolling_circle_frequency_hz": rolling_fit.frequency_hz,
            "rolling_circle_turns": rolling_fit.turns,
            "rolling_circle_angular_coverage_deg": rolling_fit.angular_coverage_deg,
            "rolling_circle_direction_consistency": (
                rolling_fit.direction_consistency
            ),
            "orbit_episode": episode_number,
            "in_orbit": episode_number > 0,
        }
    )
    if coordinates.shape[1] > 2:
        derived[f"position_z_{unit_key}"] = coordinates[:, 2]
        derived[f"position_z_smoothed_{unit_key}"] = smoothed[:, 2]
        derived[f"orbit_band_z_{unit_key}"] = oscillatory[:, 2]
    camera_pixel_columns = (
        "front_image_x_raw_px",
        "front_image_y_raw_px",
        "side_image_x_raw_px",
        "side_image_y_raw_px",
    )
    if set(camera_pixel_columns) <= set(measured.columns):
        present = source_rows >= 0
        camera_grid = []
        for column in camera_pixel_columns:
            source_values = numeric_column(measured, column)
            grid_values = np.full(len(grid_time), np.nan, dtype=float)
            grid_values[present] = source_values[source_rows[present]]
            missing_values = ~np.isfinite(grid_values)
            if np.any(missing_values):
                grid_values[missing_values] = np.interp(
                    grid_time[missing_values],
                    grid_time[~missing_values],
                    grid_values[~missing_values],
                )
            derived[column] = grid_values
            camera_grid.append(grid_values)
        direct_smoothed = signal.savgol_filter(
            np.column_stack(camera_grid), smooth_samples, 2, axis=0
        )
        for column_index, column in enumerate(camera_pixel_columns):
            derived[column.replace("_raw_px", "_direct_smooth_px")] = direct_smoothed[
                :, column_index
            ]
        projected_columns = (
            "front_image_x_smooth_px",
            "front_image_y_smooth_px",
            "side_image_x_smooth_px",
            "side_image_y_smooth_px",
        )
        for column in projected_columns:
            if column not in measured:
                continue
            source_values = numeric_column(measured, column)
            grid_values = np.full(len(grid_time), np.nan, dtype=float)
            grid_values[present] = source_values[source_rows[present]]
            missing_values = ~np.isfinite(grid_values)
            if np.any(missing_values):
                grid_values[missing_values] = np.interp(
                    grid_time[missing_values],
                    grid_time[~missing_values],
                    grid_values[~missing_values],
                )
            derived[column.replace("_smooth_px", "_calibration_projected_smooth_px")] = grid_values
    derived.to_csv(output_dir / "orbit_timeseries.csv", index=False)

    frequency_values = episodes["frequency_hz"].to_numpy(dtype=float) if len(episodes) else np.array([])
    period_values = episodes["period_s"].to_numpy(dtype=float) if len(episodes) else np.array([])
    radius_column = f"radius_median_{unit_key}"
    radius_values = episodes[radius_column].to_numpy(dtype=float) if len(episodes) else np.array([])
    circle_fit_radius_column = f"circle_fit_radius_{unit_key}"
    circle_fit_radius_values = (
        episodes[circle_fit_radius_column].to_numpy(dtype=float)
        if len(episodes)
        else np.array([])
    )
    circle_fit_error_values = (
        episodes["circle_fit_normalized_radial_rmse"].to_numpy(dtype=float)
        if len(episodes)
        else np.array([])
    )
    centers = (
        (episodes["start_s"] + episodes["end_s"]).to_numpy(dtype=float) / 2
        if len(episodes)
        else np.array([])
    )
    recurrence = np.diff(centers)
    orbit_duration = sum(grid_time[end] - grid_time[start] for start, end in intervals)
    candidate_duration = sum(
        grid_time[end] - grid_time[start] for start, end in candidate_intervals
    )
    recording_duration = float(grid_time[-1] - grid_time[0])
    camera_horizontal_proxy = (
        x_col.startswith("front_image_x_") and y_col.startswith("side_image_x_")
    )
    method_notes = [
        "The source CSV is not modified.",
        "Only gaps at or below max_gap_samples are interpolated on the derived analysis grid.",
        "The band-pass RMS envelope is an oscillation-amplitude candidate signal, not a geometric radius.",
        "Candidate windows are validated with an explicit harmonic fit whose center may move linearly within each fit window.",
        "Circle-fit radius is the mean distance from the fitted moving center; radial RMSE measures departure from that circle.",
        "Axis ratio, angular coverage, and winding consistency reject line-like or partial-arc motion.",
        "An RMS candidate is accepted as an episode only when the configured fraction of its local geometry windows pass; per-sample fit status remains in the derived CSV.",
        "Frequency is the fitted winding rate of ellipse-normalized unwrapped XY phase.",
        "Boundary sensitivity reruns the RMS candidate stage at 0.8x, 1.0x, and 1.5x the primary threshold.",
        "Reported position units are inherited from the selected input columns or --position-unit; calibration error is not estimated by this script.",
    ]
    if camera_horizontal_proxy:
        method_notes.append(
            "The displayed horizontal path combines the directly measured front-camera and side-camera horizontal pixel coordinates. "
            "It is an uncalibrated timing/shape proxy, not a reconstructed physical 3-D trajectory."
        )
    if measurement_source_mode == "paired_camera_tracks":
        method_notes.append(
            "The analysis uses the paired front_background_tracks.csv and side_background_tracks.csv files. "
            "This preserves camera measurements removed from tracks_3d.csv by its calibration-dependent 3-D jump filter."
        )
    summary = {
        "label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "git_revision": git_revision(),
        "source_csv": str(args.csv.resolve()),
        "source_sha256": file_sha256(args.csv),
        "measurement_source_mode": measurement_source_mode,
        "analysis_sources": analysis_sources,
        "source_video": str(source_video.expanduser().resolve()) if source_video else None,
        "browser_video": str(browser_video.resolve()) if browser_video else None,
        "selected_particle": selected_particle,
        "selected_columns": {"time": time_col, "frame": frame_col, "x": x_col, "y": y_col, "z": z_col},
        "trajectory_mode": "camera_pixel_views" if camera_horizontal_proxy else "selected_coordinates",
        "position_unit": unit,
        "sample_rate_hz": sample_rate_hz,
        "recording_start_s": float(grid_time[0]),
        "recording_end_s": float(grid_time[-1]),
        "candidate_episode_count": int(len(candidate_intervals)),
        "episode_count": int(len(episodes)),
        "candidate_duty_cycle": (
            candidate_duration / recording_duration if recording_duration > 0 else 0.0
        ),
        "orbit_duty_cycle": orbit_duration / recording_duration if recording_duration > 0 else 0.0,
        "frequency_hz": summary_stats(frequency_values),
        "period_s": summary_stats(period_values),
        f"radius_{unit_key}": summary_stats(radius_values),
        f"circle_fit_radius_{unit_key}": summary_stats(circle_fit_radius_values),
        "circle_fit_normalized_radial_rmse": summary_stats(
            circle_fit_error_values
        ),
        "recurrence_s": summary_stats(recurrence),
        "threshold": {"value": threshold, "unit": unit, "method": threshold_method},
        "boundary_sensitivity": sensitivity,
        "audit": audit,
        "parameters": {
            "band_low_hz": args.band_low_hz,
            "band_high_hz": args.band_high_hz,
            "minimum_episode_s": args.minimum_episode_s,
            "merge_gap_s": args.merge_gap_s,
            "smooth_window_s": args.smooth_window_s,
            "radius_window_s": args.radius_window_s,
            "fit_window_s": fit_window_s,
            "fit_step_s": fit_step_s,
            "fit_minimum_turns": fit_minimum_turns,
            "fit_minimum_axis_ratio": fit_minimum_axis_ratio,
            "fit_maximum_normalized_radial_rmse": (
                fit_maximum_normalized_radial_rmse
            ),
            "fit_minimum_direction_consistency": (
                fit_minimum_direction_consistency
            ),
            "fit_minimum_accepted_fraction": fit_minimum_accepted_fraction,
            "geometric_validation_enabled": not rms_only,
            "max_gap_samples": args.max_gap_samples,
        },
        "method_notes": method_notes,
    }
    summary = json.loads(json.dumps(summary, default=finite_or_none, allow_nan=False))
    (output_dir / "orbit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    coordinate_names = (
        ("front horizontal", "side horizontal")
        if camera_horizontal_proxy
        else tuple(name for name in ("x", "y", "z")[: smoothed.shape[1]])
    )
    plot_path_title = (
        f"{label}: detected front-X × side-X episodes"
        if camera_horizontal_proxy
        else None
    )
    make_plots(
        output_dir,
        label,
        grid_time,
        smoothed,
        oscillatory,
        radius_envelope,
        threshold,
        candidate_intervals,
        intervals,
        sample_rate_hz,
        (args.band_low_hz, args.band_high_hz),
        unit,
        coordinate_names,
        plot_path_title,
    )
    if not args.no_html:
        make_html_report(
            output_dir,
            label,
            summary,
            episodes,
            derived,
            unit,
            browser_video,
        )
    return summary


def main() -> int:
    args = parse_args()
    summary = analyze(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
