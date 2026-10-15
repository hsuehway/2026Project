#!/usr/bin/env python3
"""Run the simple geometric circle/ellipse/spiral detector on a CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simple_orbit_statistics import ORBIT_TYPES, make_intervals, rolling_statistics


COLORS = {
    "circle": "#1f77b4", "ellipse": "#ff7f0e", "spiral_in": "#2ca02c",
    "spiral_out": "#d62728", "not_orbit": "#7f7f7f",
}
SETTING_NAMES = (
    "window_s", "step_s", "smooth_samples", "minimum_radius", "minimum_turns",
    "minimum_direction_consistency", "circle_ratio", "minimum_ellipse_ratio",
    "maximum_ellipse_error", "maximum_return_error", "spiral_change",
    "minimum_spiral_r2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--time-col", default="time_s")
    parser.add_argument("--x-col", default="x_mm")
    parser.add_argument("--y-col", default="y_mm")
    parser.add_argument("--position-unit", default="mm")
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--step-s", type=float, default=0.25)
    parser.add_argument("--smooth-samples", type=int, default=5)
    parser.add_argument("--minimum-radius", type=float, default=0.02)
    parser.add_argument("--minimum-turns", type=float, default=1.0)
    parser.add_argument("--minimum-direction-consistency", type=float, default=0.75)
    parser.add_argument("--circle-ratio", type=float, default=0.80)
    parser.add_argument("--minimum-ellipse-ratio", type=float, default=0.35)
    parser.add_argument("--maximum-ellipse-error", type=float, default=0.30)
    parser.add_argument("--maximum-return-error", type=float, default=0.55)
    parser.add_argument("--spiral-change", type=float, default=0.20)
    parser.add_argument("--minimum-spiral-r2", type=float, default=0.50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or None


def save_plot(windows: pd.DataFrame, path: Path, unit: str) -> None:
    colors = [COLORS[value] for value in windows["classification"]]
    time = windows["window_center_s"]
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].scatter(time, windows[f"equivalent_radius_{unit}"], c=colors, s=10)
    axes[0].set_ylabel(f"Radius ({unit})")
    axes[1].scatter(time, windows["period_s"], c=colors, s=10)
    axes[1].set_ylabel("Period (s)")
    axes[2].scatter(time, windows["axis_ratio"], c=colors, s=10)
    axes[2].set_ylabel("Minor / major")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source = args.csv.resolve()
    frame = pd.read_csv(source)
    columns = [args.time_col, args.x_col, args.y_col]
    if not set(columns) <= set(frame.columns):
        raise ValueError(f"CSV must contain {columns}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    values = values.sort_values(args.time_col)
    time_s = values[args.time_col].to_numpy(float)
    xy = values[[args.x_col, args.y_col]].to_numpy(float)
    intervals_s = pd.Series(time_s).diff().dropna()
    if (intervals_s <= 0).any():
        raise ValueError("Time values must be unique and increasing")

    settings = {name: getattr(args, name) for name in SETTING_NAMES}
    windows = rolling_statistics(time_s, xy, settings)
    if windows.empty:
        raise ValueError("No windows contained at least 20 valid samples")
    intervals = make_intervals(windows, args.step_s)
    unit = args.position_unit
    radius_names = ("major_radius", "minor_radius", "equivalent_radius",
                    "center_x", "center_y", "center_drift_speed")
    windows = windows.rename(columns={name: f"{name}_{unit}" for name in radius_names})
    intervals = intervals.rename(columns={"median_radius": f"median_radius_{unit}"})

    output = (args.output_dir or source.with_name(source.stem + "_simple_orbits")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output / "rolling_orbit_windows.csv", index=False)
    intervals.to_csv(output / "rolling_orbit_intervals.csv", index=False)
    save_plot(windows, output / "rolling_orbit_diagnostics.png", unit)

    accepted = windows[windows["classification"].isin(ORBIT_TYPES)]
    script = Path(__file__).resolve()
    stats_script = script.with_name("simple_orbit_statistics.py")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source), "source_sha256": sha256(source),
        "analysis_script": str(script), "analysis_script_sha256": sha256(script),
        "statistics_script": str(stats_script), "statistics_script_sha256": sha256(stats_script),
        "git_revision": git_revision(),
        "input_columns": columns, "input_rows": int(len(frame)),
        "finite_rows_used": int(len(values)), "settings": settings,
        "position_unit": unit,
        "median_sample_interval_s": float(intervals_s.median()),
        "median_sampling_rate_hz": float(1 / intervals_s.median()),
        "calibration": "Not used directly; input positions are already in the declared unit.",
        "window_count": int(len(windows)),
        "classification_counts": {
            key: int(value) for key, value in windows["classification"].value_counts().items()
        },
        "orbit_window_fraction": float(len(accepted) / len(windows)),
        "orbit_radius_median": (
            float(accepted[f"equivalent_radius_{unit}"].median()) if len(accepted) else None
        ),
        "orbit_period_median_s": float(accepted["period_s"].median()) if len(accepted) else None,
        "notes": [
            "Overlapping windows are not independent measurements.",
            "PCA radii are calculated after removing linear center drift.",
            "Return error tests whether positions one fitted period apart are nearby.",
            "Thresholds require comparison with the source video before scientific use.",
        ],
    }
    with (output / "rolling_orbit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2))
