"""Small geometric circle, ellipse, and spiral detector.

For one short X/Y trajectory, remove linear center drift, estimate ellipse axes
with PCA, and test turns, direction, coordinate revisits, and radius change.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


ORBIT_TYPES = {"circle", "ellipse", "spiral_in", "spiral_out"}


def _smooth(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values.copy()
    samples += samples % 2 == 0
    return pd.DataFrame(values).rolling(samples, center=True, min_periods=1).mean().to_numpy()


def _line(time_s: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(time_s, values, 1)
    residual = np.sum((values - (slope * time_s + intercept)) ** 2)
    total = np.sum((values - np.mean(values)) ** 2)
    return float(slope), float(1 - residual / total) if total > 0 else 0.0


def window_statistics(
    time_s: np.ndarray,
    xy: np.ndarray,
    settings: dict[str, float | int],
) -> dict[str, Any]:
    """Fit and classify one X/Y trajectory window."""
    xy = _smooth(xy, int(settings["smooth_samples"]))
    time = time_s - np.mean(time_s)

    # Subtract a straight moving center.
    design = np.column_stack((np.ones(len(time)), time))
    center_motion = np.linalg.lstsq(design, xy, rcond=None)[0]
    centered = xy - design @ center_motion

    # PCA finds the best ellipse orientation and its two radii.
    variance, axes = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(variance)[::-1]
    radii = np.sqrt(2 * np.maximum(variance[order], np.finfo(float).eps))
    major_radius, minor_radius = map(float, radii)
    axis_ratio = minor_radius / major_radius
    normalized_xy = centered @ axes[:, order] / radii

    # Going around means the angle unwraps steadily through full turns.
    angle = np.unwrap(np.arctan2(normalized_xy[:, 1], normalized_xy[:, 0]))
    angular_speed, _ = _line(time, angle)
    frequency_hz = abs(angular_speed) / (2 * np.pi)
    period_s = 1 / frequency_hz if frequency_hz > 0 else math.inf
    turns = abs(float(angle[-1] - angle[0])) / (2 * np.pi)
    dt = float(np.median(np.diff(time_s)))
    direction_lag = min(max(1, round(period_s / (20 * dt))), len(angle) - 1)
    changes = angle[direction_lag:] - angle[:-direction_lag]
    direction = float(np.mean(changes * np.sign(angular_speed or 1) > 0))

    # A closed orbit should revisit almost the same coordinate one period later.
    period_lag = max(1, round(period_s / dt))
    return_error = math.inf
    if period_lag < len(xy):
        return_error = float(np.median(np.linalg.norm(
            normalized_xy[period_lag:] - normalized_xy[:-period_lag], axis=1
        )))

    normalized_radius = np.linalg.norm(normalized_xy, axis=1)
    ellipse_error = float(np.sqrt(np.mean((normalized_radius - 1) ** 2)))
    radius_slope, radius_r2 = _line(time, normalized_radius)
    radius_change = float(
        radius_slope * (time[-1] - time[0]) / max(np.median(normalized_radius), 1e-12)
    )

    tests = [
        (major_radius < settings["minimum_radius"], "not_orbit", "radius too small"),
        (turns < settings["minimum_turns"], "not_orbit", "not enough rotation"),
        (axis_ratio < settings["minimum_ellipse_ratio"], "not_orbit", "back-and-forth motion"),
        (direction < settings["minimum_direction_consistency"], "not_orbit", "direction reverses"),
        (
            abs(radius_change) >= settings["spiral_change"]
            and radius_r2 >= settings["minimum_spiral_r2"],
            "spiral_out" if radius_change > 0 else "spiral_in",
            "radius changes steadily",
        ),
        (ellipse_error > settings["maximum_ellipse_error"], "not_orbit", "poor ellipse fit"),
        (return_error > settings["maximum_return_error"], "not_orbit", "does not revisit path"),
    ]
    classification, reason = "circle", "closed path with similar axes"
    for failed, label, message in tests:
        if failed:
            classification, reason = label, message
            break
    else:
        if axis_ratio < settings["circle_ratio"]:
            classification, reason = "ellipse", "closed path with unequal axes"

    return {
        "classification": classification,
        "reason": reason,
        "frequency_hz": frequency_hz,
        "period_s": period_s,
        "turns": turns,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "equivalent_radius": float(np.sqrt(np.mean(radii**2))),
        "axis_ratio": axis_ratio,
        "ellipse_error": ellipse_error,
        "return_error": return_error,
        "direction_consistency": direction,
        "radius_change_fraction": radius_change,
        "radius_trend_r2": radius_r2,
        "center_x": float(center_motion[0, 0]),
        "center_y": float(center_motion[0, 1]),
        "center_drift_speed": float(np.linalg.norm(center_motion[1])),
    }


def rolling_statistics(
    time_s: np.ndarray,
    xy: np.ndarray,
    settings: dict[str, float | int],
) -> pd.DataFrame:
    """Apply the geometric test to overlapping windows."""
    width, step = float(settings["window_s"]), float(settings["step_s"])
    centers = np.arange(time_s[0] + width / 2, time_s[-1] - width / 2 + step / 2, step)
    rows = []
    for center in centers:
        selected = (time_s >= center - width / 2) & (time_s <= center + width / 2)
        if selected.sum() < 20:
            continue
        row = window_statistics(time_s[selected], xy[selected], settings)
        row.update(window_start_s=center - width / 2, window_center_s=center,
                   window_end_s=center + width / 2, sample_count=int(selected.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def make_intervals(windows: pd.DataFrame, step_s: float) -> pd.DataFrame:
    """Merge neighboring windows that have the same classification."""
    rows = []
    groups = (windows["classification"] != windows["classification"].shift()).cumsum()
    for _, part in windows.groupby(groups, sort=False):
        rows.append({
            "classification": part["classification"].iloc[0],
            "start_s": float(part["window_center_s"].iloc[0] - step_s / 2),
            "end_s": float(part["window_center_s"].iloc[-1] + step_s / 2),
            "window_count": int(len(part)),
            "median_radius": float(part["equivalent_radius"].median()),
            "median_period_s": float(part["period_s"].median()),
        })
    return pd.DataFrame(rows)
