"""Statistical calculations for circular-motion analysis.

This module contains only numerical operations.  CSV loading, plotting, HTML
generation, and command-line handling belong in ``analyze_orbits_from_csv``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class OrbitStatisticsResult:
    """Numerical outputs needed by the CSV/report layer."""

    smoothed: np.ndarray
    oscillatory: np.ndarray
    radius_envelope: np.ndarray
    smooth_samples: int
    threshold: float
    threshold_method: str
    intervals: list[tuple[int, int]]
    boundary_sensitivity: dict[str, list[dict[str, float]]]
    episode_rows: list[dict[str, Any]]


def robust_sigma(values: np.ndarray) -> float:
    """Return a median-absolute-deviation estimate of standard deviation."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return math.nan
    median = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - median)))


def odd_window(seconds: float, sample_rate_hz: float, count: int) -> int:
    """Convert a duration to an odd sample count accepted by Savitzky-Golay."""
    samples = max(5, int(round(seconds * sample_rate_hz)))
    if samples % 2 == 0:
        samples += 1
    maximum = count if count % 2 else count - 1
    return min(samples, maximum)


def otsu_threshold(values: np.ndarray) -> float:
    """Choose an amplitude threshold using an Otsu split in log space."""
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) < 10:
        raise ValueError("Too few positive amplitude samples to choose an automatic threshold")
    log_values = np.log10(positive)
    low, high = np.percentile(log_values, [1, 99])
    if not high > low:
        return float(np.median(positive))
    histogram, edges = np.histogram(log_values, bins=256, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    weight_left = np.cumsum(histogram)
    weight_right = np.cumsum(histogram[::-1])[::-1]
    mean_left = np.cumsum(histogram * centers) / np.maximum(weight_left, 1)
    mean_right = (
        np.cumsum((histogram * centers)[::-1]) / np.maximum(weight_right[::-1], 1)
    )[::-1]
    variance = (
        weight_left[:-1]
        * weight_right[1:]
        * (mean_left[:-1] - mean_right[1:]) ** 2
    )
    threshold_log = centers[int(np.argmax(variance))]
    return float(10**threshold_log)


def contiguous_intervals(
    mask: np.ndarray,
    sample_rate_hz: float,
    minimum_s: float,
    merge_gap_s: float,
) -> list[tuple[int, int]]:
    """Return inclusive intervals after merging short gaps and removing short runs."""
    closing = max(1, int(round(merge_gap_s * sample_rate_hz)))
    closed = ndimage.binary_closing(mask, structure=np.ones(closing, dtype=bool))
    labels, count = ndimage.label(closed)
    minimum = max(1, int(round(minimum_s * sample_rate_hz)))
    intervals = []
    for label in range(1, count + 1):
        indexes = np.flatnonzero(labels == label)
        if len(indexes) >= minimum:
            intervals.append((int(indexes[0]), int(indexes[-1])))
    return intervals


def episode_metrics(
    episode: int,
    start: int,
    end: int,
    time_s: np.ndarray,
    positions: np.ndarray,
    oscillatory: np.ndarray,
    radius_envelope: np.ndarray,
    position_unit: str,
) -> dict[str, Any]:
    """Calculate period, radius, shape, direction, center, and drift for one episode."""
    section = slice(start, end + 1)
    local_time = time_s[section]
    xy = oscillatory[section, :2].copy()
    xy -= np.mean(xy, axis=0)
    covariance = np.cov(xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    projected = xy @ eigenvectors
    if np.linalg.det(eigenvectors) < 0:
        projected[:, 1] *= -1
    safe = np.maximum(eigenvalues, np.finfo(float).eps)
    angle = np.unwrap(
        np.arctan2(
            projected[:, 1] / np.sqrt(safe[1]),
            projected[:, 0] / np.sqrt(safe[0]),
        )
    )
    slope, intercept = np.polyfit(local_time, angle, 1)
    frequency_hz = abs(float(slope)) / (2 * np.pi)
    period_s = 1.0 / frequency_hz if frequency_hz > 0 else math.nan
    fitted = slope * local_time + intercept
    increments = np.diff(angle)
    winding_sign = 1.0 if slope >= 0 else -1.0

    radius_instantaneous = np.sqrt(np.sum(xy**2, axis=1))
    radius_median = float(np.median(radius_envelope[section]))
    radius_sigma = robust_sigma(radius_instantaneous)
    velocity = np.gradient(xy, local_time, axis=0)
    speed = np.sqrt(np.sum(velocity**2, axis=1))
    center = np.mean(positions[section], axis=0)
    center_slopes = [
        np.polyfit(local_time, positions[section, axis], 1)[0]
        for axis in range(2)
    ]

    metric = {
        "episode": episode,
        "start_s": float(local_time[0]),
        "end_s": float(local_time[-1]),
        "duration_s": float(local_time[-1] - local_time[0]),
        "cycles": abs(float(angle[-1] - angle[0])) / (2 * np.pi),
        "frequency_hz": frequency_hz,
        "period_s": period_s,
        "direction_xy": "clockwise" if slope < 0 else "counterclockwise",
        "direction_consistency": float(np.mean(np.sign(increments) == winding_sign)),
        "xy_circularity": float(np.sqrt(safe[1] / safe[0])),
        f"ellipse_major_radius_{position_unit}": float(np.sqrt(2 * safe[0])),
        f"ellipse_minor_radius_{position_unit}": float(np.sqrt(2 * safe[1])),
        f"radius_median_{position_unit}": radius_median,
        f"radius_robust_sigma_{position_unit}": radius_sigma,
        "radial_coefficient_of_variation": (
            radius_sigma / radius_median if radius_median else math.nan
        ),
        f"center_x_{position_unit}": float(center[0]),
        f"center_y_{position_unit}": float(center[1]),
        "phase_fit_residual_rad": float(np.std(angle - fitted)),
        f"median_orbit_speed_{position_unit}_per_s": float(np.median(speed)),
        f"center_drift_speed_{position_unit}_per_s": float(np.hypot(*center_slopes)),
    }
    if positions.shape[1] > 2:
        metric[f"center_z_{position_unit}"] = float(
            np.mean(positions[section, 2])
        )
    return metric


def summary_stats(values: np.ndarray) -> dict[str, float | None]:
    """Return finite mean, sample standard deviation, and median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "standard_deviation": None, "median": None}
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": (
            float(np.std(values, ddof=1)) if len(values) > 1 else None
        ),
        "median": float(np.median(values)),
    }


def analyze_orbit_statistics(
    time_s: np.ndarray,
    coordinates: np.ndarray,
    sample_rate_hz: float,
    position_unit: str,
    *,
    band_low_hz: float,
    band_high_hz: float,
    threshold_setting: str,
    minimum_episode_s: float,
    merge_gap_s: float,
    smooth_window_s: float,
    radius_window_s: float,
) -> OrbitStatisticsResult:
    """Run the complete numerical orbit-analysis pipeline."""
    if len(time_s) != len(coordinates):
        raise ValueError("Time and coordinate arrays must have the same length")
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("Orbit analysis requires at least two coordinate columns")
    if not 0 < band_low_hz < band_high_hz < sample_rate_hz / 2:
        raise ValueError("The frequency band must satisfy 0 < low < high < Nyquist frequency")

    smooth_samples = odd_window(smooth_window_s, sample_rate_hz, len(time_s))
    if smooth_samples < 5:
        raise ValueError("Recording is too short for the requested smoothing window")
    smoothed = signal.savgol_filter(coordinates, smooth_samples, 2, axis=0)
    sos = signal.butter(
        4,
        (band_low_hz, band_high_hz),
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    oscillatory = signal.sosfiltfilt(sos, smoothed, axis=0)
    radius_samples = max(3, int(round(radius_window_s * sample_rate_hz)))
    radius_power = ndimage.uniform_filter1d(
        oscillatory[:, 0] ** 2 + oscillatory[:, 1] ** 2,
        size=radius_samples,
        mode="nearest",
    )
    radius_envelope = np.sqrt(np.maximum(radius_power, 0))

    if threshold_setting.lower() == "auto":
        threshold = otsu_threshold(radius_envelope)
        threshold_method = "automatic Otsu split of log radius envelope"
    else:
        try:
            threshold = float(threshold_setting)
        except ValueError as exc:
            raise ValueError(
                "--threshold must be a positive number or 'auto'"
            ) from exc
        if threshold <= 0:
            raise ValueError("--threshold must be positive")
        threshold_method = "user supplied"

    intervals = contiguous_intervals(
        radius_envelope >= threshold,
        sample_rate_hz,
        minimum_episode_s,
        merge_gap_s,
    )
    sensitivity: dict[str, list[dict[str, float]]] = {}
    for factor in (0.8, 1.0, 1.5):
        candidate = threshold * factor
        sensitivity[f"{candidate:.9g}"] = [
            {"start_s": float(time_s[start]), "end_s": float(time_s[end])}
            for start, end in contiguous_intervals(
                radius_envelope >= candidate,
                sample_rate_hz,
                minimum_episode_s,
                merge_gap_s,
            )
        ]

    episode_rows = [
        episode_metrics(
            number,
            start,
            end,
            time_s,
            smoothed,
            oscillatory,
            radius_envelope,
            position_unit,
        )
        for number, (start, end) in enumerate(intervals, start=1)
    ]
    return OrbitStatisticsResult(
        smoothed=smoothed,
        oscillatory=oscillatory,
        radius_envelope=radius_envelope,
        smooth_samples=smooth_samples,
        threshold=threshold,
        threshold_method=threshold_method,
        intervals=intervals,
        boundary_sensitivity=sensitivity,
        episode_rows=episode_rows,
    )
