"""Statistical calculations for circular-motion analysis.

This module contains only numerical operations.  CSV loading, plotting, HTML
generation, and command-line handling belong in ``analyze_orbits_from_csv``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage, optimize, signal


@dataclass(frozen=True)
class RollingMovingCenterResult:
    """Per-sample local estimates from overlapping moving-center fits."""

    available: np.ndarray
    accepted: np.ndarray
    status: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    center_vx: np.ndarray
    center_vy: np.ndarray
    radius: np.ndarray
    major_radius: np.ndarray
    minor_radius: np.ndarray
    axis_ratio: np.ndarray
    radial_rmse: np.ndarray
    normalized_radial_rmse: np.ndarray
    position_rmse: np.ndarray
    frequency_hz: np.ndarray
    turns: np.ndarray
    angular_coverage_deg: np.ndarray
    direction_consistency: np.ndarray


@dataclass(frozen=True)
class OrbitStatisticsResult:
    """Numerical outputs needed by the CSV/report layer."""

    smoothed: np.ndarray
    oscillatory: np.ndarray
    radius_envelope: np.ndarray
    smooth_samples: int
    threshold: float
    threshold_method: str
    candidate_intervals: list[tuple[int, int]]
    intervals: list[tuple[int, int]]
    boundary_sensitivity: dict[str, list[dict[str, float]]]
    episode_rows: list[dict[str, Any]]
    rolling_fit: RollingMovingCenterResult


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


def fit_moving_center_orbit(
    time_s: np.ndarray,
    xy: np.ndarray,
    *,
    band_low_hz: float,
    band_high_hz: float,
    minimum_turns: float,
    minimum_axis_ratio: float,
    maximum_normalized_radial_rmse: float,
    minimum_direction_consistency: float,
) -> dict[str, Any]:
    """Fit a linearly drifting center plus a harmonic local orbit.

    For each trial frequency, X and Y are solved simultaneously as a linear
    combination of ``1``, local time, cosine, and sine.  The first two terms
    define the moving center.  The harmonic coefficient matrix defines the
    fitted ellipse; its singular values are the semi-axis radii.  Circle-fit
    residuals are then measured from the explicitly estimated moving center.
    """
    time_s = np.asarray(time_s, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if len(time_s) < 12 or xy.shape != (len(time_s), 2):
        raise ValueError("A moving-center fit requires at least 12 finite X/Y samples")
    if not np.isfinite(time_s).all() or not np.isfinite(xy).all():
        raise ValueError("Moving-center fit inputs must be finite")
    local_time = time_s - float(np.mean(time_s))
    duration_s = float(np.ptp(local_time))
    if duration_s <= 0:
        raise ValueError("Moving-center fit requires a positive time span")

    def solve(frequency_hz: float) -> tuple[float, np.ndarray, np.ndarray]:
        phase = 2 * np.pi * frequency_hz * local_time
        design = np.column_stack(
            (np.ones(len(local_time)), local_time, np.cos(phase), np.sin(phase))
        )
        coefficients = np.linalg.lstsq(design, xy, rcond=None)[0]
        residual = xy - design @ coefficients
        objective = float(np.mean(np.sum(residual**2, axis=1)))
        return objective, coefficients, residual

    optimized = optimize.minimize_scalar(
        lambda frequency: solve(float(frequency))[0],
        bounds=(band_low_hz, band_high_hz),
        method="bounded",
        options={"xatol": max(1e-6, (band_high_hz - band_low_hz) * 1e-5)},
    )
    frequency_hz = float(optimized.x)
    _, coefficients, residual = solve(frequency_hz)
    center = np.column_stack((np.ones(len(local_time)), local_time)) @ coefficients[:2]
    centered = xy - center
    harmonic_matrix = coefficients[2:].T
    singular_values = np.linalg.svd(harmonic_matrix, compute_uv=False)
    major_radius = float(np.max(singular_values))
    minor_radius = float(np.min(singular_values))
    axis_ratio = minor_radius / major_radius if major_radius > 0 else 0.0

    instantaneous_radius = np.linalg.norm(centered, axis=1)
    radius = float(np.mean(instantaneous_radius))
    radial_residual = instantaneous_radius - radius
    radial_rmse = float(np.sqrt(np.mean(radial_residual**2)))
    normalized_radial_rmse = radial_rmse / radius if radius > 0 else math.inf
    position_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    raw_angle = np.unwrap(np.arctan2(centered[:, 1], centered[:, 0]))
    angular_slope, angular_intercept = np.polyfit(local_time, raw_angle, 1)
    fitted_angle = angular_slope * local_time + angular_intercept
    turns = abs(float(raw_angle[-1] - raw_angle[0])) / (2 * np.pi)
    angular_coverage_deg = abs(float(raw_angle[-1] - raw_angle[0])) * 180 / np.pi
    dt = float(np.median(np.diff(time_s)))
    direction_lag = min(
        max(1, int(round(1.0 / max(frequency_hz, 1e-12) / (20 * dt)))),
        len(raw_angle) - 1,
    )
    changes = raw_angle[direction_lag:] - raw_angle[:-direction_lag]
    winding_sign = 1.0 if angular_slope >= 0 else -1.0
    direction_consistency = float(np.mean(changes * winding_sign > 0))
    direction = "counterclockwise" if angular_slope >= 0 else "clockwise"

    checks = (
        (not optimized.success, "frequency optimization failed"),
        (not np.isfinite(radius) or radius <= np.finfo(float).eps, "radius is zero"),
        (turns < minimum_turns, "insufficient angular coverage"),
        (axis_ratio < minimum_axis_ratio, "path is too line-like"),
        (
            normalized_radial_rmse > maximum_normalized_radial_rmse,
            "radial residual is too large",
        ),
        (
            direction_consistency < minimum_direction_consistency,
            "winding direction is inconsistent",
        ),
    )
    status = "accepted"
    accepted = True
    for failed, reason in checks:
        if failed:
            accepted = False
            status = reason
            break

    return {
        "accepted": accepted,
        "status": status,
        "center_x": float(coefficients[0, 0]),
        "center_y": float(coefficients[0, 1]),
        "center_vx": float(coefficients[1, 0]),
        "center_vy": float(coefficients[1, 1]),
        "center_drift_speed": float(np.hypot(coefficients[1, 0], coefficients[1, 1])),
        "radius": radius,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "axis_ratio": axis_ratio,
        "radial_rmse": radial_rmse,
        "normalized_radial_rmse": normalized_radial_rmse,
        "position_rmse": position_rmse,
        "frequency_hz": frequency_hz,
        "period_s": 1.0 / frequency_hz,
        "turns": turns,
        "angular_coverage_deg": angular_coverage_deg,
        "direction": direction,
        "direction_consistency": direction_consistency,
        "phase_fit_residual_rad": float(np.std(raw_angle - fitted_angle)),
        "window_duration_s": duration_s,
    }


def rolling_moving_center_fits(
    time_s: np.ndarray,
    xy: np.ndarray,
    candidate_intervals: list[tuple[int, int]],
    *,
    sample_rate_hz: float,
    window_s: float,
    step_s: float,
    band_low_hz: float,
    band_high_hz: float,
    minimum_turns: float,
    minimum_axis_ratio: float,
    maximum_normalized_radial_rmse: float,
    minimum_direction_consistency: float,
) -> RollingMovingCenterResult:
    """Apply the explicit center/radius fit across candidate intervals.

    Fits are evaluated at regularly spaced window centers.  Each candidate
    sample receives the nearest local estimate, making the rolling quantities
    available alongside every time-series row without pretending that
    overlapping fits are independent measurements.
    """
    count = len(time_s)
    numeric_names = (
        "center_x", "center_y", "center_vx", "center_vy", "radius",
        "major_radius", "minor_radius", "axis_ratio", "radial_rmse",
        "normalized_radial_rmse", "position_rmse", "frequency_hz", "turns",
        "angular_coverage_deg", "direction_consistency",
    )
    arrays = {name: np.full(count, np.nan, dtype=float) for name in numeric_names}
    available = np.zeros(count, dtype=bool)
    accepted = np.zeros(count, dtype=bool)
    status = np.full(count, "not a candidate", dtype=object)
    half_window = max(6, int(round(window_s * sample_rate_hz / 2)))
    step = max(1, int(round(step_s * sample_rate_hz)))

    for interval_start, interval_end in candidate_intervals:
        centers = np.arange(interval_start, interval_end + 1, step, dtype=int)
        if not len(centers) or centers[-1] != interval_end:
            centers = np.append(centers, interval_end)
        center_results: list[tuple[int, dict[str, Any]]] = []
        for center_index in centers:
            start = max(0, int(center_index) - half_window)
            end = min(count, int(center_index) + half_window + 1)
            try:
                result = fit_moving_center_orbit(
                    time_s[start:end],
                    xy[start:end, :2],
                    band_low_hz=band_low_hz,
                    band_high_hz=band_high_hz,
                    minimum_turns=minimum_turns,
                    minimum_axis_ratio=minimum_axis_ratio,
                    maximum_normalized_radial_rmse=maximum_normalized_radial_rmse,
                    minimum_direction_consistency=minimum_direction_consistency,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                result = {"accepted": False, "status": str(exc)}
                result.update({name: math.nan for name in numeric_names})
            center_results.append((int(center_index), result))

        center_indexes = np.asarray([item[0] for item in center_results], dtype=int)
        sample_indexes = np.arange(interval_start, interval_end + 1, dtype=int)
        insertion = np.searchsorted(center_indexes, sample_indexes)
        insertion = np.clip(insertion, 0, len(center_indexes) - 1)
        previous = np.clip(insertion - 1, 0, len(center_indexes) - 1)
        choose_previous = (
            np.abs(sample_indexes - center_indexes[previous])
            <= np.abs(sample_indexes - center_indexes[insertion])
        )
        nearest = np.where(choose_previous, previous, insertion)
        for sample_index, result_index in zip(sample_indexes, nearest):
            result = center_results[int(result_index)][1]
            available[sample_index] = np.isfinite(result.get("radius", math.nan))
            accepted[sample_index] = bool(result["accepted"])
            status[sample_index] = str(result["status"])
            for name in numeric_names:
                arrays[name][sample_index] = float(result.get(name, math.nan))

    return RollingMovingCenterResult(
        available=available,
        accepted=accepted,
        status=status,
        **arrays,
    )


def episode_metrics(
    episode: int,
    start: int,
    end: int,
    time_s: np.ndarray,
    positions: np.ndarray,
    oscillatory: np.ndarray,
    radius_envelope: np.ndarray,
    position_unit: str,
    rolling_fit: RollingMovingCenterResult | None = None,
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
    if rolling_fit is not None:
        def local_median(values: np.ndarray) -> float:
            local = np.asarray(values[section], dtype=float)
            local = local[np.isfinite(local)]
            return float(np.median(local)) if len(local) else math.nan

        metric.update(
            {
                f"circle_fit_radius_{position_unit}": local_median(rolling_fit.radius),
                f"circle_fit_major_radius_{position_unit}": local_median(
                    rolling_fit.major_radius
                ),
                f"circle_fit_minor_radius_{position_unit}": local_median(
                    rolling_fit.minor_radius
                ),
                "circle_fit_axis_ratio": local_median(rolling_fit.axis_ratio),
                f"circle_fit_radial_rmse_{position_unit}": local_median(
                    rolling_fit.radial_rmse
                ),
                "circle_fit_normalized_radial_rmse": local_median(
                    rolling_fit.normalized_radial_rmse
                ),
                f"circle_fit_position_rmse_{position_unit}": local_median(
                    rolling_fit.position_rmse
                ),
                "circle_fit_frequency_hz": local_median(rolling_fit.frequency_hz),
                "circle_fit_turns": local_median(rolling_fit.turns),
                "circle_fit_angular_coverage_deg": local_median(
                    rolling_fit.angular_coverage_deg
                ),
                "circle_fit_direction_consistency": local_median(
                    rolling_fit.direction_consistency
                ),
                f"circle_fit_center_x_{position_unit}": local_median(
                    rolling_fit.center_x
                ),
                f"circle_fit_center_y_{position_unit}": local_median(
                    rolling_fit.center_y
                ),
                f"circle_fit_center_vx_{position_unit}_per_s": local_median(
                    rolling_fit.center_vx
                ),
                f"circle_fit_center_vy_{position_unit}_per_s": local_median(
                    rolling_fit.center_vy
                ),
                "circle_fit_acceptance_fraction": float(
                    np.mean(rolling_fit.accepted[section])
                ),
            }
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
    fit_window_s: float = 2.0,
    fit_step_s: float = 0.25,
    fit_minimum_turns: float = 1.5,
    fit_minimum_axis_ratio: float = 0.70,
    fit_maximum_normalized_radial_rmse: float = 0.25,
    fit_minimum_direction_consistency: float = 0.80,
    fit_minimum_accepted_fraction: float = 0.60,
    validate_geometry: bool = True,
) -> OrbitStatisticsResult:
    """Run the complete numerical orbit-analysis pipeline."""
    if len(time_s) != len(coordinates):
        raise ValueError("Time and coordinate arrays must have the same length")
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("Orbit analysis requires at least two coordinate columns")
    if not 0 < band_low_hz < band_high_hz < sample_rate_hz / 2:
        raise ValueError("The frequency band must satisfy 0 < low < high < Nyquist frequency")
    if fit_window_s <= 0 or fit_step_s <= 0:
        raise ValueError("Moving-center fit window and step must be positive")
    if fit_minimum_turns <= 0:
        raise ValueError("Moving-center minimum turns must be positive")
    if not 0 < fit_minimum_axis_ratio <= 1:
        raise ValueError("Moving-center minimum axis ratio must be in (0, 1]")
    if fit_maximum_normalized_radial_rmse <= 0:
        raise ValueError("Moving-center maximum normalized radial RMSE must be positive")
    if not 0 < fit_minimum_direction_consistency <= 1:
        raise ValueError("Moving-center minimum direction consistency must be in (0, 1]")
    if not 0 < fit_minimum_accepted_fraction <= 1:
        raise ValueError("Moving-center minimum accepted fraction must be in (0, 1]")

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

    candidate_intervals = contiguous_intervals(
        radius_envelope >= threshold,
        sample_rate_hz,
        minimum_episode_s,
        merge_gap_s,
    )
    rolling_fit = rolling_moving_center_fits(
        time_s,
        smoothed[:, :2],
        candidate_intervals,
        sample_rate_hz=sample_rate_hz,
        window_s=fit_window_s,
        step_s=fit_step_s,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        minimum_turns=fit_minimum_turns,
        minimum_axis_ratio=fit_minimum_axis_ratio,
        maximum_normalized_radial_rmse=fit_maximum_normalized_radial_rmse,
        minimum_direction_consistency=fit_minimum_direction_consistency,
    )
    if validate_geometry:
        intervals = [
            (start, end)
            for start, end in candidate_intervals
            if float(np.mean(rolling_fit.accepted[start : end + 1]))
            >= fit_minimum_accepted_fraction
        ]
    else:
        intervals = candidate_intervals
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
            rolling_fit,
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
        candidate_intervals=candidate_intervals,
        intervals=intervals,
        boundary_sensitivity=sensitivity,
        episode_rows=episode_rows,
        rolling_fit=rolling_fit,
    )
