#!/usr/bin/env python3
"""Build compact browser data and an uncertainty audit for the SP6 report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
SOURCE_TRACKS = (
    ROOT
    / "Video Tracking Chin Lab"
    / "outputs_background"
    / "tracks"
    / "SP6_modes_20261004"
    / "tracking_full_57s_end"
    / "tracks_3d.csv"
)
POSITIONS = STUDY / "data" / "SP6_position_time.csv"
INTERVALS = STUDY / "data" / "SP6_circular_intervals.csv"
SUMMARY = STUDY / "data" / "SP6_mode_summary.json"
TEMPERATURE = STUDY / "data" / "SP6_temperature_trace_1Hz.csv"
AUDIT_RERUN = Path("/tmp/sp6-audit-20261004/SP6_position_time.csv")


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = pd.read_csv(SOURCE_TRACKS).sort_values("frame").reset_index(drop=True)
    positions = pd.read_csv(POSITIONS).sort_values("frame").reset_index(drop=True)
    intervals = pd.read_csv(INTERVALS)
    temperature = pd.read_csv(TEMPERATURE)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    frame_delta = positions["frame"].diff().dropna().to_numpy(dtype=int)
    timestamp_error = np.abs(
        positions["time_s"].to_numpy(dtype=float)
        - positions["frame"].to_numpy(dtype=float) / 120.0
    )
    coordinate_sigma = {
        axis: robust_sigma(
            source[f"{axis}_raw_mm"].to_numpy(dtype=float)
            - source[f"{axis}_smooth_mm"].to_numpy(dtype=float)
        )
        for axis in ("x", "y", "z")
    }
    pixel_sigma = {
        camera: {
            axis: robust_sigma(
                source[f"{camera}_image_{axis}_raw_px"].to_numpy(dtype=float)
                - source[f"{camera}_image_{axis}_smooth_px"].to_numpy(dtype=float)
            )
            for axis in ("x", "y")
        }
        for camera in ("front", "side")
    }

    frequency = intervals["frequency_hz"].to_numpy(dtype=float)
    period = intervals["period_s"].to_numpy(dtype=float)
    radius = intervals["orbit_rms_median_mm"].to_numpy(dtype=float)
    recurrence = np.diff(
        ((intervals["start_s"] + intervals["end_s"]) / 2).to_numpy(dtype=float)
    )
    dof = len(frequency) - 1

    temp_values = temperature["bottom_temperature_c"].to_numpy(dtype=float)
    temp_times = temperature["time_s"].to_numpy(dtype=float)
    peaks, _ = find_peaks(temp_values, prominence=0.8, distance=30)
    temp_periods = np.diff(temp_times[peaks])

    audit_hash = sha256(POSITIONS)
    rerun_hash = sha256(AUDIT_RERUN) if AUDIT_RERUN.exists() else None
    audit = {
        "source_video": "Gianni Levitation File/New Deets/SP_Ceramic/SP6",
        "source_tracks": str(SOURCE_TRACKS.relative_to(ROOT)),
        "fps": 120.0,
        "row_count": int(len(positions)),
        "start_s": float(positions["time_s"].iloc[0]),
        "end_s": float(positions["time_s"].iloc[-1]),
        "duplicate_frames": int(positions["frame"].duplicated().sum()),
        "timestamps_monotonic": bool(positions["time_s"].is_monotonic_increasing),
        "max_timestamp_vs_frame_error_s": float(timestamp_error.max()),
        "rows_after_frame_gaps": int(np.sum(frame_delta > 1)),
        "maximum_frame_gap": int(frame_delta.max()),
        "missing_raw_xyz_rows": int(
            positions[["x_mm", "y_mm", "z_mm"]].isna().any(axis=1).sum()
        ),
        "missing_smoothed_xyz_rows": int(
            positions[["x_smooth_mm", "y_smooth_mm", "z_smooth_mm"]]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "export_sha256": audit_hash,
        "independent_rerun_sha256": rerun_hash,
        "independent_rerun_exact_match": bool(rerun_hash == audit_hash),
        "analysis_interpolation": (
            "Only gaps of at most two frames are interpolated for filtering and mode "
            "detection. Exported rows remain measured rows from the tracker."
        ),
    }

    boundary_sets = summary["boundary_sensitivity"]
    episode_data = []
    for index, row in intervals.iterrows():
        boundary_candidates = [values[index] for values in boundary_sets.values()]
        mask = source["time_s"].between(row["start_s"], row["end_s"])
        subset = source.loc[mask]
        episode_data.append(
            {
                "episode": int(row["episode"]),
                "start": float(row["start_s"]),
                "end": float(row["end_s"]),
                "possibleStartMin": float(min(x["start_s"] for x in boundary_candidates)),
                "possibleStartMax": float(max(x["start_s"] for x in boundary_candidates)),
                "possibleEndMin": float(min(x["end_s"] for x in boundary_candidates)),
                "possibleEndMax": float(max(x["end_s"] for x in boundary_candidates)),
                "frequency": float(row["frequency_hz"]),
                "period": float(row["period_s"]),
                "radius": float(row["orbit_rms_median_mm"]),
                "circularity": float(row["xy_circularity"]),
                "center": [
                    float(row["center_x_mm"]),
                    float(row["center_y_mm"]),
                    float(row["center_z_mm"]),
                ],
                "frontCenterPx": [
                    float(subset["front_image_x_px"].mean() + 1.0),
                    float(subset["front_image_y_px"].mean() + 3.0),
                ],
                "sideCenterPx": [
                    float(subset["side_image_x_px"].mean() + 908.0),
                    float(subset["side_image_y_px"].mean()),
                ],
            }
        )

    episode_number = np.zeros(len(source), dtype=int)
    for episode in episode_data:
        mask = source["time_s"].between(episode["start"], episode["end"])
        episode_number[mask.to_numpy()] = episode["episode"]

    source_episode = source.copy()
    source_episode["episode"] = episode_number
    source_episode["bottom_temperature_c"] = np.interp(
        source_episode["time_s"], temp_times, temp_values
    )
    orbit_rms_by_frame = positions.set_index("frame")["orbit_rms_mm"]
    source_episode["orbit_rms_mm"] = source_episode["frame"].map(orbit_rms_by_frame)

    compact = source_episode.iloc[::4].copy()
    if compact.index[-1] != source_episode.index[-1]:
        compact = pd.concat([compact, source_episode.iloc[[-1]]])
    tracks = []
    for row in compact.itertuples(index=False):
        tracks.append(
            [
                round(float(row.time_s), 4),
                round(float(row.front_image_x_px + 1.0), 2),
                round(float(row.front_image_y_px + 3.0), 2),
                round(float(row.side_image_x_px + 908.0), 2),
                round(float(row.side_image_y_px), 2),
                round(float(row.x_smooth_mm), 5),
                round(float(row.y_smooth_mm), 5),
                round(float(row.z_smooth_mm), 5),
                round(float(row.orbit_rms_mm), 5),
                int(row.episode),
                round(float(row.bottom_temperature_c), 3),
            ]
        )

    samples = []
    for row in positions.itertuples(index=False):
        samples.append(
            [
                int(row.frame),
                round(float(row.time_s), 6),
                round(float(row.x_mm), 6),
                round(float(row.y_mm), 6),
                round(float(row.z_mm), 6),
                round(float(row.x_smooth_mm), 6),
                round(float(row.y_smooth_mm), 6),
                round(float(row.z_smooth_mm), 6),
                round(float(row.orbit_rms_mm), 6),
                int(row.circular_episode),
                round(float(row.bottom_temperature_c), 4),
            ]
        )

    circular_rms = positions.loc[
        positions["in_circular_episode"].astype(bool), "orbit_rms_mm"
    ].to_numpy(dtype=float)
    quiet_rms = positions.loc[
        ~positions["in_circular_episode"].astype(bool), "orbit_rms_mm"
    ].to_numpy(dtype=float)

    uncertainty = {
        "coordinateRandomSigmaMm": coordinate_sigma,
        "pixelRandomSigmaPx": pixel_sigma,
        "stereoZOffsetMedianMm": float(np.median(source["z_error_mm"])),
        "stereoZOffsetScatterSigmaMm": robust_sigma(source["z_error_mm"]),
        "frequencyMeanHz": float(frequency.mean()),
        "frequencyEpisodeSigmaHz": float(frequency.std(ddof=1)),
        "frequencyMean95CiHalfHz": float(
            student_t.ppf(0.975, dof) * frequency.std(ddof=1) / np.sqrt(len(frequency))
        ),
        "periodMeanS": float(period.mean()),
        "periodEpisodeSigmaS": float(period.std(ddof=1)),
        "radiusMeanMm": float(radius.mean()),
        "radiusEpisodeSigmaMm": float(radius.std(ddof=1)),
        "recurrenceMeanS": float(recurrence.mean()),
        "recurrenceSigmaS": float(recurrence.std(ddof=1)),
        "temperatureCycleMeanS": float(temp_periods.mean()),
        "temperatureCycleSigmaS": float(temp_periods.std(ddof=1)),
        "circularRmsMedianMm": float(np.median(circular_rms)),
        "quietRmsMedianMm": float(np.median(quiet_rms)),
        "absoluteCalibration": (
            "Not numerically bounded: the original SP6 calibration file is missing. "
            "Absolute millimeter coordinates have an additional unknown systematic error."
        ),
    }

    data = {
        "duration": 343.475,
        "reliableTrackingStart": 57.0,
        "temperature": [
            [round(float(time_s), 3), round(float(temp_c), 3)]
            for time_s, temp_c in zip(temp_times, temp_values)
        ],
        "trackFields": [
            "time",
            "frontX",
            "frontY",
            "sideX",
            "sideY",
            "xMm",
            "yMm",
            "zMm",
            "orbitRmsMm",
            "episode",
            "temperatureC",
        ],
        "tracks": tracks,
        "sampleFields": [
            "frame",
            "time",
            "xRawMm",
            "yRawMm",
            "zRawMm",
            "xMm",
            "yMm",
            "zMm",
            "orbitRmsMm",
            "episode",
            "temperatureC",
        ],
        "samples": samples,
        "episodes": episode_data,
        "uncertainty": uncertainty,
        "audit": audit,
    }

    audit_path = STUDY / "data" / "SP6_uncertainty_audit.json"
    audit_path.write_text(
        json.dumps({"uncertainty": uncertainty, "audit": audit}, indent=2) + "\n",
        encoding="utf-8",
    )
    js_path = STUDY / "orbit-data.js"
    js_path.write_text(
        "window.SP6_DATA=" + json.dumps(data, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({"data_js": str(js_path), "audit_json": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
