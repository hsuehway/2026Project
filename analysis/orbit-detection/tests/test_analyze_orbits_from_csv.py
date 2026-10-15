from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "code" / "analyze_orbits_from_csv.py"
SPEC = importlib.util.spec_from_file_location("analyze_orbits_from_csv", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_detects_two_synthetic_orbit_bursts(tmp_path: Path) -> None:
    sample_rate_hz = 100.0
    time_s = np.arange(0, 32, 1 / sample_rate_hz)
    envelope = np.zeros_like(time_s)
    envelope[(time_s >= 5) & (time_s <= 13)] = 1.8
    envelope[(time_s >= 19) & (time_s <= 28)] = 1.8
    phase = 2 * np.pi * 2.0 * time_s
    rng = np.random.default_rng(20261004)
    x = envelope * np.cos(phase) + rng.normal(0, 0.015, len(time_s))
    y = envelope * np.sin(phase) + rng.normal(0, 0.015, len(time_s))
    source = tmp_path / "synthetic.csv"
    pd.DataFrame({"time_s": time_s, "x_mm": x, "y_mm": y}).to_csv(source, index=False)
    output = tmp_path / "results"
    args = Namespace(
        csv=source,
        output_dir=output,
        label="synthetic",
        time_col=None,
        frame_col=None,
        fps=None,
        x_col=None,
        y_col=None,
        z_col=None,
        position_unit=None,
        band_low_hz=1.0,
        band_high_hz=4.0,
        threshold="0.7",
        minimum_episode_s=2.0,
        merge_gap_s=0.4,
        smooth_window_s=0.05,
        radius_window_s=0.5,
        max_gap_samples=2,
        no_html=False,
    )
    summary = MODULE.analyze(args)
    derived = pd.read_csv(output / "orbit_timeseries.csv")
    intervals = pd.read_csv(output / "orbit_intervals.csv")
    assert summary["candidate_episode_count"] == 2
    assert summary["episode_count"] == 2
    assert abs(summary["frequency_hz"]["mean"] - 2.0) < 0.03
    assert abs(summary["radius_mm"]["mean"] - 1.8) < 0.08
    assert abs(summary["circle_fit_radius_mm"]["mean"] - 1.8) < 0.08
    assert intervals["circle_fit_normalized_radial_rmse"].max() < 0.10
    assert "rolling_center_x_mm" in derived
    assert "rolling_circle_radius_mm" in derived
    assert "rolling_circle_fit_accepted" in derived
    assert (output / "orbit_intervals.csv").is_file()
    assert (output / "orbit_timeseries.csv").is_file()
    assert (output / "orbit_summary.json").is_file()
    assert (output / "orbit_report.html").is_file()


def test_requires_fps_when_only_frames_are_available(tmp_path: Path) -> None:
    source = tmp_path / "frames.csv"
    pd.DataFrame({"frame": range(30), "x_mm": range(30), "y_mm": range(30)}).to_csv(source, index=False)
    args = Namespace(
        csv=source,
        output_dir=tmp_path / "results",
        label=None,
        time_col=None,
        frame_col=None,
        fps=None,
        x_col=None,
        y_col=None,
        z_col=None,
        position_unit=None,
        band_low_hz=1.0,
        band_high_hz=4.0,
        threshold="auto",
        minimum_episode_s=2.0,
        merge_gap_s=0.75,
        smooth_window_s=0.1,
        radius_window_s=0.5,
        max_gap_samples=2,
        no_html=True,
    )
    try:
        MODULE.analyze(args)
    except SystemExit as error:
        assert "supply --fps" in str(error)
    else:
        raise AssertionError("Expected missing FPS to be rejected")


def test_tracks_3d_prefers_complete_paired_camera_measurements(tmp_path: Path) -> None:
    sample_rate_hz = 100.0
    frame = np.arange(1000)
    time_s = frame / sample_rate_hz
    phase = 2 * np.pi * 2.0 * time_s
    common = {"seed_id": 0, "particle": 0, "frame": frame, "time_s": time_s}
    front = pd.DataFrame(
        {
            **common,
            "front_image_x_px": 400 + np.cos(phase),
            "front_image_y_px": 100 + np.sin(phase),
            "front_confidence": 20.0,
            "front_peak_difference": 50.0,
        }
    )
    side = pd.DataFrame(
        {
            **common,
            "side_image_x_px": 440 + np.sin(phase),
            "side_image_y_px": 90 + np.sin(phase),
            "side_confidence": 30.0,
            "side_peak_difference": 100.0,
        }
    )
    front.to_csv(tmp_path / "front_background_tracks.csv", index=False)
    side.to_csv(tmp_path / "side_background_tracks.csv", index=False)
    filtered = frame % 10 != 0
    pd.DataFrame(
        {
            "frame": frame[filtered],
            "time_s": time_s[filtered],
            "front_image_x_raw_px": front["front_image_x_px"].to_numpy()[filtered],
            "side_image_x_raw_px": side["side_image_x_px"].to_numpy()[filtered],
        }
    ).to_csv(tmp_path / "tracks_3d.csv", index=False)
    args = Namespace(
        csv=tmp_path / "tracks_3d.csv",
        output_dir=tmp_path / "results",
        label="paired",
        time_col=None,
        frame_col=None,
        fps=sample_rate_hz,
        x_col=None,
        y_col=None,
        z_col=None,
        position_unit=None,
        band_low_hz=1.0,
        band_high_hz=4.0,
        threshold="0.2",
        minimum_episode_s=1.0,
        merge_gap_s=0.2,
        smooth_window_s=0.1,
        radius_window_s=0.5,
        max_gap_samples=2,
        no_html=False,
    )
    summary = MODULE.analyze(args)
    derived = pd.read_csv(args.output_dir / "orbit_timeseries.csv")
    assert summary["measurement_source_mode"] == "paired_camera_tracks"
    assert summary["audit"]["input_rows"] == 1000
    assert summary["audit"]["inserted_analysis_samples"] == 0
    assert summary["selected_columns"]["x"] == "front_image_x_raw_px"
    assert "front_image_x_direct_smooth_px" in derived
    assert "front_image_x_calibration_projected_smooth_px" not in derived


def test_no_orbit_still_writes_valid_empty_results(tmp_path: Path) -> None:
    sample_rate_hz = 50.0
    time_s = np.arange(0, 10, 1 / sample_rate_hz)
    source = tmp_path / "quiet.csv"
    pd.DataFrame(
        {
            "time_s": time_s,
            "x_mm": 0.002 * np.sin(2 * np.pi * 2 * time_s),
            "y_mm": 0.002 * np.cos(2 * np.pi * 2 * time_s),
        }
    ).to_csv(source, index=False)
    args = Namespace(
        csv=source,
        output_dir=tmp_path / "results",
        label="quiet",
        time_col=None,
        frame_col=None,
        fps=None,
        x_col=None,
        y_col=None,
        z_col=None,
        position_unit=None,
        band_low_hz=1.0,
        band_high_hz=4.0,
        threshold="0.1",
        minimum_episode_s=2.0,
        merge_gap_s=0.75,
        smooth_window_s=0.1,
        radius_window_s=0.5,
        max_gap_samples=2,
        no_html=False,
    )
    summary = MODULE.analyze(args)
    intervals = pd.read_csv(args.output_dir / "orbit_intervals.csv")
    assert summary["episode_count"] == 0
    assert intervals.empty
    assert "period_s" in intervals.columns
    assert (args.output_dir / "orbit_report.html").is_file()
