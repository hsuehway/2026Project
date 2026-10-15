from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from orbit_statistics import (  # noqa: E402
    analyze_orbit_statistics,
    contiguous_intervals,
    episode_metrics,
    fit_moving_center_orbit,
    summary_stats,
)


class OrbitStatisticsTests(unittest.TestCase):
    def test_complete_statistics_pipeline_detects_synthetic_orbit(self) -> None:
        sample_rate_hz = 100.0
        time_s = np.arange(0, 6, 1 / sample_rate_hz)
        phase = 2 * np.pi * 2.0 * time_s
        coordinates = np.column_stack((np.cos(phase), np.sin(phase)))
        result = analyze_orbit_statistics(
            time_s,
            coordinates,
            sample_rate_hz,
            "mm",
            band_low_hz=1.0,
            band_high_hz=4.0,
            threshold_setting="0.4",
            minimum_episode_s=1.0,
            merge_gap_s=0.2,
            smooth_window_s=0.05,
            radius_window_s=0.5,
        )
        self.assertEqual(len(result.intervals), 1)
        self.assertEqual(len(result.episode_rows), 1)
        self.assertAlmostEqual(result.episode_rows[0]["frequency_hz"], 2.0, places=2)
        self.assertAlmostEqual(
            result.episode_rows[0]["circle_fit_radius_mm"], 1.0, places=2
        )

    def test_moving_center_fit_recovers_center_drift_and_radius(self) -> None:
        sample_rate_hz = 100.0
        time_s = np.arange(0, 4, 1 / sample_rate_hz)
        centered_time = time_s - np.mean(time_s)
        phase = 2 * np.pi * 2.0 * time_s
        radius_mm = 1.5
        xy = np.column_stack(
            (
                5.0 + 0.30 * centered_time + radius_mm * np.cos(phase),
                3.0 - 0.20 * centered_time + radius_mm * np.sin(phase),
            )
        )
        fit = fit_moving_center_orbit(
            time_s,
            xy,
            band_low_hz=1.0,
            band_high_hz=4.0,
            minimum_turns=1.5,
            minimum_axis_ratio=0.70,
            maximum_normalized_radial_rmse=0.25,
            minimum_direction_consistency=0.80,
        )
        self.assertTrue(fit["accepted"])
        self.assertAlmostEqual(fit["center_x"], 5.0, places=3)
        self.assertAlmostEqual(fit["center_y"], 3.0, places=3)
        self.assertAlmostEqual(fit["center_vx"], 0.30, places=3)
        self.assertAlmostEqual(fit["center_vy"], -0.20, places=3)
        self.assertAlmostEqual(fit["radius"], radius_mm, places=3)
        self.assertAlmostEqual(fit["frequency_hz"], 2.0, places=3)

    def test_moving_center_fit_rejects_back_and_forth_line(self) -> None:
        sample_rate_hz = 100.0
        time_s = np.arange(0, 4, 1 / sample_rate_hz)
        phase = 2 * np.pi * 2.0 * time_s
        xy = np.column_stack((1.5 * np.cos(phase), np.zeros_like(time_s)))
        fit = fit_moving_center_orbit(
            time_s,
            xy,
            band_low_hz=1.0,
            band_high_hz=4.0,
            minimum_turns=1.5,
            minimum_axis_ratio=0.70,
            maximum_normalized_radial_rmse=0.25,
            minimum_direction_consistency=0.80,
        )
        self.assertFalse(fit["accepted"])
        self.assertEqual(fit["status"], "insufficient angular coverage")
        self.assertLess(fit["axis_ratio"], 0.01)

    def test_rms_candidate_line_is_not_a_validated_orbit(self) -> None:
        sample_rate_hz = 100.0
        time_s = np.arange(0, 6, 1 / sample_rate_hz)
        phase = 2 * np.pi * 2.0 * time_s
        coordinates = np.column_stack((np.cos(phase), np.zeros_like(time_s)))
        result = analyze_orbit_statistics(
            time_s,
            coordinates,
            sample_rate_hz,
            "mm",
            band_low_hz=1.0,
            band_high_hz=4.0,
            threshold_setting="0.4",
            minimum_episode_s=1.0,
            merge_gap_s=0.2,
            smooth_window_s=0.05,
            radius_window_s=0.5,
        )
        self.assertEqual(len(result.candidate_intervals), 1)
        self.assertEqual(result.intervals, [])

    def test_contiguous_intervals_merges_short_gap(self) -> None:
        mask = np.zeros(100, dtype=bool)
        mask[10:30] = True
        mask[32:60] = True
        self.assertEqual(contiguous_intervals(mask, 10.0, 2.0, 0.4), [(10, 59)])

    def test_episode_metrics_recovers_circle_frequency_and_radius(self) -> None:
        sample_rate_hz = 100.0
        time_s = np.arange(0, 4, 1 / sample_rate_hz)
        frequency_hz = 2.0
        radius_mm = 1.5
        phase = 2 * np.pi * frequency_hz * time_s
        xy = np.column_stack((radius_mm * np.cos(phase), radius_mm * np.sin(phase)))
        metric = episode_metrics(
            1,
            0,
            len(time_s) - 1,
            time_s,
            xy,
            xy,
            np.full(len(time_s), radius_mm),
            "mm",
        )
        self.assertAlmostEqual(metric["frequency_hz"], frequency_hz, places=3)
        self.assertAlmostEqual(metric["period_s"], 1 / frequency_hz, places=3)
        self.assertAlmostEqual(metric["radius_median_mm"], radius_mm, places=6)

    def test_summary_stats_ignores_nonfinite_values(self) -> None:
        result = summary_stats(np.array([1.0, 2.0, 3.0, np.nan]))
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["standard_deviation"], 1.0)


if __name__ == "__main__":
    unittest.main()
