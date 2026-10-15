from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "code" / "simple_orbit_statistics.py"
SPEC = importlib.util.spec_from_file_location("simple_orbit_statistics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SETTINGS = {
    "smooth_samples": 5,
    "minimum_radius": 0.02,
    "minimum_turns": 1.0,
    "minimum_direction_consistency": 0.75,
    "circle_ratio": 0.80,
    "minimum_ellipse_ratio": 0.35,
    "maximum_ellipse_error": 0.30,
    "maximum_return_error": 0.55,
    "spiral_change": 0.20,
    "minimum_spiral_r2": 0.50,
}


def path(minor: float = 1.0, spiral: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261008)
    time_s = np.arange(0, 3, 0.01)
    phase = 2 * np.pi * 2 * time_s
    radius = 1 + spiral * (time_s / time_s[-1] - 0.5)
    xy = np.column_stack(
        (
            0.02 * time_s + radius * np.cos(phase),
            -0.01 * time_s + minor * radius * np.sin(phase),
        )
    )
    return time_s, xy + rng.normal(0, 0.01, xy.shape)


def test_circle_radius_and_period() -> None:
    time_s, xy = path()
    result = MODULE.window_statistics(time_s, xy, SETTINGS)
    assert result["classification"] == "circle"
    assert abs(result["equivalent_radius"] - 1.0) < 0.03
    assert abs(result["period_s"] - 0.5) < 0.01
    assert result["return_error"] < 0.10


def test_ellipse_line_and_spiral() -> None:
    time_s, xy = path(minor=0.6)
    assert MODULE.window_statistics(time_s, xy, SETTINGS)["classification"] == "ellipse"

    time_s, xy = path(minor=0.02)
    assert MODULE.window_statistics(time_s, xy, SETTINGS)["classification"] == "not_orbit"

    time_s, xy = path(spiral=0.7)
    assert MODULE.window_statistics(time_s, xy, SETTINGS)["classification"] == "spiral_out"


def test_noise_is_not_an_orbit() -> None:
    rng = np.random.default_rng(37)
    time_s = np.arange(0, 3, 0.01)
    xy = rng.normal(0, 0.01, (len(time_s), 2))
    assert MODULE.window_statistics(time_s, xy, SETTINGS)["classification"] == "not_orbit"
