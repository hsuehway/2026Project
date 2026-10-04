"""Shared continuous-segment trajectory smoothing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


RAW_POSITION_COLUMNS = ("x_mm", "y_mm", "z_mm")
RAW_POSITION_ALIAS_COLUMNS = ("x_raw_mm", "y_raw_mm", "z_raw_mm")
SMOOTH_POSITION_COLUMNS = ("x_smooth_mm", "y_smooth_mm", "z_smooth_mm")
SMOOTH_WINDOW_COLUMN = "position_smooth_window"
SMOOTH_POLYORDER_COLUMN = "position_smooth_polyorder"


def validate_smoothing_parameters(smooth_window: int, smooth_polyorder: int) -> None:
    if smooth_window < 0:
        raise ValueError("smooth window must be zero or positive")
    if smooth_polyorder < 0:
        raise ValueError("smooth polynomial order must be zero or positive")
    if smooth_window not in (0, 1) and smooth_window % 2 == 0:
        raise ValueError("smooth window must be odd, or 0/1 to disable")
    if smooth_window > 1 and smooth_polyorder >= smooth_window:
        raise ValueError("smooth polynomial order must be smaller than the window")


def continuous_segment_slices(frame: np.ndarray) -> list[tuple[int, int]]:
    """Return slices that never cross a missing-frame gap."""
    if len(frame) == 0:
        return []
    split_points = np.flatnonzero(np.diff(frame) != 1) + 1
    starts = np.concatenate(([0], split_points))
    stops = np.concatenate((split_points, [len(frame)]))
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def effective_savgol_window(
    point_count: int,
    smooth_window: int,
    smooth_polyorder: int,
) -> int | None:
    if smooth_window <= 1 or point_count < 3:
        return None
    window = min(smooth_window, point_count)
    if window % 2 == 0:
        window -= 1
    if window <= smooth_polyorder or window < 3:
        return None
    return window


def smooth_positions(
    position: np.ndarray,
    smooth_window: int,
    smooth_polyorder: int,
) -> np.ndarray:
    """Smooth one uninterrupted Nx3 trajectory without changing its length."""
    window = effective_savgol_window(
        len(position),
        smooth_window,
        smooth_polyorder,
    )
    if window is None:
        return position.copy()

    try:
        from scipy.signal import savgol_filter
    except Exception:
        return position.copy()

    return savgol_filter(
        position,
        window_length=window,
        polyorder=smooth_polyorder,
        axis=0,
    )


def add_smoothed_track_positions(
    tracks: pd.DataFrame,
    smooth_window: int,
    smooth_polyorder: int,
    group_column: str = "particle",
) -> pd.DataFrame:
    """Add raw aliases and smoothed positions, splitting every track at gaps."""
    validate_smoothing_parameters(smooth_window, smooth_polyorder)
    required = (group_column, "frame", *RAW_POSITION_COLUMNS)
    missing = [column for column in required if column not in tracks]
    if missing:
        raise ValueError(f"tracks are missing smoothing columns: {missing}")

    result = tracks.copy()
    raw_position = result.loc[:, RAW_POSITION_COLUMNS].to_numpy(dtype=float)
    result.loc[:, RAW_POSITION_ALIAS_COLUMNS] = raw_position
    smoothed = np.full_like(raw_position, np.nan, dtype=float)

    for _, group in result.groupby(group_column, sort=False):
        ordered = group.sort_values("frame")
        ordered_indexes = ordered.index.to_numpy()
        frames = ordered["frame"].to_numpy(dtype=float)
        positions = ordered.loc[:, RAW_POSITION_COLUMNS].to_numpy(dtype=float)
        ordered_smoothed = np.full_like(positions, np.nan, dtype=float)
        for start, stop in continuous_segment_slices(frames):
            ordered_smoothed[start:stop] = smooth_positions(
                positions[start:stop],
                smooth_window,
                smooth_polyorder,
            )
        index_positions = result.index.get_indexer(ordered_indexes)
        smoothed[index_positions] = ordered_smoothed

    result.loc[:, SMOOTH_POSITION_COLUMNS] = smoothed
    result[SMOOTH_WINDOW_COLUMN] = int(smooth_window)
    result[SMOOTH_POLYORDER_COLUMN] = int(smooth_polyorder)
    return result
