"""Detect and review likely particle identity switches in a tracking CSV."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from background_subtraction_tracker import (
    CHANNELS,
    frame_to_gray,
    open_video,
    parse_frame_ranges,
    sanitize_output_name,
)
from track_orthogonal_3d import choose_video, read_calibration

REQUIRED_COLUMNS = ("particle", "frame", "x_mm", "y_mm", "z_mm")
REVIEW_MODES = ("manual", "report", "auto")
SIZE_CAMERAS = ("front", "side", "both")
SIZE_POLARITIES = ("bright", "dark", "abs")
MANUAL_CORRECTION_FLAG_COLUMNS = (
    "manual_correction_reviewed",
    "manual_correction_applied",
    "front_manual_correction_reviewed",
    "front_manual_correction_applied",
    "side_manual_correction_reviewed",
    "side_manual_correction_applied",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect likely particle ID switches from abnormal 3D jumps and "
            "optionally write a corrected tracking CSV."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="tracks_3d.csv or corrected tracking CSV. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--mode",
        choices=REVIEW_MODES,
        default="manual",
        help=(
            "manual shows each suspicious frame for review; report only writes "
            "candidates; auto applies all detected switch suggestions."
        ),
    )
    parser.add_argument(
        "--particles",
        type=parse_particle_pair,
        default=(0, 1),
        help="Two particle IDs to inspect as A,B. Default: 0,1.",
    )
    parser.add_argument(
        "--jump-threshold-mm",
        type=float,
        default=0.25,
        help="Minimum keep-ID jump cost required before a frame is suspicious.",
    )
    parser.add_argument(
        "--improvement-ratio",
        type=float,
        default=1.5,
        help="Require keep-ID cost / swap-ID cost to be at least this large.",
    )
    parser.add_argument(
        "--min-improvement-mm",
        type=float,
        default=0.05,
        help="Require swap-ID cost to improve the frame by at least this many mm.",
    )
    parser.add_argument(
        "--swap-scope",
        choices=("from-frame", "single-frame"),
        default="from-frame",
        help="In auto mode, apply each accepted swap from that frame onward or only that frame.",
    )
    parser.add_argument(
        "--review-frames",
        type=parse_frame_ranges,
        default=(),
        metavar="RANGES",
        help=(
            "Force manual/report review for frame range(s), even if the automatic "
            "jump detector does not flag them. Example: 3800-4300."
        ),
    )
    parser.add_argument(
        "--review-frame-step",
        type=int,
        default=1,
        help="When --review-frames is used, review every Nth complete frame.",
    )
    manual_group = parser.add_mutually_exclusive_group()
    manual_group.add_argument(
        "--review-manual-corrections",
        dest="review_manual_corrections",
        action="store_true",
        help=(
            "Add every frame marked as manually reviewed/corrected by the "
            "background tracker to the review queue. Default: enabled only "
            "when using the default manual-corrections-only candidate mode."
        ),
    )
    manual_group.add_argument(
        "--no-review-manual-corrections",
        dest="review_manual_corrections",
        action="store_false",
        help="Do not add tracker manual-correction frames to the review queue.",
    )
    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument(
        "--manual-corrections-only",
        dest="manual_corrections_only",
        action="store_true",
        default=True,
        help=(
            "Only review frames marked by tracker manual correction flags. "
            "Default: enabled."
        ),
    )
    candidate_group.add_argument(
        "--include-auto-candidates",
        dest="manual_corrections_only",
        action="store_false",
        help=(
            "Include automatic jump, forced-range, and size-signature candidates "
            "instead of the default manual-correction-only review. Add "
            "--review-manual-corrections if you also want tracker manual "
            "correction frames in the queue."
        ),
    )
    parser.add_argument(
        "--size-signature",
        action="store_true",
        help=(
            "Also flag possible ID switches from apparent particle radius. This "
            "opens the video, asks you to click the particle IDs on a calibration "
            "frame, and adds size-based review candidates."
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Original combined tracking video for --size-signature. If omitted, "
            "File Explorer opens."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Calibration JSON containing front_roi and side_roi.",
    )
    parser.add_argument(
        "--size-camera",
        choices=SIZE_CAMERAS,
        default="both",
        help="Camera view(s) used for apparent-radius signatures. Default: both.",
    )
    parser.add_argument(
        "--size-channel",
        choices=CHANNELS,
        default="green",
        help="Video color channel used for size signatures. Default: green.",
    )
    parser.add_argument(
        "--size-polarity",
        choices=SIZE_POLARITIES,
        default="bright",
        help="Whether particles are bright, dark, or either relative to the local patch.",
    )
    parser.add_argument(
        "--size-calibration-frame",
        type=int,
        help=(
            "Frame where you will click particle IDs for size calibration. If "
            "omitted, the script prompts with the first tracked frame as default."
        ),
    )
    parser.add_argument(
        "--size-search-radius-px",
        type=int,
        default=18,
        help="Patch radius around each clicked/tracked point for apparent-radius sizing.",
    )
    parser.add_argument(
        "--size-threshold-fraction",
        type=float,
        default=0.35,
        help="Fraction of local contrast used to segment the particle blob.",
    )
    parser.add_argument(
        "--size-min-area-px",
        type=int,
        default=2,
        help="Ignore apparent-size blobs smaller than this area.",
    )
    parser.add_argument(
        "--size-min-improvement-px",
        type=float,
        default=0.75,
        help="Minimum radius-cost improvement, in pixels, needed for a size candidate.",
    )
    parser.add_argument(
        "--size-improvement-ratio",
        type=float,
        default=1.25,
        help="Minimum keep/swap radius-cost ratio needed for a size candidate.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional name for the correction output folder.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic PNG plots.",
    )
    parser.set_defaults(review_manual_corrections=None)
    args = parser.parse_args()
    if args.review_manual_corrections is None:
        args.review_manual_corrections = bool(args.manual_corrections_only)
    return args


def parse_particle_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use exactly two particle IDs as A,B.")
    try:
        first = int(parts[0])
        second = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Particle IDs must be integers.") from exc
    if first == second:
        raise argparse.ArgumentTypeError("Particle IDs must be different.")
    return first, second


def main() -> int:
    args = parse_args()
    validate_args(args)

    csv_path = args.csv or choose_csv()
    tracks = pd.read_csv(csv_path)
    validate_tracks(tracks, csv_path)
    if args.review_manual_corrections and not manual_correction_flag_columns(tracks):
        message = (
            "No manual correction flag columns were found in this CSV. "
            "Rerun background_subtraction_tracker.py after this update to create "
            "manual-correction flags, or use --review-frames for older tracks."
        )
        if args.manual_corrections_only:
            raise SystemExit(message)
        print(f"Warning: {message}")

    particle_a, particle_b = args.particles
    size_signature: dict[str, object] | None = None
    size_video_path: Path | None = None
    if args.size_signature:
        size_video_path = args.video or choose_video(
            "Choose original combined video for particle size signatures"
        )
        tracks, size_signature = add_size_signature_observations(
            tracks=tracks,
            video_path=size_video_path,
            particle_a=particle_a,
            particle_b=particle_b,
            args=args,
        )

    candidates = detect_switch_candidates(
        tracks,
        particle_a=particle_a,
        particle_b=particle_b,
        jump_threshold_mm=args.jump_threshold_mm,
        improvement_ratio=args.improvement_ratio,
        min_improvement_mm=args.min_improvement_mm,
        review_frames=args.review_frames,
        review_frame_step=args.review_frame_step,
        review_manual_corrections=args.review_manual_corrections,
        manual_corrections_only=args.manual_corrections_only,
        size_signature=size_signature,
    )

    corrected, actions = review_or_apply_candidates(
        tracks,
        candidates,
        particle_a=particle_a,
        particle_b=particle_b,
        mode=args.mode,
        swap_scope=args.swap_scope,
        thresholds={
            "jump_threshold_mm": args.jump_threshold_mm,
            "improvement_ratio": args.improvement_ratio,
            "min_improvement_mm": args.min_improvement_mm,
            "review_frames": args.review_frames,
            "review_frame_step": args.review_frame_step,
            "review_manual_corrections": args.review_manual_corrections,
            "manual_corrections_only": args.manual_corrections_only,
            "size_signature": size_signature,
        },
    )

    output_dir = make_correction_output_dir(csv_path, args.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_stem = prefix_with_trial_index(
        f"{csv_path.stem}_identity_corrected",
        infer_trial_index(csv_path),
    )
    corrected_path = output_dir / f"{corrected_stem}.csv"
    candidates_path = output_dir / "identity_switch_candidates.csv"
    actions_path = output_dir / "identity_review_actions.csv"
    log_path = output_dir / "identity_correction_log.json"

    corrected.to_csv(corrected_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    pd.DataFrame(actions).to_csv(actions_path, index=False)
    write_log(
        log_path,
        csv_path,
        corrected_path,
        args,
        candidates,
        actions,
        size_video_path=size_video_path,
        size_signature=size_signature,
    )
    if not args.no_plots:
        plot_candidates(candidates, output_dir / "identity_switch_diagnostic_plot.png")

    print("\nDone.")
    print(f"Input CSV:      {csv_path.resolve()}")
    print(f"Candidates:     {len(candidates)}")
    print(f"Actions:        {len(actions)}")
    print(f"Corrected CSV:  {corrected_path.resolve()}")
    print(f"Output folder:  {output_dir.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not args.review_manual_corrections:
        args.manual_corrections_only = False
    if args.jump_threshold_mm < 0:
        raise SystemExit("--jump-threshold-mm must be non-negative.")
    if args.improvement_ratio < 1:
        raise SystemExit("--improvement-ratio must be at least 1.")
    if args.min_improvement_mm < 0:
        raise SystemExit("--min-improvement-mm must be non-negative.")
    if args.review_frame_step <= 0:
        raise SystemExit("--review-frame-step must be positive.")
    if args.size_calibration_frame is not None and args.size_calibration_frame < 0:
        raise SystemExit("--size-calibration-frame must be zero or positive.")
    if args.size_search_radius_px <= 0:
        raise SystemExit("--size-search-radius-px must be positive.")
    if not 0 < args.size_threshold_fraction <= 1:
        raise SystemExit("--size-threshold-fraction must be between 0 and 1.")
    if args.size_min_area_px <= 0:
        raise SystemExit("--size-min-area-px must be positive.")
    if args.size_min_improvement_px < 0:
        raise SystemExit("--size-min-improvement-px must be non-negative.")
    if args.size_improvement_ratio < 1:
        raise SystemExit("--size-improvement-ratio must be at least 1.")


def choose_csv() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(f"Pass --csv because File Explorer is unavailable ({exc}).") from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose tracks_3d.csv to inspect for ID switches",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No CSV selected.")
    return Path(path)


def validate_tracks(tracks: pd.DataFrame, csv_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in tracks.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {missing}")
    if tracks.empty:
        raise SystemExit(f"{csv_path} is empty.")


def detect_switch_candidates(
    tracks: pd.DataFrame,
    particle_a: int,
    particle_b: int,
    jump_threshold_mm: float,
    improvement_ratio: float,
    min_improvement_mm: float,
    review_frames: tuple[tuple[int, int], ...] = (),
    review_frame_step: int = 1,
    review_manual_corrections: bool = False,
    manual_corrections_only: bool = False,
    size_signature: dict[str, object] | None = None,
) -> pd.DataFrame:
    pair = tracks.loc[tracks["particle"].astype(int).isin([particle_a, particle_b])]
    rows: list[dict[str, float | int | str]] = []
    frames = complete_frames_for_pair(pair, particle_a, particle_b)

    if not manual_corrections_only:
        for previous_frame, current_frame in zip(frames[:-1], frames[1:]):
            row = switch_candidate_metrics(
                pair,
                previous_frame,
                current_frame,
                particle_a,
                particle_b,
            )
            if row is None:
                continue
            keep_cost = float(row["keep_cost_mm"])
            improvement = float(row["improvement_mm"])
            ratio = float(row["improvement_ratio"])
            suspicious = (
                keep_cost >= jump_threshold_mm
                and improvement >= min_improvement_mm
                and ratio >= improvement_ratio
            )
            if suspicious:
                row["suggested_action"] = "swap"
                row["is_suspicious"] = True
                row["candidate_source"] = "auto_jump"
                rows.append(row)

        forced = build_forced_review_candidates(
            pair,
            frames,
            particle_a,
            particle_b,
            review_frames,
            review_frame_step,
        )
        rows.extend(forced)
        if size_signature is not None:
            rows.extend(
                build_size_signature_candidates(
                    pair,
                    frames,
                    particle_a,
                    particle_b,
                    size_signature,
                )
            )

    if review_manual_corrections:
        rows.extend(
            build_manual_correction_review_candidates(
                pair,
                frames,
                particle_a,
                particle_b,
            )
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates["source_priority"] = candidates["candidate_source"].map(
        {
            "manual_correction": 0,
            "auto_jump": 1,
            "size_signature": 2,
            "manual_range": 3,
        }
    ).fillna(9)
    return candidates.sort_values(
        ["switch_frame", "source_priority", "improvement_ratio"],
        ascending=[True, True, False],
    ).drop_duplicates("switch_frame", keep="first").drop(
        columns=["source_priority"],
    ).reset_index(drop=True)


def complete_frames_for_pair(
    pair: pd.DataFrame,
    particle_a: int,
    particle_b: int,
) -> list[int]:
    frames: list[int] = []
    for frame, group in pair.groupby(pair["frame"].astype(int), sort=True):
        particles = set(group["particle"].astype(int))
        if particle_a in particles and particle_b in particles:
            frames.append(int(frame))
    return frames


def switch_candidate_metrics(
    tracks: pd.DataFrame,
    previous_frame: int,
    current_frame: int,
    particle_a: int,
    particle_b: int,
) -> dict[str, float | int | str | bool] | None:
    previous = positions_for_frame(tracks, previous_frame, particle_a, particle_b)
    current = positions_for_frame(tracks, current_frame, particle_a, particle_b)
    if previous is None or current is None:
        return None

    previous_a, previous_b = previous
    current_a, current_b = current
    a_keep_jump = distance(previous_a, current_a)
    b_keep_jump = distance(previous_b, current_b)
    a_swap_jump = distance(previous_a, current_b)
    b_swap_jump = distance(previous_b, current_a)
    keep_cost = a_keep_jump + b_keep_jump
    swap_cost = a_swap_jump + b_swap_jump
    improvement = keep_cost - swap_cost
    ratio = keep_cost / max(swap_cost, 1e-12)
    return {
        "previous_frame": int(previous_frame),
        "switch_frame": int(current_frame),
        "frame_gap": int(current_frame) - int(previous_frame),
        "particle_a": int(particle_a),
        "particle_b": int(particle_b),
        "a_keep_jump_mm": a_keep_jump,
        "b_keep_jump_mm": b_keep_jump,
        "a_swap_jump_mm": a_swap_jump,
        "b_swap_jump_mm": b_swap_jump,
        "keep_cost_mm": keep_cost,
        "swap_cost_mm": swap_cost,
        "improvement_mm": improvement,
        "improvement_ratio": ratio,
        "suggested_action": "keep",
        "is_suspicious": False,
        "candidate_source": "none",
        "cost_units": "mm",
    }


def build_forced_review_candidates(
    tracks: pd.DataFrame,
    complete_frames: list[int],
    particle_a: int,
    particle_b: int,
    review_frames: tuple[tuple[int, int], ...],
    review_frame_step: int,
) -> list[dict[str, float | int | str | bool]]:
    if not review_frames:
        return []
    rows: list[dict[str, float | int | str | bool]] = []
    review_index = 0
    for previous_frame, current_frame in zip(complete_frames[:-1], complete_frames[1:]):
        if not frame_in_ranges(current_frame, review_frames):
            continue
        if review_index % review_frame_step != 0:
            review_index += 1
            continue
        review_index += 1
        row = switch_candidate_metrics(
            tracks,
            previous_frame,
            current_frame,
            particle_a,
            particle_b,
        )
        if row is None:
            continue
        row["suggested_action"] = "manual_review"
        row["is_suspicious"] = True
        row["candidate_source"] = "manual_range"
        rows.append(row)
    return rows


def build_manual_correction_review_candidates(
    tracks: pd.DataFrame,
    complete_frames: list[int],
    particle_a: int,
    particle_b: int,
) -> list[dict[str, float | int | str | bool]]:
    flag_columns = manual_correction_flag_columns(tracks)
    if not flag_columns:
        return []

    frame_flags = manual_correction_frame_flags(tracks, flag_columns)
    previous_by_frame = {
        int(current_frame): int(previous_frame)
        for previous_frame, current_frame in zip(complete_frames[:-1], complete_frames[1:])
    }
    complete_frame_set = set(int(frame) for frame in complete_frames)
    rows: list[dict[str, float | int | str | bool]] = []

    for current_frame in sorted(frame_flags):
        if current_frame not in complete_frame_set:
            continue
        previous_frame = previous_by_frame.get(current_frame)
        if previous_frame is None:
            continue
        row = switch_candidate_metrics(
            tracks,
            previous_frame,
            current_frame,
            particle_a,
            particle_b,
        )
        if row is None:
            continue
        row["suggested_action"] = "manual_review"
        row["is_suspicious"] = True
        row["candidate_source"] = "manual_correction"
        row["manual_correction_flag_columns"] = ",".join(flag_columns)
        row["manual_correction_cameras"] = manual_correction_cameras_for_frame(
            tracks,
            current_frame,
        )
        rows.append(row)
    return rows


def manual_correction_flag_columns(tracks: pd.DataFrame) -> list[str]:
    return [
        column
        for column in MANUAL_CORRECTION_FLAG_COLUMNS
        if column in tracks.columns
    ]


def manual_correction_frame_flags(
    tracks: pd.DataFrame,
    flag_columns: list[str],
) -> set[int]:
    if not flag_columns:
        return set()

    any_flag = pd.Series(False, index=tracks.index)
    for column in flag_columns:
        any_flag = any_flag | truthy_values(tracks[column])
    flagged = tracks.loc[any_flag, "frame"].astype(int)
    return set(int(frame) for frame in flagged)


def truthy_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.fillna("").astype(str).str.lower().isin(
        {"true", "1", "yes", "y", "reviewed", "applied"}
    )


def manual_correction_cameras_for_frame(tracks: pd.DataFrame, frame: int) -> str:
    frame_rows = tracks.loc[tracks["frame"].astype(int) == int(frame)]
    cameras: list[str] = []
    for camera in ("front", "side"):
        reviewed_col = f"{camera}_manual_correction_reviewed"
        applied_col = f"{camera}_manual_correction_applied"
        camera_flag = False
        for column in (reviewed_col, applied_col):
            if column in frame_rows.columns and truthy_values(frame_rows[column]).any():
                camera_flag = True
        if camera_flag:
            cameras.append(camera)
    if cameras:
        return ",".join(cameras)
    if "manual_correction_cameras" not in frame_rows.columns:
        return ""
    values = [
        str(value)
        for value in frame_rows["manual_correction_cameras"].dropna().unique()
        if str(value)
    ]
    return ",".join(sorted(set(",".join(values).split(",")))) if values else ""


def add_size_signature_observations(
    tracks: pd.DataFrame,
    video_path: Path,
    particle_a: int,
    particle_b: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    (
        _geometry,
        _front_cal,
        _side_cal,
        front_roi,
        side_roi,
        _front_focus_roi,
        _side_focus_roi,
    ) = read_calibration(args.calibration)
    cameras = cameras_from_size_choice(args.size_camera)
    for camera_name in cameras:
        roi_for_camera(camera_name, front_roi, side_roi, args.calibration)
        x_column = f"{camera_name}_image_x_px"
        y_column = f"{camera_name}_image_y_px"
        missing = [column for column in (x_column, y_column) if column not in tracks.columns]
        if missing:
            raise SystemExit(
                f"{Path(args.csv).name if args.csv else 'Selected CSV'} is missing "
                f"required {camera_name} size-signature columns: {missing}"
            )

    calibration_frame = choose_size_calibration_frame(tracks, args)
    video = open_video(video_path)
    try:
        calibration_image = np.asarray(video[calibration_frame])
        expected_radius_px: dict[str, dict[int, float]] = {}
        click_points_px: dict[str, dict[int, tuple[float, float]]] = {}
        for camera_name in cameras:
            roi = roi_for_camera(camera_name, front_roi, side_roi, args.calibration)
            gray = frame_to_gray(calibration_image, args.size_channel, roi)
            points = click_size_signature_points(
                gray,
                camera_name,
                calibration_frame,
                (particle_a, particle_b),
            )
            click_points_px[camera_name] = points
            expected_radius_px[camera_name] = {}
            for particle, (x_px, y_px) in points.items():
                measurement = measure_apparent_particle_size(
                    gray,
                    x_px,
                    y_px,
                    search_radius_px=args.size_search_radius_px,
                    threshold_fraction=args.size_threshold_fraction,
                    min_area_px=args.size_min_area_px,
                    polarity=args.size_polarity,
                )
                if not measurement["valid"]:
                    raise SystemExit(
                        f"Could not measure particle {particle} in the {camera_name} "
                        f"view on frame {calibration_frame}. Try a clearer "
                        "--size-calibration-frame or a larger --size-search-radius-px."
                    )
                expected_radius_px[camera_name][particle] = float(measurement["radius_px"])

        corrected = tracks.copy()
        for camera_name in cameras:
            corrected[f"{camera_name}_size_radius_px"] = np.nan
            corrected[f"{camera_name}_size_area_px"] = np.nan
            corrected[f"{camera_name}_size_centroid_x_px"] = np.nan
            corrected[f"{camera_name}_size_centroid_y_px"] = np.nan

        pair_mask = corrected["particle"].astype(int).isin([particle_a, particle_b])
        frame_numbers = sorted(corrected.loc[pair_mask, "frame"].astype(int).unique())
        if not frame_numbers:
            raise SystemExit("No selected particles were found in the tracking CSV.")

        print(
            f"Measuring apparent radius for {len(frame_numbers)} tracked frames "
            f"from {video_path.name}..."
        )
        for index, frame_number in enumerate(frame_numbers, start=1):
            if index == 1 or index % 500 == 0 or index == len(frame_numbers):
                print(f"  size frame {index}/{len(frame_numbers)}: {frame_number}")
            try:
                frame_image = np.asarray(video[int(frame_number)])
            except Exception:
                continue
            for camera_name in cameras:
                roi = roi_for_camera(camera_name, front_roi, side_roi, args.calibration)
                gray = frame_to_gray(frame_image, args.size_channel, roi)
                x_column = f"{camera_name}_image_x_px"
                y_column = f"{camera_name}_image_y_px"
                if x_column not in corrected.columns or y_column not in corrected.columns:
                    continue
                row_mask = pair_mask & (corrected["frame"].astype(int) == int(frame_number))
                for row_index, row in corrected.loc[row_mask].iterrows():
                    x_px = float(row[x_column])
                    y_px = float(row[y_column])
                    if not np.isfinite(x_px) or not np.isfinite(y_px):
                        continue
                    measurement = measure_apparent_particle_size(
                        gray,
                        x_px,
                        y_px,
                        search_radius_px=args.size_search_radius_px,
                        threshold_fraction=args.size_threshold_fraction,
                        min_area_px=args.size_min_area_px,
                        polarity=args.size_polarity,
                    )
                    if not measurement["valid"]:
                        continue
                    corrected.at[row_index, f"{camera_name}_size_radius_px"] = float(
                        measurement["radius_px"]
                    )
                    corrected.at[row_index, f"{camera_name}_size_area_px"] = float(
                        measurement["area_px"]
                    )
                    corrected.at[row_index, f"{camera_name}_size_centroid_x_px"] = float(
                        measurement["centroid_x_px"]
                    )
                    corrected.at[row_index, f"{camera_name}_size_centroid_y_px"] = float(
                        measurement["centroid_y_px"]
                    )
    finally:
        close_video(video)

    signature = {
        "enabled": True,
        "video": str(video_path.resolve()),
        "calibration_frame": int(calibration_frame),
        "cameras": cameras,
        "channel": args.size_channel,
        "polarity": args.size_polarity,
        "search_radius_px": int(args.size_search_radius_px),
        "threshold_fraction": float(args.size_threshold_fraction),
        "min_area_px": int(args.size_min_area_px),
        "min_improvement_px": float(args.size_min_improvement_px),
        "improvement_ratio": float(args.size_improvement_ratio),
        "expected_radius_px": expected_radius_px,
        "click_points_px": {
            camera: {
                int(particle): [float(point[0]), float(point[1])]
                for particle, point in points.items()
            }
            for camera, points in click_points_px.items()
        },
    }
    return corrected, signature


def cameras_from_size_choice(choice: str) -> tuple[str, ...]:
    if choice == "both":
        return ("front", "side")
    return (choice,)


def roi_for_camera(
    camera_name: str,
    front_roi,
    side_roi,
    calibration_path: Path,
):
    roi = front_roi if camera_name == "front" else side_roi
    if roi is None:
        raise SystemExit(
            f"{calibration_path} must contain {camera_name}_roi for size signatures."
        )
    return roi


def choose_size_calibration_frame(tracks: pd.DataFrame, args: argparse.Namespace) -> int:
    if args.size_calibration_frame is not None:
        return int(args.size_calibration_frame)
    first_frame = int(tracks["frame"].astype(int).min())
    try:
        value = input(
            f"Frame for size-signature clicks [{first_frame}]: "
        ).strip()
    except EOFError:
        value = ""
    if not value:
        return first_frame
    try:
        frame = int(value)
    except ValueError as exc:
        raise SystemExit("Size-signature calibration frame must be an integer.") from exc
    if frame < 0:
        raise SystemExit("Size-signature calibration frame must be zero or positive.")
    return frame


def click_size_signature_points(
    gray: np.ndarray,
    camera_name: str,
    frame_number: int,
    particles: tuple[int, int],
) -> dict[int, tuple[float, float]]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(gray, cmap="gray")
    ax.set_title(
        f"{camera_name} frame {frame_number}: click particle {particles[0]}, "
        f"then particle {particles[1]}"
    )
    ax.set_xlabel("x px")
    ax.set_ylabel("y px")
    points = plt.ginput(len(particles), timeout=0)
    plt.close(fig)
    if len(points) != len(particles):
        raise SystemExit("Size-signature calibration was cancelled before all clicks.")
    return {
        int(particle): (float(point[0]), float(point[1]))
        for particle, point in zip(particles, points)
    }


def measure_apparent_particle_size(
    gray: np.ndarray,
    x_px: float,
    y_px: float,
    search_radius_px: int,
    threshold_fraction: float,
    min_area_px: int,
    polarity: str,
) -> dict[str, float | bool]:
    height, width = gray.shape[:2]
    x_center = int(round(x_px))
    y_center = int(round(y_px))
    x0 = max(0, x_center - search_radius_px)
    x1 = min(width, x_center + search_radius_px + 1)
    y0 = max(0, y_center - search_radius_px)
    y1 = min(height, y_center + search_radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        return {"valid": False}

    patch = gray[y0:y1, x0:x1].astype(float)
    if patch.size == 0:
        return {"valid": False}
    if polarity == "bright":
        baseline = float(np.percentile(patch, 20))
        signal = patch - baseline
    elif polarity == "dark":
        baseline = float(np.percentile(patch, 80))
        signal = baseline - patch
    else:
        baseline = float(np.median(patch))
        signal = np.abs(patch - baseline)
    signal = np.clip(signal, 0, None)
    peak = float(np.max(signal))
    if peak <= 0:
        return {"valid": False}

    mask = signal >= (peak * threshold_fraction)
    try:
        from scipy import ndimage
    except Exception as exc:
        raise SystemExit("scipy is required for size-signature measurements.") from exc

    labels, count = ndimage.label(mask)
    if count == 0:
        return {"valid": False}

    center = np.array([float(y_px - y0), float(x_px - x0)])
    best: dict[str, float | bool] | None = None
    best_score = float("inf")
    for label_id in range(1, count + 1):
        coords = np.argwhere(labels == label_id)
        area_px = int(len(coords))
        if area_px < min_area_px:
            continue
        centroid_y, centroid_x = coords.mean(axis=0)
        distance_px = float(np.linalg.norm(np.array([centroid_y, centroid_x]) - center))
        score = distance_px - area_px * 1e-4
        if score >= best_score:
            continue
        radius_px = math.sqrt(area_px / math.pi)
        best_score = score
        best = {
            "valid": True,
            "area_px": float(area_px),
            "radius_px": float(radius_px),
            "diameter_px": float(radius_px * 2.0),
            "centroid_x_px": float(centroid_x + x0),
            "centroid_y_px": float(centroid_y + y0),
            "distance_from_track_px": distance_px,
            "local_peak": peak,
        }
    return best or {"valid": False}


def close_video(video) -> None:
    close = getattr(video, "close", None)
    if callable(close):
        close()


def build_size_signature_candidates(
    tracks: pd.DataFrame,
    complete_frames: list[int],
    particle_a: int,
    particle_b: int,
    size_signature: dict[str, object],
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for previous_frame, current_frame in zip(complete_frames[:-1], complete_frames[1:]):
        costs = size_signature_costs_for_frame(
            tracks,
            current_frame,
            particle_a,
            particle_b,
            size_signature,
        )
        if costs is None:
            continue
        keep_cost = float(costs["keep_cost_px"])
        swap_cost = float(costs["swap_cost_px"])
        improvement = keep_cost - swap_cost
        ratio = keep_cost / max(swap_cost, 1e-12)
        if (
            improvement < float(size_signature["min_improvement_px"])
            or ratio < float(size_signature["improvement_ratio"])
        ):
            continue
        row = switch_candidate_metrics(
            tracks,
            previous_frame,
            current_frame,
            particle_a,
            particle_b,
        )
        if row is None:
            continue
        row["jump_keep_cost_mm"] = float(row["keep_cost_mm"])
        row["jump_swap_cost_mm"] = float(row["swap_cost_mm"])
        row["keep_cost_mm"] = keep_cost
        row["swap_cost_mm"] = swap_cost
        row["improvement_mm"] = improvement
        row["improvement_ratio"] = ratio
        row["suggested_action"] = "swap"
        row["is_suspicious"] = True
        row["candidate_source"] = "size_signature"
        row["cost_units"] = "radius_px"
        row.update(costs)
        rows.append(row)
    return rows


def size_signature_costs_for_frame(
    tracks: pd.DataFrame,
    frame_number: int,
    particle_a: int,
    particle_b: int,
    size_signature: dict[str, object],
) -> dict[str, float | int | str] | None:
    expected = size_signature["expected_radius_px"]
    cameras = tuple(size_signature["cameras"])
    frame_tracks = tracks.loc[tracks["frame"].astype(int) == int(frame_number)]
    rows_a = frame_tracks.loc[frame_tracks["particle"].astype(int) == particle_a]
    rows_b = frame_tracks.loc[frame_tracks["particle"].astype(int) == particle_b]
    if rows_a.empty or rows_b.empty:
        return None
    row_a = rows_a.sort_index().iloc[0]
    row_b = rows_b.sort_index().iloc[0]

    keep_cost = 0.0
    swap_cost = 0.0
    used_cameras: list[str] = []
    details: dict[str, float | int | str] = {}
    for camera_name in cameras:
        radius_column = f"{camera_name}_size_radius_px"
        if radius_column not in tracks.columns:
            continue
        expected_camera = expected.get(camera_name, {})
        expected_a = expected_camera.get(particle_a)
        expected_b = expected_camera.get(particle_b)
        observed_a = float(row_a.get(radius_column, np.nan))
        observed_b = float(row_b.get(radius_column, np.nan))
        if not all_finite(expected_a, expected_b, observed_a, observed_b):
            continue
        camera_keep = abs(observed_a - float(expected_a)) + abs(
            observed_b - float(expected_b)
        )
        camera_swap = abs(observed_a - float(expected_b)) + abs(
            observed_b - float(expected_a)
        )
        keep_cost += camera_keep
        swap_cost += camera_swap
        used_cameras.append(camera_name)
        details[f"{camera_name}_particle_{particle_a}_radius_px"] = observed_a
        details[f"{camera_name}_particle_{particle_b}_radius_px"] = observed_b
        details[f"{camera_name}_expected_particle_{particle_a}_radius_px"] = float(
            expected_a
        )
        details[f"{camera_name}_expected_particle_{particle_b}_radius_px"] = float(
            expected_b
        )
        details[f"{camera_name}_size_keep_cost_px"] = float(camera_keep)
        details[f"{camera_name}_size_swap_cost_px"] = float(camera_swap)

    if not used_cameras:
        return None
    details["size_cameras"] = ",".join(used_cameras)
    details["keep_cost_px"] = float(keep_cost)
    details["swap_cost_px"] = float(swap_cost)
    details["size_keep_cost_px"] = float(keep_cost)
    details["size_swap_cost_px"] = float(swap_cost)
    details["size_improvement_px"] = float(keep_cost - swap_cost)
    return details


def all_finite(*values: object) -> bool:
    for value in values:
        try:
            if not np.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def frame_in_ranges(
    frame_number: int,
    ranges: tuple[tuple[int, int], ...],
) -> bool:
    return any(start <= frame_number <= end for start, end in ranges)


def positions_for_frame(
    tracks: pd.DataFrame,
    frame: int,
    particle_a: int,
    particle_b: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    frame_tracks = tracks.loc[tracks["frame"].astype(int) == int(frame)]
    rows_a = frame_tracks.loc[frame_tracks["particle"].astype(int) == particle_a]
    rows_b = frame_tracks.loc[frame_tracks["particle"].astype(int) == particle_b]
    if rows_a.empty or rows_b.empty:
        return None
    point_a = rows_a.sort_index().iloc[0][["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    point_b = rows_b.sort_index().iloc[0][["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    return point_a, point_b


def distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def review_or_apply_candidates(
    tracks: pd.DataFrame,
    initial_candidates: pd.DataFrame,
    particle_a: int,
    particle_b: int,
    mode: Literal["manual", "report", "auto"],
    swap_scope: Literal["from-frame", "single-frame"],
    thresholds: dict[str, object],
) -> tuple[pd.DataFrame, list[dict[str, float | int | str | bool]]]:
    corrected = tracks.copy()
    actions: list[dict[str, float | int | str | bool]] = []
    if mode == "report" or initial_candidates.empty:
        return add_correction_metadata(corrected, actions), actions

    ignored: set[int] = set()
    while True:
        candidates = detect_switch_candidates(
            corrected,
            particle_a=particle_a,
            particle_b=particle_b,
            jump_threshold_mm=thresholds["jump_threshold_mm"],
            improvement_ratio=thresholds["improvement_ratio"],
            min_improvement_mm=thresholds["min_improvement_mm"],
            review_frames=thresholds["review_frames"],
            review_frame_step=int(thresholds["review_frame_step"]),
            review_manual_corrections=bool(thresholds.get("review_manual_corrections", False)),
            manual_corrections_only=bool(thresholds.get("manual_corrections_only", False)),
            size_signature=thresholds.get("size_signature"),
        )
        if candidates.empty:
            break
        candidates = candidates.loc[~candidates["switch_frame"].astype(int).isin(ignored)]
        if candidates.empty:
            break
        candidate = candidates.iloc[0]
        frame = int(candidate["switch_frame"])

        if mode == "auto":
            decision = "swap_from_here" if swap_scope == "from-frame" else "swap_this_frame"
        else:
            decision = review_candidate_window(corrected, candidate, particle_a, particle_b)

        action = {
            "switch_frame": frame,
            "previous_frame": int(candidate["previous_frame"]),
            "particle_a": int(particle_a),
            "particle_b": int(particle_b),
            "decision": decision,
            "keep_cost_mm": float(candidate["keep_cost_mm"]),
            "swap_cost_mm": float(candidate["swap_cost_mm"]),
            "improvement_ratio": float(candidate["improvement_ratio"]),
            "frame_gap": int(candidate.get("frame_gap", 1)),
            "candidate_source": str(candidate.get("candidate_source", "")),
            "manual_correction_cameras": str(
                candidate.get("manual_correction_cameras", "")
            ),
            "manual_correction_flag_columns": str(
                candidate.get("manual_correction_flag_columns", "")
            ),
            "cost_units": str(candidate.get("cost_units", "mm")),
        }

        if decision == "stop":
            actions.append(action)
            break
        if decision == "keep":
            ignored.add(frame)
            actions.append(action)
            continue
        if decision == "swap_this_frame":
            corrected = swap_particle_ids_for_frames(
                corrected,
                [frame],
                particle_a,
                particle_b,
                action_label="manual_single_frame",
            )
            ignored.add(frame)
            actions.append(action)
            continue
        if decision == "swap_from_here":
            max_frame = int(corrected["frame"].max())
            frames = list(range(frame, max_frame + 1))
            corrected = swap_particle_ids_for_frames(
                corrected,
                frames,
                particle_a,
                particle_b,
                action_label="manual_from_frame",
            )
            ignored.add(frame)
            actions.append(action)
            continue
        raise RuntimeError(f"Unknown review decision: {decision}")

    return add_correction_metadata(corrected, actions), actions


def review_candidate_window(
    tracks: pd.DataFrame,
    candidate: pd.Series,
    particle_a: int,
    particle_b: int,
) -> Literal["swap_from_here", "swap_this_frame", "keep", "stop"]:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    previous_frame = int(candidate["previous_frame"])
    switch_frame = int(candidate["switch_frame"])
    previous = positions_for_frame(tracks, previous_frame, particle_a, particle_b)
    current = positions_for_frame(tracks, switch_frame, particle_a, particle_b)
    if previous is None or current is None:
        return "keep"

    decision = {"value": "keep"}
    previous_a, previous_b = previous
    current_a, current_b = current
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.subplots_adjust(bottom=0.24)
    cost_units = str(candidate.get("cost_units", "mm"))
    fig.suptitle(
        f"Suspicious ID switch at frame {switch_frame}\n"
        f"keep cost {candidate['keep_cost_mm']:.4g} {cost_units}, "
        f"swap cost {candidate['swap_cost_mm']:.4g} {cost_units}, "
        f"ratio {candidate['improvement_ratio']:.3g}, "
        f"source {candidate.get('candidate_source', 'auto')}, "
        f"gap {candidate.get('frame_gap', 1)}"
    )
    projections = [
        (0, 1, "XY", "x_mm", "y_mm"),
        (0, 2, "XZ", "x_mm", "z_mm"),
        (1, 2, "YZ", "y_mm", "z_mm"),
    ]
    for ax, (i, j, title, xlabel, ylabel) in zip(axes, projections):
        plot_review_projection(
            ax,
            previous_a,
            current_a,
            previous_b,
            current_b,
            particle_a,
            particle_b,
            i,
            j,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    def choose(value: str):
        def handler(_event) -> None:
            decision["value"] = value
            plt.close(fig)

        return handler

    buttons = [
        ([0.08, 0.06, 0.20, 0.08], "Swap From Here", "swap_from_here"),
        ([0.31, 0.06, 0.20, 0.08], "Swap This Frame", "swap_this_frame"),
        ([0.54, 0.06, 0.16, 0.08], "Keep", "keep"),
        ([0.74, 0.06, 0.16, 0.08], "Stop Review", "stop"),
    ]
    for bounds, label, value in buttons:
        button = Button(fig.add_axes(bounds), label)
        button.on_clicked(choose(value))
        # Keep a reference alive for matplotlib.
        setattr(fig, f"_button_{value}", button)

    plt.show(block=True)
    return decision["value"]


def plot_review_projection(
    ax,
    previous_a: np.ndarray,
    current_a: np.ndarray,
    previous_b: np.ndarray,
    current_b: np.ndarray,
    particle_a: int,
    particle_b: int,
    axis_i: int,
    axis_j: int,
) -> None:
    colors = {particle_a: "tab:blue", particle_b: "tab:orange"}
    for particle, previous, current in (
        (particle_a, previous_a, current_a),
        (particle_b, previous_b, current_b),
    ):
        color = colors[particle]
        ax.plot(
            [previous[axis_i], current[axis_i]],
            [previous[axis_j], current[axis_j]],
            color=color,
            linewidth=1.5,
            alpha=0.75,
        )
        ax.scatter(previous[axis_i], previous[axis_j], color=color, marker="x", s=60)
        ax.scatter(current[axis_i], current[axis_j], color=color, marker="o", s=55)
        ax.text(current[axis_i], current[axis_j], f" {particle}", color=color)
    ax.plot(
        [previous_a[axis_i], current_b[axis_i]],
        [previous_a[axis_j], current_b[axis_j]],
        color="gray",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    ax.plot(
        [previous_b[axis_i], current_a[axis_i]],
        [previous_b[axis_j], current_a[axis_j]],
        color="gray",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )


def swap_particle_ids_for_frames(
    tracks: pd.DataFrame,
    frames: list[int],
    particle_a: int,
    particle_b: int,
    action_label: str,
) -> pd.DataFrame:
    corrected = tracks.copy()
    frame_set = set(int(frame) for frame in frames)
    frame_mask = corrected["frame"].astype(int).isin(frame_set)
    particle_mask = corrected["particle"].astype(int).isin([particle_a, particle_b])
    mask = frame_mask & particle_mask
    if "identity_auto_correction_applied" not in corrected.columns:
        corrected["identity_auto_correction_applied"] = False
        corrected["identity_auto_correction_action"] = ""
        corrected["pre_identity_auto_particle"] = pd.NA
    corrected.loc[mask, "pre_identity_auto_particle"] = corrected.loc[mask, "particle"]
    corrected.loc[mask, "particle"] = corrected.loc[mask, "particle"].astype(int).map(
        {particle_a: particle_b, particle_b: particle_a}
    )
    if "seed_id" in corrected.columns:
        seed_mask = frame_mask & corrected["seed_id"].astype(int).isin([particle_a, particle_b])
        if "pre_identity_auto_seed_id" not in corrected.columns:
            corrected["pre_identity_auto_seed_id"] = pd.NA
        corrected.loc[seed_mask, "pre_identity_auto_seed_id"] = corrected.loc[
            seed_mask,
            "seed_id",
        ]
        corrected.loc[seed_mask, "seed_id"] = corrected.loc[seed_mask, "seed_id"].astype(
            int
        ).map({particle_a: particle_b, particle_b: particle_a})
    corrected.loc[mask, "identity_auto_correction_applied"] = True
    corrected.loc[mask, "identity_auto_correction_action"] = action_label
    return corrected.sort_values(["particle", "frame"]).reset_index(drop=True)


def add_correction_metadata(
    tracks: pd.DataFrame,
    actions: list[dict[str, float | int | str | bool]],
) -> pd.DataFrame:
    corrected = tracks.copy()
    if "identity_auto_correction_applied" not in corrected.columns:
        corrected["identity_auto_correction_applied"] = False
    corrected["identity_auto_correction_reviewed"] = True
    corrected["identity_auto_correction_action_count"] = len(
        [action for action in actions if str(action.get("decision", "")).startswith("swap")]
    )
    return corrected


def make_correction_output_dir(csv_path: Path, output_name: str | None) -> Path:
    trial_dir = infer_trial_dir(csv_path)
    parent = trial_dir / "corrections"
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        sanitize_output_name(output_name)
        if output_name
        else f"identity_correction_{timestamp}"
    )
    folder_name = prefix_with_trial_index(base_name, infer_trial_index(csv_path))
    output_dir = parent / folder_name
    if not output_dir.exists():
        return output_dir
    suffix = 2
    while True:
        candidate = parent / f"{folder_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def infer_trial_dir(csv_path: Path) -> Path:
    parent = csv_path.resolve().parent
    if "identity_correction" in parent.name and parent.parent.name == "corrections":
        return parent.parent.parent
    if parent.name == "corrections":
        return parent.parent
    return parent


def infer_trial_index(path: Path) -> str:
    candidates = [path.stem] + [part.name for part in path.resolve().parents]
    for name in candidates:
        match = re.search(r"(MI\d+)", name, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def prefix_with_trial_index(name: str, trial_index: str) -> str:
    if not trial_index:
        return sanitize_output_name(name)
    safe_name = sanitize_output_name(name)
    if safe_name.upper().startswith(f"{trial_index.upper()}_"):
        return safe_name
    return f"{trial_index}_{safe_name}"


def write_log(
    log_path: Path,
    input_csv: Path,
    corrected_csv: Path,
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    actions: list[dict[str, float | int | str | bool]],
    size_video_path: Path | None = None,
    size_signature: dict[str, object] | None = None,
) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(input_csv.resolve()),
                "corrected_csv": str(corrected_csv.resolve()),
                "trial_index": infer_trial_index(input_csv),
                "mode": args.mode,
                "particles": list(args.particles),
                "jump_threshold_mm": args.jump_threshold_mm,
                "improvement_ratio": args.improvement_ratio,
                "min_improvement_mm": args.min_improvement_mm,
                "swap_scope": args.swap_scope,
                "review_manual_corrections": bool(args.review_manual_corrections),
                "manual_corrections_only": bool(args.manual_corrections_only),
                "size_signature_enabled": bool(args.size_signature),
                "size_signature_video": str(size_video_path.resolve())
                if size_video_path is not None
                else None,
                "size_signature": size_signature,
                "candidate_count": int(len(candidates)),
                "actions": actions,
            },
            f,
            indent=2,
        )
        f.write("\n")


def plot_candidates(candidates: pd.DataFrame, output_path: Path) -> None:
    if candidates.empty:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(candidates["switch_frame"], candidates["keep_cost_mm"], label="keep")
    axes[0].plot(candidates["switch_frame"], candidates["swap_cost_mm"], label="swap")
    axes[0].set_ylabel("cost")
    axes[0].legend()
    axes[1].plot(candidates["switch_frame"], candidates["improvement_ratio"])
    axes[1].set_ylabel("improvement_ratio")
    axes[1].set_xlabel("frame")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
