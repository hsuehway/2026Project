"""Overlay particle state-vector arrows on a background-subtracted preview video."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from background_subtraction_tracker import sanitize_output_name
from track_orthogonal_3d import (
    CameraCalibration,
    ChamberGeometry,
    calibration_path_for_video_id,
    infer_video_id,
    read_calibration,
)
from verify_tracking_overlay import COLORS, choose_file, draw_frame_label, draw_marker

VIEW_CHOICES = ("both", "front", "side")
VECTOR_CHOICES = ("velocity", "acceleration", "both")
STATE_SOURCE_CHOICES = ("smoothed", "raw", "both")
TRACK_COLUMNS = (
    "particle",
    "frame",
    "x_mm",
    "y_mm",
    "z_mm",
    "front_image_x_px",
    "front_image_y_px",
    "side_image_x_px",
    "side_image_y_px",
)
STATE_VECTOR_COLUMNS = (
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "ax_mm_s2",
    "ay_mm_s2",
    "az_mm_s2",
)
RAW_STATE_COLUMNS = (
    "x_raw_mm",
    "y_raw_mm",
    "z_raw_mm",
    "vx_raw_mm_s",
    "vy_raw_mm_s",
    "vz_raw_mm_s",
    "ax_raw_mm_s2",
    "ay_raw_mm_s2",
    "az_raw_mm_s2",
)
SMOOTH_STATE_COLUMNS = (
    "x_smooth_mm",
    "y_smooth_mm",
    "z_smooth_mm",
    "vx_smooth_mm_s",
    "vy_smooth_mm_s",
    "vz_smooth_mm_s",
    "ax_smooth_mm_s2",
    "ay_smooth_mm_s2",
    "az_smooth_mm_s2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge tracks_3d.csv pixel positions with state-vector velocities/"
            "accelerations and draw scaled plus normalized vector overlays."
        )
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Preview video. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--tracks-csv",
        type=Path,
        help="tracks_3d.csv with front/side pixel positions. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--state-csv",
        type=Path,
        help=(
            "particle_state_vectors.csv or velocities_accelerations.csv with "
            "velocity/acceleration columns. If omitted, File Explorer opens."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Calibration JSON. If omitted, the script tries "
            "calibrations\\<trial>_calibration.json for an MI## or SP## trial ID "
            "from the preview/tracks filename, then calibration.json."
        ),
    )
    parser.add_argument(
        "--calibrations-dir",
        type=Path,
        default=Path("calibrations"),
        help="Folder containing per-MI calibration JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "state_vector_overlays",
        help="Parent folder for timestamped overlay outputs.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output folder name. Defaults to preview name plus timestamp.",
    )
    parser.add_argument(
        "--preview-start-frame",
        type=int,
        default=1,
        help="Original tracking frame corresponding to preview video frame 0.",
    )
    parser.add_argument(
        "--preview-frame-step",
        type=int,
        default=1,
        help="Original frame increment between consecutive preview video frames.",
    )
    parser.add_argument(
        "--view",
        choices=VIEW_CHOICES,
        default="both",
        help="Which preview panel(s) to draw.",
    )
    parser.add_argument(
        "--layout",
        choices=("side-by-side", "stacked"),
        default="side-by-side",
        help="Preview panel layout.",
    )
    parser.add_argument(
        "--vectors",
        choices=VECTOR_CHOICES,
        default="both",
        help="Draw velocity arrows, acceleration arrows, or both.",
    )
    parser.add_argument(
        "--state-source",
        choices=STATE_SOURCE_CHOICES,
        default="both",
        help=(
            "Choose output sources. Default both writes four separate videos: "
            "raw scaled/normalized and smoothed scaled/normalized."
        ),
    )
    parser.add_argument(
        "--velocity-scale-s",
        type=float,
        default=0.01,
        help=(
            "Scaled-video velocity multiplier in seconds. Larger values draw "
            "longer velocity arrows."
        ),
    )
    parser.add_argument(
        "--acceleration-scale-s2",
        type=float,
        default=0.0005,
        help=(
            "Scaled-video acceleration multiplier in seconds^2. Larger values "
            "draw longer acceleration arrows."
        ),
    )
    parser.add_argument(
        "--max-arrow-length-px",
        type=float,
        default=70.0,
        help="Maximum arrow length in the scaled video.",
    )
    parser.add_argument(
        "--velocity-normalized-length-px",
        type=float,
        default=38.0,
        help="Velocity arrow length in the normalized video.",
    )
    parser.add_argument(
        "--acceleration-normalized-length-px",
        type=float,
        default=28.0,
        help="Acceleration arrow length in the normalized video.",
    )
    parser.add_argument(
        "--marker-radius-px",
        type=int,
        default=5,
        help="Position marker radius in pixels.",
    )
    parser.add_argument(
        "--line-width-px",
        type=int,
        default=2,
        help="Arrow and marker line width.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum preview frames to write.",
    )
    parser.add_argument(
        "--front-width",
        type=int,
        help="Override front preview panel width in pixels.",
    )
    parser.add_argument(
        "--front-height",
        type=int,
        help="Override front preview panel height in pixels.",
    )
    parser.add_argument(
        "--side-width",
        type=int,
        help="Override side preview panel width in pixels.",
    )
    parser.add_argument(
        "--side-height",
        type=int,
        help="Override side preview panel height in pixels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    preview_path = args.preview or choose_file(
        "Choose preview video",
        [("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
    )
    tracks_path = args.tracks_csv or choose_file(
        "Choose tracks_3d.csv",
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )
    state_path = args.state_csv or choose_file(
        "Choose particle_state_vectors.csv or velocities_accelerations.csv",
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )

    calibration_path = resolve_calibration_path(args, preview_path, tracks_path)
    geometry, front_cal, side_cal, front_roi, side_roi, _, _ = read_calibration(
        calibration_path
    )
    if front_roi is None or side_roi is None:
        raise SystemExit(f"{calibration_path} must contain front_roi and side_roi.")

    tracks = pd.read_csv(tracks_path)
    states = prepare_state_source_columns(pd.read_csv(state_path))
    validate_csvs(tracks, states, tracks_path, state_path)
    overlay_rows = merge_tracks_and_states(tracks, states)
    if overlay_rows.empty:
        raise SystemExit(
            "No matching particle/frame rows between tracks_3d.csv and state vectors."
        )

    panel = resolve_panel_geometry(args, front_roi, side_roi)
    output_dir = make_output_dir(args.output_dir, args.output_name, preview_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = sanitize_output_name(preview_path.stem)
    video_outputs: dict[str, dict[str, object]] = {}
    for source in output_state_sources(args.state_source):
        scaled_path = output_dir / f"{base_name}_state_vectors_{source}_scaled.mp4"
        normalized_path = (
            output_dir / f"{base_name}_state_vectors_{source}_normalized.mp4"
        )
        source_args_values = vars(args).copy()
        source_args_values["state_source"] = source
        source_args = argparse.Namespace(**source_args_values)
        frames_written, scaled_arrows, normalized_arrows = write_overlay_videos(
            preview_path=preview_path,
            scaled_path=scaled_path,
            normalized_path=normalized_path,
            rows=overlay_rows,
            panel=panel,
            geometry=geometry,
            front_cal=front_cal,
            side_cal=side_cal,
            args=source_args,
        )
        video_outputs[source] = {
            "scaled_video": str(scaled_path.resolve()),
            "normalized_video": str(normalized_path.resolve()),
            "frames_written": int(frames_written),
            "scaled_arrows": int(scaled_arrows),
            "normalized_arrows": int(normalized_arrows),
        }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preview": str(preview_path.resolve()),
                "tracks_csv": str(tracks_path.resolve()),
                "state_csv": str(state_path.resolve()),
                "calibration": str(calibration_path.resolve()),
                "preview_start_frame": args.preview_start_frame,
                "preview_frame_step": args.preview_frame_step,
                "vectors": args.vectors,
                "state_source": args.state_source,
                "video_outputs": video_outputs,
                "view": args.view,
                "layout": args.layout,
                "velocity_scale_s": args.velocity_scale_s,
                "acceleration_scale_s2": args.acceleration_scale_s2,
                "max_arrow_length_px": args.max_arrow_length_px,
                "velocity_normalized_length_px": args.velocity_normalized_length_px,
                "acceleration_normalized_length_px": args.acceleration_normalized_length_px,
                "matched_rows": int(len(overlay_rows)),
                "panel": panel,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print("\nDone.")
    print(f"Preview:          {preview_path.resolve()}")
    print(f"Tracks CSV:       {tracks_path.resolve()}")
    print(f"State CSV:        {state_path.resolve()}")
    print(f"Calibration:      {calibration_path.resolve()}")
    print(f"Matched rows:     {len(overlay_rows)}")
    for source, output in video_outputs.items():
        print(f"{source.title()} frames:    {output['frames_written']}")
        print(f"{source.title()} scaled:    {output['scaled_video']}")
        print(f"{source.title()} normalized:{output['normalized_video']}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.preview_start_frame < 0:
        raise SystemExit("--preview-start-frame must be zero or positive.")
    if args.preview_frame_step <= 0:
        raise SystemExit("--preview-frame-step must be positive.")
    if args.velocity_scale_s < 0:
        raise SystemExit("--velocity-scale-s must be non-negative.")
    if args.acceleration_scale_s2 < 0:
        raise SystemExit("--acceleration-scale-s2 must be non-negative.")
    if args.max_arrow_length_px <= 0:
        raise SystemExit("--max-arrow-length-px must be positive.")
    if args.velocity_normalized_length_px <= 0:
        raise SystemExit("--velocity-normalized-length-px must be positive.")
    if args.acceleration_normalized_length_px <= 0:
        raise SystemExit("--acceleration-normalized-length-px must be positive.")
    if args.marker_radius_px <= 0:
        raise SystemExit("--marker-radius-px must be positive.")
    if args.line_width_px <= 0:
        raise SystemExit("--line-width-px must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")
    for name in ("front_width", "front_height", "side_width", "side_height"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")


def resolve_calibration_path(
    args: argparse.Namespace,
    preview_path: Path,
    tracks_path: Path,
) -> Path:
    if args.calibration is not None:
        return args.calibration
    for path in (preview_path, tracks_path):
        video_id = infer_video_id(path)
        if video_id is None:
            continue
        candidate = calibration_path_for_video_id(video_id, args.calibrations_dir)
        if candidate.exists():
            return candidate
    fallback = Path("calibration.json")
    if fallback.exists():
        return fallback
    raise SystemExit(
        "Could not infer a calibration file. Pass --calibration with the JSON used "
        "for this video."
    )


def validate_csvs(
    tracks: pd.DataFrame,
    states: pd.DataFrame,
    tracks_path: Path,
    state_path: Path,
) -> None:
    missing_tracks = [column for column in TRACK_COLUMNS if column not in tracks.columns]
    if missing_tracks:
        raise SystemExit(f"{tracks_path} is missing required columns: {missing_tracks}")
    missing_states = [
        column
        for column in ("particle", "frame", *STATE_VECTOR_COLUMNS)
        if column not in states.columns
    ]
    if missing_states:
        raise SystemExit(f"{state_path} is missing required columns: {missing_states}")
    if tracks.empty:
        raise SystemExit(f"{tracks_path} is empty.")
    if states.empty:
        raise SystemExit(f"{state_path} is empty.")


def prepare_state_source_columns(states: pd.DataFrame) -> pd.DataFrame:
    states = states.copy()
    aliases = {
        "x_raw_mm": ("x_raw_mm", "x_mm"),
        "y_raw_mm": ("y_raw_mm", "y_mm"),
        "z_raw_mm": ("z_raw_mm", "z_mm"),
        "x_smooth_mm": ("x_smooth_mm", "x_mm"),
        "y_smooth_mm": ("y_smooth_mm", "y_mm"),
        "z_smooth_mm": ("z_smooth_mm", "z_mm"),
        "vx_raw_mm_s": ("vx_raw_mm_s", "vx_mm_s"),
        "vy_raw_mm_s": ("vy_raw_mm_s", "vy_mm_s"),
        "vz_raw_mm_s": ("vz_raw_mm_s", "vz_mm_s"),
        "ax_raw_mm_s2": ("ax_raw_mm_s2", "ax_mm_s2"),
        "ay_raw_mm_s2": ("ay_raw_mm_s2", "ay_mm_s2"),
        "az_raw_mm_s2": ("az_raw_mm_s2", "az_mm_s2"),
        "vx_smooth_mm_s": ("vx_smooth_mm_s", "vx_mm_s"),
        "vy_smooth_mm_s": ("vy_smooth_mm_s", "vy_mm_s"),
        "vz_smooth_mm_s": ("vz_smooth_mm_s", "vz_mm_s"),
        "ax_smooth_mm_s2": ("ax_smooth_mm_s2", "ax_mm_s2"),
        "ay_smooth_mm_s2": ("ay_smooth_mm_s2", "ay_mm_s2"),
        "az_smooth_mm_s2": ("az_smooth_mm_s2", "az_mm_s2"),
    }
    for target, candidates in aliases.items():
        if target in states.columns:
            continue
        source = next((column for column in candidates if column in states.columns), None)
        if source is None:
            raise SystemExit(
                f"State CSV cannot provide {target}; rerun calculate_motion.py and "
                "build_state_vectors.py."
            )
        states[target] = states[source]
    return states


def merge_tracks_and_states(tracks: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    track_columns = list(dict.fromkeys(TRACK_COLUMNS))
    state_columns = list(
        dict.fromkeys(
            [
                "particle",
                "frame",
                *STATE_VECTOR_COLUMNS,
                *RAW_STATE_COLUMNS,
                *SMOOTH_STATE_COLUMNS,
            ]
        )
    )
    left = tracks.loc[:, track_columns].copy()
    right = states.loc[:, state_columns].copy()
    for table in (left, right):
        table["particle"] = table["particle"].astype(int)
        table["frame"] = table["frame"].astype(int)
    merged = left.merge(right, on=["particle", "frame"], how="inner")
    return merged.sort_values(["frame", "particle"]).reset_index(drop=True)


def resolve_panel_geometry(args: argparse.Namespace, front_roi, side_roi) -> dict[str, int]:
    front_width = args.front_width or front_roi.width
    front_height = args.front_height or front_roi.height
    side_width = args.side_width or side_roi.width
    side_height = args.side_height or side_roi.height
    if args.layout == "stacked":
        return {
            "front_x": 0,
            "front_y": 0,
            "side_x": 0,
            "side_y": front_height,
            "front_width": front_width,
            "front_height": front_height,
            "side_width": side_width,
            "side_height": side_height,
        }
    return {
        "front_x": 0,
        "front_y": 0,
        "side_x": front_width,
        "side_y": 0,
        "front_width": front_width,
        "front_height": front_height,
        "side_width": side_width,
        "side_height": side_height,
    }


def make_output_dir(parent_dir: Path, output_name: str | None, preview_path: Path) -> Path:
    parent_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        sanitize_output_name(output_name)
        if output_name
        else f"{sanitize_output_name(preview_path.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = parent_dir / base_name
    if not output_dir.exists():
        return output_dir
    suffix = 2
    while True:
        candidate = parent_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def write_overlay_videos(
    preview_path: Path,
    scaled_path: Path,
    normalized_path: Path,
    rows: pd.DataFrame,
    panel: dict[str, int],
    geometry: ChamberGeometry,
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    reader = imageio.get_reader(str(preview_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    rows_by_frame = {
        int(frame): group.copy()
        for frame, group in rows.groupby("frame", sort=False)
    }

    frames_written = 0
    scaled_arrows = 0
    normalized_arrows = 0
    with imageio.get_writer(str(scaled_path), fps=fps, codec="libx264", quality=8) as scaled_writer:
        with imageio.get_writer(str(normalized_path), fps=fps, codec="libx264", quality=8) as normalized_writer:
            try:
                for preview_index, frame in enumerate(reader):
                    if args.max_frames is not None and frames_written >= args.max_frames:
                        break
                    track_frame = (
                        args.preview_start_frame
                        + preview_index * args.preview_frame_step
                    )
                    source = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
                    scaled_image = source.copy()
                    normalized_image = source.copy()
                    frame_rows = rows_by_frame.get(track_frame)
                    if frame_rows is not None and not frame_rows.empty:
                        scaled_arrows += draw_state_vectors(
                            ImageDraw.Draw(scaled_image),
                            frame_rows,
                            panel,
                            geometry,
                            front_cal,
                            side_cal,
                            args,
                            normalized=False,
                        )
                        normalized_arrows += draw_state_vectors(
                            ImageDraw.Draw(normalized_image),
                            frame_rows,
                            panel,
                            geometry,
                            front_cal,
                            side_cal,
                            args,
                            normalized=True,
                        )
                    draw_frame_label(ImageDraw.Draw(scaled_image), track_frame)
                    draw_frame_label(ImageDraw.Draw(normalized_image), track_frame)
                    draw_state_source_legend(
                        ImageDraw.Draw(scaled_image),
                        args.state_source,
                    )
                    draw_state_source_legend(
                        ImageDraw.Draw(normalized_image),
                        args.state_source,
                    )
                    scaled_writer.append_data(np.asarray(scaled_image))
                    normalized_writer.append_data(np.asarray(normalized_image))
                    frames_written += 1
            finally:
                reader.close()

    return frames_written, scaled_arrows, normalized_arrows


def draw_state_vectors(
    draw: ImageDraw.ImageDraw,
    rows: pd.DataFrame,
    panel: dict[str, int],
    geometry: ChamberGeometry,
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    args: argparse.Namespace,
    normalized: bool,
) -> int:
    arrows_drawn = 0
    for _, row in rows.iterrows():
        particle = int(row["particle"])
        color = COLORS[particle % len(COLORS)]
        if args.view in ("both", "front"):
            arrows_drawn += draw_row_state_sources_for_camera(
                draw,
                row,
                panel,
                front_cal,
                geometry,
                camera="front",
                horizontal_axis="x",
                particle=particle,
                particle_color=color,
                args=args,
                normalized=normalized,
            )
        if args.view in ("both", "side"):
            arrows_drawn += draw_row_state_sources_for_camera(
                draw,
                row,
                panel,
                side_cal,
                geometry,
                camera="side",
                horizontal_axis="y",
                particle=particle,
                particle_color=color,
                args=args,
                normalized=normalized,
            )
    return arrows_drawn


def draw_row_state_sources_for_camera(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    panel: dict[str, int],
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
    *,
    camera: Literal["front", "side"],
    horizontal_axis: Literal["x", "y"],
    particle: int,
    particle_color: tuple[int, int, int],
    args: argparse.Namespace,
    normalized: bool,
) -> int:
    arrows = 0
    sources = state_sources_for_mode(args.state_source)
    for source in sources:
        start_x, start_y = state_position_to_pixels(
            row,
            panel,
            calibration,
            geometry,
            camera=camera,
            horizontal_axis=horizontal_axis,
            state_source=source,
        )
        if not math.isfinite(start_x) or not math.isfinite(start_y):
            continue
        comparison = len(sources) > 1
        marker_color = (
            tuple(max(0, int(channel * 0.55)) for channel in particle_color)
            if source == "raw" and comparison
            else particle_color
        )
        marker_label = f"P{particle}{'R' if source == 'raw' else 'S'}" if comparison else f"P{particle}"
        draw_marker(
            draw,
            start_x,
            start_y,
            max(2, args.marker_radius_px - 1) if source == "raw" and comparison else args.marker_radius_px,
            marker_color,
            marker_label,
            max(1, args.line_width_px - 1) if source == "raw" and comparison else args.line_width_px,
        )
        arrows += draw_row_vectors_for_camera(
            draw,
            row,
            start_x,
            start_y,
            calibration,
            geometry,
            horizontal_axis=horizontal_axis,
            args=args,
            normalized=normalized,
            state_source=source,
            comparison=comparison,
        )
    return arrows


def state_sources_for_mode(mode: str) -> tuple[Literal["raw", "smoothed"], ...]:
    if mode == "raw":
        return ("raw",)
    if mode == "smoothed":
        return ("smoothed",)
    if mode == "both":
        return ("raw", "smoothed")
    raise ValueError(f"Unknown state source: {mode}")


def output_state_sources(mode: str) -> tuple[Literal["raw", "smoothed"], ...]:
    if mode == "both":
        return ("raw", "smoothed")
    if mode == "raw":
        return ("raw",)
    if mode == "smoothed":
        return ("smoothed",)
    raise ValueError(f"Unknown state source: {mode}")


def state_position_to_pixels(
    row: pd.Series,
    panel: dict[str, int],
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
    *,
    camera: Literal["front", "side"],
    horizontal_axis: Literal["x", "y"],
    state_source: Literal["raw", "smoothed"],
) -> tuple[float, float]:
    if state_source == "raw":
        return (
            panel[f"{camera}_x"] + float(row[f"{camera}_image_x_px"]),
            panel[f"{camera}_y"] + float(row[f"{camera}_image_y_px"]),
        )

    z_mm = state_value(row, "z_mm", state_source)
    horizontal_mm = state_value(row, f"{horizontal_axis}_mm", state_source)
    image_x = float(calibration.center_x_at_z(z_mm, geometry)) + horizontal_mm * float(
        calibration.px_per_mm_at_z(z_mm, geometry)
    )
    bottom_y = float(calibration.bottom_center_px[1])
    top_y = float(calibration.top_center_px[1])
    image_y = bottom_y + (z_mm / geometry.gap_mm) * (top_y - bottom_y)
    return panel[f"{camera}_x"] + image_x, panel[f"{camera}_y"] + image_y


def state_value(
    row: pd.Series,
    canonical_column: str,
    state_source: Literal["raw", "smoothed"],
) -> float:
    suffix = "raw" if state_source == "raw" else "smooth"
    if canonical_column.endswith("_mm_s2"):
        source_column = canonical_column.replace("_mm_s2", f"_{suffix}_mm_s2")
    elif canonical_column.endswith("_mm_s"):
        source_column = canonical_column.replace("_mm_s", f"_{suffix}_mm_s")
    elif canonical_column.endswith("_mm"):
        source_column = canonical_column.replace("_mm", f"_{suffix}_mm")
    else:
        source_column = canonical_column
    column = source_column if source_column in row.index else canonical_column
    return float(row[column])


def draw_state_source_legend(draw: ImageDraw.ImageDraw, mode: str) -> None:
    if mode == "both":
        text = "R raw/frame difference (dim)   S smooth curve (bright)"
    elif mode == "raw":
        text = "state source: raw position + frame differences"
    else:
        text = "state source: smoothed position + curve derivatives"
    draw.rectangle([0, 52, 380, 72], fill=(0, 0, 0))
    draw.text((6, 56), text, fill=(235, 235, 235))


def draw_row_vectors_for_camera(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    start_x: float,
    start_y: float,
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
    horizontal_axis: Literal["x", "y"],
    args: argparse.Namespace,
    normalized: bool,
    state_source: Literal["raw", "smoothed"] = "smoothed",
    comparison: bool = False,
) -> int:
    arrows = 0
    if args.vectors in ("velocity", "both"):
        dx, dy = vector_components_to_pixels(
            row,
            calibration,
            geometry,
            horizontal_axis=horizontal_axis,
            kind="velocity",
            normalized=normalized,
            args=args,
            state_source=state_source,
        )
        if dx is not None and dy is not None:
            color = (
                (45, 120, 140)
                if state_source == "raw" and comparison
                else (60, 230, 255)
            )
            draw_arrow(
                draw,
                start_x,
                start_y,
                dx,
                dy,
                color=color,
                width=max(1, args.line_width_px - 1)
                if state_source == "raw" and comparison
                else args.line_width_px,
                label=f"v{'R' if state_source == 'raw' else 'S'}" if comparison else "v",
            )
            arrows += 1
    if args.vectors in ("acceleration", "both"):
        dx, dy = vector_components_to_pixels(
            row,
            calibration,
            geometry,
            horizontal_axis=horizontal_axis,
            kind="acceleration",
            normalized=normalized,
            args=args,
            state_source=state_source,
        )
        if dx is not None and dy is not None:
            color = (
                (145, 105, 35)
                if state_source == "raw" and comparison
                else (255, 230, 60)
            )
            draw_arrow(
                draw,
                start_x,
                start_y,
                dx,
                dy,
                color=color,
                width=max(1, args.line_width_px - 1),
                label=f"a{'R' if state_source == 'raw' else 'S'}" if comparison else "a",
            )
            arrows += 1
    return arrows


def vector_components_to_pixels(
    row: pd.Series,
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
    horizontal_axis: Literal["x", "y"],
    kind: Literal["velocity", "acceleration"],
    normalized: bool,
    args: argparse.Namespace,
    state_source: Literal["raw", "smoothed"] = "smoothed",
) -> tuple[float | None, float | None]:
    prefix = "v" if kind == "velocity" else "a"
    unit = "mm_s" if kind == "velocity" else "mm_s2"
    canonical_horizontal = f"{prefix}{horizontal_axis}_{unit}"
    canonical_z = f"{prefix}z_{unit}"
    horizontal_component = state_value(
        row,
        canonical_horizontal,
        state_source,
    )
    z_component = state_value(row, canonical_z, state_source)
    z_mm = state_value(row, "z_mm", state_source)
    px_per_mm = float(calibration.px_per_mm_at_z(z_mm, geometry))
    z_px_per_mm = z_pixel_slope(calibration, geometry)
    dx = horizontal_component * px_per_mm
    dy = z_component * z_px_per_mm
    if not math.isfinite(dx) or not math.isfinite(dy):
        return None, None
    if normalized:
        length = (
            args.velocity_normalized_length_px
            if kind == "velocity"
            else args.acceleration_normalized_length_px
        )
        return normalize_vector(dx, dy, length)

    scale = (
        args.velocity_scale_s
        if kind == "velocity"
        else args.acceleration_scale_s2
    )
    dx *= scale
    dy *= scale
    return cap_vector(dx, dy, args.max_arrow_length_px)


def z_pixel_slope(
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
) -> float:
    bottom_y = calibration.bottom_center_px[1]
    top_y = calibration.top_center_px[1]
    return float((top_y - bottom_y) / geometry.gap_mm)


def normalize_vector(
    dx: float,
    dy: float,
    length: float,
) -> tuple[float | None, float | None]:
    magnitude = math.hypot(dx, dy)
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        return None, None
    factor = length / magnitude
    return dx * factor, dy * factor


def cap_vector(
    dx: float,
    dy: float,
    max_length: float,
) -> tuple[float | None, float | None]:
    magnitude = math.hypot(dx, dy)
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        return None, None
    if magnitude <= max_length:
        return dx, dy
    factor = max_length / magnitude
    return dx * factor, dy * factor


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
    color: tuple[int, int, int],
    width: int,
    label: str,
) -> None:
    end_x = start_x + dx
    end_y = start_y + dy
    draw.line([start_x, start_y, end_x, end_y], fill=color, width=width)

    angle = math.atan2(dy, dx)
    head_length = max(7.0, width * 4.5)
    head_angle = math.radians(150)
    left = (
        end_x + head_length * math.cos(angle + head_angle),
        end_y + head_length * math.sin(angle + head_angle),
    )
    right = (
        end_x + head_length * math.cos(angle - head_angle),
        end_y + head_length * math.sin(angle - head_angle),
    )
    draw.polygon([(end_x, end_y), left, right], fill=color)
    if label:
        draw.text((end_x + 4, end_y + 4), label, fill=color)


if __name__ == "__main__":
    raise SystemExit(main())
