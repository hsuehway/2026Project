"""Overlay reduced rotation axes on a background-subtracted preview video."""

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
AXIS_CHOICES = ("raw-2d", "reconstructed-3d", "both")
POSITION_COLUMNS = (
    "particle",
    "frame",
    "front_image_x_px",
    "front_image_y_px",
    "side_image_x_px",
    "side_image_y_px",
)
OPTIONAL_TRACK_COLUMNS = ("x_mm", "y_mm", "z_mm")
ROTATION_COLUMNS = (
    "particle",
    "frame",
    "n_x",
    "n_y",
    "n_z",
)
RAW_AXIS_COLUMNS = (
    "front_axis_u_raw",
    "front_axis_v_raw",
    "side_axis_u_raw",
    "side_axis_v_raw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw raw projected axes and reconstructed 3D reduced-attitude axes "
            "from rotation_state_vectors.csv on a preview video."
        )
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Preview video. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--rotation-csv",
        type=Path,
        help=(
            "rotation_state_vectors.csv or particle_reduced_state_vectors.csv. "
            "If omitted, File Explorer opens."
        ),
    )
    parser.add_argument(
        "--tracks-csv",
        type=Path,
        help=(
            "Optional tracks_3d.csv with front/side pixel positions. Required "
            "only when the rotation CSV does not include image-position columns."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Calibration JSON. If omitted, the script tries "
            "calibrations\\<trial>_calibration.json for an MI## or SP## trial ID "
            "from the preview/CSV filename, then calibration.json."
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
        default=Path("outputs_background") / "rotation_overlays",
        help="Parent folder for timestamped rotation overlay outputs.",
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
        "--axes",
        choices=AXIS_CHOICES,
        default="both",
        help="Draw raw 2D PCA axes, reconstructed 3D projected axes, or both.",
    )
    parser.add_argument(
        "--axis-half-length-px",
        type=float,
        default=26.0,
        help="Half-length of axis line segments in preview pixels.",
    )
    parser.add_argument(
        "--omega-normalized-length-px",
        type=float,
        default=36.0,
        help="Normalized angular-velocity arrow length when --draw-omega is used.",
    )
    parser.add_argument(
        "--draw-omega",
        action="store_true",
        help="Also draw observable tumbling angular velocity as a magenta arrow.",
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
        default=3,
        help="Axis and marker line width.",
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
    parser.add_argument(
        "--no-zoom-panels",
        action="store_true",
        help="Do not replace the lower preview area with clean per-particle zoom crops.",
    )
    parser.add_argument(
        "--zoom-crop-width-px",
        type=int,
        default=81,
        help="Native camera-panel crop width for each clean zoom view.",
    )
    parser.add_argument(
        "--zoom-crop-height-px",
        type=int,
        default=81,
        help="Native camera-panel crop height for each clean zoom view.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    preview_path = args.preview or choose_file(
        "Choose preview video",
        [("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
    )
    rotation_path = args.rotation_csv or choose_file(
        "Choose rotation_state_vectors.csv",
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )

    calibration_path = resolve_calibration_path(args, preview_path, rotation_path)
    geometry, front_cal, side_cal, front_roi, side_roi, _, _ = read_calibration(
        calibration_path
    )
    if front_roi is None or side_roi is None:
        raise SystemExit(f"{calibration_path} must contain front_roi and side_roi.")

    rotation = pd.read_csv(rotation_path)
    tracks = pd.read_csv(args.tracks_csv) if args.tracks_csv is not None else None
    rows = prepare_overlay_rows(rotation, tracks, rotation_path, args.tracks_csv)
    if rows.empty:
        raise SystemExit("No matching particle/frame rows are available for overlay.")

    panel = resolve_panel_geometry(args, front_roi, side_roi)
    output_dir = make_output_dir(args.output_dir, args.output_name, preview_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sanitize_output_name(preview_path.stem)}_rotation_overlay.mp4"

    (
        frames_written,
        markers_drawn,
        axis_lines_drawn,
        omega_arrows_drawn,
        zoom_panels_drawn,
    ) = write_overlay_video(
        preview_path=preview_path,
        output_path=output_path,
        rows=rows,
        panel=panel,
        geometry=geometry,
        front_cal=front_cal,
        side_cal=side_cal,
        args=args,
    )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preview": str(preview_path.resolve()),
                "rotation_csv": str(rotation_path.resolve()),
                "tracks_csv": str(args.tracks_csv.resolve()) if args.tracks_csv else None,
                "calibration": str(calibration_path.resolve()),
                "output_video": str(output_path.resolve()),
                "preview_start_frame": args.preview_start_frame,
                "preview_frame_step": args.preview_frame_step,
                "view": args.view,
                "layout": args.layout,
                "axes": args.axes,
                "axis_half_length_px": args.axis_half_length_px,
                "draw_omega": bool(args.draw_omega),
                "matched_rows": int(len(rows)),
                "frames_written": int(frames_written),
                "markers_drawn": int(markers_drawn),
                "axis_lines_drawn": int(axis_lines_drawn),
                "omega_arrows_drawn": int(omega_arrows_drawn),
                "zoom_panels_enabled": not args.no_zoom_panels,
                "zoom_crop_width_px": int(args.zoom_crop_width_px),
                "zoom_crop_height_px": int(args.zoom_crop_height_px),
                "zoom_panels_drawn": int(zoom_panels_drawn),
                "panel": panel,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print("\nDone.")
    print(f"Preview:           {preview_path.resolve()}")
    print(f"Rotation CSV:      {rotation_path.resolve()}")
    if args.tracks_csv is not None:
        print(f"Tracks CSV:        {args.tracks_csv.resolve()}")
    print(f"Calibration:       {calibration_path.resolve()}")
    print(f"Matched rows:      {len(rows)}")
    print(f"Frames written:    {frames_written}")
    print(f"Markers drawn:     {markers_drawn}")
    print(f"Axis lines drawn:  {axis_lines_drawn}")
    if args.draw_omega:
        print(f"Omega arrows:      {omega_arrows_drawn}")
    if not args.no_zoom_panels:
        print(f"Zoom panels drawn: {zoom_panels_drawn}")
    print(f"Output video:      {output_path.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.preview_start_frame < 0:
        raise SystemExit("--preview-start-frame must be zero or positive.")
    if args.preview_frame_step <= 0:
        raise SystemExit("--preview-frame-step must be positive.")
    if args.axis_half_length_px <= 0:
        raise SystemExit("--axis-half-length-px must be positive.")
    if args.omega_normalized_length_px <= 0:
        raise SystemExit("--omega-normalized-length-px must be positive.")
    if args.marker_radius_px <= 0:
        raise SystemExit("--marker-radius-px must be positive.")
    if args.line_width_px <= 0:
        raise SystemExit("--line-width-px must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")
    if args.zoom_crop_width_px <= 2:
        raise SystemExit("--zoom-crop-width-px must be greater than 2.")
    if args.zoom_crop_height_px <= 2:
        raise SystemExit("--zoom-crop-height-px must be greater than 2.")
    for name in ("front_width", "front_height", "side_width", "side_height"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")


def resolve_calibration_path(
    args: argparse.Namespace,
    preview_path: Path,
    rotation_path: Path,
) -> Path:
    if args.calibration is not None:
        return args.calibration
    candidates = [preview_path, rotation_path]
    if args.tracks_csv is not None:
        candidates.append(args.tracks_csv)
    for path in candidates:
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


def prepare_overlay_rows(
    rotation: pd.DataFrame,
    tracks: pd.DataFrame | None,
    rotation_path: Path,
    tracks_path: Path | None,
) -> pd.DataFrame:
    missing_rotation = [column for column in ROTATION_COLUMNS if column not in rotation.columns]
    if missing_rotation:
        raise SystemExit(f"{rotation_path} is missing required columns: {missing_rotation}")
    if rotation.empty:
        raise SystemExit(f"{rotation_path} is empty.")

    rotation = rotation.copy()
    rotation["particle"] = rotation["particle"].astype(int)
    rotation["frame"] = rotation["frame"].astype(int)

    missing_positions = [
        column for column in POSITION_COLUMNS if column not in rotation.columns
    ]
    if missing_positions:
        if tracks is None or tracks_path is None:
            raise SystemExit(
                f"{rotation_path} is missing image-position columns {missing_positions}. "
                "Pass --tracks-csv with the matching tracks_3d.csv."
            )
        missing_tracks = [column for column in POSITION_COLUMNS if column not in tracks.columns]
        if missing_tracks:
            raise SystemExit(f"{tracks_path} is missing required columns: {missing_tracks}")
        left = rotation.copy()
        track_columns = list(dict.fromkeys(POSITION_COLUMNS))
        track_columns.extend(
            column
            for column in OPTIONAL_TRACK_COLUMNS
            if column in tracks.columns and column not in left.columns
        )
        right = tracks.loc[:, track_columns].copy()
        right["particle"] = right["particle"].astype(int)
        right["frame"] = right["frame"].astype(int)
        rows = left.merge(right, on=["particle", "frame"], how="inner", suffixes=("", "_track"))
    else:
        rows = rotation

    if "rotation_valid" not in rows.columns:
        rows["rotation_valid"] = np.isfinite(rows[["n_x", "n_y", "n_z"]]).all(axis=1)
    if "invalid_reason" not in rows.columns:
        rows["invalid_reason"] = ""
    for column in RAW_AXIS_COLUMNS:
        if column not in rows.columns:
            rows[column] = np.nan
    for column in (
        "omega_x_rad_s",
        "omega_y_rad_s",
        "omega_z_rad_s",
        "orientation_confidence",
    ):
        if column not in rows.columns:
            rows[column] = np.nan

    return rows.sort_values(["frame", "particle"]).reset_index(drop=True)


def resolve_panel_geometry(args: argparse.Namespace, front_roi, side_roi) -> dict[str, float]:
    front_width = float(args.front_width or front_roi.width)
    front_height = float(args.front_height or front_roi.height)
    side_width = float(args.side_width or side_roi.width)
    side_height = float(args.side_height or side_roi.height)
    panel: dict[str, float] = {
        "front_width": front_width,
        "front_height": front_height,
        "side_width": side_width,
        "side_height": side_height,
        "front_scale_x": front_width / float(front_roi.width),
        "front_scale_y": front_height / float(front_roi.height),
        "side_scale_x": side_width / float(side_roi.width),
        "side_scale_y": side_height / float(side_roi.height),
    }
    if args.layout == "stacked":
        panel.update(
            {
                "front_x": 0.0,
                "front_y": 0.0,
                "side_x": 0.0,
                "side_y": front_height,
            }
        )
    else:
        panel.update(
            {
                "front_x": 0.0,
                "front_y": 0.0,
                "side_x": front_width,
                "side_y": 0.0,
            }
        )
    return panel


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


def write_overlay_video(
    preview_path: Path,
    output_path: Path,
    rows: pd.DataFrame,
    panel: dict[str, float],
    geometry: ChamberGeometry,
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    args: argparse.Namespace,
) -> tuple[int, int, int, int, int]:
    reader = imageio.get_reader(str(preview_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    rows_by_frame = {
        int(frame): group.copy()
        for frame, group in rows.groupby("frame", sort=False)
    }
    zoom_particle_ids = sorted(int(particle) for particle in rows["particle"].dropna().unique())
    zoom_state: dict[int, dict[str, Image.Image]] = {}

    frames_written = 0
    markers_drawn = 0
    axis_lines_drawn = 0
    omega_arrows_drawn = 0
    zoom_panels_drawn = 0
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        try:
            for preview_index, frame in enumerate(reader):
                if args.max_frames is not None and frames_written >= args.max_frames:
                    break
                track_frame = args.preview_start_frame + preview_index * args.preview_frame_step
                source_image = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
                image = source_image.copy()
                draw = ImageDraw.Draw(image)
                frame_rows = rows_by_frame.get(track_frame)
                if frame_rows is not None and not frame_rows.empty:
                    counts = draw_rotation_rows(
                        draw,
                        frame_rows,
                        panel,
                        geometry,
                        front_cal,
                        side_cal,
                        args,
                    )
                    markers_drawn += counts[0]
                    axis_lines_drawn += counts[1]
                    omega_arrows_drawn += counts[2]
                    if not args.no_zoom_panels:
                        update_zoom_panel_state(
                            zoom_state,
                            source_image,
                            frame_rows,
                            panel,
                            args,
                        )
                if not args.no_zoom_panels:
                    zoom_panels_drawn += draw_held_zoom_panels(
                        image,
                        zoom_state,
                        zoom_particle_ids,
                        panel,
                    )
                draw_frame_label(draw, track_frame)
                writer.append_data(np.asarray(image))
                frames_written += 1
        finally:
            reader.close()

    return frames_written, markers_drawn, axis_lines_drawn, omega_arrows_drawn, zoom_panels_drawn


def draw_rotation_rows(
    draw: ImageDraw.ImageDraw,
    rows: pd.DataFrame,
    panel: dict[str, float],
    geometry: ChamberGeometry,
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    markers_drawn = 0
    axis_lines_drawn = 0
    omega_arrows_drawn = 0
    for _, row in rows.iterrows():
        particle = int(row["particle"])
        marker_color = COLORS[particle % len(COLORS)]
        valid = bool(row["rotation_valid"])
        raw_color = (60, 230, 255) if valid else (255, 90, 90)
        reconstructed_color = (255, 230, 60) if valid else (255, 90, 90)
        omega_color = (255, 80, 220)
        label = f"P{particle}"
        if not valid and str(row.get("invalid_reason", "")):
            label = f"{label} !"

        if args.view in ("both", "front"):
            center = scaled_point(row, panel, "front")
            draw_marker(
                draw,
                center[0],
                center[1],
                args.marker_radius_px,
                marker_color,
                label,
                max(1, args.line_width_px - 1),
            )
            markers_drawn += 1
            axis_lines_drawn += draw_camera_axes(
                draw,
                row,
                center,
                panel,
                geometry,
                front_cal,
                camera="front",
                raw_color=raw_color,
                reconstructed_color=reconstructed_color,
                args=args,
            )
            if args.draw_omega:
                omega_arrows_drawn += draw_projected_vector(
                    draw,
                    row,
                    center,
                    panel,
                    geometry,
                    front_cal,
                    camera="front",
                    vector_columns=("omega_x_rad_s", "omega_z_rad_s"),
                    color=omega_color,
                    label="w",
                    length_px=args.omega_normalized_length_px,
                    width=max(1, args.line_width_px - 1),
                )

        if args.view in ("both", "side"):
            center = scaled_point(row, panel, "side")
            draw_marker(
                draw,
                center[0],
                center[1],
                args.marker_radius_px,
                marker_color,
                label,
                max(1, args.line_width_px - 1),
            )
            markers_drawn += 1
            axis_lines_drawn += draw_camera_axes(
                draw,
                row,
                center,
                panel,
                geometry,
                side_cal,
                camera="side",
                raw_color=raw_color,
                reconstructed_color=reconstructed_color,
                args=args,
            )
            if args.draw_omega:
                omega_arrows_drawn += draw_projected_vector(
                    draw,
                    row,
                    center,
                    panel,
                    geometry,
                    side_cal,
                    camera="side",
                    vector_columns=("omega_y_rad_s", "omega_z_rad_s"),
                    color=omega_color,
                    label="w",
                    length_px=args.omega_normalized_length_px,
                    width=max(1, args.line_width_px - 1),
                )

    return markers_drawn, axis_lines_drawn, omega_arrows_drawn


def draw_clean_zoom_panels(
    image: Image.Image,
    source_image: Image.Image,
    rows: pd.DataFrame,
    panel: dict[str, float],
    args: argparse.Namespace,
) -> int:
    particle_ids = sorted(int(particle) for particle in rows["particle"].dropna().unique())
    zoom_state: dict[int, dict[str, Image.Image]] = {}
    update_zoom_panel_state(zoom_state, source_image, rows, panel, args)
    return draw_held_zoom_panels(image, zoom_state, particle_ids, panel)


def update_zoom_panel_state(
    zoom_state: dict[int, dict[str, Image.Image]],
    source_image: Image.Image,
    rows: pd.DataFrame,
    panel: dict[str, float],
    args: argparse.Namespace,
) -> None:
    if rows.empty:
        return
    particle_rows = [
        group.sort_values("frame").iloc[-1]
        for _, group in rows.groupby("particle", sort=True)
    ]
    for row in particle_rows:
        particle = int(row["particle"])
        particle_state = zoom_state.setdefault(particle, {})
        front_crop = crop_clean_zoom(source_image, row, panel, args, "front")
        side_crop = crop_clean_zoom(source_image, row, panel, args, "side")
        if front_crop is not None:
            particle_state["front"] = front_crop.copy()
        if side_crop is not None:
            particle_state["side"] = side_crop.copy()


def draw_held_zoom_panels(
    image: Image.Image,
    zoom_state: dict[int, dict[str, Image.Image]],
    particle_ids: list[int],
    panel: dict[str, float],
) -> int:
    bottom_top = int(
        math.ceil(
            max(
                panel["front_y"] + panel["front_height"],
                panel["side_y"] + panel["side_height"],
            )
        )
    )
    image_width, image_height = image.size
    bottom_height = image_height - bottom_top
    if bottom_height < 40:
        return 0

    if not particle_ids:
        return 0

    draw = ImageDraw.Draw(image)
    draw.rectangle([0, bottom_top, image_width, image_height], fill=(0, 0, 0))

    margin = 8
    slot_count = len(particle_ids)
    slot_width = image_width / slot_count
    panels_drawn = 0
    for slot_index, particle in enumerate(particle_ids):
        slot_x0 = int(round(slot_index * slot_width))
        slot_x1 = int(round((slot_index + 1) * slot_width))
        slot_y0 = bottom_top
        slot_y1 = image_height
        inner_x0 = slot_x0 + margin
        inner_x1 = slot_x1 - margin
        inner_y0 = slot_y0 + margin
        inner_y1 = slot_y1 - margin
        if inner_x1 <= inner_x0 or inner_y1 <= inner_y0:
            continue

        gutter = margin
        half_width = (inner_x1 - inner_x0 - gutter) // 2
        if half_width <= 0:
            continue
        front_box = (inner_x0, inner_y0, inner_x0 + half_width, inner_y1)
        side_box = (inner_x0 + half_width + gutter, inner_y0, inner_x1, inner_y1)

        particle_state = zoom_state.get(particle, {})
        front_crop = particle_state.get("front")
        side_crop = particle_state.get("side")
        if front_crop is not None:
            paste_contained(image, front_crop, front_box)
            panels_drawn += 1
        if side_crop is not None:
            paste_contained(image, side_crop, side_box)
            panels_drawn += 1
    return panels_drawn


def crop_clean_zoom(
    source_image: Image.Image,
    row: pd.Series,
    panel: dict[str, float],
    args: argparse.Namespace,
    camera: Literal["front", "side"],
) -> Image.Image | None:
    center_x, center_y = scaled_point(row, panel, camera)
    crop_width = float(args.zoom_crop_width_px) * panel[f"{camera}_scale_x"]
    crop_height = float(args.zoom_crop_height_px) * panel[f"{camera}_scale_y"]
    if not all(math.isfinite(value) for value in (center_x, center_y, crop_width, crop_height)):
        return None
    if crop_width <= 1 or crop_height <= 1:
        return None
    left = int(round(center_x - crop_width / 2.0))
    top = int(round(center_y - crop_height / 2.0))
    right = int(round(center_x + crop_width / 2.0))
    bottom = int(round(center_y + crop_height / 2.0))
    if right <= left or bottom <= top:
        return None
    return source_image.crop((left, top, right, bottom))


def paste_contained(
    target: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    box_width = x1 - x0
    box_height = y1 - y0
    if box_width <= 0 or box_height <= 0 or source.width <= 0 or source.height <= 0:
        return
    scale = min(box_width / source.width, box_height / source.height)
    new_width = max(1, int(round(source.width * scale)))
    new_height = max(1, int(round(source.height * scale)))
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
    resized = source.resize((new_width, new_height), resample=resampling)
    paste_x = x0 + (box_width - new_width) // 2
    paste_y = y0 + (box_height - new_height) // 2
    target.paste(resized, (paste_x, paste_y))


def scaled_point(
    row: pd.Series,
    panel: dict[str, float],
    camera: Literal["front", "side"],
) -> tuple[float, float]:
    return (
        panel[f"{camera}_x"] + float(row[f"{camera}_image_x_px"]) * panel[f"{camera}_scale_x"],
        panel[f"{camera}_y"] + float(row[f"{camera}_image_y_px"]) * panel[f"{camera}_scale_y"],
    )


def draw_camera_axes(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    center: tuple[float, float],
    panel: dict[str, float],
    geometry: ChamberGeometry,
    calibration: CameraCalibration,
    *,
    camera: Literal["front", "side"],
    raw_color: tuple[int, int, int],
    reconstructed_color: tuple[int, int, int],
    args: argparse.Namespace,
) -> int:
    lines = 0
    if args.axes in ("raw-2d", "both"):
        raw_vector = raw_axis_vector(row, panel, camera)
        if raw_vector is not None:
            draw_axis_line(
                draw,
                center,
                raw_vector,
                args.axis_half_length_px,
                raw_color,
                args.line_width_px,
            )
            lines += 1

    if args.axes in ("reconstructed-3d", "both"):
        projected = projected_reconstructed_axis(row, panel, geometry, calibration, camera)
        if projected is not None:
            draw_axis_line(
                draw,
                center,
                projected,
                args.axis_half_length_px * 0.75 if args.axes == "both" else args.axis_half_length_px,
                reconstructed_color,
                max(1, args.line_width_px - 1),
            )
            lines += 1
    return lines


def raw_axis_vector(
    row: pd.Series,
    panel: dict[str, float],
    camera: Literal["front", "side"],
) -> tuple[float, float] | None:
    u = float(row[f"{camera}_axis_u_raw"])
    v = float(row[f"{camera}_axis_v_raw"])
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return (u * panel[f"{camera}_scale_x"], v * panel[f"{camera}_scale_y"])


def projected_reconstructed_axis(
    row: pd.Series,
    panel: dict[str, float],
    geometry: ChamberGeometry,
    calibration: CameraCalibration,
    camera: Literal["front", "side"],
) -> tuple[float, float] | None:
    n_x = float(row["n_x"])
    n_y = float(row["n_y"])
    n_z = float(row["n_z"])
    z_mm = float(row["z_mm"]) if "z_mm" in row else math.nan
    if not all(math.isfinite(value) for value in (n_x, n_y, n_z, z_mm)):
        return None
    horizontal = n_x if camera == "front" else n_y
    px_per_mm = float(calibration.px_per_mm_at_z(z_mm, geometry))
    z_px_per_mm = z_pixel_slope(calibration, geometry)
    dx = horizontal * px_per_mm * panel[f"{camera}_scale_x"]
    dy = n_z * z_px_per_mm * panel[f"{camera}_scale_y"]
    if not math.isfinite(dx) or not math.isfinite(dy):
        return None
    return (dx, dy)


def draw_projected_vector(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    center: tuple[float, float],
    panel: dict[str, float],
    geometry: ChamberGeometry,
    calibration: CameraCalibration,
    *,
    camera: Literal["front", "side"],
    vector_columns: tuple[str, str],
    color: tuple[int, int, int],
    label: str,
    length_px: float,
    width: int,
) -> int:
    horizontal = float(row[vector_columns[0]])
    vertical = float(row[vector_columns[1]])
    z_mm = float(row["z_mm"]) if "z_mm" in row else math.nan
    if not all(math.isfinite(value) for value in (horizontal, vertical, z_mm)):
        return 0
    dx = horizontal * float(calibration.px_per_mm_at_z(z_mm, geometry)) * panel[f"{camera}_scale_x"]
    dy = vertical * z_pixel_slope(calibration, geometry) * panel[f"{camera}_scale_y"]
    vector = normalize_to_length((dx, dy), length_px)
    if vector is None:
        return 0
    draw_arrow(draw, center, vector, color, width, label)
    return 1


def draw_axis_line(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    vector: tuple[float, float],
    half_length: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    normalized = normalize_to_length(vector, half_length)
    if normalized is None:
        return
    dx, dy = normalized
    x, y = center
    draw.line([x - dx, y - dy, x + dx, y + dy], fill=color, width=width)


def normalize_to_length(
    vector: tuple[float, float],
    length: float,
) -> tuple[float, float] | None:
    dx, dy = vector
    magnitude = math.hypot(dx, dy)
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        return None
    scale = length / magnitude
    return dx * scale, dy * scale


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    vector: tuple[float, float],
    color: tuple[int, int, int],
    width: int,
    label: str,
) -> None:
    start_x, start_y = center
    dx, dy = vector
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


def z_pixel_slope(
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
) -> float:
    bottom_y = calibration.bottom_center_px[1]
    top_y = calibration.top_center_px[1]
    return float((top_y - bottom_y) / geometry.gap_mm)


if __name__ == "__main__":
    raise SystemExit(main())
