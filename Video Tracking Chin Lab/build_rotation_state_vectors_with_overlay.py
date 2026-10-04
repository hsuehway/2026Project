"""Build combined rotation state vectors and a preview overlay from existing CSVs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import estimate_particle_rotation
import verify_rotation_overlay
import verify_state_vectors_overlay
from background_subtraction_tracker import sanitize_output_name
from verify_tracking_overlay import COLORS, draw_frame_label, draw_marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge an existing rotation_state_vectors.csv with an existing "
            "velocities_accelerations.csv, then create a rotation preview overlay."
        )
    )
    parser.add_argument(
        "--rotation-csv",
        type=Path,
        help="rotation_state_vectors.csv. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--motion-csv",
        "--csv",
        dest="motion_csv",
        type=Path,
        help=(
            "velocities_accelerations.csv. If omitted, File Explorer opens. "
            "--csv is accepted as a short alias."
        ),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Background-subtracted preview video. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Calibration JSON. If omitted, the script tries "
            "calibrations\\<trial>_calibration.json for an MI## or SP## trial ID "
            "from the preview/CSV filenames, then calibration.json."
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
        default=Path("outputs_background") / "rotation_state_vectors",
        help="Parent folder for combined state-vector output folders.",
    )
    parser.add_argument(
        "--overlay-output-dir",
        type=Path,
        default=Path("outputs_background") / "rotation_overlays",
        help="Parent folder for preview overlay output folders.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional run folder name used under both output parents.",
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
        choices=verify_rotation_overlay.VIEW_CHOICES,
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
        choices=verify_rotation_overlay.AXIS_CHOICES,
        default="reconstructed-3d",
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
        help="Normalized angular-velocity arrow length.",
    )
    parser.add_argument(
        "--draw-omega",
        action="store_true",
        default=True,
        help="Draw observable tumbling angular velocity as a magenta arrow. Default: on.",
    )
    parser.add_argument(
        "--no-omega",
        dest="draw_omega",
        action="store_false",
        help="Do not draw observable tumbling angular velocity arrows.",
    )
    parser.add_argument(
        "--motion-vector-mode",
        choices=("normalized", "scaled"),
        default="normalized",
        help="Draw velocity/acceleration as fixed-length direction arrows or scaled physical arrows.",
    )
    parser.add_argument(
        "--velocity-scale-s",
        type=float,
        default=0.01,
        help="Scaled velocity multiplier in seconds when --motion-vector-mode scaled is used.",
    )
    parser.add_argument(
        "--acceleration-scale-s2",
        type=float,
        default=0.0005,
        help="Scaled acceleration multiplier in seconds^2 when --motion-vector-mode scaled is used.",
    )
    parser.add_argument(
        "--max-arrow-length-px",
        type=float,
        default=70.0,
        help="Maximum velocity/acceleration arrow length in scaled mode.",
    )
    parser.add_argument(
        "--velocity-normalized-length-px",
        type=float,
        default=38.0,
        help="Velocity arrow length in normalized mode.",
    )
    parser.add_argument(
        "--acceleration-normalized-length-px",
        type=float,
        default=28.0,
        help="Acceleration arrow length in normalized mode.",
    )
    parser.add_argument(
        "--no-state-readout",
        action="store_true",
        help="Do not draw numeric x/v/a/q/w readouts on the overlay.",
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
        help="Native camera-panel crop width for each clean zoom panel.",
    )
    parser.add_argument(
        "--zoom-crop-height-px",
        type=int,
        default=81,
        help="Native camera-panel crop height for each clean zoom panel.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Build particle_reduced_state_vectors.csv without writing the preview overlay.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    rotation_path = args.rotation_csv or choose_csv("Choose rotation_state_vectors.csv")
    motion_path = args.motion_csv or choose_csv("Choose velocities_accelerations.csv")
    preview_path = args.preview
    if preview_path is None and not args.no_overlay:
        preview_path = verify_rotation_overlay.choose_file(
            "Choose background-subtracted preview video",
            [("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )

    rotation = pd.read_csv(rotation_path)
    motion = pd.read_csv(motion_path)
    reduced = estimate_particle_rotation.build_reduced_state_vectors(rotation, motion)

    state_output_dir, overlay_output_dir, run_name = make_output_dirs(args, rotation_path)
    state_output_dir.mkdir(parents=True, exist_ok=True)
    reduced_path = state_output_dir / "particle_reduced_state_vectors.csv"
    reduced.to_csv(reduced_path, index=False)

    overlay_path = None
    overlay_metadata_path = None
    overlay_counts: dict[str, int] = {}
    if not args.no_overlay:
        if preview_path is None:
            raise SystemExit("--preview is required unless --no-overlay is used.")
        overlay_output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path, overlay_metadata_path, overlay_counts = write_preview_overlay(
            args=args,
            rotation=rotation,
            motion=motion,
            reduced=reduced,
            rotation_path=rotation_path,
            motion_path=motion_path,
            preview_path=preview_path,
            output_dir=overlay_output_dir,
        )

    write_metadata(
        state_output_dir / "metadata.json",
        run_name=run_name,
        rotation_path=rotation_path,
        motion_path=motion_path,
        preview_path=preview_path,
        reduced_path=reduced_path,
        state_output_dir=state_output_dir,
        overlay_output_dir=overlay_output_dir if not args.no_overlay else None,
        overlay_path=overlay_path,
        overlay_metadata_path=overlay_metadata_path,
        reduced_rows=len(reduced),
        overlay_counts=overlay_counts,
    )

    print("\nDone.")
    print(f"Rotation CSV:    {rotation_path.resolve()}")
    print(f"Motion CSV:      {motion_path.resolve()}")
    if preview_path is not None:
        print(f"Preview:         {preview_path.resolve()}")
    print(f"Reduced states:  {reduced_path.resolve()}")
    print(f"State folder:    {state_output_dir.resolve()}")
    if overlay_path is not None:
        print(f"Overlay video:   {overlay_path.resolve()}")
        print(f"Overlay folder:  {overlay_output_dir.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    overlay_args = SimpleNamespace(
        preview_start_frame=args.preview_start_frame,
        preview_frame_step=args.preview_frame_step,
        axis_half_length_px=args.axis_half_length_px,
        omega_normalized_length_px=args.omega_normalized_length_px,
        marker_radius_px=args.marker_radius_px,
        line_width_px=args.line_width_px,
        max_frames=args.max_frames,
        zoom_crop_width_px=args.zoom_crop_width_px,
        zoom_crop_height_px=args.zoom_crop_height_px,
        front_width=args.front_width,
        front_height=args.front_height,
        side_width=args.side_width,
        side_height=args.side_height,
    )
    verify_rotation_overlay.validate_args(overlay_args)
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


def choose_csv(title: str) -> Path:
    return verify_rotation_overlay.choose_file(
        title,
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )


def make_output_dirs(
    args: argparse.Namespace,
    rotation_path: Path,
) -> tuple[Path, Path, str]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        args.overlay_output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        base_name = sanitize_output_name(args.output_name)
    else:
        source_name = sanitize_output_name(rotation_path.parent.name or rotation_path.stem)
        base_name = f"rotation_state_vectors_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    suffix = 1
    while True:
        run_name = base_name if suffix == 1 else f"{base_name}_{suffix}"
        state_output_dir = args.output_dir / run_name
        overlay_output_dir = args.overlay_output_dir / run_name
        overlay_ok = args.no_overlay or not overlay_output_dir.exists()
        if not state_output_dir.exists() and overlay_ok:
            return state_output_dir, overlay_output_dir, run_name
        suffix += 1


def write_preview_overlay(
    *,
    args: argparse.Namespace,
    rotation: pd.DataFrame,
    motion: pd.DataFrame,
    reduced: pd.DataFrame,
    rotation_path: Path,
    motion_path: Path,
    preview_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, int]]:
    calibration_path = verify_rotation_overlay.resolve_calibration_path(
        SimpleNamespace(
            calibration=args.calibration,
            tracks_csv=motion_path,
            calibrations_dir=args.calibrations_dir,
        ),
        preview_path,
        rotation_path,
    )
    geometry, front_cal, side_cal, front_roi, side_roi, _, _ = (
        verify_rotation_overlay.read_calibration(calibration_path)
    )
    if front_roi is None or side_roi is None:
        raise SystemExit(f"{calibration_path} must contain front_roi and side_roi.")

    rows = verify_rotation_overlay.prepare_overlay_rows(
        reduced,
        None,
        rotation_path,
        None,
    )
    if rows.empty:
        raise SystemExit(
            "No matching particle/frame rows between rotation CSV and motion CSV."
        )

    overlay_args = SimpleNamespace(
        preview_start_frame=args.preview_start_frame,
        preview_frame_step=args.preview_frame_step,
        view=args.view,
        layout=args.layout,
        axes=args.axes,
        axis_half_length_px=args.axis_half_length_px,
        omega_normalized_length_px=args.omega_normalized_length_px,
        draw_omega=args.draw_omega,
        motion_vector_mode=args.motion_vector_mode,
        velocity_scale_s=args.velocity_scale_s,
        acceleration_scale_s2=args.acceleration_scale_s2,
        max_arrow_length_px=args.max_arrow_length_px,
        velocity_normalized_length_px=args.velocity_normalized_length_px,
        acceleration_normalized_length_px=args.acceleration_normalized_length_px,
        no_state_readout=args.no_state_readout,
        marker_radius_px=args.marker_radius_px,
        line_width_px=args.line_width_px,
        max_frames=args.max_frames,
        front_width=args.front_width,
        front_height=args.front_height,
        side_width=args.side_width,
        side_height=args.side_height,
        no_zoom_panels=args.no_zoom_panels,
        zoom_crop_width_px=args.zoom_crop_width_px,
        zoom_crop_height_px=args.zoom_crop_height_px,
    )
    panel = verify_rotation_overlay.resolve_panel_geometry(
        overlay_args,
        front_roi,
        side_roi,
    )
    output_path = output_dir / f"{sanitize_output_name(preview_path.stem)}_rotation_state_vectors_overlay.mp4"
    overlay_counts = write_full_state_overlay_video(
        preview_path=preview_path,
        output_path=output_path,
        rows=rows,
        panel=panel,
        geometry=geometry,
        front_cal=front_cal,
        side_cal=side_cal,
        args=overlay_args,
    )

    metadata_path = output_dir / "preview_rotation_overlay_metadata.json"
    metadata = {
        "preview": str(preview_path.resolve()),
        "rotation_csv": str(rotation_path.resolve()),
        "motion_csv": str(motion_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "output_video": str(output_path.resolve()),
        "preview_start_frame": args.preview_start_frame,
        "preview_frame_step": args.preview_frame_step,
        "view": args.view,
        "layout": args.layout,
        "axes": args.axes,
        "axis_half_length_px": args.axis_half_length_px,
        "draw_omega": bool(args.draw_omega),
        "motion_vector_mode": args.motion_vector_mode,
        "velocity_scale_s": args.velocity_scale_s,
        "acceleration_scale_s2": args.acceleration_scale_s2,
        "velocity_normalized_length_px": args.velocity_normalized_length_px,
        "acceleration_normalized_length_px": args.acceleration_normalized_length_px,
        "state_readout_enabled": not args.no_state_readout,
        "matched_rows": int(len(rows)),
        **overlay_counts,
        "zoom_panels_enabled": not args.no_zoom_panels,
        "zoom_crop_width_px": int(args.zoom_crop_width_px),
        "zoom_crop_height_px": int(args.zoom_crop_height_px),
        "panel": panel,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return output_path, metadata_path, {"matched_rows": int(len(rows)), **overlay_counts}


def write_full_state_overlay_video(
    *,
    preview_path: Path,
    output_path: Path,
    rows: pd.DataFrame,
    panel: dict[str, float],
    geometry,
    front_cal,
    side_cal,
    args: argparse.Namespace,
) -> dict[str, int]:
    reader = imageio.get_reader(str(preview_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    rows_by_frame = {
        int(frame): group.copy()
        for frame, group in rows.groupby("frame", sort=False)
    }
    particle_ids = sorted(int(particle) for particle in rows["particle"].dropna().unique())
    zoom_state: dict[int, dict[str, Image.Image]] = {}
    latest_state_rows: dict[int, pd.Series] = {}
    counts = {
        "frames_written": 0,
        "markers_drawn": 0,
        "velocity_arrows_drawn": 0,
        "acceleration_arrows_drawn": 0,
        "axis_lines_drawn": 0,
        "omega_arrows_drawn": 0,
        "zoom_panels_drawn": 0,
        "state_readout_rows_drawn": 0,
    }

    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        try:
            for preview_index, frame in enumerate(reader):
                if args.max_frames is not None and counts["frames_written"] >= args.max_frames:
                    break
                track_frame = args.preview_start_frame + preview_index * args.preview_frame_step
                source_image = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
                image = source_image.copy()
                draw = ImageDraw.Draw(image)
                frame_rows = rows_by_frame.get(track_frame)

                if frame_rows is not None and not frame_rows.empty:
                    for _, row in frame_rows.groupby("particle", sort=True).tail(1).iterrows():
                        latest_state_rows[int(row["particle"])] = row
                    frame_counts = draw_full_state_rows(
                        draw,
                        frame_rows,
                        panel,
                        geometry,
                        front_cal,
                        side_cal,
                        args,
                    )
                    for key, value in frame_counts.items():
                        counts[key] += value
                    if not args.no_zoom_panels:
                        verify_rotation_overlay.update_zoom_panel_state(
                            zoom_state,
                            source_image,
                            frame_rows,
                            panel,
                            args,
                        )

                if not args.no_zoom_panels:
                    counts["zoom_panels_drawn"] += verify_rotation_overlay.draw_held_zoom_panels(
                        image,
                        zoom_state,
                        particle_ids,
                        panel,
                    )
                if not args.no_state_readout:
                    counts["state_readout_rows_drawn"] += draw_state_readout(
                        image,
                        latest_state_rows,
                    )
                draw_frame_label(draw, track_frame)
                writer.append_data(np.asarray(image))
                counts["frames_written"] += 1
        finally:
            reader.close()

    return counts


def draw_full_state_rows(
    draw: ImageDraw.ImageDraw,
    rows: pd.DataFrame,
    panel: dict[str, float],
    geometry,
    front_cal,
    side_cal,
    args: argparse.Namespace,
) -> dict[str, int]:
    counts = {
        "markers_drawn": 0,
        "velocity_arrows_drawn": 0,
        "acceleration_arrows_drawn": 0,
        "axis_lines_drawn": 0,
        "omega_arrows_drawn": 0,
    }
    for _, row in rows.iterrows():
        particle = int(row["particle"])
        marker_color = COLORS[particle % len(COLORS)]
        valid = bool(row.get("rotation_valid", True))
        raw_color = (120, 230, 255) if valid else (255, 90, 90)
        reconstructed_color = (255, 230, 60) if valid else (255, 90, 90)

        if args.view in ("both", "front"):
            center = verify_rotation_overlay.scaled_point(row, panel, "front")
            draw_marker(
                draw,
                center[0],
                center[1],
                args.marker_radius_px,
                marker_color,
                f"P{particle}",
                max(1, args.line_width_px - 1),
            )
            counts["markers_drawn"] += 1
            counts["axis_lines_drawn"] += verify_rotation_overlay.draw_camera_axes(
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
            counts["velocity_arrows_drawn"] += draw_motion_arrow(
                draw,
                row,
                center,
                front_cal,
                geometry,
                horizontal_axis="x",
                kind="velocity",
                args=args,
            )
            counts["acceleration_arrows_drawn"] += draw_motion_arrow(
                draw,
                row,
                center,
                front_cal,
                geometry,
                horizontal_axis="x",
                kind="acceleration",
                args=args,
            )
            if args.draw_omega:
                counts["omega_arrows_drawn"] += verify_rotation_overlay.draw_projected_vector(
                    draw,
                    row,
                    center,
                    panel,
                    geometry,
                    front_cal,
                    camera="front",
                    vector_columns=("omega_x_rad_s", "omega_z_rad_s"),
                    color=(255, 80, 220),
                    label="w",
                    length_px=args.omega_normalized_length_px,
                    width=max(1, args.line_width_px - 1),
                )

        if args.view in ("both", "side"):
            center = verify_rotation_overlay.scaled_point(row, panel, "side")
            draw_marker(
                draw,
                center[0],
                center[1],
                args.marker_radius_px,
                marker_color,
                f"P{particle}",
                max(1, args.line_width_px - 1),
            )
            counts["markers_drawn"] += 1
            counts["axis_lines_drawn"] += verify_rotation_overlay.draw_camera_axes(
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
            counts["velocity_arrows_drawn"] += draw_motion_arrow(
                draw,
                row,
                center,
                side_cal,
                geometry,
                horizontal_axis="y",
                kind="velocity",
                args=args,
            )
            counts["acceleration_arrows_drawn"] += draw_motion_arrow(
                draw,
                row,
                center,
                side_cal,
                geometry,
                horizontal_axis="y",
                kind="acceleration",
                args=args,
            )
            if args.draw_omega:
                counts["omega_arrows_drawn"] += verify_rotation_overlay.draw_projected_vector(
                    draw,
                    row,
                    center,
                    panel,
                    geometry,
                    side_cal,
                    camera="side",
                    vector_columns=("omega_y_rad_s", "omega_z_rad_s"),
                    color=(255, 80, 220),
                    label="w",
                    length_px=args.omega_normalized_length_px,
                    width=max(1, args.line_width_px - 1),
                )
    return counts


def draw_motion_arrow(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    center: tuple[float, float],
    calibration,
    geometry,
    *,
    horizontal_axis: str,
    kind: str,
    args: argparse.Namespace,
) -> int:
    dx, dy = verify_state_vectors_overlay.vector_components_to_pixels(
        row,
        calibration,
        geometry,
        horizontal_axis=horizontal_axis,
        kind=kind,
        normalized=args.motion_vector_mode == "normalized",
        args=args,
    )
    if dx is None or dy is None:
        return 0
    color = (60, 230, 255) if kind == "velocity" else (255, 145, 40)
    label = "v" if kind == "velocity" else "a"
    width = args.line_width_px if kind == "velocity" else max(1, args.line_width_px - 1)
    verify_state_vectors_overlay.draw_arrow(
        draw,
        center[0],
        center[1],
        dx,
        dy,
        color=color,
        width=width,
        label=label,
    )
    return 1


def draw_state_readout(
    image: Image.Image,
    latest_state_rows: dict[int, pd.Series],
) -> int:
    if not latest_state_rows:
        return 0
    draw = ImageDraw.Draw(image)
    lines = ["state overlay: pos=P  v=cyan  a=orange  n/q=yellow/text  w=magenta"]
    for particle in sorted(latest_state_rows):
        row = latest_state_rows[particle]
        lines.append(
            f"P{particle} x={format_triplet(row, ('x_mm', 'y_mm', 'z_mm'))} "
            f"v={format_triplet(row, ('vx_mm_s', 'vy_mm_s', 'vz_mm_s'))} "
            f"a={format_triplet(row, ('ax_mm_s2', 'ay_mm_s2', 'az_mm_s2'))}"
        )
        lines.append(
            f"   q={format_tuple(row, ('q_w', 'q_x', 'q_y', 'q_z'))} "
            f"w={format_triplet(row, ('omega_x_rad_s', 'omega_y_rad_s', 'omega_z_rad_s'))}"
        )

    x = 10
    y = 48
    line_height = 13
    max_width = max((text_size(draw, line)[0] for line in lines), default=0)
    box_height = line_height * len(lines) + 8
    draw.rectangle(
        [x - 4, y - 4, x + max_width + 8, y + box_height],
        fill=(0, 0, 0),
    )
    for index, line in enumerate(lines):
        fill = (240, 240, 240)
        if line.startswith("P"):
            try:
                particle = int(line.split()[0][1:])
                fill = COLORS[particle % len(COLORS)]
            except (ValueError, IndexError):
                fill = (240, 240, 240)
        draw.text((x, y + index * line_height), line, fill=fill)
    return len(latest_state_rows)


def text_size(draw: ImageDraw.ImageDraw, text: str) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text)
        return right - left, bottom - top
    return draw.textsize(text)


def format_triplet(row: pd.Series, columns: tuple[str, str, str]) -> str:
    return "(" + ",".join(format_number(row.get(column, math.nan)) for column in columns) + ")"


def format_tuple(row: pd.Series, columns: tuple[str, ...]) -> str:
    return "(" + ",".join(format_number(row.get(column, math.nan)) for column in columns) + ")"


def format_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number:.3g}"


def write_metadata(
    output_path: Path,
    *,
    run_name: str,
    rotation_path: Path,
    motion_path: Path,
    preview_path: Path | None,
    reduced_path: Path,
    state_output_dir: Path,
    overlay_output_dir: Path | None,
    overlay_path: Path | None,
    overlay_metadata_path: Path | None,
    reduced_rows: int,
    overlay_counts: dict[str, int],
) -> None:
    metadata = {
        "run_name": run_name,
        "rotation_csv": str(rotation_path.resolve()),
        "motion_csv": str(motion_path.resolve()),
        "preview": str(preview_path.resolve()) if preview_path is not None else None,
        "reduced_state_csv": str(reduced_path.resolve()),
        "state_output_dir": str(state_output_dir.resolve()),
        "overlay_output_dir": (
            str(overlay_output_dir.resolve()) if overlay_output_dir is not None else None
        ),
        "overlay_video": str(overlay_path.resolve()) if overlay_path is not None else None,
        "overlay_metadata": (
            str(overlay_metadata_path.resolve())
            if overlay_metadata_path is not None
            else None
        ),
        "reduced_rows": int(reduced_rows),
        **overlay_counts,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
