"""Estimate reduced particle rotation states from tracked two-camera videos."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import imageio.v2 as imageio
import numpy as np
import pandas as pd

import verify_rotation_overlay as preview_rotation_overlay
from background_subtraction_tracker import (
    difference_image,
    read_background_frame,
    sanitize_output_name,
)
from particle_rotation import (
    Axis2DConfig,
    ReconstructionConfig,
    accumulate_minimum_twist_quaternions,
    estimate_axis_2d,
    observable_tumbling_angular_velocity,
    reconstruct_axis_3d,
)
from track_orthogonal_3d import (
    ROI,
    CameraCalibration,
    ChamberGeometry,
    calibration_path_for_video_id,
    infer_video_id,
    read_calibration,
)


CHANNELS = ("red", "green", "blue", "mean")
DIFFERENCE_MODES = ("abs", "bright", "dark")
OVERLAY_VIEWS = ("both", "front", "side")
OVERLAY_AXES = ("raw-2d", "reconstructed-3d", "both")
REQUIRED_TRACK_COLUMNS = (
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
MOTION_COLUMNS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "ax_mm_s2",
    "ay_mm_s2",
    "az_mm_s2",
)
ROTATION_STATE_COLUMNS = (
    "n_x",
    "n_y",
    "n_z",
    "omega_x_rad_s",
    "omega_y_rad_s",
    "omega_z_rad_s",
)
REDUCED_STATE_COLUMNS = (*MOTION_COLUMNS, *ROTATION_STATE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate minimum-twist reduced-attitude quaternions and observable "
            "tumbling angular velocity from two orthogonal camera views."
        )
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--video",
        type=Path,
        help="Combined video containing front and side camera panels. If omitted, File Explorer opens.",
    )
    input_group.add_argument(
        "--front",
        type=Path,
        help="Front-camera video for the two-video workflow. Requires --side.",
    )
    parser.add_argument(
        "--side",
        type=Path,
        help="Side-camera video for the two-video workflow.",
    )
    parser.add_argument(
        "--tracks-csv",
        type=Path,
        help="tracks_3d.csv from track_orthogonal_3d.py or background_subtraction_tracker.py.",
    )
    parser.add_argument(
        "--motion-csv",
        type=Path,
        help="Optional velocities_accelerations.csv to merge into reduced state vectors.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Calibration JSON. If omitted, calibrations\\<trial>_calibration.json "
            "is inferred from an MI## or SP## trial ID."
        ),
    )
    parser.add_argument(
        "--calibrations-dir",
        type=Path,
        default=Path("calibrations"),
        help="Folder containing per-video calibration JSON files.",
    )
    parser.add_argument(
        "--background-file",
        type=Path,
        help=(
            "Background image/video. For combined videos this should share the same "
            "layout as --video. Videos use --background-frame."
        ),
    )
    parser.add_argument(
        "--front-background-file",
        type=Path,
        help="Optional front-camera background image/video for --front/--side mode.",
    )
    parser.add_argument(
        "--side-background-file",
        type=Path,
        help="Optional side-camera background image/video for --front/--side mode.",
    )
    parser.add_argument(
        "--background-frame",
        type=int,
        default=0,
        help="Frame index used when reading a background from a video.",
    )
    parser.add_argument(
        "--video-frame-offset",
        type=int,
        default=0,
        help=(
            "Video frame index offset: video_index = track frame + offset. "
            "Use -1 if tracks frame 1 corresponds to video frame 0."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Fallback frame rate when tracks do not contain time_s.",
    )
    parser.add_argument(
        "--channel",
        choices=CHANNELS,
        default="green",
        help="Color channel used for background subtraction and PCA.",
    )
    parser.add_argument(
        "--difference-mode",
        choices=DIFFERENCE_MODES,
        default="abs",
        help="abs, bright=frame-background, or dark=background-frame.",
    )
    parser.add_argument(
        "--particles",
        type=parse_particle_list,
        help="Optional comma-separated particle IDs. Default: all particles in tracks CSV.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        help="Optional first track frame to process.",
    )
    parser.add_argument(
        "--frame-stop",
        type=int,
        help="Optional last track frame to process, inclusive.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum distinct track frames to process.",
    )
    parser.add_argument(
        "--crop-width-px",
        type=int,
        default=41,
        help="Particle crop width in native camera-panel pixels.",
    )
    parser.add_argument(
        "--crop-height-px",
        type=int,
        default=41,
        help="Particle crop height in native camera-panel pixels.",
    )
    parser.add_argument(
        "--front-crop-width-px",
        type=int,
        help="Override --crop-width-px for the front camera.",
    )
    parser.add_argument(
        "--front-crop-height-px",
        type=int,
        help="Override --crop-height-px for the front camera.",
    )
    parser.add_argument(
        "--side-crop-width-px",
        type=int,
        help="Override --crop-width-px for the side camera.",
    )
    parser.add_argument(
        "--side-crop-height-px",
        type=int,
        help="Override --crop-height-px for the side camera.",
    )
    parser.add_argument(
        "--gaussian-sigma-px",
        type=float,
        default=0.8,
        help="Small blur applied before threshold/PCA. Use 0 to disable.",
    )
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=0.35,
        help="Foreground threshold as a fraction of robust normalized crop maximum.",
    )
    parser.add_argument(
        "--front-threshold-fraction",
        type=float,
        help="Override --threshold-fraction for the front camera.",
    )
    parser.add_argument(
        "--side-threshold-fraction",
        type=float,
        help="Override --threshold-fraction for the side camera.",
    )
    parser.add_argument(
        "--minimum-foreground-pixels",
        type=int,
        default=20,
        help="Minimum crop foreground pixels required for a 2D axis.",
    )
    parser.add_argument(
        "--front-minimum-foreground-pixels",
        type=int,
        help="Override --minimum-foreground-pixels for the front camera.",
    )
    parser.add_argument(
        "--side-minimum-foreground-pixels",
        type=int,
        help="Override --minimum-foreground-pixels for the side camera.",
    )
    parser.add_argument(
        "--minimum-anisotropy",
        type=float,
        default=0.15,
        help="Minimum 2D covariance anisotropy for each projected axis.",
    )
    parser.add_argument(
        "--front-minimum-anisotropy",
        type=float,
        help="Override --minimum-anisotropy for the front camera.",
    )
    parser.add_argument(
        "--side-minimum-anisotropy",
        type=float,
        help="Override --minimum-anisotropy for the side camera.",
    )
    parser.add_argument(
        "--maximum-estimator-disagreement-deg",
        type=float,
        default=20.0,
        help="Reject frames where binary and intensity PCA axes disagree more than this.",
    )
    parser.add_argument(
        "--maximum-reprojection-error-deg",
        type=float,
        default=15.0,
        help="Maximum per-camera reprojection line error for 3D axis validity.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "rotation",
        help="Parent folder for timestamped rotation output folders.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output folder name.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic PNG plots.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip writing a diagnostic overlay video for combined-video inputs.",
    )
    parser.add_argument(
        "--overlay-axis-length-px",
        type=float,
        default=24.0,
        help="Half-length of the diagnostic 2D axis overlay lines.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help=(
            "Background-subtracted preview video for verification overlay. If omitted, "
            "File Explorer opens unless --no-preview-overlay is used."
        ),
    )
    parser.add_argument(
        "--no-preview-overlay",
        action="store_true",
        help="Skip the background-subtracted preview verification overlay.",
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
        choices=OVERLAY_VIEWS,
        default="both",
        help="Which preview panel(s) to draw in the preview verification overlay.",
    )
    parser.add_argument(
        "--layout",
        choices=("side-by-side", "stacked"),
        default="side-by-side",
        help="Preview panel layout for the preview verification overlay.",
    )
    parser.add_argument(
        "--axes",
        choices=OVERLAY_AXES,
        default="both",
        help="Draw raw 2D axes, reconstructed projected 3D axes, or both in the preview overlay.",
    )
    parser.add_argument(
        "--axis-half-length-px",
        type=float,
        default=26.0,
        help="Half-length of preview verification axis line segments in preview pixels.",
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
        help="Also draw observable tumbling angular velocity on the preview overlay.",
    )
    parser.add_argument(
        "--marker-radius-px",
        type=int,
        default=5,
        help="Preview overlay position marker radius in pixels.",
    )
    parser.add_argument(
        "--line-width-px",
        type=int,
        default=3,
        help="Preview overlay axis and marker line width.",
    )
    parser.add_argument(
        "--preview-overlay-max-frames",
        type=int,
        help="Maximum preview frames to write. Defaults to the whole preview video.",
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
        help="Do not replace the lower overlay area with clean per-particle zoom crops.",
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
    return parser.parse_args()


def parse_particle_list(value: str) -> tuple[int, ...]:
    try:
        particles = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Particles must be comma-separated integers.") from exc
    if not particles:
        raise argparse.ArgumentTypeError("Choose at least one particle ID.")
    if len(set(particles)) != len(particles):
        raise argparse.ArgumentTypeError("Particle IDs must not repeat.")
    return particles


def main() -> int:
    args = parse_args()
    validate_args(args)

    resolve_video_inputs(args)
    tracks_path = args.tracks_csv or choose_file(
        "Choose tracks_3d.csv",
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )
    calibration_path = resolve_calibration_path(args, tracks_path)
    geometry, front_cal, side_cal, front_roi, side_roi, _, _ = read_calibration(
        calibration_path
    )
    if args.video is not None and (front_roi is None or side_roi is None):
        raise SystemExit(
            f"{calibration_path} must contain front_roi and side_roi for --video mode."
        )

    tracks = load_tracks(tracks_path, args)
    axis_config = Axis2DConfig(
        crop_width_px=args.crop_width_px,
        crop_height_px=args.crop_height_px,
        gaussian_sigma_px=args.gaussian_sigma_px,
        threshold_fraction=args.threshold_fraction,
        minimum_foreground_pixels=args.minimum_foreground_pixels,
        minimum_anisotropy=args.minimum_anisotropy,
        maximum_estimator_disagreement_rad=math.radians(
            args.maximum_estimator_disagreement_deg
        ),
    )
    front_axis_config = replace(
        axis_config,
        crop_width_px=args.front_crop_width_px or axis_config.crop_width_px,
        crop_height_px=args.front_crop_height_px or axis_config.crop_height_px,
        threshold_fraction=(
            args.front_threshold_fraction
            if args.front_threshold_fraction is not None
            else axis_config.threshold_fraction
        ),
        minimum_foreground_pixels=(
            args.front_minimum_foreground_pixels
            if args.front_minimum_foreground_pixels is not None
            else axis_config.minimum_foreground_pixels
        ),
        minimum_anisotropy=(
            args.front_minimum_anisotropy
            if args.front_minimum_anisotropy is not None
            else axis_config.minimum_anisotropy
        ),
    )
    side_axis_config = replace(
        axis_config,
        crop_width_px=args.side_crop_width_px or axis_config.crop_width_px,
        crop_height_px=args.side_crop_height_px or axis_config.crop_height_px,
        threshold_fraction=(
            args.side_threshold_fraction
            if args.side_threshold_fraction is not None
            else axis_config.threshold_fraction
        ),
        minimum_foreground_pixels=(
            args.side_minimum_foreground_pixels
            if args.side_minimum_foreground_pixels is not None
            else axis_config.minimum_foreground_pixels
        ),
        minimum_anisotropy=(
            args.side_minimum_anisotropy
            if args.side_minimum_anisotropy is not None
            else axis_config.minimum_anisotropy
        ),
    )
    reconstruction_config = ReconstructionConfig(
        maximum_reprojection_error_rad=math.radians(args.maximum_reprojection_error_deg)
    )

    rotation = run_rotation_estimation(
        args=args,
        tracks=tracks,
        geometry=geometry,
        front_cal=front_cal,
        side_cal=side_cal,
        front_roi=front_roi,
        side_roi=side_roi,
        front_axis_config=front_axis_config,
        side_axis_config=side_axis_config,
        reconstruction_config=reconstruction_config,
    )

    output_dir = make_output_dir(args.output_dir, args.output_name, tracks_path, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_output_dir = make_overlay_output_dir(output_dir)
    rotation_path = output_dir / "rotation_state_vectors.csv"
    rotation.to_csv(rotation_path, index=False)

    reduced_path = None
    if args.motion_csv is not None:
        reduced = build_reduced_state_vectors(rotation, pd.read_csv(args.motion_csv))
        reduced_path = output_dir / "particle_reduced_state_vectors.csv"
        reduced.to_csv(reduced_path, index=False)

    plots_path = None
    if not args.no_plots:
        plots_path = output_dir / "rotation_diagnostics.png"
        plot_rotation_diagnostics(rotation, plots_path)

    overlay_path = None
    if not args.no_overlay and args.video is not None:
        overlay_output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_output_dir / "rotation_axis_overlay.mp4"
        write_overlay_video(args, rotation, front_roi, side_roi, overlay_path)

    preview_overlay_path = None
    preview_overlay_metadata_path = None
    if not args.no_preview_overlay:
        overlay_output_dir.mkdir(parents=True, exist_ok=True)
        preview_overlay_path, preview_overlay_metadata_path = write_preview_overlay_outputs(
            args=args,
            rotation=rotation,
            rotation_path=rotation_path,
            tracks_path=tracks_path,
            calibration_path=calibration_path,
            geometry=geometry,
            front_cal=front_cal,
            side_cal=side_cal,
            front_roi=front_roi,
            side_roi=side_roi,
            output_dir=overlay_output_dir,
        )

    metadata_path = output_dir / "metadata.json"
    write_metadata(
        metadata_path,
        args,
        tracks_path,
        calibration_path,
        rotation,
        front_axis_config,
        side_axis_config,
        reconstruction_config,
        rotation_path,
        reduced_path,
        plots_path,
        overlay_path,
        preview_overlay_path,
        preview_overlay_metadata_path,
    )

    print("\nDone.")
    print(f"Tracks CSV:       {tracks_path.resolve()}")
    print(f"Calibration:      {calibration_path.resolve()}")
    print(f"Rows:             {len(rotation)}")
    print(f"Valid rotations:  {int(rotation['rotation_valid'].sum())}")
    print(f"Output folder:    {output_dir.resolve()}")
    print(f"Rotation CSV:     {rotation_path.resolve()}")
    if reduced_path is not None:
        print(f"Reduced states:   {reduced_path.resolve()}")
    if overlay_path is not None:
        print(f"Overlay video:    {overlay_path.resolve()}")
    if preview_overlay_path is not None:
        print(f"Preview overlay:  {preview_overlay_path.resolve()}")
    if overlay_path is not None or preview_overlay_path is not None:
        print(f"Overlay folder:   {overlay_output_dir.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.front is not None and args.side is None:
        raise SystemExit("--front requires --side.")
    if args.side is not None and args.front is None:
        raise SystemExit("--side requires --front.")
    if args.background_frame < 0:
        raise SystemExit("--background-frame must be zero or positive.")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.crop_width_px <= 2 or args.crop_height_px <= 2:
        raise SystemExit("--crop-width-px and --crop-height-px must be greater than 2.")
    for name in (
        "front_crop_width_px",
        "front_crop_height_px",
        "side_crop_width_px",
        "side_crop_height_px",
    ):
        value = getattr(args, name)
        if value is not None and value <= 2:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than 2.")
    for name in ("front_threshold_fraction", "side_threshold_fraction"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative.")
    if args.minimum_foreground_pixels <= 0:
        raise SystemExit("--minimum-foreground-pixels must be positive.")
    for name in ("front_minimum_foreground_pixels", "side_minimum_foreground_pixels"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if args.minimum_anisotropy < 0:
        raise SystemExit("--minimum-anisotropy must be non-negative.")
    for name in ("front_minimum_anisotropy", "side_minimum_anisotropy"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative.")
    if args.maximum_estimator_disagreement_deg <= 0:
        raise SystemExit("--maximum-estimator-disagreement-deg must be positive.")
    if args.maximum_reprojection_error_deg <= 0:
        raise SystemExit("--maximum-reprojection-error-deg must be positive.")
    if args.frame_start is not None and args.frame_stop is not None:
        if args.frame_stop < args.frame_start:
            raise SystemExit("--frame-stop must be greater than or equal to --frame-start.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")
    if args.overlay_axis_length_px <= 0:
        raise SystemExit("--overlay-axis-length-px must be positive.")
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
    if args.preview_overlay_max_frames is not None and args.preview_overlay_max_frames <= 0:
        raise SystemExit("--preview-overlay-max-frames must be positive.")
    if args.zoom_crop_width_px <= 2:
        raise SystemExit("--zoom-crop-width-px must be greater than 2.")
    if args.zoom_crop_height_px <= 2:
        raise SystemExit("--zoom-crop-height-px must be greater than 2.")
    for name in ("front_width", "front_height", "side_width", "side_height"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")


def resolve_video_inputs(args: argparse.Namespace) -> None:
    if args.video is not None or args.front is not None:
        return
    args.video = choose_file(
        "Choose combined front/side video",
        [("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
    )


def choose_file(title: str, filetypes: list[tuple[str, str]]) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(f"{title}: pass an explicit path because Tk is unavailable ({exc}).") from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not path:
        raise SystemExit(f"No file selected for: {title}")
    return Path(path)


def resolve_calibration_path(args: argparse.Namespace, tracks_path: Path) -> Path:
    if args.calibration is not None:
        return args.calibration
    for path in (args.video, args.front, tracks_path):
        if path is None:
            continue
        video_id = infer_video_id(path)
        if video_id is None:
            continue
        candidate = calibration_path_for_video_id(video_id, args.calibrations_dir)
        if candidate.exists():
            return candidate
    fallback = Path("calibration.json")
    if fallback.exists():
        return fallback
    raise SystemExit("Could not infer calibration. Pass --calibration.")


def load_tracks(tracks_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    tracks = pd.read_csv(tracks_path)
    missing = [column for column in REQUIRED_TRACK_COLUMNS if column not in tracks.columns]
    if missing:
        raise SystemExit(f"{tracks_path} is missing required columns: {missing}")
    if tracks.empty:
        raise SystemExit(f"{tracks_path} is empty.")

    tracks = tracks.copy()
    tracks["particle"] = tracks["particle"].astype(int)
    tracks["frame"] = tracks["frame"].astype(int)
    if args.particles is not None:
        tracks = tracks.loc[tracks["particle"].isin(args.particles)].copy()
    if args.frame_start is not None:
        tracks = tracks.loc[tracks["frame"] >= args.frame_start].copy()
    if args.frame_stop is not None:
        tracks = tracks.loc[tracks["frame"] <= args.frame_stop].copy()
    if args.max_frames is not None:
        frames = sorted(tracks["frame"].unique().tolist())[: args.max_frames]
        tracks = tracks.loc[tracks["frame"].isin(frames)].copy()
    if tracks.empty:
        raise SystemExit("No track rows remain after filtering.")
    if "time_s" not in tracks.columns or tracks["time_s"].isna().all():
        tracks["time_s"] = tracks["frame"].astype(float) / float(args.fps)
    return tracks.sort_values(["frame", "particle"]).reset_index(drop=True)


def run_rotation_estimation(
    *,
    args: argparse.Namespace,
    tracks: pd.DataFrame,
    geometry: ChamberGeometry,
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    front_roi: ROI | None,
    side_roi: ROI | None,
    front_axis_config: Axis2DConfig,
    side_axis_config: Axis2DConfig,
    reconstruction_config: ReconstructionConfig,
) -> pd.DataFrame:
    provider = FrameProvider(
        args=args,
        front_roi=front_roi,
        side_roi=side_roi,
    )
    front_background, side_background = provider.read_background_panels()
    previous_axis_by_particle: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    try:
        for frame_number, frame_rows in tracks.groupby("frame", sort=True):
            video_index = int(frame_number) + int(args.video_frame_offset)
            if video_index < 0:
                continue
            front_gray, side_gray = provider.read_camera_panels(video_index)
            front_diff = difference_image(front_gray, front_background, args.difference_mode)
            side_diff = difference_image(side_gray, side_background, args.difference_mode)

            for _, track_row in frame_rows.sort_values("particle").iterrows():
                particle = int(track_row["particle"])
                front_axis = estimate_axis_2d(
                    front_diff,
                    float(track_row["front_image_x_px"]),
                    float(track_row["front_image_y_px"]),
                    front_axis_config,
                )
                side_axis = estimate_axis_2d(
                    side_diff,
                    float(track_row["side_image_x_px"]),
                    float(track_row["side_image_y_px"]),
                    side_axis_config,
                )
                reconstruction = reconstruct_axis_3d(
                    front_axis,
                    side_axis,
                    front_calibration=front_cal,
                    side_calibration=side_cal,
                    geometry=geometry,
                    z_mm=float(track_row["z_mm"]),
                    config=reconstruction_config,
                    previous_axis=previous_axis_by_particle.get(particle),
                )
                raw_axis = np.array(
                    [reconstruction.n_x, reconstruction.n_y, reconstruction.n_z],
                    dtype=float,
                )
                if reconstruction.valid and np.isfinite(raw_axis).all():
                    previous_axis_by_particle[particle] = raw_axis
                rows.append(
                    build_output_row(
                        track_row,
                        video_index,
                        front_axis,
                        side_axis,
                        reconstruction,
                    )
                )
    finally:
        provider.close()

    rotation = pd.DataFrame(rows)
    if rotation.empty:
        raise SystemExit("No rotation rows were produced.")
    return add_temporal_rotation_quantities(rotation)


class FrameProvider:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        front_roi: ROI | None,
        side_roi: ROI | None,
    ) -> None:
        self.args = args
        self.front_roi = front_roi
        self.side_roi = side_roi
        self.combined_reader = imageio.get_reader(str(args.video)) if args.video else None
        self.front_reader = imageio.get_reader(str(args.front)) if args.front else None
        self.side_reader = imageio.get_reader(str(args.side)) if args.side else None

    def read_background_panels(self) -> tuple[np.ndarray, np.ndarray]:
        args = self.args
        if args.video is not None:
            if args.background_file is not None:
                background_frame = read_background_frame(args.background_file, args.background_frame)
            else:
                background_frame = self.read_combined_frame(args.background_frame)
            return (
                frame_to_panel_gray(background_frame, args.channel, self.front_roi),
                frame_to_panel_gray(background_frame, args.channel, self.side_roi),
            )

        front_background_path = args.front_background_file or args.background_file
        side_background_path = args.side_background_file or args.background_file
        if front_background_path is None:
            front_background_frame = self.read_front_frame(args.background_frame)
        else:
            front_background_frame = read_background_frame(
                front_background_path,
                args.background_frame,
            )
        if side_background_path is None:
            side_background_frame = self.read_side_frame(args.background_frame)
        else:
            side_background_frame = read_background_frame(
                side_background_path,
                args.background_frame,
            )
        return (
            frame_to_panel_gray(front_background_frame, args.channel, self.front_roi),
            frame_to_panel_gray(side_background_frame, args.channel, self.side_roi),
        )

    def read_camera_panels(self, video_index: int) -> tuple[np.ndarray, np.ndarray]:
        args = self.args
        if args.video is not None:
            frame = self.read_combined_frame(video_index)
            return (
                frame_to_panel_gray(frame, args.channel, self.front_roi),
                frame_to_panel_gray(frame, args.channel, self.side_roi),
            )
        return (
            frame_to_panel_gray(self.read_front_frame(video_index), args.channel, self.front_roi),
            frame_to_panel_gray(self.read_side_frame(video_index), args.channel, self.side_roi),
        )

    def read_combined_frame(self, index: int) -> np.ndarray:
        if self.combined_reader is None:
            raise RuntimeError("Combined reader is not available.")
        return np.asarray(self.combined_reader.get_data(index))

    def read_front_frame(self, index: int) -> np.ndarray:
        if self.front_reader is None:
            raise RuntimeError("Front reader is not available.")
        return np.asarray(self.front_reader.get_data(index))

    def read_side_frame(self, index: int) -> np.ndarray:
        if self.side_reader is None:
            raise RuntimeError("Side reader is not available.")
        return np.asarray(self.side_reader.get_data(index))

    def close(self) -> None:
        for reader in (self.combined_reader, self.front_reader, self.side_reader):
            if reader is not None:
                reader.close()


def frame_to_panel_gray(
    frame: np.ndarray,
    channel: Literal["red", "green", "blue", "mean"],
    roi: ROI | None,
) -> np.ndarray:
    arr = np.asarray(frame)
    if roi is not None:
        arr = roi.crop(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if channel == "mean":
        return arr[:, :, :3].mean(axis=2).astype(np.float32)
    channel_index = {"red": 0, "green": 1, "blue": 2}[channel]
    return arr[:, :, channel_index].astype(np.float32)


def build_output_row(
    track_row: pd.Series,
    video_index: int,
    front_axis,
    side_axis,
    reconstruction,
) -> dict[str, Any]:
    invalid_reason = ""
    if not reconstruction.valid:
        invalid_reason = reconstruction.invalid_reason or "invalid"
    return {
        "particle": int(track_row["particle"]),
        "frame": int(track_row["frame"]),
        "video_frame_index": int(video_index),
        "time_s": float(track_row["time_s"]),
        "x_mm": float(track_row["x_mm"]),
        "y_mm": float(track_row["y_mm"]),
        "z_mm": float(track_row["z_mm"]),
        "front_image_x_px": float(track_row["front_image_x_px"]),
        "front_image_y_px": float(track_row["front_image_y_px"]),
        "side_image_x_px": float(track_row["side_image_x_px"]),
        "side_image_y_px": float(track_row["side_image_y_px"]),
        "front_axis_u_raw": front_axis.axis_u,
        "front_axis_v_raw": front_axis.axis_v,
        "side_axis_u_raw": side_axis.axis_u,
        "side_axis_v_raw": side_axis.axis_v,
        "front_axis_angle_rad": front_axis.angle_rad,
        "side_axis_angle_rad": side_axis.angle_rad,
        "front_anisotropy": front_axis.anisotropy,
        "side_anisotropy": side_axis.anisotropy,
        "front_axis_ratio": front_axis.axis_ratio,
        "side_axis_ratio": side_axis.axis_ratio,
        "front_foreground_area_px": int(front_axis.foreground_area_px),
        "side_foreground_area_px": int(side_axis.foreground_area_px),
        "front_signal_p95": front_axis.signal_p95,
        "side_signal_p95": side_axis.signal_p95,
        "front_crop_origin_u": int(front_axis.crop_origin_u),
        "front_crop_origin_v": int(front_axis.crop_origin_v),
        "side_crop_origin_u": int(side_axis.crop_origin_u),
        "side_crop_origin_v": int(side_axis.crop_origin_v),
        "front_axis_valid": bool(front_axis.valid),
        "side_axis_valid": bool(side_axis.valid),
        "front_axis_invalid_reason": front_axis.invalid_reason,
        "side_axis_invalid_reason": side_axis.invalid_reason,
        "front_axis_estimator": front_axis.estimator,
        "side_axis_estimator": side_axis.estimator,
        "n_x_raw": reconstruction.n_x,
        "n_y_raw": reconstruction.n_y,
        "n_z_raw": reconstruction.n_z,
        "front_reprojection_error_rad": reconstruction.front_reprojection_error_rad,
        "side_reprojection_error_rad": reconstruction.side_reprojection_error_rad,
        "orientation_confidence": reconstruction.orientation_confidence,
        "rotation_valid": bool(reconstruction.valid),
        "invalid_reason": invalid_reason,
        "orientation_type": "minimum_twist_reduced_attitude",
        "angular_velocity_type": "observable_tumbling_component",
        "axial_spin_observable": False,
        "axial_spin_assumption_rad_s": 0.0,
        "calibration_mode": "orthographic_two_camera",
    }


def add_temporal_rotation_quantities(rotation: pd.DataFrame) -> pd.DataFrame:
    rotation = rotation.sort_values(["particle", "frame"]).reset_index(drop=True).copy()
    for column in (
        "n_x",
        "n_y",
        "n_z",
        "q_w",
        "q_x",
        "q_y",
        "q_z",
        "omega_x_rad_s",
        "omega_y_rad_s",
        "omega_z_rad_s",
        "omega_mag_rad_s",
    ):
        rotation[column] = np.nan

    for _, group in rotation.groupby("particle", sort=True):
        indices = group.index.to_numpy()
        raw_axes = group[["n_x_raw", "n_y_raw", "n_z_raw"]].to_numpy(dtype=float)
        valid = group["rotation_valid"].to_numpy(dtype=bool) & np.isfinite(raw_axes).all(axis=1)
        signed_axes = np.full_like(raw_axes, np.nan, dtype=float)
        previous_axis: np.ndarray | None = None
        for i, axis in enumerate(raw_axes):
            if not valid[i]:
                continue
            axis = axis / np.linalg.norm(axis)
            if previous_axis is not None and float(np.dot(axis, previous_axis)) < 0:
                axis = -axis
            signed_axes[i] = axis
            previous_axis = axis

        quaternions = accumulate_minimum_twist_quaternions(signed_axes, valid)
        omega = observable_tumbling_angular_velocity(
            signed_axes,
            group["time_s"].to_numpy(dtype=float),
            valid,
        )
        rotation.loc[indices, ["n_x", "n_y", "n_z"]] = signed_axes
        rotation.loc[indices, ["q_w", "q_x", "q_y", "q_z"]] = quaternions
        rotation.loc[indices, ["omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s"]] = omega
        rotation.loc[indices, "omega_mag_rad_s"] = np.linalg.norm(omega, axis=1)

    return rotation.sort_values(["frame", "particle"]).reset_index(drop=True)


def build_reduced_state_vectors(rotation: pd.DataFrame, motion: pd.DataFrame) -> pd.DataFrame:
    missing_motion = [
        column for column in ("particle", "frame", *MOTION_COLUMNS) if column not in motion.columns
    ]
    if missing_motion:
        raise SystemExit(f"Motion CSV is missing required columns: {missing_motion}")
    keep_rotation = [
        "particle",
        "frame",
        "time_s",
        *ROTATION_STATE_COLUMNS,
        "q_w",
        "q_x",
        "q_y",
        "q_z",
        "rotation_valid",
        "invalid_reason",
        "orientation_type",
        "angular_velocity_type",
        "axial_spin_observable",
        "axial_spin_assumption_rad_s",
        "orientation_confidence",
    ]
    left = motion.copy()
    right = rotation.loc[:, keep_rotation].copy()
    for table in (left, right):
        table["particle"] = table["particle"].astype(int)
        table["frame"] = table["frame"].astype(int)
    merged = left.merge(right, on=["particle", "frame"], how="inner", suffixes=("", "_rotation"))
    if "time_s_rotation" in merged.columns:
        merged["time_s"] = merged["time_s"].where(merged["time_s"].notna(), merged["time_s_rotation"])
        merged = merged.drop(columns=["time_s_rotation"])
    merged["state_vector_order"] = ",".join(REDUCED_STATE_COLUMNS)
    merged["state_vector_json"] = [
        vector_to_json(row)
        for row in merged.loc[:, REDUCED_STATE_COLUMNS].to_numpy(dtype=float)
    ]
    return merged.sort_values(["particle", "frame"]).reset_index(drop=True)


def vector_to_json(values: np.ndarray) -> str:
    clean = []
    for value in np.asarray(values, dtype=float):
        clean.append(float(value) if np.isfinite(value) else None)
    return json.dumps(clean, separators=(",", ":"))


def make_output_dir(
    parent: Path,
    output_name: str | None,
    tracks_path: Path,
    args: argparse.Namespace,
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    if output_name:
        base_name = sanitize_output_name(output_name)
    else:
        source = args.video or args.front or tracks_path
        base_name = (
            f"rotation_{sanitize_output_name(source.stem)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir = parent / base_name
    if not output_dir.exists():
        return output_dir
    suffix = 2
    while True:
        candidate = parent / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def make_overlay_output_dir(rotation_output_dir: Path) -> Path:
    parent = rotation_output_dir.parent
    if parent.name == "rotation":
        overlay_parent = parent.parent / "rotation_overlays"
    else:
        overlay_parent = parent / "rotation_overlays"
    return overlay_parent / rotation_output_dir.name


def plot_rotation_diagnostics(rotation: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for particle, group in rotation.groupby("particle", sort=True):
        label = f"particle {particle}"
        axes[0].plot(group["time_s"], group["n_x"], label=f"{label} n_x")
        axes[0].plot(group["time_s"], group["n_y"], label=f"{label} n_y")
        axes[0].plot(group["time_s"], group["n_z"], label=f"{label} n_z")
        axes[1].plot(group["time_s"], group["q_w"], label=f"{label} q_w")
        axes[1].plot(group["time_s"], group["q_x"], label=f"{label} q_x")
        axes[1].plot(group["time_s"], group["q_y"], label=f"{label} q_y")
        axes[1].plot(group["time_s"], group["q_z"], label=f"{label} q_z")
        axes[2].plot(group["time_s"], group["omega_mag_rad_s"], label=label)
        axes[3].plot(group["time_s"], group["front_anisotropy"], label=f"{label} front")
        axes[3].plot(group["time_s"], group["side_anisotropy"], label=f"{label} side")

    axes[0].set_ylabel("3D axis n")
    axes[1].set_ylabel("quaternion")
    axes[2].set_ylabel("omega mag [rad/s]")
    axes[3].set_ylabel("anisotropy")
    axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_preview_overlay_outputs(
    *,
    args: argparse.Namespace,
    rotation: pd.DataFrame,
    rotation_path: Path,
    tracks_path: Path,
    calibration_path: Path,
    geometry: ChamberGeometry,
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    front_roi: ROI | None,
    side_roi: ROI | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    if front_roi is None or side_roi is None:
        raise SystemExit("Calibration must contain front_roi and side_roi for preview overlay.")
    preview_path = args.preview or choose_file(
        "Choose background-subtracted preview video for rotation overlay",
        [("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
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
        marker_radius_px=args.marker_radius_px,
        line_width_px=args.line_width_px,
        max_frames=args.preview_overlay_max_frames,
        front_width=args.front_width,
        front_height=args.front_height,
        side_width=args.side_width,
        side_height=args.side_height,
        no_zoom_panels=args.no_zoom_panels,
        zoom_crop_width_px=args.zoom_crop_width_px,
        zoom_crop_height_px=args.zoom_crop_height_px,
    )
    rows = preview_rotation_overlay.prepare_overlay_rows(
        rotation,
        tracks=None,
        rotation_path=rotation_path,
        tracks_path=None,
    )
    if rows.empty:
        raise SystemExit("No matching particle/frame rows are available for preview overlay.")

    panel = preview_rotation_overlay.resolve_panel_geometry(overlay_args, front_roi, side_roi)
    output_path = output_dir / f"{sanitize_output_name(preview_path.stem)}_rotation_overlay.mp4"
    (
        frames_written,
        markers_drawn,
        axis_lines_drawn,
        omega_arrows_drawn,
        zoom_panels_drawn,
    ) = preview_rotation_overlay.write_overlay_video(
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
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preview": str(preview_path.resolve()),
                "rotation_csv": str(rotation_path.resolve()),
                "tracks_csv": str(tracks_path.resolve()),
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
    return output_path, metadata_path


def write_overlay_video(
    args: argparse.Namespace,
    rotation: pd.DataFrame,
    front_roi: ROI | None,
    side_roi: ROI | None,
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Skipping overlay video because Pillow is unavailable.")
        return

    if args.video is None:
        return
    if front_roi is None or side_roi is None:
        return

    reader = imageio.get_reader(str(args.video))
    fps = float(reader.get_meta_data().get("fps") or args.fps)
    rows_by_frame = {
        int(frame): group.copy()
        for frame, group in rotation.groupby("frame", sort=False)
    }
    panel = native_overlay_panel_geometry(front_roi, side_roi)
    zoom_particle_ids = sorted(int(particle) for particle in rotation["particle"].dropna().unique())
    zoom_state: dict[int, dict[str, Any]] = {}
    try:
        with imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for frame_number in sorted(rows_by_frame):
                video_index = int(frame_number) + int(args.video_frame_offset)
                if video_index < 0:
                    continue
                try:
                    frame = reader.get_data(video_index)
                except Exception:
                    continue
                source_image = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
                image = source_image.copy()
                draw = ImageDraw.Draw(image)
                frame_rows = rows_by_frame[frame_number]
                for _, row in frame_rows.iterrows():
                    draw_camera_axis(
                        draw,
                        row,
                        roi=front_roi,
                        x_col="front_image_x_px",
                        y_col="front_image_y_px",
                        axis_u_col="front_axis_u_raw",
                        axis_v_col="front_axis_v_raw",
                        length_px=args.overlay_axis_length_px,
                        color=(40, 230, 255) if bool(row["rotation_valid"]) else (255, 80, 80),
                    )
                    draw_camera_axis(
                        draw,
                        row,
                        roi=side_roi,
                        x_col="side_image_x_px",
                        y_col="side_image_y_px",
                        axis_u_col="side_axis_u_raw",
                        axis_v_col="side_axis_v_raw",
                        length_px=args.overlay_axis_length_px,
                        color=(40, 230, 255) if bool(row["rotation_valid"]) else (255, 80, 80),
                    )
                if not args.no_zoom_panels:
                    preview_rotation_overlay.update_zoom_panel_state(
                        zoom_state,
                        source_image,
                        frame_rows,
                        panel,
                        args,
                    )
                    preview_rotation_overlay.draw_held_zoom_panels(
                        image,
                        zoom_state,
                        zoom_particle_ids,
                        panel,
                    )
                draw.text((12, 12), f"frame {frame_number}", fill=(255, 255, 255))
                writer.append_data(np.asarray(image))
    finally:
        reader.close()


def native_overlay_panel_geometry(front_roi: ROI, side_roi: ROI) -> dict[str, float]:
    return {
        "front_x": float(front_roi.x),
        "front_y": float(front_roi.y),
        "front_width": float(front_roi.width),
        "front_height": float(front_roi.height),
        "front_scale_x": 1.0,
        "front_scale_y": 1.0,
        "side_x": float(side_roi.x),
        "side_y": float(side_roi.y),
        "side_width": float(side_roi.width),
        "side_height": float(side_roi.height),
        "side_scale_x": 1.0,
        "side_scale_y": 1.0,
    }


def draw_camera_axis(
    draw,
    row: pd.Series,
    *,
    roi: ROI,
    x_col: str,
    y_col: str,
    axis_u_col: str,
    axis_v_col: str,
    length_px: float,
    color: tuple[int, int, int],
) -> None:
    axis_u = float(row[axis_u_col])
    axis_v = float(row[axis_v_col])
    if not math.isfinite(axis_u) or not math.isfinite(axis_v):
        return
    center_x = roi.x + float(row[x_col])
    center_y = roi.y + float(row[y_col])
    norm = math.hypot(axis_u, axis_v)
    if norm <= 1e-12:
        return
    dx = axis_u / norm * length_px
    dy = axis_v / norm * length_px
    draw.line(
        [(center_x - dx, center_y - dy), (center_x + dx, center_y + dy)],
        fill=color,
        width=3,
    )
    draw.ellipse(
        [(center_x - 3, center_y - 3), (center_x + 3, center_y + 3)],
        outline=color,
        width=2,
    )


def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    tracks_path: Path,
    calibration_path: Path,
    rotation: pd.DataFrame,
    front_axis_config: Axis2DConfig,
    side_axis_config: Axis2DConfig,
    reconstruction_config: ReconstructionConfig,
    rotation_path: Path,
    reduced_path: Path | None,
    plots_path: Path | None,
    overlay_path: Path | None,
    preview_overlay_path: Path | None,
    preview_overlay_metadata_path: Path | None,
) -> None:
    reference_axes: dict[str, list[float]] = {}
    for particle, group in rotation.loc[rotation["rotation_valid"]].groupby("particle", sort=True):
        first = group.iloc[0]
        reference_axes[str(int(particle))] = [
            float(first["n_x"]),
            float(first["n_y"]),
            float(first["n_z"]),
        ]
    metadata = {
        "source_video": str(args.video.resolve()) if args.video else None,
        "front_video": str(args.front.resolve()) if args.front else None,
        "side_video": str(args.side.resolve()) if args.side else None,
        "tracks_csv": str(tracks_path.resolve()),
        "motion_csv": str(args.motion_csv.resolve()) if args.motion_csv else None,
        "calibration": str(calibration_path.resolve()),
        "background_file": str(args.background_file.resolve()) if args.background_file else None,
        "front_background_file": str(args.front_background_file.resolve())
        if args.front_background_file
        else None,
        "side_background_file": str(args.side_background_file.resolve())
        if args.side_background_file
        else None,
        "rotation_csv": str(rotation_path.resolve()),
        "reduced_state_csv": str(reduced_path.resolve()) if reduced_path else None,
        "plots": str(plots_path.resolve()) if plots_path else None,
        "overlay_video": str(overlay_path.resolve()) if overlay_path else None,
        "preview_overlay_video": str(preview_overlay_path.resolve())
        if preview_overlay_path
        else None,
        "preview_overlay_metadata": str(preview_overlay_metadata_path.resolve())
        if preview_overlay_metadata_path
        else None,
        "rows": int(len(rotation)),
        "valid_rotation_rows": int(rotation["rotation_valid"].sum()),
        "particles": int(rotation["particle"].nunique()),
        "front_axis_2d_config": asdict(front_axis_config),
        "side_axis_2d_config": asdict(side_axis_config),
        "reconstruction_config": asdict(reconstruction_config),
        "fps": float(args.fps),
        "video_frame_offset": int(args.video_frame_offset),
        "difference_mode": args.difference_mode,
        "channel": args.channel,
        "reference_axes_lab_frame": reference_axes,
        "coordinate_convention": {
            "vectors": "[x, y, z] in laboratory frame",
            "quaternions": "scalar-first [q_w, q_x, q_y, q_z]",
            "q_lab_from_body": (
                "active minimum-twist reduced-attitude rotation, relative to "
                "the first reliable observed axis for each particle"
            ),
            "angular_velocity_frame": "laboratory frame",
        },
        "orientation_type": "minimum_twist_reduced_attitude",
        "angular_velocity_type": "observable_tumbling_component",
        "axial_spin_observable": False,
        "axial_spin_assumption_rad_s": 0.0,
        "observability_note": (
            "Only tumbling of one visible 3D body axis is estimated. Spin about "
            "that axis is not measured in this reduced-attitude pipeline."
        ),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
