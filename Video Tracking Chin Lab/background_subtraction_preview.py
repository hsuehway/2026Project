"""Create a preview video of background-subtracted camera panels.

This is a visual debugging helper for ``background_subtraction_tracker.py``.
It uses the same combined video, calibration JSON, and front/side ROIs, then
writes an MP4 showing frames after the background frame with the background
subtracted.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from background_subtraction_tracker import (
    DIFFERENCE_MODES,
    FOCUS_BOX_MODES,
    CHANNELS,
    apply_focus_mask,
    apply_focus_mask_with_particle_windows,
    assign_unique_component_candidates,
    close_video,
    difference_image,
    frame_to_gray,
    interactive_particle_seeds_masked,
    locate_component_candidates,
    open_video,
    parse_seed_points,
    print_focus_summary,
    read_background_frame,
    resolve_background_path,
    resolve_focus_rois,
    sanitize_output_name,
    should_select_focus_box,
    validate_seed_points_in_focus,
)
from track_orthogonal_3d import (
    ChamberGeometry,
    ROI,
    calibration_path_for_video,
    choose_missing_rois,
    choose_video,
    infer_video_id,
    interactive_calibration,
    read_calibration,
    write_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write an MP4 preview of the background-subtracted camera panels."
    )
    parser.add_argument("--video", type=Path, help="Combined screen-recording video.")
    parser.add_argument(
        "--background-video",
        type=Path,
        help=(
            "Optional separate video used only for the background frame. "
            "Use this for a clean chamber/background recording."
        ),
    )
    parser.add_argument(
        "--background-image",
        type=Path,
        help="Optional still image used as the background frame.",
    )
    parser.add_argument(
        "--background-file",
        type=Path,
        help="Optional background image or video. Videos use --background-frame.",
    )
    parser.add_argument(
        "--choose-background",
        action="store_true",
        help=(
            "Open File Explorer to choose a background image or video. This is "
            "the default when no explicit background path is passed."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Calibration JSON override. By default the preview uses "
            "calibrations\\<trial>_calibration.json when the video name contains "
            "an MI## or SP## trial ID."
        ),
    )
    parser.add_argument(
        "--calibrations-dir",
        type=Path,
        default=Path("calibrations"),
        help="Folder for video-specific calibration files.",
    )
    parser.add_argument(
        "--force-calibration",
        action="store_true",
        help=(
            "Re-click plate/camera calibration for this video and overwrite the "
            "resolved calibration file. Saved front/side panel ROIs are reused."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Preview MP4 output path. Defaults to a unique file in outputs_background/previews.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "previews",
        help="Folder for timestamped preview MP4 files when --output is not given.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional base name for the preview file when --output is not given.",
    )
    parser.add_argument(
        "--channel",
        choices=CHANNELS,
        default="green",
        help="Video channel to subtract.",
    )
    parser.add_argument(
        "--difference-mode",
        choices=DIFFERENCE_MODES,
        default="abs",
        help="Subtraction mode: abs, bright, or dark.",
    )
    parser.add_argument(
        "--background-frame",
        type=int,
        default=0,
        help="Frame index used as the background image.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        help="First frame to show. Defaults to background-frame + 1.",
    )
    parser.add_argument(
        "--front-focus-box",
        type=ROI.from_string,
        help=(
            "Front camera allowed box as x,y,width,height in front-crop pixels. "
            "By default this masks the preview using follow mode."
        ),
    )
    parser.add_argument(
        "--side-focus-box",
        type=ROI.from_string,
        help=(
            "Side camera allowed box as x,y,width,height in side-crop pixels. "
            "By default this masks the preview using follow mode."
        ),
    )
    parser.add_argument(
        "--select-focus-box",
        action="store_true",
        help="Click a front and side focus box before writing the preview. This is the default.",
    )
    parser.add_argument(
        "--focus-box-mode",
        choices=FOCUS_BOX_MODES,
        default="follow",
        help=(
            "follow zeroes outside the box except around already-tracked "
            "particles. hard zeroes outside the box every frame. seed-only "
            "draws the box without masking preview frames."
        ),
    )
    parser.add_argument(
        "--no-focus-box",
        action="store_true",
        help="Do not use a focus box.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of preview frames to write. Default 0 writes the whole video.",
    )
    parser.add_argument(
        "--seed-frame",
        type=int,
        help=(
            "Frame shown for clicking particles in follow mode. Defaults to "
            "--start-frame."
        ),
    )
    parser.add_argument(
        "--particle-count",
        type=int,
        default=2,
        help="Number of particles to click in each camera view in follow mode.",
    )
    parser.add_argument(
        "--front-seeds",
        type=parse_seed_points,
        help=(
            "Front-camera crop-local particle positions as x,y;x,y. Supply this "
            "with --side-seeds to skip the graphical seed windows."
        ),
    )
    parser.add_argument(
        "--side-seeds",
        type=parse_seed_points,
        help="Side-camera crop-local particle positions as x,y;x,y.",
    )
    parser.add_argument(
        "--search-radius-px",
        type=int,
        default=12,
        help="Local search radius around tracked particles in follow mode.",
    )
    parser.add_argument(
        "--particle-window-radius-px",
        type=int,
        default=48,
        help=(
            "In follow mode, keep this radius visible around each tracked "
            "particle when it leaves the focus box."
        ),
    )
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=0.35,
        help="Keep local difference pixels at least this fraction of the local peak.",
    )
    parser.add_argument(
        "--min-difference",
        type=float,
        default=4.0,
        help="Minimum local difference intensity required to move a preview window.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=120,
        help="How many missed frames to keep a preview particle window alive.",
    )
    parser.add_argument(
        "--missing-search-expansion-px",
        type=float,
        default=1.0,
        help="How much to expand a preview window after each missed frame.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Preview video FPS. Defaults to source FPS when available.",
    )
    parser.add_argument(
        "--scale-percentile",
        type=float,
        default=99.5,
        help="Per-frame percentile used to scale the difference image for display.",
    )
    parser.add_argument(
        "--display-threshold",
        type=float,
        default=8.0,
        help="Set difference values below this to black before display.",
    )
    parser.add_argument(
        "--outside-window-display-threshold",
        type=float,
        default=12.0,
        help=(
            "In follow mode, require outside-the-box moving-window pixels to "
            "reach this difference before they are shown."
        ),
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=0.0,
        help="Optional Gaussian blur radius applied before thresholding.",
    )
    parser.add_argument(
        "--cleanup-pixels",
        type=int,
        default=0,
        help="Remove connected bright regions smaller than this many pixels after thresholding.",
    )
    parser.add_argument(
        "--fixed-scale",
        type=float,
        default=40.0,
        help="Use this fixed difference value as white. Use 0 for automatic scaling.",
    )
    parser.add_argument(
        "--layout",
        choices=("side-by-side", "stacked"),
        default="side-by-side",
        help="How to arrange front and side panels in the preview.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    video_path = args.video or choose_video("Choose combined video")
    (
        geometry,
        front_cal,
        side_cal,
        front_roi,
        side_roi,
        calibration_path,
        calibration_written,
    ) = load_or_create_preview_calibration(args, video_path)
    print(f"Calibration file: {calibration_path.resolve()}")
    front_focus_roi, side_focus_roi = resolve_focus_rois(
        video_path=video_path,
        channel=args.channel,
        front_roi=front_roi,
        side_roi=side_roi,
        saved_front_focus_roi=None,
        saved_side_focus_roi=None,
        front_focus_box=args.front_focus_box,
        side_focus_box=args.side_focus_box,
        select_focus_box=should_select_focus_box(args),
        no_focus_box=args.no_focus_box,
    )
    print_focus_summary(front_focus_roi, side_focus_roi)
    print(f"Focus box mode: {args.focus_box_mode}")

    start_frame = args.start_frame
    if start_frame is None:
        start_frame = args.background_frame + 1
    seed_frame = args.seed_frame if args.seed_frame is not None else start_frame
    if args.focus_box_mode == "follow" and start_frame < seed_frame:
        raise SystemExit("--start-frame must be greater than or equal to --seed-frame.")

    front_states = None
    side_states = None
    if args.focus_box_mode == "follow":
        if args.front_seeds is not None:
            front_seed_points = list(args.front_seeds)
            side_seed_points = list(args.side_seeds)
            validate_seed_points_in_focus(front_seed_points, front_focus_roi, "front")
            validate_seed_points_in_focus(side_seed_points, side_focus_roi, "side")
            print("Using particle seed coordinates supplied by Video Analysis Studio.")
        else:
            front_seed_points = interactive_particle_seeds_masked(
                video_path=video_path,
                camera_name="front",
                channel=args.channel,
                roi=front_roi,
                seed_frame=seed_frame,
                particle_count=args.particle_count,
                focus_roi=front_focus_roi,
            )
            side_seed_points = interactive_particle_seeds_masked(
                video_path=video_path,
                camera_name="side",
                channel=args.channel,
                roi=side_roi,
                seed_frame=seed_frame,
                particle_count=args.particle_count,
                focus_roi=side_focus_roi,
            )
        front_states = make_particle_states(front_seed_points)
        side_states = make_particle_states(side_seed_points)

    raw_frames = open_video(video_path)
    source_fps = get_source_fps(video_path)
    fps = args.fps or source_fps or 30.0

    background_path = resolve_background_path(args)
    background_frame = read_background_frame(background_path, args.background_frame)
    print(f"Background source: {background_path.resolve()}")
    front_background = frame_to_gray(background_frame, args.channel, front_roi)
    side_background = frame_to_gray(background_frame, args.channel, side_roi)

    output_path = resolve_preview_output_path(args, video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_preview_video(
            raw_frames=raw_frames,
            output_path=output_path,
            front_roi=front_roi,
            side_roi=side_roi,
            front_focus_roi=front_focus_roi,
            side_focus_roi=side_focus_roi,
            front_states=front_states,
            side_states=side_states,
            front_background=front_background,
            side_background=side_background,
            channel=args.channel,
            difference_mode=args.difference_mode,
            start_frame=start_frame,
            max_frames=args.max_frames,
            fps=fps,
            scale_percentile=args.scale_percentile,
            display_threshold=args.display_threshold,
            blur_radius=args.blur_radius,
            cleanup_pixels=args.cleanup_pixels,
            fixed_scale=args.fixed_scale,
            focus_box_mode=args.focus_box_mode,
            seed_frame=seed_frame,
            search_radius_px=args.search_radius_px,
            particle_window_radius_px=args.particle_window_radius_px,
            outside_window_display_threshold=args.outside_window_display_threshold,
            threshold_fraction=args.threshold_fraction,
            min_difference=args.min_difference,
            memory=args.memory,
            missing_search_expansion_px=args.missing_search_expansion_px,
            layout=args.layout,
        )
    finally:
        close_video(raw_frames)
    write_preview_metadata(
        output_path=output_path,
        video_path=video_path,
        background_path=background_path,
        calibration_path=calibration_path,
        calibration_written=calibration_written,
        args=args,
        start_frame=start_frame,
        seed_frame=seed_frame,
        fps=fps,
        front_focus_roi=front_focus_roi,
        side_focus_roi=side_focus_roi,
    )
    print(f"Wrote preview: {output_path.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    background_sources = [
        source
        for source in (args.background_video, args.background_image, args.background_file)
        if source is not None
    ]
    if len(background_sources) > 1:
        raise SystemExit(
            "Use only one of --background-video, --background-image, or --background-file."
        )
    if args.choose_background and background_sources:
        raise SystemExit(
            "--choose-background cannot be combined with an explicit background path."
        )
    if args.background_frame < 0:
        raise SystemExit("--background-frame must be zero or positive.")
    if args.start_frame is not None and args.start_frame < 0:
        raise SystemExit("--start-frame must be zero or positive.")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be zero or positive.")
    if args.seed_frame is not None and args.seed_frame < 0:
        raise SystemExit("--seed-frame must be zero or positive.")
    if args.particle_count <= 0:
        raise SystemExit("--particle-count must be positive.")
    if (args.front_seeds is None) != (args.side_seeds is None):
        raise SystemExit("--front-seeds and --side-seeds must be supplied together.")
    if args.front_seeds is not None:
        if len(args.front_seeds) != args.particle_count:
            raise SystemExit(
                "--front-seeds must contain exactly --particle-count coordinates."
            )
        if len(args.side_seeds) != args.particle_count:
            raise SystemExit(
                "--side-seeds must contain exactly --particle-count coordinates."
            )
    if args.search_radius_px <= 0:
        raise SystemExit("--search-radius-px must be positive.")
    if args.particle_window_radius_px <= 0:
        raise SystemExit("--particle-window-radius-px must be positive.")
    if not 0 < args.threshold_fraction <= 1:
        raise SystemExit("--threshold-fraction must be in the range (0, 1].")
    if args.min_difference < 0:
        raise SystemExit("--min-difference must be non-negative.")
    if args.memory < 0:
        raise SystemExit("--memory must be zero or positive.")
    if args.missing_search_expansion_px < 0:
        raise SystemExit("--missing-search-expansion-px must be non-negative.")
    if args.fps is not None and args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if not 0 < args.scale_percentile <= 100:
        raise SystemExit("--scale-percentile must be in the range (0, 100].")
    if args.display_threshold < 0:
        raise SystemExit("--display-threshold must be non-negative.")
    if args.outside_window_display_threshold < 0:
        raise SystemExit("--outside-window-display-threshold must be non-negative.")
    if args.blur_radius < 0:
        raise SystemExit("--blur-radius must be non-negative.")
    if args.cleanup_pixels < 0:
        raise SystemExit("--cleanup-pixels must be zero or positive.")
    if args.fixed_scale < 0:
        raise SystemExit("--fixed-scale must be non-negative.")
    if args.no_focus_box and (
        args.front_focus_box is not None
        or args.side_focus_box is not None
        or args.select_focus_box
    ):
        raise SystemExit(
            "--no-focus-box cannot be combined with focus-box selection."
        )


def load_or_create_preview_calibration(
    args: argparse.Namespace,
    video_path: Path,
):
    calibration_path = resolve_preview_calibration_path(args, video_path)
    existing = read_calibration_if_available(calibration_path)
    needs_write = args.force_calibration or existing is None
    if not needs_write:
        (
            geometry,
            front_cal,
            side_cal,
            front_roi,
            side_roi,
            _front_focus_roi,
            _side_focus_roi,
        ) = existing
        if front_roi is None or side_roi is None:
            raise SystemExit(
                f"{calibration_path} exists but is missing front_roi/side_roi. "
                "Rerun preview with --force-calibration so the panel split can be "
                "chosen and saved."
            )
        return geometry, front_cal, side_cal, front_roi, side_roi, calibration_path, False

    fallback = existing or read_calibration_if_available(Path("calibration.json"))
    if fallback is not None:
        geometry, _old_front_cal, _old_side_cal, front_roi, side_roi, *_ = fallback
    else:
        geometry = ChamberGeometry()
        front_roi = None
        side_roi = None

    if front_roi is None or side_roi is None:
        print("No saved front/side panel split was found; choose it once now.")
    else:
        print("Reusing saved front/side panel split for calibration.")
    front_roi, side_roi = choose_missing_rois(
        video_path,
        args.channel,
        front_roi,
        side_roi,
    )
    print(
        "Click front calibration points: bottom-left, bottom-right, "
        "top-left, top-right."
    )
    front_cal = interactive_calibration(video_path, "front", args.channel, front_roi)
    print(
        "Click side calibration points: bottom-left, bottom-right, "
        "top-left, top-right."
    )
    side_cal = interactive_calibration(video_path, "side", args.channel, side_roi)
    write_calibration(
        calibration_path,
        geometry,
        front_cal,
        side_cal,
        front_roi=front_roi,
        side_roi=side_roi,
    )
    print(f"Saved calibration: {calibration_path.resolve()}")
    return geometry, front_cal, side_cal, front_roi, side_roi, calibration_path, True


def resolve_preview_calibration_path(args: argparse.Namespace, video_path: Path) -> Path:
    if args.calibration is not None:
        return args.calibration
    video_calibration = calibration_path_for_video(video_path, args.calibrations_dir)
    if video_calibration is not None:
        return video_calibration
    return Path("calibration.json")


def read_calibration_if_available(path: Path):
    if not path.exists():
        return None
    return read_calibration(path)


def write_preview_metadata(
    output_path: Path,
    video_path: Path,
    background_path: Path,
    calibration_path: Path,
    calibration_written: bool,
    args: argparse.Namespace,
    start_frame: int,
    seed_frame: int,
    fps: float,
    front_focus_roi: ROI | None,
    side_focus_roi: ROI | None,
) -> None:
    import json

    metadata = {
        "video": str(video_path.resolve()),
        "video_id": infer_video_id(video_path),
        "background": str(background_path.resolve()),
        "calibration_file_used": str(calibration_path.resolve()),
        "calibration_written_by_preview": bool(calibration_written),
        "channel": args.channel,
        "difference_mode": args.difference_mode,
        "background_frame": int(args.background_frame),
        "start_frame": int(start_frame),
        "seed_frame": int(seed_frame),
        "fps": float(fps),
        "focus_box_mode": args.focus_box_mode,
        "front_focus_roi": roi_to_metadata(front_focus_roi),
        "side_focus_roi": roi_to_metadata(side_focus_roi),
        "display_threshold": float(args.display_threshold),
        "outside_window_display_threshold": float(args.outside_window_display_threshold),
        "threshold_fraction": float(args.threshold_fraction),
        "min_difference": float(args.min_difference),
        "search_radius_px": int(args.search_radius_px),
        "particle_window_radius_px": int(args.particle_window_radius_px),
        "cleanup_pixels": int(args.cleanup_pixels),
    }
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def roi_to_metadata(roi: ROI | None):
    return None if roi is None else roi.to_dict()


def get_source_fps(video_path: Path) -> float | None:
    try:
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.average_rate:
                return float(stream.average_rate)
    except Exception:
        return None
    return None


def resolve_preview_output_path(args: argparse.Namespace, video_path: Path) -> Path:
    if args.output is not None:
        return args.output

    video_name = sanitize_output_name(video_path.stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = (
        sanitize_output_name(args.output_name)
        if args.output_name
        else f"{video_name}_{timestamp}"
    )
    output_dir = args.output_dir / folder_name
    output_path = output_dir / f"{video_name}_preview.mp4"
    if not output_dir.exists():
        return output_path

    suffix = 2
    while True:
        candidate_dir = args.output_dir / f"{folder_name}_{suffix}"
        if not candidate_dir.exists():
            return candidate_dir / f"{video_name}_preview.mp4"
        suffix += 1


def make_particle_states(
    seed_points: list[tuple[float, float]],
) -> dict[int, dict[str, float | int | bool]]:
    return {
        seed_id: {
            "x": float(point[0]),
            "y": float(point[1]),
            "missing": 0,
            "active": True,
        }
        for seed_id, point in enumerate(seed_points)
    }


def update_follow_states(
    diff: np.ndarray,
    states: dict[int, dict[str, float | int | bool]],
    search_radius_px: int,
    threshold_fraction: float,
    min_difference: float,
    memory: int,
    missing_search_expansion_px: float,
) -> None:
    candidate_by_seed: dict[
        int,
        list[dict[str, float | tuple[int, int]]],
    ] = {}
    active_seed_ids: list[int] = []

    for seed_id, state in states.items():
        if not state["active"]:
            continue
        active_seed_ids.append(seed_id)
        current_radius = int(
            round(search_radius_px + int(state["missing"]) * missing_search_expansion_px)
        )
        radius = max(search_radius_px, current_radius)
        candidate_by_seed[seed_id] = locate_component_candidates(
            diff,
            x=float(state["x"]),
            y=float(state["y"]),
            radius=radius,
            threshold_fraction=threshold_fraction,
            min_difference=min_difference,
        )

    assignments = assign_unique_component_candidates(candidate_by_seed)

    for seed_id in active_seed_ids:
        state = states[seed_id]
        result = assignments.get(seed_id)
        if result is None:
            state["missing"] = int(state["missing"]) + 1
            if int(state["missing"]) > memory:
                state["active"] = False
            continue

        state.update(
            {
                "x": float(result["x"]),
                "y": float(result["y"]),
                "missing": 0,
                "active": True,
            }
        )

def write_preview_video(
    raw_frames,
    output_path: Path,
    front_roi,
    side_roi,
    front_focus_roi,
    side_focus_roi,
    front_states,
    side_states,
    front_background: np.ndarray,
    side_background: np.ndarray,
    channel: Literal["red", "green", "blue", "mean"],
    difference_mode: Literal["abs", "bright", "dark"],
    start_frame: int,
    max_frames: int,
    fps: float,
    scale_percentile: float,
    display_threshold: float,
    blur_radius: float,
    cleanup_pixels: int,
    fixed_scale: float,
    focus_box_mode: Literal["follow", "hard", "seed-only"],
    seed_frame: int,
    search_radius_px: int,
    particle_window_radius_px: int,
    outside_window_display_threshold: float,
    threshold_fraction: float,
    min_difference: float,
    memory: int,
    missing_search_expansion_px: float,
    layout: Literal["side-by-side", "stacked"],
) -> None:
    import imageio.v2 as imageio

    frame_limit = None if max_frames == 0 else max_frames
    written = 0
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8) as writer:
        for frame_index in range(start_frame, len(raw_frames)):
            if frame_limit is not None and written >= frame_limit:
                break
            frame = raw_frames[frame_index]
            front = frame_to_gray(frame, channel, front_roi)
            side = frame_to_gray(frame, channel, side_roi)
            front_diff = difference_image(front, front_background, difference_mode)
            side_diff = difference_image(side, side_background, difference_mode)
            if focus_box_mode == "follow":
                if frame_index > seed_frame:
                    update_follow_states(
                        front_diff,
                        front_states,
                        search_radius_px=search_radius_px,
                        threshold_fraction=threshold_fraction,
                        min_difference=min_difference,
                        memory=memory,
                        missing_search_expansion_px=missing_search_expansion_px,
                    )
                    update_follow_states(
                        side_diff,
                        side_states,
                        search_radius_px=search_radius_px,
                        threshold_fraction=threshold_fraction,
                        min_difference=min_difference,
                        memory=memory,
                        missing_search_expansion_px=missing_search_expansion_px,
                    )
                front_diff = apply_focus_mask_with_particle_windows(
                    front_diff,
                    front_focus_roi,
                    front_states,
                    search_radius_px=search_radius_px,
                    missing_search_expansion_px=missing_search_expansion_px,
                    particle_window_radius_px=particle_window_radius_px,
                    outside_window_threshold=outside_window_display_threshold,
                    name="front focus box",
                )
                side_diff = apply_focus_mask_with_particle_windows(
                    side_diff,
                    side_focus_roi,
                    side_states,
                    search_radius_px=search_radius_px,
                    missing_search_expansion_px=missing_search_expansion_px,
                    particle_window_radius_px=particle_window_radius_px,
                    outside_window_threshold=outside_window_display_threshold,
                    name="side focus box",
                )
            elif focus_box_mode == "hard":
                front_diff = apply_focus_mask(
                    front_diff, front_focus_roi, "front focus box"
                )
                side_diff = apply_focus_mask(side_diff, side_focus_roi, "side focus box")
            front_img = difference_to_rgb(
                front_diff,
                label=f"front frame {frame_index}",
                focus_roi=front_focus_roi,
                scale_percentile=scale_percentile,
                display_threshold=display_threshold,
                blur_radius=blur_radius,
                cleanup_pixels=cleanup_pixels,
                fixed_scale=fixed_scale,
            )
            side_img = difference_to_rgb(
                side_diff,
                label=f"side frame {frame_index}",
                focus_roi=side_focus_roi,
                scale_percentile=scale_percentile,
                display_threshold=display_threshold,
                blur_radius=blur_radius,
                cleanup_pixels=cleanup_pixels,
                fixed_scale=fixed_scale,
            )
            writer.append_data(pad_to_macroblock(compose_panels(front_img, side_img, layout)))
            written += 1

    if written == 0:
        raise SystemExit("No preview frames were written. Check --start-frame/--max-frames.")


def difference_to_rgb(
    diff: np.ndarray,
    label: str,
    focus_roi: ROI | None,
    scale_percentile: float,
    display_threshold: float,
    blur_radius: float,
    cleanup_pixels: int,
    fixed_scale: float,
) -> np.ndarray:
    display = prepare_display_difference(
        diff,
        display_threshold=display_threshold,
        blur_radius=blur_radius,
        cleanup_pixels=cleanup_pixels,
    )
    scale = fixed_scale if fixed_scale > 0 else float(np.percentile(display, scale_percentile))
    scale = max(scale, 1.0)
    gray = np.clip(display / scale * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=(255, 255, 0), width=2)
    draw.rectangle([0, 0, 230, 24], fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 0))
    return np.asarray(img)


def prepare_display_difference(
    diff: np.ndarray,
    display_threshold: float,
    blur_radius: float,
    cleanup_pixels: int,
) -> np.ndarray:
    display = diff.astype(np.float32, copy=True)
    if blur_radius > 0:
        try:
            from scipy.ndimage import gaussian_filter

            display = gaussian_filter(display, sigma=blur_radius)
        except Exception:
            pass
    if display_threshold > 0:
        display[display < display_threshold] = 0
    if cleanup_pixels > 0:
        display = remove_small_regions(display, cleanup_pixels)
    return display


def remove_small_regions(display: np.ndarray, min_pixels: int) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:
        return display
    mask = display > 0
    labels, count = ndimage.label(mask)
    if count == 0:
        return display
    sizes = np.bincount(labels.ravel())
    keep_labels = np.flatnonzero(sizes >= min_pixels)
    keep = np.isin(labels, keep_labels)
    keep[labels == 0] = False
    return np.where(keep, display, 0)


def compose_panels(
    front: np.ndarray,
    side: np.ndarray,
    layout: Literal["side-by-side", "stacked"],
) -> np.ndarray:
    front_img = Image.fromarray(front)
    side_img = Image.fromarray(side)
    if layout == "stacked":
        width = max(front_img.width, side_img.width)
        height = front_img.height + side_img.height
        canvas = Image.new("RGB", (width, height), "black")
        canvas.paste(front_img, (0, 0))
        canvas.paste(side_img, (0, front_img.height))
    else:
        width = front_img.width + side_img.width
        height = max(front_img.height, side_img.height)
        canvas = Image.new("RGB", (width, height), "black")
        canvas.paste(front_img, (0, 0))
        canvas.paste(side_img, (front_img.width, 0))
    return np.asarray(canvas)


def pad_to_macroblock(frame: np.ndarray, block_size: int = 16) -> np.ndarray:
    height, width = frame.shape[:2]
    padded_width = int(np.ceil(width / block_size) * block_size)
    padded_height = int(np.ceil(height / block_size) * block_size)
    if padded_width == width and padded_height == height:
        return frame
    padded = np.zeros((padded_height, padded_width, 3), dtype=frame.dtype)
    padded[:height, :width, :] = frame
    return padded


if __name__ == "__main__":
    raise SystemExit(main())
