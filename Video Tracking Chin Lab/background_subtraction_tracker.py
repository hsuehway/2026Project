"""Track clicked particles by subtracting a background frame.

This is a separate workflow from ``track.py``. It uses the same calibration JSON
and camera-panel ROIs, but it does not use Trackpy feature linking. Instead, it
subtracts a background frame, follows local moving intensity near each clicked
particle, and reconstructs 3D coordinates from the front/side camera views.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache")
)

from track_orthogonal_3d import (  # noqa: E402
    CameraCalibration,
    ChamberGeometry,
    ROI,
    calibration_path_for_video,
    choose_video,
    infer_video_id,
    interactive_focus_roi,
    read_calibration,
)
from trajectory_smoothing import (  # noqa: E402
    add_smoothed_track_positions,
    validate_smoothing_parameters,
)

CHANNELS = ("red", "green", "blue", "mean")
DIFFERENCE_MODES = ("abs", "bright", "dark")
FOCUS_BOX_MODES = ("follow", "hard", "seed-only")
Z_SOURCES = ("average", "front", "side")
IDENTITY_AUTHORITIES = ("auto", "none", "front", "side")
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".tif", ".tiff"}


def parse_frame_ranges(value: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
        else:
            start_text = end_text = chunk
        try:
            start = int(start_text.strip())
            end = int(end_text.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid frame range {chunk!r}; use START-END or FRAME."
            ) from exc
        if start < 0 or end < 0:
            raise argparse.ArgumentTypeError("Frame ranges must be non-negative.")
        if end < start:
            raise argparse.ArgumentTypeError(
                f"Invalid frame range {chunk!r}; end is before start."
            )
        ranges.append((start, end))
    return tuple(ranges)


def frame_in_ranges(frame_number: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= frame_number <= end for start, end in ranges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track clicked particles by subtracting a background frame from a "
            "single combined front/side video."
        )
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
            "Calibration JSON override. By default the tracker requires "
            "calibrations\\<trial>_calibration.json for videos whose names contain "
            "an MI## or SP## trial ID."
        ),
    )
    parser.add_argument(
        "--calibrations-dir",
        type=Path,
        default=Path("calibrations"),
        help="Folder containing video-specific calibration files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "tracks",
        help="Parent directory where each run output folder is created.",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Optional name for this run's output folder inside --output-dir. "
            "Defaults to the video name plus a timestamp."
        ),
    )
    parser.add_argument(
        "--channel",
        choices=CHANNELS,
        default="green",
        help="Video channel used for tracking.",
    )
    parser.add_argument(
        "--difference-mode",
        choices=DIFFERENCE_MODES,
        default="abs",
        help=(
            "How to subtract the background: abs=absolute difference, "
            "bright=frame-background, dark=background-frame."
        ),
    )
    parser.add_argument(
        "--background-frame",
        type=int,
        default=0,
        help="Frame index used as the background image.",
    )
    parser.add_argument(
        "--seed-frame",
        type=int,
        default=1,
        help="Frame shown for clicking the particles to track.",
    )
    parser.add_argument(
        "--particle-count",
        type=int,
        default=2,
        help="Number of particles to click in each camera view.",
    )
    parser.add_argument(
        "--front-focus-box",
        type=ROI.from_string,
        help=(
            "Front camera allowed box as x,y,width,height in front-crop pixels. "
            "By default this only cleans up the seed-click image."
        ),
    )
    parser.add_argument(
        "--side-focus-box",
        type=ROI.from_string,
        help=(
            "Side camera allowed box as x,y,width,height in side-crop pixels. "
            "By default this only cleans up the seed-click image."
        ),
    )
    parser.add_argument(
        "--select-focus-box",
        action="store_true",
        help="Click a front and side focus box before selecting particles. This is the default.",
    )
    parser.add_argument(
        "--focus-box-mode",
        choices=FOCUS_BOX_MODES,
        default="follow",
        help=(
            "follow zeroes outside the box except around already-tracked "
            "particles. hard zeroes outside the box every frame. seed-only "
            "uses the box only for initial particle selection."
        ),
    )
    parser.add_argument(
        "--no-focus-box",
        action="store_true",
        help="Do not use a focus box.",
    )
    parser.add_argument(
        "--search-radius-px",
        type=int,
        default=12,
        help="Local search radius around the previous particle location.",
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
        help="Minimum local difference intensity required to accept a particle update.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=120,
        help="How many missed frames to tolerate before ending a particle track.",
    )
    parser.add_argument(
        "--missing-search-expansion-px",
        type=float,
        default=1.0,
        help=(
            "How much to expand the local search radius for each consecutive "
            "missed frame."
        ),
    )
    parser.add_argument(
        "--no-manual-correction",
        action="store_true",
        help=(
            "Disable interactive correction prompts. By default the tracker "
            "asks you to re-click particles when a track is lost or IDs may swap."
        ),
    )
    parser.add_argument(
        "--manual-correction-after-misses",
        type=int,
        default=3,
        help=(
            "Prompt for manual correction after this many consecutive missed "
            "frames for a particle."
        ),
    )
    parser.add_argument(
        "--manual-correction-close-px",
        type=float,
        default=0.0,
        help=(
            "Prompt for manual correction when same-camera particles are this "
            "close in pixels. Use 0 to disable close-particle prompts."
        ),
    )
    parser.add_argument(
        "--manual-correction-cooldown-frames",
        type=int,
        default=30,
        help=(
            "Minimum frames between close-particle manual correction prompts. "
            "Lost-track prompts can still appear sooner."
        ),
    )
    parser.add_argument(
        "--front-overlap-frames",
        type=parse_frame_ranges,
        default=(),
        metavar="RANGES",
        help=(
            "Inclusive frame range(s) where front-camera particles are allowed "
            "to share one visible component, such as 2200-2350 or "
            "2200-2350,2600-2680."
        ),
    )
    parser.add_argument(
        "--side-overlap-frames",
        type=parse_frame_ranges,
        default=(),
        metavar="RANGES",
        help=(
            "Inclusive frame range(s) where side-camera particles are allowed "
            "to share one visible component, such as 2200-2350 or "
            "2200-2350,2600-2680."
        ),
    )
    parser.add_argument(
        "--identity-authority",
        choices=IDENTITY_AUTHORITIES,
        default="auto",
        help=(
            "Use a clear camera to preserve particle IDs during overlap ranges. "
            "auto uses side during front-overlap frames and front during "
            "side-overlap frames. none disables authority remapping."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Analyze only the first N frames.",
    )
    parser.add_argument(
        "--z-source",
        choices=Z_SOURCES,
        default="average",
        help="Which camera determines final z after front/side tracks are paired.",
    )
    parser.add_argument(
        "--z-pairing-check",
        action="store_true",
        help=(
            "Optional off-by-default check before 3D merge: compare same-ID "
            "front/side z pairing against the best z-agreement pairing and "
            "remap side IDs only when the z-error improvement is clear."
        ),
    )
    parser.add_argument(
        "--z-pairing-min-improvement-mm",
        type=float,
        default=0.05,
        help=(
            "Minimum total front/side z-error improvement required before "
            "--z-pairing-check remaps IDs."
        ),
    )
    parser.add_argument(
        "--max-jump-mm",
        type=float,
        default=0.15,
        help=(
            "Drop points after frame-to-frame 3D jumps larger than this. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help=(
            "Odd Savitzky-Golay window used to add smooth 3D coordinates to "
            "tracks_3d.csv. Use 0 or 1 to copy the raw positions unchanged."
        ),
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order for the saved smooth track.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    video_path = args.video or choose_video("Choose combined video")
    calibration_path = resolve_required_tracker_calibration(args, video_path)
    print(f"Calibration file: {calibration_path.resolve()}")
    (
        geometry,
        front_cal,
        side_cal,
        front_roi,
        side_roi,
        _,
        _,
    ) = read_calibration(calibration_path)
    if front_roi is None or side_roi is None:
        raise SystemExit(
            f"{calibration_path} must include front_roi and side_roi. Run "
            "background_subtraction_preview.py with --force-calibration to recreate it."
        )

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

    front_seed_points = interactive_particle_seeds_masked(
        video_path=video_path,
        camera_name="front",
        channel=args.channel,
        roi=front_roi,
        seed_frame=args.seed_frame,
        particle_count=args.particle_count,
        focus_roi=front_focus_roi,
    )
    side_seed_points = interactive_particle_seeds_masked(
        video_path=video_path,
        camera_name="side",
        channel=args.channel,
        roi=side_roi,
        seed_frame=args.seed_frame,
        particle_count=args.particle_count,
        focus_roi=side_focus_roi,
    )

    background_path = resolve_background_path(args)
    background_frame = read_background_frame(background_path, args.background_frame)
    print(f"Background source: {background_path.resolve()}")

    raw_frames = open_video(video_path)
    try:
        front_tracks = track_camera(
            raw_frames=raw_frames,
            background_frame=background_frame,
            roi=front_roi,
            calibration=front_cal,
            geometry=geometry,
            axis="x",
            camera_name="front",
            seed_points=front_seed_points,
            focus_roi=front_focus_roi,
            args=args,
        )
    finally:
        close_video(raw_frames)
    raw_frames = open_video(video_path)
    try:
        side_tracks = track_camera(
            raw_frames=raw_frames,
            background_frame=background_frame,
            roi=side_roi,
            calibration=side_cal,
            geometry=geometry,
            axis="y",
            camera_name="side",
            seed_points=side_seed_points,
            focus_roi=side_focus_roi,
            args=args,
        )
    finally:
        close_video(raw_frames)
    manual_stop_frame = getattr(args, "manual_stop_frame", None)
    if manual_stop_frame is not None:
        manual_stop_frame = int(manual_stop_frame)
        front_tracks = front_tracks.loc[
            front_tracks["frame"] <= manual_stop_frame
        ].reset_index(drop=True)
        side_tracks = side_tracks.loc[
            side_tracks["frame"] <= manual_stop_frame
        ].reset_index(drop=True)
        print(f"Manual stop frame: keeping frames through {manual_stop_frame}.")

    front_tracks, side_tracks, identity_authority_report = apply_identity_authority(
        front_tracks,
        side_tracks,
        args.identity_authority,
        args.front_overlap_frames,
        args.side_overlap_frames,
    )
    front_tracks, side_tracks, z_pairing_report = apply_z_pairing_check(
        front_tracks,
        side_tracks,
        enabled=args.z_pairing_check,
        min_improvement_mm=args.z_pairing_min_improvement_mm,
    )

    tracks_3d = combine_camera_tracks(front_tracks, side_tracks, geometry, args.z_source)
    if args.max_jump_mm > 0:
        tracks_3d = filter_large_3d_jumps(tracks_3d, args.max_jump_mm)
    tracks_3d = add_tracker_smoothing(
        tracks_3d,
        geometry=geometry,
        front_calibration=front_cal,
        side_calibration=side_cal,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )

    output_dir = make_run_output_dir(args.output_dir, args.output_name, video_path)
    write_outputs(
        output_dir,
        geometry,
        front_tracks,
        side_tracks,
        tracks_3d,
        make_plots=not args.no_plots,
        front_overlap_frames=args.front_overlap_frames,
        side_overlap_frames=args.side_overlap_frames,
        identity_authority_report=identity_authority_report,
        z_pairing_report=z_pairing_report,
        calibration_path=calibration_path,
        video_path=video_path,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )
    print_summary(output_dir, front_tracks, side_tracks, tracks_3d)
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
    if args.seed_frame < 0:
        raise SystemExit("--seed-frame must be zero or positive.")
    if args.particle_count <= 0:
        raise SystemExit("--particle-count must be positive.")
    if args.search_radius_px <= 0:
        raise SystemExit("--search-radius-px must be positive.")
    if not 0 < args.threshold_fraction <= 1:
        raise SystemExit("--threshold-fraction must be in the range (0, 1].")
    if args.min_difference < 0:
        raise SystemExit("--min-difference must be non-negative.")
    if args.memory < 0:
        raise SystemExit("--memory must be zero or positive.")
    if args.missing_search_expansion_px < 0:
        raise SystemExit("--missing-search-expansion-px must be non-negative.")
    if args.manual_correction_after_misses <= 0:
        raise SystemExit("--manual-correction-after-misses must be positive.")
    if args.manual_correction_close_px < 0:
        raise SystemExit("--manual-correction-close-px must be non-negative.")
    if args.manual_correction_cooldown_frames < 0:
        raise SystemExit("--manual-correction-cooldown-frames must be non-negative.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")
    if args.z_pairing_min_improvement_mm < 0:
        raise SystemExit("--z-pairing-min-improvement-mm must be non-negative.")
    if args.max_jump_mm < 0:
        raise SystemExit("--max-jump-mm must be non-negative.")
    try:
        validate_smoothing_parameters(args.smooth_window, args.smooth_polyorder)
    except ValueError as exc:
        raise SystemExit(f"Invalid tracker smoothing settings: {exc}.") from exc
    if args.no_focus_box and (
        args.front_focus_box is not None
        or args.side_focus_box is not None
        or args.select_focus_box
    ):
        raise SystemExit(
            "--no-focus-box cannot be combined with focus-box selection."
        )


def resolve_required_tracker_calibration(args: argparse.Namespace, video_path: Path) -> Path:
    if args.calibration is not None:
        if not args.calibration.exists():
            raise SystemExit(
                f"Calibration file does not exist: {args.calibration}\n"
                "Run background_subtraction_preview.py first, or pass an existing "
                "--calibration file."
            )
        return args.calibration

    video_id = infer_video_id(video_path)
    if video_id is not None:
        calibration_path = calibration_path_for_video(video_path, args.calibrations_dir)
        if calibration_path is None or not calibration_path.exists():
            preview_command = (
                f'.\\.venv\\Scripts\\python.exe background_subtraction_preview.py '
                f'--video "{video_path}"'
            )
            raise SystemExit(
                f"No calibration found for {video_id}.\n\n"
                f"Expected:\n{calibration_path}\n\n"
                "Run the preview first so it can create the calibration:\n"
                f"{preview_command}"
            )
        return calibration_path

    fallback = Path("calibration.json")
    if fallback.exists():
        return fallback
    raise SystemExit(
        "No video-specific calibration could be inferred because the video name "
        "does not contain an MI## or SP## trial ID, and calibration.json does not exist.\n"
        "Run background_subtraction_preview.py first, or pass --calibration."
    )


def open_video(video_path: Path):
    try:
        import pims
    except ImportError as exc:
        raise SystemExit(
            "PIMS is required. Install dependencies with "
            "`\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from exc
    return pims.open(str(video_path))


def close_video(video) -> None:
    """Close PIMS/ImageIO readers without noisy ffmpeg destructor tracebacks."""
    import contextlib
    import io

    if video is None:
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        reader = getattr(video, "reader", None)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
            try:
                delattr(video, "reader")
            except Exception:
                pass
            return
        close = getattr(video, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise SystemExit("PyAV is required to read background frames.") from exc

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise SystemExit(f"Could not read frame {frame_index} from {video_path}.")


def read_background_frame(background_path: Path, frame_index: int) -> np.ndarray:
    if not background_path.exists():
        raise SystemExit(f"Background file does not exist: {background_path}")
    if is_image_file(background_path):
        return read_image_frame(background_path)
    if is_video_file(background_path):
        return read_video_frame(background_path, frame_index)
    raise SystemExit(
        f"Unsupported background file type: {background_path.suffix or '(no extension)'}"
    )


def read_image_frame(image_path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required to read background image files.") from exc

    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def resolve_background_path(args: argparse.Namespace) -> Path:
    if args.choose_background:
        return choose_background_file()
    if args.background_file is not None:
        return args.background_file
    if args.background_image is not None:
        return args.background_image
    if args.background_video is not None:
        return args.background_video

    return choose_background_file()


def choose_background_file(title: str = "Choose background image or video") -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(
            f"{title}: pass a path with --background-file because Tk file dialogs "
            f"are unavailable ({exc})."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Background images/videos", "*.avi *.mp4 *.mov *.mkv *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("Video files", "*.avi *.mp4 *.mov *.mkv *.tif *.tiff"),
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise SystemExit(f"No file selected for: {title}")
    return Path(path)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def frame_to_gray(
    frame,
    channel: Literal["red", "green", "blue", "mean"],
    roi: ROI,
) -> np.ndarray:
    arr = roi.crop(np.asarray(frame))
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if channel == "mean":
        return arr[:, :, :3].mean(axis=2).astype(np.float32)
    channel_index = {"red": 0, "green": 1, "blue": 2}[channel]
    return arr[:, :, channel_index].astype(np.float32)


def difference_image(
    frame: np.ndarray,
    background: np.ndarray,
    mode: Literal["abs", "bright", "dark"],
) -> np.ndarray:
    if mode == "bright":
        return np.clip(frame - background, 0, None)
    if mode == "dark":
        return np.clip(background - frame, 0, None)
    return np.abs(frame - background)


def should_select_focus_box(args: argparse.Namespace) -> bool:
    if args.no_focus_box:
        return False
    if args.select_focus_box:
        return True
    if args.front_focus_box is not None or args.side_focus_box is not None:
        return False
    return True


def resolve_focus_rois(
    video_path: Path,
    channel: Literal["red", "green", "blue", "mean"],
    front_roi: ROI,
    side_roi: ROI,
    saved_front_focus_roi: ROI | None,
    saved_side_focus_roi: ROI | None,
    front_focus_box: ROI | None,
    side_focus_box: ROI | None,
    select_focus_box: bool,
    no_focus_box: bool,
) -> tuple[ROI | None, ROI | None]:
    if no_focus_box:
        return None, None

    front_focus_roi = front_focus_box or saved_front_focus_roi
    side_focus_roi = side_focus_box or saved_side_focus_roi
    if select_focus_box:
        front_focus_roi = interactive_focus_roi(video_path, "front", channel, front_roi)
        side_focus_roi = interactive_focus_roi(video_path, "side", channel, side_roi)

    validate_focus_roi(front_focus_roi, front_roi, "front")
    validate_focus_roi(side_focus_roi, side_roi, "side")
    return front_focus_roi, side_focus_roi


def validate_focus_roi(focus_roi: ROI | None, camera_roi: ROI, camera_name: str) -> None:
    if focus_roi is None:
        return
    focus_roi.validate_against_frame(
        (camera_roi.height, camera_roi.width),
        f"{camera_name} focus box",
    )


def print_focus_summary(front_focus_roi: ROI | None, side_focus_roi: ROI | None) -> None:
    def describe(roi: ROI | None) -> str:
        return "none" if roi is None else f"{roi.x},{roi.y},{roi.width},{roi.height}"

    print(
        "Focus masks: "
        f"front={describe(front_focus_roi)}, side={describe(side_focus_roi)}"
    )


def apply_focus_mask(
    image: np.ndarray,
    focus_roi: ROI | None,
    name: str = "focus",
) -> np.ndarray:
    if focus_roi is None:
        return image
    focus_roi.validate_against_frame(image.shape, name)
    masked = np.zeros_like(image)
    y0 = focus_roi.y
    y1 = focus_roi.y + focus_roi.height
    x0 = focus_roi.x
    x1 = focus_roi.x + focus_roi.width
    masked[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    return masked


def apply_focus_mask_with_window(
    image: np.ndarray,
    focus_roi: ROI | None,
    x: float,
    y: float,
    radius: int,
    name: str = "focus",
) -> np.ndarray:
    if focus_roi is None:
        return image
    masked = apply_focus_mask(image, focus_roi, name)
    copy_search_window(masked, image, x=x, y=y, radius=radius)
    return masked


def apply_focus_mask_with_particle_windows(
    image: np.ndarray,
    focus_roi: ROI | None,
    states: dict[int, dict[str, float | int | bool]],
    search_radius_px: int,
    missing_search_expansion_px: float,
    particle_window_radius_px: int | None = None,
    outside_window_threshold: float | None = None,
    name: str = "focus",
) -> np.ndarray:
    if focus_roi is None:
        return image
    masked = apply_focus_mask(image, focus_roi, name)
    for state in states.values():
        if not state["active"]:
            continue
        radius = int(
            round(search_radius_px + int(state["missing"]) * missing_search_expansion_px)
        )
        if particle_window_radius_px is not None:
            radius = max(radius, particle_window_radius_px)
        radius = max(search_radius_px, radius)
        if outside_window_threshold is None:
            copy_search_window(
                masked,
                image,
                x=float(state["x"]),
                y=float(state["y"]),
                radius=radius,
            )
        else:
            copy_thresholded_outside_window(
                masked,
                image,
                focus_roi=focus_roi,
                x=float(state["x"]),
                y=float(state["y"]),
                radius=radius,
                threshold=outside_window_threshold,
            )
    return masked


def copy_thresholded_outside_window(
    target: np.ndarray,
    source: np.ndarray,
    focus_roi: ROI,
    x: float,
    y: float,
    radius: int,
    threshold: float,
) -> None:
    height, width = source.shape[:2]
    x0 = max(0, int(round(x)) - radius)
    x1 = min(width, int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height, int(round(y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    patch = source[y0:y1, x0:x1]
    keep = patch >= threshold

    focus_x0 = max(focus_roi.x, x0)
    focus_x1 = min(focus_roi.x + focus_roi.width, x1)
    focus_y0 = max(focus_roi.y, y0)
    focus_y1 = min(focus_roi.y + focus_roi.height, y1)
    if focus_x0 < focus_x1 and focus_y0 < focus_y1:
        keep[
            focus_y0 - y0 : focus_y1 - y0,
            focus_x0 - x0 : focus_x1 - x0,
        ] = False

    target_patch = target[y0:y1, x0:x1]
    target_patch[keep] = patch[keep]


def copy_search_window(
    target: np.ndarray,
    source: np.ndarray,
    x: float,
    y: float,
    radius: int,
) -> None:
    height, width = source.shape[:2]
    x0 = max(0, int(round(x)) - radius)
    x1 = min(width, int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height, int(round(y)) + radius + 1)
    if x0 < x1 and y0 < y1:
        target[y0:y1, x0:x1] = source[y0:y1, x0:x1]


def interactive_particle_seeds_masked(
    video_path: Path,
    camera_name: str,
    channel: Literal["red", "green", "blue", "mean"],
    roi: ROI,
    seed_frame: int,
    particle_count: int,
    focus_roi: ROI | None,
) -> list[tuple[float, float]]:
    import matplotlib.pyplot as plt

    frames = open_video(video_path)
    try:
        if seed_frame >= len(frames):
            raise SystemExit(
                f"--seed-frame {seed_frame} is outside {camera_name} video length "
                f"{len(frames)}."
            )

        frame = frame_to_gray(frames[seed_frame], channel, roi)
    finally:
        close_video(frames)

    display = apply_focus_mask(frame, focus_roi, f"{camera_name} focus box")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(display, cmap="gray")
    ax.set_title(
        f"{camera_name}: click {particle_count} particle(s), in the same order as the other view"
    )
    ax.set_xlabel("Camera-crop x [px]")
    ax.set_ylabel("Camera-crop y [px]")
    points = plt.ginput(particle_count, timeout=0)
    plt.close(fig)
    if len(points) != particle_count:
        raise SystemExit(
            f"Expected {particle_count} clicked particles for {camera_name}; got "
            f"{len(points)}."
        )
    seed_points = [(float(x), float(y)) for x, y in points]
    validate_seed_points_in_focus(seed_points, focus_roi, camera_name)
    return seed_points


def validate_seed_points_in_focus(
    seed_points: list[tuple[float, float]],
    focus_roi: ROI | None,
    camera_name: str,
) -> None:
    if focus_roi is None:
        return
    outside: list[int] = []
    x0 = focus_roi.x
    x1 = focus_roi.x + focus_roi.width
    y0 = focus_roi.y
    y1 = focus_roi.y + focus_roi.height
    for index, (x, y) in enumerate(seed_points, start=1):
        if not (x0 <= x < x1 and y0 <= y < y1):
            outside.append(index)
    if outside:
        numbers = ", ".join(str(index) for index in outside)
        raise SystemExit(
            f"{camera_name} particle click(s) {numbers} are outside the focus box. "
            "Rerun with a larger focus box or click inside the boxed area."
        )


def interactive_manual_correction(
    diff: np.ndarray,
    camera_name: Literal["front", "side"],
    frame_number: int,
    states: dict[int, dict[str, float | int | bool]],
    reason: str,
) -> tuple[dict[int, tuple[float, float]], str] | None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    particle_ids = sorted(states)
    id_text = ", ".join(str(seed_id) for seed_id in particle_ids)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(bottom=0.18)
    positive = diff[np.isfinite(diff) & (diff > 0)]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    ax.imshow(diff, cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(
        f"{camera_name} frame {frame_number}: click particle ID(s) {id_text}\n"
        f"Manual correction: {reason}"
    )
    ax.set_xlabel("Camera-crop x [px]")
    ax.set_ylabel("Camera-crop y [px]")
    status_text = fig.text(
        0.08,
        0.08,
        f"Next click: particle ID {particle_ids[0]}",
        ha="left",
        va="center",
    )

    for seed_id, state in states.items():
        if not state["active"]:
            continue
        x = float(state["x"])
        y = float(state["y"])
        ax.scatter([x], [y], marker="x", s=55, linewidths=1.5)
        ax.text(x + 4, y + 4, str(seed_id), color="yellow", fontsize=10)

    clicked_points: list[tuple[float, float]] = []
    stop_requested = False
    accept_current_requested = False
    completed = False

    def on_frame_click(event) -> None:
        nonlocal completed
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if len(clicked_points) >= len(particle_ids):
            return
        seed_id = particle_ids[len(clicked_points)]
        x = float(event.xdata)
        y = float(event.ydata)
        clicked_points.append((x, y))
        ax.scatter([x], [y], marker="o", s=45, facecolors="none", edgecolors="cyan")
        ax.text(x + 4, y + 4, str(seed_id), color="cyan", fontsize=10)
        if len(clicked_points) == len(particle_ids):
            completed = True
            plt.close(fig)
        else:
            status_text.set_text(
                f"Next click: particle ID {particle_ids[len(clicked_points)]}"
            )
            fig.canvas.draw_idle()

    def on_stop(_event) -> None:
        nonlocal stop_requested
        stop_requested = True
        plt.close(fig)

    def on_accept_current(_event) -> None:
        nonlocal accept_current_requested
        accept_current_requested = True
        plt.close(fig)

    accept_button_ax = fig.add_axes([0.48, 0.045, 0.20, 0.075])
    accept_button = Button(accept_button_ax, "Accept current")
    accept_button.on_clicked(on_accept_current)

    button_ax = fig.add_axes([0.71, 0.045, 0.21, 0.075])
    stop_button = Button(button_ax, "End track here")
    stop_button.on_clicked(on_stop)
    fig.canvas.mpl_connect("button_press_event", on_frame_click)
    plt.show(block=True)

    if stop_requested:
        return None
    if accept_current_requested:
        return (
            {
                seed_id: (float(states[seed_id]["x"]), float(states[seed_id]["y"]))
                for seed_id in particle_ids
            },
            "accept_current",
        )
    if not completed or len(clicked_points) != len(particle_ids):
        raise SystemExit(
            f"Expected {len(particle_ids)} manual correction click(s) for "
            f"{camera_name} frame {frame_number}; got {len(clicked_points)}."
        )
    return (
        {
            seed_id: (float(x), float(y))
            for seed_id, (x, y) in zip(particle_ids, clicked_points)
        },
        "clicked",
    )


def close_particle_pair(
    states: dict[int, dict[str, float | int | bool]],
    threshold_px: float,
) -> tuple[int, int, float] | None:
    if threshold_px <= 0:
        return None
    active_items = [
        (seed_id, state)
        for seed_id, state in states.items()
        if state["active"]
    ]
    for index, (seed_a, state_a) in enumerate(active_items):
        for seed_b, state_b in active_items[index + 1 :]:
            distance = math.hypot(
                float(state_a["x"]) - float(state_b["x"]),
                float(state_a["y"]) - float(state_b["y"]),
            )
            if distance <= threshold_px:
                return seed_a, seed_b, distance
    return None


def peak_difference_near(diff: np.ndarray, x: float, y: float, radius: int) -> float:
    height, width = diff.shape
    x0 = max(0, int(round(x)) - radius)
    x1 = min(width, int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height, int(round(y)) + radius + 1)
    patch = diff[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.max(patch))


def locate_component_candidates(
    diff: np.ndarray,
    x: float,
    y: float,
    radius: int,
    threshold_fraction: float,
    min_difference: float,
) -> list[dict[str, float | tuple[int, int]]]:
    from scipy import ndimage

    height, width = diff.shape
    x0 = max(0, int(round(x)) - radius)
    x1 = min(width, int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height, int(round(y)) + radius + 1)
    patch = diff[y0:y1, x0:x1]
    if patch.size == 0:
        return []

    peak = float(np.max(patch))
    if peak < min_difference:
        return []

    threshold = max(min_difference, peak * threshold_fraction)
    mask = patch >= threshold
    labels, count = ndimage.label(mask)
    if count == 0:
        return []

    yy, xx = np.indices(patch.shape)
    candidates: list[dict[str, float | tuple[int, int]]] = []
    for label_id in range(1, count + 1):
        component_mask = labels == label_id
        if not np.any(component_mask):
            continue

        weights = np.where(component_mask, patch, 0.0)
        total = float(np.sum(weights))
        if total <= 0:
            continue

        component_values = patch[component_mask]
        component_peak = float(np.max(component_values))
        local_peak_index = int(np.argmax(np.where(component_mask, patch, -np.inf)))
        peak_y, peak_x = np.unravel_index(local_peak_index, patch.shape)
        cx = float(np.sum((xx + x0) * weights) / total)
        cy = float(np.sum((yy + y0) * weights) / total)
        candidates.append(
            {
                "x": cx,
                "y": cy,
                "confidence": component_peak / (float(np.mean(patch)) + 1e-6),
                "peak_difference": component_peak,
                "distance": math.hypot(cx - x, cy - y),
                "component_key": (int(peak_x + x0), int(peak_y + y0)),
            }
        )

    candidates.sort(
        key=lambda candidate: (
            float(candidate["distance"]),
            -float(candidate["peak_difference"]),
        )
    )
    return candidates


def assign_unique_component_candidates(
    candidate_by_seed: dict[int, list[dict[str, float | tuple[int, int]]]],
) -> dict[int, dict[str, float | tuple[int, int]]]:
    if not candidate_by_seed:
        return {}

    seed_ids = sorted(candidate_by_seed)
    component_keys: list[tuple[int, int]] = []
    seen_keys: set[tuple[int, int]] = set()
    best_candidate: dict[
        tuple[int, tuple[int, int]],
        dict[str, float | tuple[int, int]],
    ] = {}

    for seed_id in seed_ids:
        for candidate in candidate_by_seed[seed_id]:
            component_key = candidate["component_key"]
            if not isinstance(component_key, tuple):
                continue
            key = (int(component_key[0]), int(component_key[1]))
            existing = best_candidate.get((seed_id, key))
            if existing is None or float(candidate["distance"]) < float(
                existing["distance"]
            ):
                best_candidate[(seed_id, key)] = candidate
            if key not in seen_keys:
                seen_keys.add(key)
                component_keys.append(key)

    if not component_keys:
        return {}

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return assign_unique_component_candidates_greedy(candidate_by_seed)

    large_cost = 1e9
    costs = np.full((len(seed_ids), len(component_keys)), large_cost, dtype=float)
    for row, seed_id in enumerate(seed_ids):
        for col, component_key in enumerate(component_keys):
            candidate = best_candidate.get((seed_id, component_key))
            if candidate is not None:
                costs[row, col] = float(candidate["distance"])

    row_indexes, column_indexes = linear_sum_assignment(costs)
    assigned: dict[int, dict[str, float | tuple[int, int]]] = {}
    for row, col in zip(row_indexes, column_indexes):
        if costs[row, col] >= large_cost:
            continue
        seed_id = seed_ids[int(row)]
        component_key = component_keys[int(col)]
        assigned[seed_id] = best_candidate[(seed_id, component_key)]
    return assigned


def assign_overlap_component_candidates(
    candidate_by_seed: dict[int, list[dict[str, float | tuple[int, int]]]],
) -> dict[int, dict[str, float | tuple[int, int]]]:
    assigned: dict[int, dict[str, float | tuple[int, int]]] = {}
    for seed_id, candidates in candidate_by_seed.items():
        if candidates:
            assigned[seed_id] = candidates[0]
    return assigned


def assign_unique_component_candidates_greedy(
    candidate_by_seed: dict[int, list[dict[str, float | tuple[int, int]]]],
) -> dict[int, dict[str, float | tuple[int, int]]]:
    choices: list[tuple[float, int, tuple[int, int], dict[str, float | tuple[int, int]]]] = []
    for seed_id, candidates in candidate_by_seed.items():
        for candidate in candidates:
            component_key = candidate["component_key"]
            if not isinstance(component_key, tuple):
                continue
            choices.append(
                (
                    float(candidate["distance"]),
                    seed_id,
                    (int(component_key[0]), int(component_key[1])),
                    candidate,
                )
            )

    assigned: dict[int, dict[str, float | tuple[int, int]]] = {}
    used_components: set[tuple[int, int]] = set()
    for _, seed_id, component_key, candidate in sorted(choices):
        if seed_id in assigned or component_key in used_components:
            continue
        assigned[seed_id] = candidate
        used_components.add(component_key)
    return assigned


def track_camera(
    raw_frames,
    background_frame,
    roi: ROI,
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
    axis: Literal["x", "y"],
    camera_name: Literal["front", "side"],
    seed_points: list[tuple[float, float]],
    focus_roi: ROI | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    frame_count = len(raw_frames)
    end_frame = min(frame_count, args.max_frames) if args.max_frames else frame_count
    manual_stop_frame = getattr(args, "manual_stop_frame", None)
    if manual_stop_frame is not None:
        end_frame = min(end_frame, int(manual_stop_frame) + 1)
    if args.seed_frame >= end_frame:
        raise SystemExit("--seed-frame is outside the analyzed frame range.")

    background = frame_to_gray(background_frame, args.channel, roi)
    validate_seed_points_in_focus(seed_points, focus_roi, camera_name)
    states = {
        seed_id: {
            "x": float(point[0]),
            "y": float(point[1]),
            "missing": 0,
            "active": True,
        }
        for seed_id, point in enumerate(seed_points)
    }
    rows: list[dict[str, float | int | bool | str]] = []
    manual_correction_enabled = not args.no_manual_correction
    manual_correction_count = 0
    last_manual_correction_frame = -args.manual_correction_cooldown_frames - 1
    overlap_ranges = (
        args.front_overlap_frames
        if camera_name == "front"
        else args.side_overlap_frames
    )
    overlap_mode_reported = False

    for seed_id, point in enumerate(seed_points):
        rows.append(
            camera_row(
                seed_id,
                args.seed_frame,
                point[0],
                point[1],
                confidence=1.0,
                peak_difference=0.0,
                calibration=calibration,
                geometry=geometry,
                axis=axis,
                camera_name=camera_name,
            )
        )

    for frame_number in range(args.seed_frame + 1, end_frame):
        overlap_frame = frame_in_ranges(frame_number, overlap_ranges)
        if overlap_frame and not overlap_mode_reported:
            print(
                f"{camera_name} overlap mode active starting at frame "
                f"{frame_number}."
            )
            overlap_mode_reported = True
        frame = frame_to_gray(raw_frames[frame_number], args.channel, roi)
        diff = difference_image(frame, background, args.difference_mode)
        if args.focus_box_mode == "hard":
            shared_diff = apply_focus_mask(diff, focus_roi, f"{camera_name} focus box")
        elif args.focus_box_mode == "seed-only":
            shared_diff = diff
        else:
            shared_diff = None
        frame_rows: list[dict[str, float | int | bool | str]] = []
        missing_triggers: list[int] = []
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
                round(
                    args.search_radius_px
                    + int(state["missing"]) * args.missing_search_expansion_px
                )
            )
            if args.focus_box_mode == "follow":
                search_diff = diff
            else:
                search_diff = shared_diff
            candidate_by_seed[seed_id] = locate_component_candidates(
                search_diff,
                x=float(state["x"]),
                y=float(state["y"]),
                radius=max(args.search_radius_px, current_radius),
                threshold_fraction=args.threshold_fraction,
                min_difference=args.min_difference,
            )

        if overlap_frame:
            assignments = assign_overlap_component_candidates(candidate_by_seed)
        else:
            assignments = assign_unique_component_candidates(candidate_by_seed)

        for seed_id in active_seed_ids:
            state = states[seed_id]
            result = assignments.get(seed_id)
            if result is None:
                state["missing"] = int(state["missing"]) + 1
                if (
                    manual_correction_enabled
                    and int(state["missing"]) >= args.manual_correction_after_misses
                ):
                    missing_triggers.append(seed_id)
                elif int(state["missing"]) > args.memory:
                    state["active"] = False
                continue

            x = float(result["x"])
            y = float(result["y"])
            confidence = float(result["confidence"])
            peak_difference = float(result["peak_difference"])
            state.update({"x": x, "y": y, "missing": 0, "active": True})
            frame_rows.append(
                camera_row(
                    seed_id,
                    frame_number,
                    x,
                    y,
                    confidence=confidence,
                    peak_difference=peak_difference,
                    calibration=calibration,
                    geometry=geometry,
                    axis=axis,
                    camera_name=camera_name,
                )
            )

        correction_reason = None
        if manual_correction_enabled:
            if missing_triggers:
                missing_text = ", ".join(str(seed_id) for seed_id in missing_triggers)
                correction_reason = f"lost particle ID(s) {missing_text}"
            elif (
                not overlap_frame
                and frame_number - last_manual_correction_frame
                >= args.manual_correction_cooldown_frames
            ):
                close_pair = close_particle_pair(
                    states,
                    args.manual_correction_close_px,
                )
                if close_pair is not None:
                    seed_a, seed_b, distance = close_pair
                    correction_reason = (
                        f"particle IDs {seed_a} and {seed_b} are "
                        f"{distance:.1f} px apart"
                    )

        if correction_reason is not None:
            print(
                f"{camera_name} frame {frame_number}: manual correction "
                f"({correction_reason})."
            )
            pre_correction_points = {
                seed_id: (float(state["x"]), float(state["y"]))
                for seed_id, state in states.items()
            }
            correction_result = interactive_manual_correction(
                diff=diff,
                camera_name=camera_name,
                frame_number=frame_number,
                states=states,
                reason=correction_reason,
            )
            if correction_result is None:
                stop_frame = max(args.seed_frame, frame_number - 1)
                setattr(args, "manual_stop_frame", stop_frame)
                print(
                    f"{camera_name} frame {frame_number}: ending tracking early; "
                    f"last saved frame is {stop_frame}."
                )
                break
            corrected_points, correction_action = correction_result
            frame_rows = []
            correction_particle_ids = ",".join(str(seed_id) for seed_id in sorted(states))
            for seed_id in sorted(states):
                x, y = corrected_points[seed_id]
                pre_x, pre_y = pre_correction_points[seed_id]
                states[seed_id].update(
                    {"x": x, "y": y, "missing": 0, "active": True}
                )
                frame_rows.append(
                    camera_row(
                        seed_id,
                        frame_number,
                        x,
                        y,
                        confidence=1.0,
                        peak_difference=peak_difference_near(
                            diff,
                            x,
                            y,
                            args.search_radius_px,
                        ),
                        calibration=calibration,
                        geometry=geometry,
                        axis=axis,
                        camera_name=camera_name,
                        manual_correction_reviewed=True,
                        manual_correction_applied=correction_action == "clicked",
                        manual_correction_action=correction_action,
                        manual_correction_reason=correction_reason,
                        manual_correction_particle_ids=correction_particle_ids,
                        pre_manual_correction_x_px=pre_x,
                        pre_manual_correction_y_px=pre_y,
                    )
                )
            manual_correction_count += 1
            last_manual_correction_frame = frame_number

        rows.extend(frame_rows)

    tracks = pd.DataFrame(rows).sort_values(["seed_id", "frame"]).reset_index(drop=True)
    print(
        f"{camera_name} background-subtracted tracks: "
        + ", ".join(
            f"seed {seed_id}: {len(group)} points, frames {int(group['frame'].min())}-{int(group['frame'].max())}"
            for seed_id, group in tracks.groupby("seed_id", sort=True)
        )
    )
    if manual_correction_enabled:
        print(f"{camera_name} manual corrections: {manual_correction_count}")
    return tracks


def locate_next_position(
    diff: np.ndarray,
    x: float,
    y: float,
    radius: int,
    threshold_fraction: float,
    min_difference: float,
) -> tuple[float, float, float, float] | None:
    candidates = locate_component_candidates(
        diff=diff,
        x=x,
        y=y,
        radius=radius,
        threshold_fraction=threshold_fraction,
        min_difference=min_difference,
    )
    if not candidates:
        return None
    best = candidates[0]
    return (
        float(best["x"]),
        float(best["y"]),
        float(best["confidence"]),
        float(best["peak_difference"]),
    )


def camera_row(
    seed_id: int,
    frame: int,
    image_x_px: float,
    image_y_px: float,
    confidence: float,
    peak_difference: float,
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
    axis: Literal["x", "y"],
    camera_name: Literal["front", "side"],
    manual_correction_reviewed: bool = False,
    manual_correction_applied: bool = False,
    manual_correction_action: str = "",
    manual_correction_reason: str = "",
    manual_correction_particle_ids: str = "",
    pre_manual_correction_x_px: float = math.nan,
    pre_manual_correction_y_px: float = math.nan,
) -> dict[str, float | int | bool | str]:
    z_mm = float(calibration.z_from_image_y(image_y_px, geometry))
    horizontal_mm = float(
        calibration.horizontal_mm_from_image_x(image_x_px, z_mm, geometry)
    )
    row = {
        "seed_id": seed_id,
        "particle": seed_id,
        "frame": frame,
        f"{camera_name}_image_x_px": image_x_px,
        f"{camera_name}_image_y_px": image_y_px,
        f"{camera_name}_z_mm": z_mm,
        f"{camera_name}_confidence": confidence,
        f"{camera_name}_peak_difference": peak_difference,
        f"{camera_name}_manual_correction_reviewed": manual_correction_reviewed,
        f"{camera_name}_manual_correction_applied": manual_correction_applied,
        f"{camera_name}_manual_correction_action": manual_correction_action,
        f"{camera_name}_manual_correction_reason": manual_correction_reason,
        f"{camera_name}_manual_correction_particle_ids": manual_correction_particle_ids,
        f"{camera_name}_pre_manual_correction_x_px": pre_manual_correction_x_px,
        f"{camera_name}_pre_manual_correction_y_px": pre_manual_correction_y_px,
        f"{camera_name}_manual_correction_x_px": image_x_px
        if manual_correction_reviewed
        else math.nan,
        f"{camera_name}_manual_correction_y_px": image_y_px
        if manual_correction_reviewed
        else math.nan,
    }
    row[f"{axis}_mm"] = horizontal_mm
    return row


def apply_identity_authority(
    front: pd.DataFrame,
    side: pd.DataFrame,
    mode: Literal["auto", "none", "front", "side"],
    front_overlap_frames: tuple[tuple[int, int], ...],
    side_overlap_frames: tuple[tuple[int, int], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    front = add_camera_identity_columns(front, "front")
    side = add_camera_identity_columns(side, "side")
    report: dict[str, object] = {
        "mode": mode,
        "enabled": mode != "none",
        "front_remapped_rows": 0,
        "side_remapped_rows": 0,
        "authority_frames": [],
    }
    if mode == "none":
        return front, side, report

    frames = sorted(set(front["frame"].astype(int)) | set(side["frame"].astype(int)))
    for frame in frames:
        authority = identity_authority_for_frame(
            frame,
            mode,
            front_overlap_frames,
            side_overlap_frames,
        )
        if authority is None:
            continue

        if authority == "side":
            target_camera = "front"
            front, remapped = remap_camera_frame_to_authority(
                target=front,
                authority=side,
                frame=frame,
                target_camera="front",
                authority_camera="side",
                target_z_col="front_z_mm",
                authority_z_col="side_z_mm",
            )
            side = mark_authority_frame(side, "side", frame, "side")
            report["front_remapped_rows"] = int(report["front_remapped_rows"]) + remapped
        else:
            target_camera = "side"
            side, remapped = remap_camera_frame_to_authority(
                target=side,
                authority=front,
                frame=frame,
                target_camera="side",
                authority_camera="front",
                target_z_col="side_z_mm",
                authority_z_col="front_z_mm",
            )
            front = mark_authority_frame(front, "front", frame, "front")
            report["side_remapped_rows"] = int(report["side_remapped_rows"]) + remapped

        report["authority_frames"].append(
            {
                "frame": frame,
                "authority_camera": authority,
                "remapped_camera": target_camera,
                "remapped_rows": remapped,
            }
        )

    total = int(report["front_remapped_rows"]) + int(report["side_remapped_rows"])
    if total:
        print(
            "Identity authority remapped rows: "
            f"front={report['front_remapped_rows']}, "
            f"side={report['side_remapped_rows']}."
        )
    return front, side, report


def add_camera_identity_columns(df: pd.DataFrame, camera_name: str) -> pd.DataFrame:
    tagged = df.copy()
    tagged[f"{camera_name}_original_seed_id"] = tagged["seed_id"].astype(int)
    tagged[f"{camera_name}_original_particle"] = tagged["particle"].astype(int)
    tagged[f"{camera_name}_identity_authority_applied"] = False
    tagged[f"{camera_name}_identity_authority_camera"] = ""
    tagged[f"{camera_name}_identity_remapped"] = False
    return tagged


def identity_authority_for_frame(
    frame: int,
    mode: Literal["auto", "none", "front", "side"],
    front_overlap_frames: tuple[tuple[int, int], ...],
    side_overlap_frames: tuple[tuple[int, int], ...],
) -> Literal["front", "side"] | None:
    if mode == "none":
        return None
    front_overlap = frame_in_ranges(frame, front_overlap_frames)
    side_overlap = frame_in_ranges(frame, side_overlap_frames)
    if mode == "front":
        return "front" if side_overlap else None
    if mode == "side":
        return "side" if front_overlap else None
    if front_overlap and not side_overlap:
        return "side"
    if side_overlap and not front_overlap:
        return "front"
    return None


def remap_camera_frame_to_authority(
    target: pd.DataFrame,
    authority: pd.DataFrame,
    frame: int,
    target_camera: str,
    authority_camera: str,
    target_z_col: str,
    authority_z_col: str,
) -> tuple[pd.DataFrame, int]:
    target_mask = target["frame"].astype(int) == frame
    authority_frame = authority.loc[authority["frame"].astype(int) == frame]
    target_frame = target.loc[target_mask]
    if target_frame.empty or authority_frame.empty:
        return target, 0

    assignments = match_rows_by_z(
        target_frame,
        authority_frame,
        target_z_col,
        authority_z_col,
    )
    if not assignments:
        return target, 0

    target = target.copy()
    remapped = 0
    for target_index, authority_index in assignments.items():
        new_particle = int(authority.loc[authority_index, "particle"])
        old_particle = int(target.loc[target_index, "particle"])
        new_seed_id = int(authority.loc[authority_index, "seed_id"])
        old_seed_id = int(target.loc[target_index, "seed_id"])
        target.loc[target_index, "particle"] = new_particle
        target.loc[target_index, "seed_id"] = new_seed_id
        target.loc[target_index, f"{target_camera}_identity_authority_applied"] = True
        target.loc[
            target_index,
            f"{target_camera}_identity_authority_camera",
        ] = authority_camera
        changed = old_particle != new_particle or old_seed_id != new_seed_id
        target.loc[target_index, f"{target_camera}_identity_remapped"] = changed
        if changed:
            remapped += 1

    return target, remapped


def mark_authority_frame(
    df: pd.DataFrame,
    camera_name: str,
    frame: int,
    authority_camera: str,
) -> pd.DataFrame:
    marked = df.copy()
    mask = marked["frame"].astype(int) == frame
    marked.loc[mask, f"{camera_name}_identity_authority_applied"] = True
    marked.loc[mask, f"{camera_name}_identity_authority_camera"] = authority_camera
    return marked


def match_rows_by_z(
    target_frame: pd.DataFrame,
    authority_frame: pd.DataFrame,
    target_z_col: str,
    authority_z_col: str,
) -> dict[int, int]:
    if target_frame.empty or authority_frame.empty:
        return {}

    target_indexes = list(target_frame.index)
    authority_indexes = list(authority_frame.index)
    target_z = target_frame[target_z_col].to_numpy(dtype=float)
    authority_z = authority_frame[authority_z_col].to_numpy(dtype=float)
    costs = np.abs(target_z[:, None] - authority_z[None, :])
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return match_rows_by_z_greedy(target_indexes, authority_indexes, costs)

    row_indexes, col_indexes = linear_sum_assignment(costs)
    return {
        int(target_indexes[int(row)]): int(authority_indexes[int(col)])
        for row, col in zip(row_indexes, col_indexes)
    }


def match_rows_by_z_greedy(
    target_indexes: list[int],
    authority_indexes: list[int],
    costs: np.ndarray,
) -> dict[int, int]:
    choices: list[tuple[float, int, int]] = []
    for row, target_index in enumerate(target_indexes):
        for col, authority_index in enumerate(authority_indexes):
            choices.append((float(costs[row, col]), target_index, authority_index))
    assignments: dict[int, int] = {}
    used_authority: set[int] = set()
    for _, target_index, authority_index in sorted(choices):
        if target_index in assignments or authority_index in used_authority:
            continue
        assignments[target_index] = authority_index
        used_authority.add(authority_index)
    return assignments


def apply_z_pairing_check(
    front: pd.DataFrame,
    side: pd.DataFrame,
    enabled: bool,
    min_improvement_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {
        "enabled": bool(enabled),
        "min_improvement_mm": float(min_improvement_mm),
        "side_remapped_rows": 0,
        "frames_checked": 0,
        "frames_remapped": [],
    }
    if not enabled:
        return front, side, report

    front = ensure_z_pairing_columns(front, "front")
    side = ensure_z_pairing_columns(side, "side")
    frames = sorted(set(front["frame"].astype(int)) & set(side["frame"].astype(int)))
    for frame in frames:
        front_frame = front.loc[front["frame"].astype(int) == frame]
        side_frame = side.loc[side["frame"].astype(int) == frame]
        if len(front_frame) < 2 or len(side_frame) < 2:
            continue

        report["frames_checked"] = int(report["frames_checked"]) + 1
        remap, current_cost, optimal_cost, improvement = z_pairing_for_frame(
            front_frame,
            side_frame,
        )
        if not remap or improvement < min_improvement_mm:
            continue

        side = side.copy()
        remapped_rows = 0
        for side_index, front_index in remap.items():
            new_particle = int(front.loc[front_index, "particle"])
            old_particle = int(side.loc[side_index, "particle"])
            new_seed_id = int(front.loc[front_index, "seed_id"])
            old_seed_id = int(side.loc[side_index, "seed_id"])
            side.loc[side_index, "particle"] = new_particle
            side.loc[side_index, "seed_id"] = new_seed_id
            side.loc[side_index, "side_z_pairing_check_applied"] = True
            side.loc[side_index, "side_z_pairing_current_cost_mm"] = current_cost
            side.loc[side_index, "side_z_pairing_optimal_cost_mm"] = optimal_cost
            side.loc[side_index, "side_z_pairing_improvement_mm"] = improvement
            changed = old_particle != new_particle or old_seed_id != new_seed_id
            side.loc[side_index, "side_z_pairing_remapped"] = changed
            if changed:
                remapped_rows += 1

        front = mark_z_pairing_front_frame(
            front,
            frame,
            current_cost,
            optimal_cost,
            improvement,
        )
        if remapped_rows:
            report["side_remapped_rows"] = int(report["side_remapped_rows"]) + remapped_rows
            report["frames_remapped"].append(
                {
                    "frame": frame,
                    "side_remapped_rows": remapped_rows,
                    "current_cost_mm": current_cost,
                    "optimal_cost_mm": optimal_cost,
                    "improvement_mm": improvement,
                }
            )

    if int(report["side_remapped_rows"]):
        print(
            "Z-pairing check remapped side rows: "
            f"{report['side_remapped_rows']}."
        )
    else:
        print("Z-pairing check enabled: no side rows remapped.")
    return front, side, report


def ensure_z_pairing_columns(df: pd.DataFrame, camera_name: str) -> pd.DataFrame:
    tagged = df.copy()
    tagged[f"{camera_name}_pre_z_pairing_seed_id"] = tagged["seed_id"].astype(int)
    tagged[f"{camera_name}_pre_z_pairing_particle"] = tagged["particle"].astype(int)
    tagged[f"{camera_name}_z_pairing_check_applied"] = False
    tagged[f"{camera_name}_z_pairing_remapped"] = False
    tagged[f"{camera_name}_z_pairing_current_cost_mm"] = np.nan
    tagged[f"{camera_name}_z_pairing_optimal_cost_mm"] = np.nan
    tagged[f"{camera_name}_z_pairing_improvement_mm"] = np.nan
    return tagged


def z_pairing_for_frame(
    front_frame: pd.DataFrame,
    side_frame: pd.DataFrame,
) -> tuple[dict[int, int], float, float, float]:
    front_indexes = list(front_frame.index)
    side_indexes = list(side_frame.index)
    front_z = front_frame["front_z_mm"].to_numpy(dtype=float)
    side_z = side_frame["side_z_mm"].to_numpy(dtype=float)
    costs = np.abs(side_z[:, None] - front_z[None, :])

    assignments = match_rows_by_z_greedy(side_indexes, front_indexes, costs)
    try:
        from scipy.optimize import linear_sum_assignment

        row_indexes, col_indexes = linear_sum_assignment(costs)
        assignments = {
            int(side_indexes[int(row)]): int(front_indexes[int(col)])
            for row, col in zip(row_indexes, col_indexes)
        }
    except ImportError:
        pass

    current_cost = current_z_pairing_cost(front_frame, side_frame)
    optimal_cost = float(
        sum(
            abs(
                float(side_frame.loc[side_index, "side_z_mm"])
                - float(front_frame.loc[front_index, "front_z_mm"])
            )
            for side_index, front_index in assignments.items()
        )
    )
    improvement = current_cost - optimal_cost
    remap = {
        side_index: front_index
        for side_index, front_index in assignments.items()
        if int(side_frame.loc[side_index, "particle"])
        != int(front_frame.loc[front_index, "particle"])
    }
    return remap, current_cost, optimal_cost, improvement


def current_z_pairing_cost(front_frame: pd.DataFrame, side_frame: pd.DataFrame) -> float:
    merged = front_frame[["particle", "front_z_mm"]].merge(
        side_frame[["particle", "side_z_mm"]],
        on="particle",
        how="inner",
    )
    if merged.empty:
        return float("inf")
    return float(np.abs(merged["front_z_mm"] - merged["side_z_mm"]).sum())


def mark_z_pairing_front_frame(
    front: pd.DataFrame,
    frame: int,
    current_cost: float,
    optimal_cost: float,
    improvement: float,
) -> pd.DataFrame:
    marked = front.copy()
    mask = marked["frame"].astype(int) == int(frame)
    marked.loc[mask, "front_z_pairing_check_applied"] = True
    marked.loc[mask, "front_z_pairing_current_cost_mm"] = current_cost
    marked.loc[mask, "front_z_pairing_optimal_cost_mm"] = optimal_cost
    marked.loc[mask, "front_z_pairing_improvement_mm"] = improvement
    return marked


def combine_camera_tracks(
    front: pd.DataFrame,
    side: pd.DataFrame,
    geometry: ChamberGeometry,
    z_source: Literal["average", "front", "side"],
) -> pd.DataFrame:
    merged = front.merge(side, on=["seed_id", "particle", "frame"], how="inner")
    if merged.empty:
        raise SystemExit("Front and side tracks have no overlapping frames.")

    front_z = merged["front_z_mm"].to_numpy(dtype=float)
    side_z = merged["side_z_mm"].to_numpy(dtype=float)
    if z_source == "front":
        z_mm = front_z
    elif z_source == "side":
        z_mm = side_z
    else:
        z_mm = (front_z + side_z) / 2.0

    x_mm = merged["x_mm"].to_numpy(dtype=float)
    y_mm = merged["y_mm"].to_numpy(dtype=float)
    radial_mm = np.sqrt(x_mm**2 + y_mm**2)
    radius_mm = geometry.radius_at_z(z_mm)

    tracks = pd.DataFrame(
        {
            "seed_id": merged["seed_id"].astype(int),
            "particle": merged["particle"].astype(int),
            "frame": merged["frame"].astype(int),
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "front_z_mm": front_z,
            "side_z_mm": side_z,
            "z_error_mm": np.abs(front_z - side_z),
            "radial_mm": radial_mm,
            "chamber_radius_mm": radius_mm,
            "inside_chamber": radial_mm <= radius_mm,
            "front_image_x_px": merged["front_image_x_px"].astype(float),
            "front_image_y_px": merged["front_image_y_px"].astype(float),
            "side_image_x_px": merged["side_image_x_px"].astype(float),
            "side_image_y_px": merged["side_image_y_px"].astype(float),
            "front_confidence": merged["front_confidence"].astype(float),
            "side_confidence": merged["side_confidence"].astype(float),
            "front_peak_difference": merged["front_peak_difference"].astype(float),
            "side_peak_difference": merged["side_peak_difference"].astype(float),
        }
    )
    identity_columns = [
        "front_original_seed_id",
        "front_original_particle",
        "front_identity_authority_applied",
        "front_identity_authority_camera",
        "front_identity_remapped",
        "front_pre_z_pairing_seed_id",
        "front_pre_z_pairing_particle",
        "front_z_pairing_check_applied",
        "front_z_pairing_remapped",
        "front_z_pairing_current_cost_mm",
        "front_z_pairing_optimal_cost_mm",
        "front_z_pairing_improvement_mm",
        "side_original_seed_id",
        "side_original_particle",
        "side_identity_authority_applied",
        "side_identity_authority_camera",
        "side_identity_remapped",
        "side_pre_z_pairing_seed_id",
        "side_pre_z_pairing_particle",
        "side_z_pairing_check_applied",
        "side_z_pairing_remapped",
        "side_z_pairing_current_cost_mm",
        "side_z_pairing_optimal_cost_mm",
        "side_z_pairing_improvement_mm",
    ]
    manual_columns = [
        "front_manual_correction_reviewed",
        "front_manual_correction_applied",
        "front_manual_correction_action",
        "front_manual_correction_reason",
        "front_manual_correction_particle_ids",
        "front_pre_manual_correction_x_px",
        "front_pre_manual_correction_y_px",
        "front_manual_correction_x_px",
        "front_manual_correction_y_px",
        "side_manual_correction_reviewed",
        "side_manual_correction_applied",
        "side_manual_correction_action",
        "side_manual_correction_reason",
        "side_manual_correction_particle_ids",
        "side_pre_manual_correction_x_px",
        "side_pre_manual_correction_y_px",
        "side_manual_correction_x_px",
        "side_manual_correction_y_px",
    ]
    for column in identity_columns:
        if column in merged.columns:
            tracks[column] = merged[column].to_numpy()
    for column in manual_columns:
        if column in merged.columns:
            tracks[column] = merged[column].to_numpy()

    def merged_bool(column: str) -> np.ndarray:
        if column not in merged.columns:
            return np.zeros(len(merged), dtype=bool)
        return merged[column].fillna(False).astype(bool).to_numpy()

    def merged_text(column: str) -> list[str]:
        if column not in merged.columns:
            return [""] * len(merged)
        return ["" if pd.isna(value) else str(value) for value in merged[column]]

    front_reviewed = merged_bool("front_manual_correction_reviewed")
    side_reviewed = merged_bool("side_manual_correction_reviewed")
    front_applied = merged_bool("front_manual_correction_applied")
    side_applied = merged_bool("side_manual_correction_applied")
    tracks["manual_correction_reviewed"] = front_reviewed | side_reviewed
    tracks["manual_correction_applied"] = front_applied | side_applied

    front_actions = merged_text("front_manual_correction_action")
    side_actions = merged_text("side_manual_correction_action")
    front_reasons = merged_text("front_manual_correction_reason")
    side_reasons = merged_text("side_manual_correction_reason")
    tracks["manual_correction_cameras"] = [
        ",".join(
            camera
            for camera, reviewed in (("front", front_flag), ("side", side_flag))
            if reviewed
        )
        for front_flag, side_flag in zip(front_reviewed, side_reviewed)
    ]
    tracks["manual_correction_actions"] = [
        ";".join(
            f"{camera}:{action}"
            for camera, reviewed, action in (
                ("front", front_flag, front_action),
                ("side", side_flag, side_action),
            )
            if reviewed and action
        )
        for front_flag, side_flag, front_action, side_action in zip(
            front_reviewed,
            side_reviewed,
            front_actions,
            side_actions,
        )
    ]
    tracks["manual_correction_reasons"] = [
        ";".join(
            f"{camera}:{reason}"
            for camera, reviewed, reason in (
                ("front", front_flag, front_reason),
                ("side", side_flag, side_reason),
            )
            if reviewed and reason
        )
        for front_flag, side_flag, front_reason, side_reason in zip(
            front_reviewed,
            side_reviewed,
            front_reasons,
            side_reasons,
        )
    ]
    return tracks.sort_values(["seed_id", "frame"]).reset_index(drop=True)


def filter_large_3d_jumps(tracks: pd.DataFrame, max_jump_mm: float) -> pd.DataFrame:
    kept_groups: list[pd.DataFrame] = []
    removed = 0
    for _, group in tracks.groupby("seed_id", sort=False):
        group = group.sort_values("frame").copy()
        jump = np.sqrt(
            group["x_mm"].diff() ** 2
            + group["y_mm"].diff() ** 2
            + group["z_mm"].diff() ** 2
        )
        keep = jump.isna() | (jump <= max_jump_mm)
        removed += int((~keep).sum())
        kept_groups.append(group.loc[keep])

    filtered = pd.concat(kept_groups, ignore_index=True) if kept_groups else tracks
    print(f"Removed {removed} points with 3D jumps larger than {max_jump_mm:g} mm.")
    return filtered.sort_values(["seed_id", "frame"]).reset_index(drop=True)


def add_tracker_smoothing(
    tracks: pd.DataFrame,
    geometry: ChamberGeometry,
    front_calibration: CameraCalibration,
    side_calibration: CameraCalibration,
    smooth_window: int,
    smooth_polyorder: int,
) -> pd.DataFrame:
    """Save raw/smooth 3D coordinates and both views' pixel projections."""
    group_column = "seed_id" if "seed_id" in tracks.columns else "particle"
    result = add_smoothed_track_positions(
        tracks,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        group_column=group_column,
    )

    raw_pixel_columns = (
        "front_image_x_px",
        "front_image_y_px",
        "side_image_x_px",
        "side_image_y_px",
    )
    for column in raw_pixel_columns:
        result[column.replace("_px", "_raw_px")] = result[column].to_numpy(dtype=float)

    x_smooth = result["x_smooth_mm"].to_numpy(dtype=float)
    y_smooth = result["y_smooth_mm"].to_numpy(dtype=float)
    z_smooth = result["z_smooth_mm"].to_numpy(dtype=float)
    result["front_image_x_smooth_px"] = (
        front_calibration.center_x_at_z(z_smooth, geometry)
        + x_smooth * front_calibration.px_per_mm_at_z(z_smooth, geometry)
    )
    result["front_image_y_smooth_px"] = image_y_from_z(
        front_calibration,
        z_smooth,
        geometry,
    )
    result["side_image_x_smooth_px"] = (
        side_calibration.center_x_at_z(z_smooth, geometry)
        + y_smooth * side_calibration.px_per_mm_at_z(z_smooth, geometry)
    )
    result["side_image_y_smooth_px"] = image_y_from_z(
        side_calibration,
        z_smooth,
        geometry,
    )
    return result


def image_y_from_z(
    calibration: CameraCalibration,
    z_mm: np.ndarray,
    geometry: ChamberGeometry,
) -> np.ndarray:
    bottom_y = calibration.bottom_center_px[1]
    top_y = calibration.top_center_px[1]
    return bottom_y + (np.asarray(z_mm, dtype=float) / geometry.gap_mm) * (
        top_y - bottom_y
    )


def make_run_output_dir(parent_dir: Path, output_name: str | None, video_path: Path) -> Path:
    parent_dir.mkdir(parents=True, exist_ok=True)
    if output_name:
        base_name = sanitize_output_name(output_name)
    else:
        video_name = sanitize_output_name(video_path.stem) or "run"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{video_name}_{timestamp}"

    output_dir = parent_dir / base_name
    if not output_dir.exists():
        return output_dir

    suffix = 2
    while True:
        candidate = parent_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def sanitize_output_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "run"


def format_frame_ranges(ranges: tuple[tuple[int, int], ...]) -> str:
    return ",".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )


def add_overlap_tags(
    tracks: pd.DataFrame,
    front_overlap_frames: tuple[tuple[int, int], ...],
    side_overlap_frames: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    tagged = tracks.copy()
    front_enabled = bool(front_overlap_frames)
    side_enabled = bool(side_overlap_frames)
    tagged["overlap_mode_enabled"] = front_enabled or side_enabled
    tagged["front_overlap_mode_enabled"] = front_enabled
    tagged["side_overlap_mode_enabled"] = side_enabled
    tagged["front_overlap_ranges"] = format_frame_ranges(front_overlap_frames)
    tagged["side_overlap_ranges"] = format_frame_ranges(side_overlap_frames)

    if "frame" in tagged.columns:
        frames = [int(frame) for frame in tagged["frame"].to_list()]
        front_flags = [
            frame_in_ranges(frame, front_overlap_frames)
            for frame in frames
        ]
        side_flags = [
            frame_in_ranges(frame, side_overlap_frames)
            for frame in frames
        ]
        tagged["front_overlap_frame"] = front_flags
        tagged["side_overlap_frame"] = side_flags
        tagged["overlap_frame"] = [
            front_flag or side_flag
            for front_flag, side_flag in zip(front_flags, side_flags)
        ]
    return tagged


def write_outputs(
    output_dir: Path,
    geometry: ChamberGeometry,
    front_tracks: pd.DataFrame,
    side_tracks: pd.DataFrame,
    tracks_3d: pd.DataFrame,
    make_plots: bool,
    front_overlap_frames: tuple[tuple[int, int], ...],
    side_overlap_frames: tuple[tuple[int, int], ...],
    identity_authority_report: dict[str, object],
    z_pairing_report: dict[str, object],
    calibration_path: Path,
    video_path: Path,
    smooth_window: int,
    smooth_polyorder: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tagged_front_tracks = add_overlap_tags(
        front_tracks,
        front_overlap_frames,
        side_overlap_frames,
    )
    tagged_side_tracks = add_overlap_tags(
        side_tracks,
        front_overlap_frames,
        side_overlap_frames,
    )
    tagged_tracks_3d = add_overlap_tags(
        tracks_3d,
        front_overlap_frames,
        side_overlap_frames,
    )
    tagged_front_tracks.to_csv(output_dir / "front_background_tracks.csv", index=False)
    tagged_side_tracks.to_csv(output_dir / "side_background_tracks.csv", index=False)
    tagged_tracks_3d.to_csv(output_dir / "tracks_3d.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "chamber": {
                    "bottom_diameter_mm": geometry.bottom_diameter_mm,
                    "top_diameter_mm": geometry.top_diameter_mm,
                    "gap_mm": geometry.gap_mm,
                },
                "video": str(video_path.resolve()),
                "video_id": infer_video_id(video_path),
                "calibration_file_used": str(calibration_path.resolve()),
                "front_points": int(len(front_tracks)),
                "side_points": int(len(side_tracks)),
                "tracks_3d_points": int(len(tracks_3d)),
                "particles": int(tracks_3d["particle"].nunique()),
                "position_smoothing": {
                    "owner": "background_subtraction_tracker",
                    "method": "Savitzky-Golay within continuous frame segments",
                    "window": int(smooth_window),
                    "polyorder": int(smooth_polyorder),
                    "split_at_frame_gaps": True,
                    "raw_coordinates_preserved": True,
                },
                "overlap_mode": {
                    "enabled": bool(front_overlap_frames or side_overlap_frames),
                    "front_enabled": bool(front_overlap_frames),
                    "side_enabled": bool(side_overlap_frames),
                    "front_ranges": format_frame_ranges(front_overlap_frames),
                    "side_ranges": format_frame_ranges(side_overlap_frames),
                },
                "identity_authority": identity_authority_report,
                "z_pairing_check": z_pairing_report,
            },
            f,
            indent=2,
        )
        f.write("\n")
    if make_plots:
        plot_projections(tracks_3d, output_dir / "track_projections.png")
        plot_chamber_scale_projections(
            tracks_3d, geometry, output_dir / "track_projections_chamber_scale.png"
        )


def plot_projections(tracks: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    projections = [
        ("x_mm", "y_mm", "XY projection"),
        ("x_mm", "z_mm", "XZ projection"),
        ("y_mm", "z_mm", "YZ projection"),
    ]
    for ax, (x_col, y_col, title) in zip(axes, projections):
        for _, group in tracks.groupby("particle", sort=False):
            ax.plot(group[x_col], group[y_col], linewidth=1.2, alpha=0.85)
            ax.scatter(group[x_col], group[y_col], s=8, alpha=0.45)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_chamber_scale_projections(
    tracks: pd.DataFrame,
    geometry: ChamberGeometry,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    xy_limit = geometry.bottom_radius_mm
    z_min = 0.0
    z_max = geometry.gap_mm

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    projections = [
        ("x_mm", "y_mm", "XY projection", (-xy_limit, xy_limit), (-xy_limit, xy_limit)),
        ("x_mm", "z_mm", "XZ projection", (-xy_limit, xy_limit), (z_min, z_max)),
        ("y_mm", "z_mm", "YZ projection", (-xy_limit, xy_limit), (z_min, z_max)),
    ]
    for ax, (x_col, y_col, title, xlim, ylim) in zip(axes, projections):
        for _, group in tracks.groupby("particle", sort=False):
            ax.plot(group[x_col], group[y_col], linewidth=1.2, alpha=0.85)
            ax.scatter(group[x_col], group[y_col], s=8, alpha=0.45)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{title} chamber scale")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(
    output_dir: Path,
    front_tracks: pd.DataFrame,
    side_tracks: pd.DataFrame,
    tracks_3d: pd.DataFrame,
) -> None:
    print("\nDone.")
    print(f"Front tracked points: {len(front_tracks)}")
    print(f"Side tracked points:  {len(side_tracks)}")
    print(f"3D tracked points:    {len(tracks_3d)}")
    print(f"Particles:            {tracks_3d['particle'].nunique()}")
    print(f"Output directory:     {output_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
