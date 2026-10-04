"""Overlay 3D tracking CSV positions on a preview video for verification."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from background_subtraction_tracker import sanitize_output_name
from track_orthogonal_3d import read_calibration

VIEW_CHOICES = ("both", "front", "side")
POSITION_SOURCES = ("raw", "smoothed")
RAW_PIXEL_COLUMNS = (
    "particle",
    "frame",
    "front_image_x_px",
    "front_image_y_px",
    "side_image_x_px",
    "side_image_y_px",
)
SMOOTH_PIXEL_COLUMNS = (
    "front_image_x_smooth_px",
    "front_image_y_smooth_px",
    "side_image_x_smooth_px",
    "side_image_y_smooth_px",
)
REQUIRED_COLUMNS = (*RAW_PIXEL_COLUMNS, *SMOOTH_PIXEL_COLUMNS)
COLORS = [
    (255, 60, 60),
    (70, 170, 255),
    (80, 255, 120),
    (255, 120, 255),
    (255, 190, 60),
    (180, 120, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay tracks_3d.csv particle positions on preview video frames."
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Preview video. If omitted, File Explorer will open.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="tracks_3d.csv file. If omitted, File Explorer will open.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Calibration JSON used to infer front/side panel sizes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "verification",
        help="Parent folder for timestamped verification outputs.",
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
        help="Which preview panel(s) to mark.",
    )
    parser.add_argument(
        "--layout",
        choices=("side-by-side", "stacked"),
        default="side-by-side",
        help="Preview panel layout.",
    )
    parser.add_argument(
        "--marker-radius-px",
        type=int,
        default=9,
        help="Circle marker radius in preview pixels.",
    )
    parser.add_argument(
        "--line-width-px",
        type=int,
        default=2,
        help="Marker line width.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="Draw this many prior same-particle positions as a trail. Use 0 for none.",
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
    csv_path = args.csv or choose_file(
        "Choose tracks_3d.csv",
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )
    tracks = pd.read_csv(csv_path)
    validate_tracks(tracks, csv_path)

    panel = resolve_panel_geometry(args)
    output_dir = make_output_dir(args.output_dir, args.output_name, preview_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_name = sanitize_output_name(preview_path.stem)
    output_paths = {
        "raw": output_dir / f"{preview_name}_verification_raw.mp4",
        "smoothed": output_dir / f"{preview_name}_verification_smoothed.mp4",
    }
    results: dict[str, dict[str, int]] = {}
    for position_source in POSITION_SOURCES:
        frames_written, markers_drawn = write_verification_video(
            preview_path=preview_path,
            output_path=output_paths[position_source],
            tracks=tracks,
            panel=panel,
            args=args,
            position_source=position_source,
        )
        results[position_source] = {
            "frames_written": frames_written,
            "markers_drawn": markers_drawn,
        }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preview": str(preview_path.resolve()),
                "csv": str(csv_path.resolve()),
                "output_videos": {
                    source: str(path.resolve())
                    for source, path in output_paths.items()
                },
                "preview_start_frame": args.preview_start_frame,
                "preview_frame_step": args.preview_frame_step,
                "view": args.view,
                "layout": args.layout,
                "results": results,
                "panel": panel,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print("\nDone.")
    print(f"Preview:        {preview_path.resolve()}")
    print(f"Tracks CSV:     {csv_path.resolve()}")
    for source in POSITION_SOURCES:
        print(
            f"{source.title()} video:  {output_paths[source].resolve()} "
            f"({results[source]['frames_written']} frames, "
            f"{results[source]['markers_drawn']} markers)"
        )
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.preview_start_frame < 0:
        raise SystemExit("--preview-start-frame must be zero or positive.")
    if args.preview_frame_step <= 0:
        raise SystemExit("--preview-frame-step must be positive.")
    if args.marker_radius_px <= 0:
        raise SystemExit("--marker-radius-px must be positive.")
    if args.line_width_px <= 0:
        raise SystemExit("--line-width-px must be positive.")
    if args.trail_length < 0:
        raise SystemExit("--trail-length must be zero or positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")
    for name in ("front_width", "front_height", "side_width", "side_height"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")


def validate_tracks(tracks: pd.DataFrame, csv_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in tracks.columns]
    if missing:
        smooth_missing = [column for column in SMOOTH_PIXEL_COLUMNS if column in missing]
        hint = (
            " Re-run background_subtraction_tracker.py to add tracker-owned "
            "smooth positions and projections."
            if smooth_missing
            else ""
        )
        raise SystemExit(f"{csv_path} is missing required columns: {missing}.{hint}")
    if tracks.empty:
        raise SystemExit(f"{csv_path} is empty.")


def choose_file(title: str, filetypes) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(f"Pass a path because File Explorer is unavailable ({exc}).") from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not path:
        raise SystemExit(f"No file selected for: {title}")
    return Path(path)


def resolve_panel_geometry(args: argparse.Namespace) -> dict[str, int]:
    _, _, _, front_roi, side_roi, _, _ = read_calibration(args.calibration)
    if front_roi is None or side_roi is None:
        raise SystemExit("calibration.json must contain front_roi and side_roi.")

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


def write_verification_video(
    preview_path: Path,
    output_path: Path,
    tracks: pd.DataFrame,
    panel: dict[str, int],
    args: argparse.Namespace,
    position_source: Literal["raw", "smoothed"] = "raw",
) -> tuple[int, int]:
    reader = imageio.get_reader(str(preview_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    tracks_by_frame = {
        int(frame): group.copy()
        for frame, group in tracks.groupby("frame", sort=False)
    }

    frames_written = 0
    markers_drawn = 0
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8) as writer:
        try:
            for preview_index, frame in enumerate(reader):
                if args.max_frames is not None and frames_written >= args.max_frames:
                    break
                track_frame = (
                    args.preview_start_frame
                    + preview_index * args.preview_frame_step
                )
                image = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
                draw = ImageDraw.Draw(image)
                rows = tracks_by_frame.get(track_frame)
                if rows is not None and not rows.empty:
                    markers_drawn += draw_markers(
                        draw=draw,
                        rows=rows,
                        panel=panel,
                        view=args.view,
                        radius=args.marker_radius_px,
                        width=args.line_width_px,
                        position_source=position_source,
                    )
                    if args.trail_length:
                        draw_trails(
                            draw=draw,
                            tracks_by_frame=tracks_by_frame,
                            current_frame=track_frame,
                            panel=panel,
                            view=args.view,
                            radius=max(1, args.marker_radius_px // 3),
                            width=1,
                            trail_length=args.trail_length,
                            position_source=position_source,
                        )
                draw_frame_label(draw, track_frame)
                draw_position_source_label(draw, position_source)
                writer.append_data(np.asarray(image))
                frames_written += 1
        finally:
            reader.close()

    return frames_written, markers_drawn


def draw_markers(
    draw: ImageDraw.ImageDraw,
    rows: pd.DataFrame,
    panel: dict[str, int],
    view: Literal["both", "front", "side"],
    radius: int,
    width: int,
    position_source: Literal["raw", "smoothed"] = "raw",
) -> int:
    front_x_column, front_y_column, side_x_column, side_y_column = pixel_columns(
        position_source
    )
    count = 0
    for _, row in rows.iterrows():
        particle = int(row["particle"])
        color = COLORS[particle % len(COLORS)]
        label = f"P{particle}"
        if view in ("both", "front"):
            x = panel["front_x"] + float(row[front_x_column])
            y = panel["front_y"] + float(row[front_y_column])
            draw_marker(draw, x, y, radius, color, label, width)
            count += 1
        if view in ("both", "side"):
            x = panel["side_x"] + float(row[side_x_column])
            y = panel["side_y"] + float(row[side_y_column])
            draw_marker(draw, x, y, radius, color, label, width)
            count += 1
    return count


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: int,
    color: tuple[int, int, int],
    label: str,
    width: int,
) -> None:
    x0 = x - radius
    x1 = x + radius
    y0 = y - radius
    y1 = y + radius
    draw.ellipse([x0, y0, x1, y1], outline=color, width=width)
    draw.line([x - radius, y, x + radius, y], fill=color, width=width)
    draw.line([x, y - radius, x, y + radius], fill=color, width=width)
    draw.text((x + radius + 3, y - radius - 2), label, fill=color)


def draw_trails(
    draw: ImageDraw.ImageDraw,
    tracks_by_frame: dict[int, pd.DataFrame],
    current_frame: int,
    panel: dict[str, int],
    view: Literal["both", "front", "side"],
    radius: int,
    width: int,
    trail_length: int,
    position_source: Literal["raw", "smoothed"] = "raw",
) -> None:
    front_x_column, front_y_column, side_x_column, side_y_column = pixel_columns(
        position_source
    )
    for frame in range(current_frame - trail_length, current_frame):
        rows = tracks_by_frame.get(frame)
        if rows is None or rows.empty:
            continue
        for _, row in rows.iterrows():
            particle = int(row["particle"])
            color = tuple(max(0, int(channel * 0.55)) for channel in COLORS[particle % len(COLORS)])
            if view in ("both", "front"):
                draw_marker(
                    draw,
                    panel["front_x"] + float(row[front_x_column]),
                    panel["front_y"] + float(row[front_y_column]),
                    radius,
                    color,
                    "",
                    width,
                )
            if view in ("both", "side"):
                draw_marker(
                    draw,
                    panel["side_x"] + float(row[side_x_column]),
                    panel["side_y"] + float(row[side_y_column]),
                    radius,
                    color,
                    "",
                    width,
                )


def draw_frame_label(draw: ImageDraw.ImageDraw, track_frame: int) -> None:
    text = f"tracking frame {track_frame}"
    draw.rectangle([0, 28, 170, 50], fill=(0, 0, 0))
    draw.text((6, 32), text, fill=(255, 255, 0), font=ImageFont.load_default())


def pixel_columns(
    position_source: Literal["raw", "smoothed"],
) -> tuple[str, str, str, str]:
    if position_source == "smoothed":
        return SMOOTH_PIXEL_COLUMNS
    if position_source == "raw":
        return RAW_PIXEL_COLUMNS[2:]
    raise ValueError(f"Unknown position source: {position_source}")


def draw_position_source_label(
    draw: ImageDraw.ImageDraw,
    position_source: Literal["raw", "smoothed"],
) -> None:
    text = f"{position_source.upper()} POSITIONS"
    draw.rectangle([0, 52, 170, 72], fill=(0, 0, 0))
    color = (255, 255, 255) if position_source == "raw" else (80, 255, 255)
    draw.text((6, 56), text, fill=color, font=ImageFont.load_default())


if __name__ == "__main__":
    raise SystemExit(main())
