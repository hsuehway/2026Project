"""Estimate particle diameters and ice masses from preview videos/images.

This tool is separate from the tracker. It reads a background-subtracted preview,
lets you choose the front or side view, thresholds bright particle blobs, converts
pixel diameter to millimeters using the existing chamber calibration, and writes
CSV outputs under outputs_background/mass.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from background_subtraction_tracker import (
    is_image_file,
    is_video_file,
    read_image_frame,
    sanitize_output_name,
)
from track_orthogonal_3d import CameraCalibration, ChamberGeometry, ROI, read_calibration

ICE_DENSITY_MG_PER_MM3 = 0.9167
VIEW_CHOICES = ("front", "side")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate particle diameters/masses from preview images or videos."
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Preview image/video. If omitted, File Explorer will open.",
    )
    parser.add_argument(
        "--view",
        choices=VIEW_CHOICES,
        help="Use the front or side preview panel for diameter estimation.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Calibration JSON used for pixel-to-mm conversion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "mass",
        help="Parent folder for per-trial diameter/mass outputs.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional trial folder name. Defaults to preview name plus timestamp.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="Preview pixel threshold, 0-255, used to segment bright particles.",
    )
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=2,
        help="Ignore connected bright regions smaller than this many pixels.",
    )
    parser.add_argument(
        "--max-area-px",
        type=int,
        default=300,
        help="Ignore connected bright regions larger than this many pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=2,
        help="Keep the largest N detections per frame. Use 0 to keep all detections.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Analyze every Nth video frame.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First video frame to analyze.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of sampled frames to analyze.",
    )
    parser.add_argument(
        "--label-height-px",
        type=int,
        default=28,
        help="Ignore this many top pixels to remove preview label text.",
    )
    parser.add_argument(
        "--label-width-px",
        type=int,
        default=240,
        help="Ignore this many left pixels in the label band.",
    )
    parser.add_argument(
        "--border-px",
        type=int,
        default=3,
        help="Ignore this many pixels around the preview panel border.",
    )
    parser.add_argument(
        "--density-mg-per-mm3",
        type=float,
        default=ICE_DENSITY_MG_PER_MM3,
        help="Particle material density. Ice is about 0.9167 mg/mm^3.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    preview_path = args.preview or choose_preview_file()
    view = args.view or prompt_view()
    (
        geometry,
        front_cal,
        side_cal,
        front_roi,
        side_roi,
        _,
        _,
    ) = read_calibration(args.calibration)
    if front_roi is None or side_roi is None:
        raise SystemExit("calibration.json must contain front_roi and side_roi.")

    calibration = front_cal if view == "front" else side_cal
    camera_roi = front_roi if view == "front" else side_roi
    output_dir = make_trial_output_dir(args.output_dir, args.output_name, preview_path)

    rows, frames_processed = estimate_from_preview(
        preview_path=preview_path,
        view=view,
        geometry=geometry,
        calibration=calibration,
        front_roi=front_roi,
        side_roi=side_roi,
        camera_roi=camera_roi,
        args=args,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detections = pd.DataFrame(rows)
    detections_path = output_dir / "diameters.csv"
    detections.to_csv(detections_path, index=False)

    summary = build_summary(
        detections=detections,
        preview_path=preview_path,
        view=view,
        frames_processed=frames_processed,
        args=args,
    )
    summary_path = output_dir / "diameter_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\nDone.")
    print(f"Preview source: {preview_path.resolve()}")
    print(f"View used:      {view}")
    print(f"Frames read:    {frames_processed}")
    print(f"Detections:     {len(detections)}")
    print(f"Output folder:  {output_dir.resolve()}")
    print(f"CSV:            {detections_path.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.threshold < 0:
        raise SystemExit("--threshold must be non-negative.")
    if args.min_area_px <= 0:
        raise SystemExit("--min-area-px must be positive.")
    if args.max_area_px < 0:
        raise SystemExit("--max-area-px must be zero or positive.")
    if args.max_particles < 0:
        raise SystemExit("--max-particles must be zero or positive.")
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be positive.")
    if args.start_frame < 0:
        raise SystemExit("--start-frame must be zero or positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")
    if args.label_height_px < 0 or args.label_width_px < 0:
        raise SystemExit("--label-height-px and --label-width-px must be non-negative.")
    if args.border_px < 0:
        raise SystemExit("--border-px must be zero or positive.")
    if args.density_mg_per_mm3 <= 0:
        raise SystemExit("--density-mg-per-mm3 must be positive.")


def choose_preview_file() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(
            f"Pass --preview because Tk file dialogs are unavailable ({exc})."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose background-subtracted preview image or video",
        filetypes=[
            ("Preview images/videos", "*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("Video files", "*.mp4 *.mov *.avi *.mkv"),
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise SystemExit("No preview file selected.")
    return Path(path)


def prompt_view() -> Literal["front", "side"]:
    while True:
        value = input("Use which view for diameter approximation? [front/side]: ").strip().lower()
        if value in VIEW_CHOICES:
            return value  # type: ignore[return-value]
        print("Please type front or side.")


def estimate_from_preview(
    preview_path: Path,
    view: Literal["front", "side"],
    geometry: ChamberGeometry,
    calibration: CameraCalibration,
    front_roi: ROI,
    side_roi: ROI,
    camera_roi: ROI,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float | int | str]], int]:
    rows: list[dict[str, float | int | str]] = []
    frames_processed = 0

    if is_image_file(preview_path):
        frame = read_image_frame(preview_path)
        rows.extend(
            estimate_frame(
                frame=frame,
                frame_number=0,
                time_s=0.0,
                view=view,
                geometry=geometry,
                calibration=calibration,
                front_roi=front_roi,
                side_roi=side_roi,
                camera_roi=camera_roi,
                args=args,
            )
        )
        return rows, 1

    if not is_video_file(preview_path):
        raise SystemExit(f"Unsupported preview file type: {preview_path.suffix}")

    import imageio.v2 as imageio

    reader = imageio.get_reader(str(preview_path))
    fps = float(reader.get_meta_data().get("fps") or 0.0)
    frame_number = args.start_frame
    sampled = 0
    try:
        while True:
            if args.max_frames is not None and sampled >= args.max_frames:
                break
            try:
                frame = reader.get_data(frame_number)
            except IndexError:
                break
            time_s = frame_number / fps if fps > 0 else float("nan")
            rows.extend(
                estimate_frame(
                    frame=frame,
                    frame_number=frame_number,
                    time_s=time_s,
                    view=view,
                    geometry=geometry,
                    calibration=calibration,
                    front_roi=front_roi,
                    side_roi=side_roi,
                    camera_roi=camera_roi,
                    args=args,
                )
            )
            frames_processed += 1
            sampled += 1
            frame_number += args.frame_step
    finally:
        reader.close()

    return rows, frames_processed


def estimate_frame(
    frame: np.ndarray,
    frame_number: int,
    time_s: float,
    view: Literal["front", "side"],
    geometry: ChamberGeometry,
    calibration: CameraCalibration,
    front_roi: ROI,
    side_roi: ROI,
    camera_roi: ROI,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    panel = extract_preview_panel(frame, view, front_roi, side_roi)
    gray = panel_to_gray(panel)
    clear_preview_overlays(gray, args)

    mask = gray >= args.threshold
    components = connected_components(mask)
    detections = []
    for component_id, coords in enumerate(components):
        area_px = int(len(coords))
        if area_px < args.min_area_px:
            continue
        if args.max_area_px and area_px > args.max_area_px:
            continue

        ys = coords[:, 0]
        xs = coords[:, 1]
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        equivalent_diameter_px = 2.0 * math.sqrt(area_px / math.pi)
        bbox_diameter_px = float(max(bbox_width, bbox_height))

        z_mm = float(calibration.z_from_image_y(centroid_y, geometry))
        px_per_mm = float(calibration.px_per_mm_at_z(z_mm, geometry))
        mm_per_px = 1.0 / px_per_mm
        diameter_mm = equivalent_diameter_px * mm_per_px
        bbox_diameter_mm = bbox_diameter_px * mm_per_px
        volume_mm3 = math.pi / 6.0 * diameter_mm**3
        mass_mg = args.density_mg_per_mm3 * volume_mm3

        detections.append(
            {
                "frame": frame_number,
                "time_s": time_s,
                "view": view,
                "component_id": component_id,
                "area_px": area_px,
                "centroid_x_px": centroid_x,
                "centroid_y_px": centroid_y,
                "bbox_width_px": bbox_width,
                "bbox_height_px": bbox_height,
                "equivalent_diameter_px": equivalent_diameter_px,
                "bbox_diameter_px": bbox_diameter_px,
                "z_mm": z_mm,
                "px_per_mm": px_per_mm,
                "mm_per_px": mm_per_px,
                "diameter_mm": diameter_mm,
                "bbox_diameter_mm": bbox_diameter_mm,
                "volume_mm3": volume_mm3,
                "mass_mg": mass_mg,
                "mass_ug": mass_mg * 1000.0,
            }
        )

    detections.sort(key=lambda row: (-int(row["area_px"]), float(row["centroid_x_px"])))
    if args.max_particles:
        detections = detections[: args.max_particles]
    for rank, row in enumerate(detections):
        row["detection_rank"] = rank
    return detections


def extract_preview_panel(
    frame: np.ndarray,
    view: Literal["front", "side"],
    front_roi: ROI,
    side_roi: ROI,
) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    front_width = front_roi.width
    front_height = front_roi.height
    side_width = side_roi.width
    side_height = side_roi.height
    if view == "front":
        return arr[:front_height, :front_width, :]
    x0 = front_width
    x1 = x0 + side_width
    return arr[:side_height, x0:x1, :]


def panel_to_gray(panel: np.ndarray) -> np.ndarray:
    if panel.ndim == 2:
        return panel.astype(np.float32)
    rgb = panel[:, :, :3].astype(np.float32)
    return rgb.max(axis=2)


def clear_preview_overlays(gray: np.ndarray, args: argparse.Namespace) -> None:
    border = args.border_px
    if border > 0:
        gray[:border, :] = 0
        gray[-border:, :] = 0
        gray[:, :border] = 0
        gray[:, -border:] = 0
    if args.label_height_px > 0 and args.label_width_px > 0:
        h = min(args.label_height_px, gray.shape[0])
        w = min(args.label_width_px, gray.shape[1])
        gray[:h, :w] = 0


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    try:
        from scipy import ndimage
    except Exception as exc:
        raise SystemExit("scipy is required for connected-component sizing.") from exc

    labels, count = ndimage.label(mask)
    components: list[np.ndarray] = []
    for label_id in range(1, count + 1):
        coords = np.argwhere(labels == label_id)
        if coords.size:
            components.append(coords)
    return components


def make_trial_output_dir(
    parent_dir: Path,
    output_name: str | None,
    preview_path: Path,
) -> Path:
    parent_dir.mkdir(parents=True, exist_ok=True)
    if output_name:
        base_name = sanitize_output_name(output_name)
    else:
        base_name = (
            f"{sanitize_output_name(preview_path.stem)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def build_summary(
    detections: pd.DataFrame,
    preview_path: Path,
    view: str,
    frames_processed: int,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "preview": str(preview_path),
        "view": view,
        "frames_processed": frames_processed,
        "detections": int(len(detections)),
        "threshold": float(args.threshold),
        "density_mg_per_mm3": float(args.density_mg_per_mm3),
    }
    if detections.empty:
        return summary

    for column in ("diameter_mm", "mass_ug", "area_px"):
        values = detections[column].astype(float)
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_median"] = float(values.median())
        summary[f"{column}_std"] = float(values.std(ddof=0))
        summary[f"{column}_min"] = float(values.min())
        summary[f"{column}_max"] = float(values.max())
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
