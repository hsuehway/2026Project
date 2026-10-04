"""Reconstruct 3D particle trajectories from orthogonal camera views.

Coordinate convention
---------------------
- Front camera horizontal coordinate -> x_mm
- Side camera horizontal coordinate  -> y_mm
- Either camera vertical coordinate  -> z_mm
- z_mm = 0 at the bottom plate and z_mm = the configured physical gap at the top plate

The chamber dimensions are encoded below, but pixel-to-mm conversion still
requires a camera calibration because video pixels depend on magnification,
cropping, and perspective. A single screen-recording video with both camera
panels is supported by selecting front and side regions of interest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache")
)


DEFAULT_BOTTOM_DIAMETER_MM = 33.7
DEFAULT_TOP_DIAMETER_MM = 25.4
DEFAULT_GAP_MM = 10.0

CHANNELS = ("red", "green", "blue", "mean")
Z_SOURCES = ("average", "front", "side")


def infer_video_id(video_path: Path) -> str | None:
    """Return a trial ID such as MI19 or SP1 from a video/background filename."""
    match = re.search(r"((?:MI|SP)\d+)", video_path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def calibration_path_for_video_id(
    video_id: str,
    calibrations_dir: Path = Path("calibrations"),
) -> Path:
    return calibrations_dir / f"{video_id}_calibration.json"


def calibration_path_for_video(
    video_path: Path,
    calibrations_dir: Path = Path("calibrations"),
) -> Path | None:
    video_id = infer_video_id(video_path)
    if video_id is None:
        return None
    return calibration_path_for_video_id(video_id, calibrations_dir)


@dataclass(frozen=True)
class ROI:
    """Rectangular crop region in full-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_string(cls, value: str) -> "ROI":
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError("ROI must be x,y,width,height.")
        try:
            x, y, width, height = [int(float(part)) for part in parts]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("ROI values must be numbers.") from exc
        roi = cls(x=x, y=y, width=width, height=height)
        roi.validate_values("ROI")
        return roi

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ROI":
        roi = cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
        roi.validate_values("ROI")
        return roi

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    def validate_values(self, name: str) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(f"{name} x/y must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{name} width/height must be positive.")

    def validate_against_frame(self, frame_shape: tuple[int, ...], name: str) -> None:
        self.validate_values(name)
        frame_height, frame_width = frame_shape[:2]
        if self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ValueError(
                f"{name} ROI {self} exceeds frame size {frame_width}x{frame_height}."
            )

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.height, self.x : self.x + self.width, ...]

    def offset_x(self, x_px: np.ndarray) -> np.ndarray:
        return x_px + self.x

    def offset_y(self, y_px: np.ndarray) -> np.ndarray:
        return y_px + self.y


@dataclass(frozen=True)
class ChamberGeometry:
    bottom_diameter_mm: float = DEFAULT_BOTTOM_DIAMETER_MM
    top_diameter_mm: float = DEFAULT_TOP_DIAMETER_MM
    gap_mm: float = DEFAULT_GAP_MM

    @property
    def bottom_radius_mm(self) -> float:
        return self.bottom_diameter_mm / 2.0

    @property
    def top_radius_mm(self) -> float:
        return self.top_diameter_mm / 2.0

    def radius_at_z(self, z_mm: float | np.ndarray) -> float | np.ndarray:
        z = np.asarray(z_mm, dtype=float)
        fraction = np.clip(z / self.gap_mm, 0.0, 1.0)
        radius = self.bottom_radius_mm + fraction * (
            self.top_radius_mm - self.bottom_radius_mm
        )
        if np.isscalar(z_mm):
            return float(radius)
        return radius


def apply_gap_override(
    geometry: ChamberGeometry,
    gap_mm: float | None,
) -> tuple[ChamberGeometry, bool]:
    """Apply an explicit physical gap without changing the plate diameters."""

    if gap_mm is None or math.isclose(
        geometry.gap_mm,
        gap_mm,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        return geometry, False
    return (
        ChamberGeometry(
            bottom_diameter_mm=geometry.bottom_diameter_mm,
            top_diameter_mm=geometry.top_diameter_mm,
            gap_mm=float(gap_mm),
        ),
        True,
    )


@dataclass(frozen=True)
class CameraCalibration:
    """Four clicked plate-edge points for one side-view camera.

    Points should be clicked in image pixel coordinates:
    bottom_left_px, bottom_right_px, top_left_px, top_right_px.
    """

    bottom_left_px: tuple[float, float]
    bottom_right_px: tuple[float, float]
    top_left_px: tuple[float, float]
    top_right_px: tuple[float, float]

    @staticmethod
    def _point(data: Iterable[float]) -> tuple[float, float]:
        x, y = data
        return (float(x), float(y))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        return cls(
            bottom_left_px=cls._point(data["bottom_left_px"]),
            bottom_right_px=cls._point(data["bottom_right_px"]),
            top_left_px=cls._point(data["top_left_px"]),
            top_right_px=cls._point(data["top_right_px"]),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "bottom_left_px": list(self.bottom_left_px),
            "bottom_right_px": list(self.bottom_right_px),
            "top_left_px": list(self.top_left_px),
            "top_right_px": list(self.top_right_px),
        }

    @property
    def bottom_center_px(self) -> tuple[float, float]:
        return midpoint(self.bottom_left_px, self.bottom_right_px)

    @property
    def top_center_px(self) -> tuple[float, float]:
        return midpoint(self.top_left_px, self.top_right_px)

    @property
    def bottom_width_px(self) -> float:
        return abs(self.bottom_right_px[0] - self.bottom_left_px[0])

    @property
    def top_width_px(self) -> float:
        return abs(self.top_right_px[0] - self.top_left_px[0])

    def validate(self, name: str) -> None:
        if self.bottom_width_px <= 0:
            raise ValueError(f"{name} bottom plate width is zero in calibration.")
        if self.top_width_px <= 0:
            raise ValueError(f"{name} top plate width is zero in calibration.")
        bottom_y = self.bottom_center_px[1]
        top_y = self.top_center_px[1]
        if math.isclose(bottom_y, top_y):
            raise ValueError(f"{name} top and bottom plate rows are identical.")

    def _fraction_from_z(self, z_mm: float | np.ndarray, geometry: ChamberGeometry):
        return np.asarray(z_mm, dtype=float) / geometry.gap_mm

    def z_from_image_y(
        self, image_y_px: float | np.ndarray, geometry: ChamberGeometry
    ) -> float | np.ndarray:
        bottom_y = self.bottom_center_px[1]
        top_y = self.top_center_px[1]
        fraction = (np.asarray(image_y_px, dtype=float) - bottom_y) / (
            top_y - bottom_y
        )
        z_mm = fraction * geometry.gap_mm
        if np.isscalar(image_y_px):
            return float(z_mm)
        return z_mm

    def center_x_at_z(
        self, z_mm: float | np.ndarray, geometry: ChamberGeometry
    ) -> float | np.ndarray:
        bottom_x = self.bottom_center_px[0]
        top_x = self.top_center_px[0]
        fraction = self._fraction_from_z(z_mm, geometry)
        center_x = bottom_x + fraction * (top_x - bottom_x)
        if np.isscalar(z_mm):
            return float(center_x)
        return center_x

    def px_per_mm_at_z(
        self, z_mm: float | np.ndarray, geometry: ChamberGeometry
    ) -> float | np.ndarray:
        bottom_px_per_mm = self.bottom_width_px / geometry.bottom_diameter_mm
        top_px_per_mm = self.top_width_px / geometry.top_diameter_mm
        fraction = self._fraction_from_z(z_mm, geometry)
        px_per_mm = bottom_px_per_mm + fraction * (
            top_px_per_mm - bottom_px_per_mm
        )
        if np.isscalar(z_mm):
            return float(px_per_mm)
        return px_per_mm

    def horizontal_mm_from_image_x(
        self,
        image_x_px: float | np.ndarray,
        z_mm: float | np.ndarray,
        geometry: ChamberGeometry,
    ) -> float | np.ndarray:
        center_x_px = self.center_x_at_z(z_mm, geometry)
        px_per_mm = self.px_per_mm_at_z(z_mm, geometry)
        horizontal_mm = (np.asarray(image_x_px, dtype=float) - center_x_px) / px_per_mm
        if np.isscalar(image_x_px) and np.isscalar(z_mm):
            return float(horizontal_mm)
        return horizontal_mm


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track particles from orthogonal camera views and reconstruct 3D "
            "coordinates in a 33.7 mm / 25.4 mm plate chamber."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Single combined video containing both camera views, such as a "
            "screen recording with front and side panels."
        ),
    )
    parser.add_argument(
        "--front",
        type=Path,
        help="Front-camera video path, for the older two-video workflow.",
    )
    parser.add_argument(
        "--side",
        type=Path,
        help="Side-camera video path, for the older two-video workflow.",
    )
    parser.add_argument(
        "--front-roi",
        type=ROI.from_string,
        help=(
            "Front camera crop in full-frame pixels as x,y,width,height. "
            "For combined videos this excludes the surrounding UI."
        ),
    )
    parser.add_argument(
        "--side-roi",
        type=ROI.from_string,
        help="Side camera crop in full-frame pixels as x,y,width,height.",
    )
    parser.add_argument(
        "--front-focus-roi",
        type=ROI.from_string,
        help=(
            "Optional smaller front search region as x,y,width,height inside "
            "the front camera crop."
        ),
    )
    parser.add_argument(
        "--side-focus-roi",
        type=ROI.from_string,
        help=(
            "Optional smaller side search region as x,y,width,height inside "
            "the side camera crop."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help=(
            "Calibration JSON path. If it does not exist, interactive calibration "
            "will be used unless --no-interactive-calibration is supplied."
        ),
    )
    parser.add_argument(
        "--gap-mm",
        type=float,
        default=None,
        help=(
            "Measured physical plate spacing in millimetres. New calibrations "
            f"default to {DEFAULT_GAP_MM:g} mm. For an existing calibration, an "
            "explicit value overrides its saved gap for this run; combine with "
            "--save-calibration to persist the correction."
        ),
    )
    parser.add_argument(
        "--save-calibration",
        action="store_true",
        help="Write clicked/generated calibration to --calibration.",
    )
    parser.add_argument(
        "--force-calibration",
        action="store_true",
        help="Ignore any existing calibration JSON and click ROI/calibration again.",
    )
    parser.add_argument(
        "--select-focus-roi",
        action="store_true",
        help="Interactively click smaller front/side particle search regions.",
    )
    parser.add_argument(
        "--select-initial-particles",
        action="store_true",
        help=(
            "On the seed frame, click the particles to keep in the front and "
            "side camera crops. Final tracks are filtered to those seeds."
        ),
    )
    parser.add_argument(
        "--seeded-method",
        choices=("template", "detections"),
        default="template",
        help=(
            "How clicked particles are followed. template follows the clicked "
            "image patch directly; detections follows Trackpy detections."
        ),
    )
    parser.add_argument(
        "--particle-count",
        type=int,
        default=2,
        help="Number of initial particles to click when --select-initial-particles is used.",
    )
    parser.add_argument(
        "--seed-frame",
        type=int,
        default=0,
        help="Frame index used for initial particle clicks.",
    )
    parser.add_argument(
        "--seed-max-distance-mm",
        type=float,
        default=3.0,
        help="Maximum 3D distance from a clicked seed to an output track at --seed-frame.",
    )
    parser.add_argument(
        "--seed-max-distance-px",
        type=float,
        default=25.0,
        help=(
            "Maximum 2D pixel distance from a clicked seed to a camera track "
            "at --seed-frame."
        ),
    )
    parser.add_argument(
        "--seed-mass-margin",
        type=float,
        default=0.5,
        help=(
            "Seeded mode keeps detections up to clicked_seed_mass * "
            "(1 + margin). Default 0.5 allows 50 percent wiggle."
        ),
    )
    parser.add_argument(
        "--seed-size-margin",
        type=float,
        default=0.5,
        help=(
            "Seeded mode keeps detections whose Trackpy size is within this "
            "fraction of the clicked seed particle sizes. Default 0.5 is 50 percent."
        ),
    )
    parser.add_argument(
        "--seed-filter-radius-px",
        type=float,
        default=12.0,
        help="Pixel radius for finding the detected feature nearest each clicked seed.",
    )
    parser.add_argument(
        "--seed-mass-radius-px",
        type=float,
        help="Deprecated alias for --seed-filter-radius-px.",
    )
    parser.add_argument(
        "--seed-mass-filter",
        action="store_true",
        help="Also apply automatic max-mass filtering from clicked seed particles.",
    )
    parser.add_argument(
        "--no-seed-size-filter",
        action="store_true",
        help="Disable automatic size filtering from clicked seed particles.",
    )
    parser.add_argument(
        "--no-seed-mass-filter",
        action="store_true",
        help="Disable automatic max-mass filtering from clicked seed particles.",
    )
    parser.add_argument(
        "--no-interactive-calibration",
        action="store_true",
        help="Fail instead of opening a click-based calibration window.",
    )
    parser.add_argument(
        "--channel",
        choices=CHANNELS,
        default="green",
        help="Color channel used before Trackpy feature finding.",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=11,
        help="Approximate particle diameter in pixels. Must be odd.",
    )
    parser.add_argument(
        "--front-diameter",
        type=int,
        help="Override --diameter for the front video.",
    )
    parser.add_argument(
        "--side-diameter",
        type=int,
        help="Override --diameter for the side video.",
    )
    parser.add_argument(
        "--minmass",
        type=float,
        default=20.0,
        help="Trackpy minimum integrated brightness/mass.",
    )
    parser.add_argument(
        "--front-minmass",
        type=float,
        help="Override --minmass for the front video.",
    )
    parser.add_argument(
        "--side-minmass",
        type=float,
        help="Override --minmass for the side video.",
    )
    parser.add_argument(
        "--max-features-per-frame",
        type=int,
        help=(
            "After Trackpy detection, keep only the strongest N features per "
            "frame in each camera crop. Use 2 when there are two particles."
        ),
    )
    parser.add_argument(
        "--min-feature-size",
        type=float,
        help="Drop detections with Trackpy size below this value.",
    )
    parser.add_argument(
        "--max-feature-size",
        type=float,
        help="Drop detections with Trackpy size above this value; useful for large blobs.",
    )
    parser.add_argument(
        "--max-mass",
        type=float,
        help="Drop detections with Trackpy mass above this value; useful for bright blobs.",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        help="Drop detections with Trackpy signal below this value.",
    )
    parser.add_argument(
        "--max-ecc",
        type=float,
        help="Drop detections with Trackpy eccentricity above this value.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional Trackpy threshold parameter.",
    )
    parser.add_argument(
        "--separation",
        type=float,
        help="Optional Trackpy minimum feature separation in pixels.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Use when particles are dark on a bright background.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Analyze only the first N frames while tuning parameters.",
    )
    parser.add_argument(
        "--processes",
        default="1",
        help='Trackpy batch process count. Use "auto" for all cores.',
    )
    parser.add_argument(
        "--z-match-tolerance-mm",
        type=float,
        default=0.5,
        help="Maximum allowed front-vs-side z disagreement for same-frame pairing.",
    )
    parser.add_argument(
        "--z-source",
        choices=Z_SOURCES,
        default="average",
        help="Which camera determines the final z coordinate after pairing.",
    )
    parser.add_argument(
        "--search-range-mm",
        type=float,
        default=1.0,
        help="Maximum 3D particle displacement per frame for Trackpy linking.",
    )
    parser.add_argument(
        "--camera-search-range-px",
        type=float,
        default=6.0,
        help=(
            "Maximum 2D particle displacement per frame in each camera when "
            "using --select-initial-particles."
        ),
    )
    parser.add_argument(
        "--template-radius-px",
        type=int,
        default=7,
        help="Half-width of the clicked-particle image patch for template tracking.",
    )
    parser.add_argument(
        "--template-min-correlation",
        type=float,
        default=0.15,
        help="Minimum normalized correlation accepted by template tracking.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=3,
        help="How many frames Trackpy may remember a missing particle.",
    )
    parser.add_argument(
        "--stub-length",
        type=int,
        default=5,
        help="Discard 3D trajectories shorter than this many frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_tracking_args(args)

    front_video, side_video, front_roi, side_roi = resolve_video_inputs(args)

    (
        geometry,
        front_cal,
        side_cal,
        front_roi,
        side_roi,
        front_focus_roi,
        side_focus_roi,
    ) = load_or_create_calibrations(
        args=args,
        front_video=front_video,
        side_video=side_video,
        front_roi=front_roi,
        side_roi=side_roi,
    )
    front_cal.validate("front")
    side_cal.validate("side")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_points_3d = None
    front_seed_points = None
    side_seed_points = None
    if args.select_initial_particles:
        front_seed_points = interactive_particle_seeds(
            video_path=front_video,
            camera_name="front",
            channel=args.channel,
            roi=front_roi,
            seed_frame=args.seed_frame,
            particle_count=args.particle_count,
        )
        side_seed_points = interactive_particle_seeds(
            video_path=side_video,
            camera_name="side",
            channel=args.channel,
            roi=side_roi,
            seed_frame=args.seed_frame,
            particle_count=args.particle_count,
        )
        seed_points_3d = particle_seed_points_to_3d(
            front_points=front_seed_points,
            side_points=side_seed_points,
            front_cal=front_cal,
            side_cal=side_cal,
            geometry=geometry,
            z_source=args.z_source,
        )

    front_features = detect_features(
        video_path=front_video,
        roi=front_roi,
        focus_roi=front_focus_roi,
        diameter=args.front_diameter or args.diameter,
        minmass=args.front_minmass if args.front_minmass is not None else args.minmass,
        args=args,
        camera_name="front",
    )
    side_features = detect_features(
        video_path=side_video,
        roi=side_roi,
        focus_roi=side_focus_roi,
        diameter=args.side_diameter or args.diameter,
        minmass=args.side_minmass if args.side_minmass is not None else args.minmass,
        args=args,
        camera_name="side",
    )

    front_mm = features_to_camera_coordinates(
        front_features,
        front_cal,
        geometry,
        axis="x",
        camera_name="front",
        source_roi=front_roi,
        detection_roi=front_focus_roi,
    )
    side_mm = features_to_camera_coordinates(
        side_features,
        side_cal,
        geometry,
        axis="y",
        camera_name="side",
        source_roi=side_roi,
        detection_roi=side_focus_roi,
    )

    if args.select_initial_particles and front_seed_points and side_seed_points:
        front_mm = apply_seed_detection_filters(
            detections=front_mm,
            camera_name="front",
            seed_points=front_seed_points,
            seed_frame=args.seed_frame,
            args=args,
        )
        side_mm = apply_seed_detection_filters(
            detections=side_mm,
            camera_name="side",
            seed_points=side_seed_points,
            seed_frame=args.seed_frame,
            args=args,
        )
        linked = link_seeded_camera_tracks_to_3d(
            front_mm=front_mm,
            side_mm=side_mm,
            front_seed_points=front_seed_points,
            side_seed_points=side_seed_points,
            geometry=geometry,
            search_range_px=args.camera_search_range_px,
            memory=args.memory,
            stub_length=args.stub_length,
            seed_frame=args.seed_frame,
            seed_max_distance_px=args.seed_max_distance_px,
            z_source=args.z_source,
        )
        matched = linked.copy()
    else:
        matched = match_orthogonal_detections(
            front_mm=front_mm,
            side_mm=side_mm,
            geometry=geometry,
            tolerance_mm=args.z_match_tolerance_mm,
            z_source=args.z_source,
        )
        linked = link_3d_tracks(
            matched,
            search_range_mm=args.search_range_mm,
            memory=args.memory,
            stub_length=args.stub_length,
        )

    write_outputs(
        output_dir=args.output_dir,
        geometry=geometry,
        front_mm=front_mm,
        side_mm=side_mm,
        matched=matched,
        linked=linked,
        seed_points_3d=seed_points_3d,
        make_plots=not args.no_plots,
    )

    print_summary(args.output_dir, front_mm, side_mm, matched, linked)
    return 0


def validate_tracking_args(args: argparse.Namespace) -> None:
    if args.video and (args.front or args.side):
        raise SystemExit("Use either --video, or --front and --side, not both.")
    if (args.front is None) != (args.side is None):
        raise SystemExit("--front and --side must be supplied together.")
    for name in ("diameter", "front_diameter", "side_diameter"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or value % 2 != 1):
            raise SystemExit(f"--{name.replace('_', '-')} must be a positive odd integer.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")
    if args.gap_mm is not None and (
        not math.isfinite(args.gap_mm) or args.gap_mm <= 0.0
    ):
        raise SystemExit("--gap-mm must be finite and positive.")
    if args.max_features_per_frame is not None and args.max_features_per_frame <= 0:
        raise SystemExit("--max-features-per-frame must be positive.")
    for name in (
        "min_feature_size",
        "max_feature_size",
        "max_mass",
        "min_signal",
        "max_ecc",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative.")
    if args.particle_count <= 0:
        raise SystemExit("--particle-count must be positive.")
    if args.seed_frame < 0:
        raise SystemExit("--seed-frame must be zero or positive.")
    if args.seed_max_distance_mm <= 0:
        raise SystemExit("--seed-max-distance-mm must be positive.")
    if args.seed_max_distance_px <= 0:
        raise SystemExit("--seed-max-distance-px must be positive.")
    if args.seed_mass_margin < 0:
        raise SystemExit("--seed-mass-margin must be non-negative.")
    if args.seed_size_margin < 0:
        raise SystemExit("--seed-size-margin must be non-negative.")
    if args.seed_filter_radius_px <= 0:
        raise SystemExit("--seed-filter-radius-px must be positive.")
    if args.seed_mass_radius_px is not None:
        if args.seed_mass_radius_px <= 0:
            raise SystemExit("--seed-mass-radius-px must be positive.")
        args.seed_filter_radius_px = args.seed_mass_radius_px
    if args.camera_search_range_px <= 0:
        raise SystemExit("--camera-search-range-px must be positive.")
    if args.z_match_tolerance_mm <= 0:
        raise SystemExit("--z-match-tolerance-mm must be positive.")
    if args.search_range_mm <= 0:
        raise SystemExit("--search-range-mm must be positive.")


def resolve_video_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, ROI | None, ROI | None]:
    if args.video:
        return args.video, args.video, args.front_roi, args.side_roi

    if args.front and args.side:
        return args.front, args.side, args.front_roi, args.side_roi

    combined_video = choose_video("Choose combined video containing both camera panels")
    return combined_video, combined_video, args.front_roi, args.side_roi


def choose_video(title: str) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(
            f"{title}: pass a path with --video or --front/--side because Tk file dialogs "
            f"are unavailable ({exc})."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Video files", "*.avi *.mp4 *.mov *.mkv *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise SystemExit(f"No file selected for: {title}")
    return Path(path)


def load_or_create_calibrations(
    args: argparse.Namespace,
    front_video: Path,
    side_video: Path,
    front_roi: ROI | None,
    side_roi: ROI | None,
    front_focus_roi: ROI | None = None,
    side_focus_roi: ROI | None = None,
) -> tuple[
    ChamberGeometry,
    CameraCalibration,
    CameraCalibration,
    ROI | None,
    ROI | None,
    ROI | None,
    ROI | None,
]:
    front_focus_roi = front_focus_roi or args.front_focus_roi
    side_focus_roi = side_focus_roi or args.side_focus_roi
    existing = (
        read_calibration(args.calibration)
        if args.calibration.exists() and not args.force_calibration
        else None
    )
    if existing:
        (
            geometry,
            front_cal,
            side_cal,
            saved_front_roi,
            saved_side_roi,
            saved_front_focus_roi,
            saved_side_focus_roi,
        ) = existing
        saved_gap_mm = geometry.gap_mm
        geometry, gap_overridden = apply_gap_override(geometry, args.gap_mm)
        if gap_overridden:
            print(
                f"Using explicit plate gap {geometry.gap_mm:g} mm instead of "
                f"{saved_gap_mm:g} mm from {args.calibration}."
            )
        elif args.gap_mm is None and not math.isclose(
            geometry.gap_mm,
            DEFAULT_GAP_MM,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            print(
                f"WARNING: {args.calibration} specifies a {geometry.gap_mm:g} mm "
                f"plate gap, while the current default is {DEFAULT_GAP_MM:g} mm. "
                "The saved calibration value will be used. Pass --gap-mm to override it."
            )
        front_roi = front_roi or saved_front_roi
        side_roi = side_roi or saved_side_roi
        front_focus_roi = front_focus_roi or saved_front_focus_roi
        side_focus_roi = side_focus_roi or saved_side_focus_roi
        if front_video == side_video and (front_roi is None or side_roi is None):
            if args.no_interactive_calibration:
                raise SystemExit(
                    "Combined-video calibration needs front_roi and side_roi. "
                    "Add them to calibration.json, pass --front-roi/--side-roi, "
                    "or rerun without --no-interactive-calibration."
                )
            front_roi, side_roi = choose_missing_rois(
                front_video, args.channel, front_roi, side_roi
            )
            if args.save_calibration:
                write_calibration(
                    args.calibration,
                    geometry,
                    front_cal,
                    side_cal,
                    front_roi=front_roi,
                    side_roi=side_roi,
                    front_focus_roi=front_focus_roi,
                    side_focus_roi=side_focus_roi,
                )
                print(f"Saved ROI updates to {args.calibration}")
        if args.select_focus_roi:
            front_focus_roi, side_focus_roi = choose_focus_rois(
                front_video=front_video,
                side_video=side_video,
                channel=args.channel,
                front_roi=front_roi,
                side_roi=side_roi,
                front_focus_roi=front_focus_roi,
                side_focus_roi=side_focus_roi,
                force=True,
            )
            if args.save_calibration:
                write_calibration(
                    args.calibration,
                    geometry,
                    front_cal,
                    side_cal,
                    front_roi=front_roi,
                    side_roi=side_roi,
                    front_focus_roi=front_focus_roi,
                    side_focus_roi=side_focus_roi,
                )
                print(f"Saved focus ROI updates to {args.calibration}")
        if gap_overridden and args.save_calibration:
            write_calibration(
                args.calibration,
                geometry,
                front_cal,
                side_cal,
                front_roi=front_roi,
                side_roi=side_roi,
                front_focus_roi=front_focus_roi,
                side_focus_roi=side_focus_roi,
            )
            print(f"Saved {geometry.gap_mm:g} mm plate gap to {args.calibration}")
        return (
            geometry,
            front_cal,
            side_cal,
            front_roi,
            side_roi,
            front_focus_roi,
            side_focus_roi,
        )

    if args.no_interactive_calibration:
        raise SystemExit(
            f"Calibration file not found: {args.calibration}. "
            "Create one from calibration.example.json or rerun without "
            "--no-interactive-calibration."
        )

    geometry = ChamberGeometry(
        gap_mm=args.gap_mm if args.gap_mm is not None else DEFAULT_GAP_MM
    )
    if front_video == side_video and (front_roi is None or side_roi is None):
        front_roi, side_roi = choose_missing_rois(
            front_video, args.channel, front_roi, side_roi
        )
    if args.select_focus_roi:
        front_focus_roi, side_focus_roi = choose_focus_rois(
            front_video=front_video,
            side_video=side_video,
            channel=args.channel,
            front_roi=front_roi,
            side_roi=side_roi,
            front_focus_roi=front_focus_roi,
            side_focus_roi=side_focus_roi,
            force=False,
        )

    front_cal = interactive_calibration(
        video_path=front_video,
        camera_name="front",
        channel=args.channel,
        roi=front_roi,
    )
    side_cal = interactive_calibration(
        video_path=side_video,
        camera_name="side",
        channel=args.channel,
        roi=side_roi,
    )

    if args.save_calibration:
        write_calibration(
            args.calibration,
            geometry,
            front_cal,
            side_cal,
            front_roi=front_roi,
            side_roi=side_roi,
            front_focus_roi=front_focus_roi,
            side_focus_roi=side_focus_roi,
        )
        print(f"Saved calibration to {args.calibration}")

    return (
        geometry,
        front_cal,
        side_cal,
        front_roi,
        side_roi,
        front_focus_roi,
        side_focus_roi,
    )


def read_calibration(
    path: Path,
) -> tuple[
    ChamberGeometry,
    CameraCalibration,
    CameraCalibration,
    ROI | None,
    ROI | None,
    ROI | None,
    ROI | None,
]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    geometry = ChamberGeometry(**data.get("chamber", {}))
    front_cal = CameraCalibration.from_dict(data["front"])
    side_cal = CameraCalibration.from_dict(data["side"])
    front_roi = ROI.from_dict(data["front_roi"]) if "front_roi" in data else None
    side_roi = ROI.from_dict(data["side_roi"]) if "side_roi" in data else None
    front_focus_roi = (
        ROI.from_dict(data["front_focus_roi"]) if "front_focus_roi" in data else None
    )
    side_focus_roi = (
        ROI.from_dict(data["side_focus_roi"]) if "side_focus_roi" in data else None
    )
    return (
        geometry,
        front_cal,
        side_cal,
        front_roi,
        side_roi,
        front_focus_roi,
        side_focus_roi,
    )


def write_calibration(
    path: Path,
    geometry: ChamberGeometry,
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    front_roi: ROI | None = None,
    side_roi: ROI | None = None,
    front_focus_roi: ROI | None = None,
    side_focus_roi: ROI | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "chamber": asdict(geometry),
        "front": front_cal.to_dict(),
        "side": side_cal.to_dict(),
    }
    if front_roi is not None:
        data["front_roi"] = front_roi.to_dict()
    if side_roi is not None:
        data["side_roi"] = side_roi.to_dict()
    if front_focus_roi is not None:
        data["front_focus_roi"] = front_focus_roi.to_dict()
    if side_focus_roi is not None:
        data["side_focus_roi"] = side_focus_roi.to_dict()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def choose_missing_rois(
    video_path: Path,
    channel: Literal["red", "green", "blue", "mean"],
    front_roi: ROI | None,
    side_roi: ROI | None,
) -> tuple[ROI, ROI]:
    if front_roi is None:
        front_roi = interactive_roi(video_path, "front", channel)
    if side_roi is None:
        side_roi = interactive_roi(video_path, "side", channel)
    return front_roi, side_roi


def interactive_roi(
    video_path: Path,
    camera_name: str,
    channel: Literal["red", "green", "blue", "mean"],
) -> ROI:
    import matplotlib.pyplot as plt

    frames = load_frames(video_path, channel)
    first_frame = np.asarray(frames[0])
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(first_frame, cmap="gray")
    ax.set_title(
        f"{camera_name}: click top-left and bottom-right corners of the camera panel"
    )
    ax.set_xlabel("Full-frame x [px]")
    ax.set_ylabel("Full-frame y [px]")
    points = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(points) != 2:
        raise SystemExit(f"Expected 2 clicked ROI points for {camera_name}; got {len(points)}.")
    (x0, y0), (x1, y1) = points
    x_min, x_max = sorted((int(round(x0)), int(round(x1))))
    y_min, y_max = sorted((int(round(y0)), int(round(y1))))
    roi = ROI(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)
    roi.validate_against_frame(first_frame.shape, f"{camera_name}")
    return roi


def choose_focus_rois(
    front_video: Path,
    side_video: Path,
    channel: Literal["red", "green", "blue", "mean"],
    front_roi: ROI | None,
    side_roi: ROI | None,
    front_focus_roi: ROI | None,
    side_focus_roi: ROI | None,
    force: bool,
) -> tuple[ROI | None, ROI | None]:
    if force or front_focus_roi is None:
        front_focus_roi = interactive_focus_roi(front_video, "front", channel, front_roi)
    if force or side_focus_roi is None:
        side_focus_roi = interactive_focus_roi(side_video, "side", channel, side_roi)
    return front_focus_roi, side_focus_roi


def interactive_focus_roi(
    video_path: Path,
    camera_name: str,
    channel: Literal["red", "green", "blue", "mean"],
    camera_roi: ROI | None,
) -> ROI:
    import matplotlib.pyplot as plt

    frames = load_frames(video_path, channel, roi=camera_roi)
    first_frame = np.asarray(frames[0])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(first_frame, cmap="gray")
    ax.set_title(
        f"{camera_name}: click top-left and bottom-right of the particle search region"
    )
    ax.set_xlabel("Camera-crop x [px]")
    ax.set_ylabel("Camera-crop y [px]")
    points = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(points) != 2:
        raise SystemExit(
            f"Expected 2 clicked focus ROI points for {camera_name}; got {len(points)}."
        )
    (x0, y0), (x1, y1) = points
    x_min, x_max = sorted((int(round(x0)), int(round(x1))))
    y_min, y_max = sorted((int(round(y0)), int(round(y1))))
    roi = ROI(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)
    roi.validate_against_frame(first_frame.shape, f"{camera_name} focus")
    return roi


def interactive_calibration(
    video_path: Path,
    camera_name: str,
    channel: Literal["red", "green", "blue", "mean"],
    roi: ROI | None = None,
) -> CameraCalibration:
    import matplotlib.pyplot as plt

    frames = load_frames(video_path, channel, roi=roi)
    first_frame = np.asarray(frames[0])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(first_frame, cmap="gray")
    ax.set_title(
        f"{camera_name}: click bottom-left, bottom-right, top-left, top-right plate edges"
    )
    ax.set_xlabel("Image x [px]")
    ax.set_ylabel("Image y [px]")
    points = plt.ginput(4, timeout=0)
    plt.close(fig)
    if len(points) != 4:
        raise SystemExit(f"Expected 4 clicked points for {camera_name}; got {len(points)}.")
    return CameraCalibration(
        bottom_left_px=CameraCalibration._point(points[0]),
        bottom_right_px=CameraCalibration._point(points[1]),
        top_left_px=CameraCalibration._point(points[2]),
        top_right_px=CameraCalibration._point(points[3]),
    )


def interactive_particle_seeds(
    video_path: Path,
    camera_name: str,
    channel: Literal["red", "green", "blue", "mean"],
    roi: ROI | None,
    seed_frame: int,
    particle_count: int,
) -> list[tuple[float, float]]:
    import matplotlib.pyplot as plt

    frames = load_frames(video_path, channel, roi=roi)
    if seed_frame >= len(frames):
        raise SystemExit(
            f"--seed-frame {seed_frame} is outside {camera_name} video length {len(frames)}."
        )
    frame = np.asarray(frames[seed_frame])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(frame, cmap="gray")
    ax.set_title(
        f"{camera_name}: click {particle_count} particle(s), in the same order as the other view"
    )
    ax.set_xlabel("Camera-crop x [px]")
    ax.set_ylabel("Camera-crop y [px]")
    points = plt.ginput(particle_count, timeout=0)
    plt.close(fig)
    if len(points) != particle_count:
        raise SystemExit(
            f"Expected {particle_count} clicked particles for {camera_name}; got {len(points)}."
        )
    return [CameraCalibration._point(point) for point in points]


def particle_seed_points_to_3d(
    front_points: list[tuple[float, float]],
    side_points: list[tuple[float, float]],
    front_cal: CameraCalibration,
    side_cal: CameraCalibration,
    geometry: ChamberGeometry,
    z_source: Literal["average", "front", "side"],
) -> pd.DataFrame:
    if len(front_points) != len(side_points):
        raise ValueError("Front and side seed point counts must match.")

    rows: list[dict[str, float | int]] = []
    for seed_id, (front_point, side_point) in enumerate(zip(front_points, side_points)):
        front_z = float(front_cal.z_from_image_y(front_point[1], geometry))
        side_z = float(side_cal.z_from_image_y(side_point[1], geometry))
        if z_source == "front":
            z_mm = front_z
        elif z_source == "side":
            z_mm = side_z
        else:
            z_mm = (front_z + side_z) / 2.0
        rows.append(
            {
                "seed_id": seed_id,
                "x_mm": float(
                    front_cal.horizontal_mm_from_image_x(front_point[0], z_mm, geometry)
                ),
                "y_mm": float(
                    side_cal.horizontal_mm_from_image_x(side_point[0], z_mm, geometry)
                ),
                "z_mm": z_mm,
                "front_seed_x_px": float(front_point[0]),
                "front_seed_y_px": float(front_point[1]),
                "side_seed_x_px": float(side_point[0]),
                "side_seed_y_px": float(side_point[1]),
                "front_z_mm": front_z,
                "side_z_mm": side_z,
                "z_error_mm": abs(front_z - side_z),
            }
        )
    seeds = pd.DataFrame(rows)
    print("Initial 3D seed points:")
    print(seeds[["seed_id", "x_mm", "y_mm", "z_mm", "z_error_mm"]].round(3).to_string(index=False))
    return seeds


def load_frames(
    video_path: Path,
    channel: Literal["red", "green", "blue", "mean"],
    roi: ROI | None = None,
    focus_roi: ROI | None = None,
) -> Any:
    try:
        import pims
    except ImportError as exc:
        raise SystemExit(
            "PIMS is required to read videos. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    @pims.pipeline
    def gray(image):
        arr = np.asarray(image)
        if roi is not None:
            arr = roi.crop(arr)
        if focus_roi is not None:
            arr = focus_roi.crop(arr)
        if arr.ndim == 2:
            return arr
        if arr.ndim != 3:
            raise ValueError(f"Expected 2D or color frame, got shape {arr.shape}.")
        if channel == "mean":
            return arr[:, :, :3].mean(axis=2).astype(arr.dtype, copy=False)
        channel_index = {"red": 0, "green": 1, "blue": 2}[channel]
        if arr.shape[2] <= channel_index:
            raise ValueError(
                f"Frame has {arr.shape[2]} channels; cannot use {channel} channel."
            )
        return arr[:, :, channel_index]

    try:
        raw_frames = pims.open(str(video_path))
        if roi is not None:
            first = np.asarray(raw_frames[0])
            roi.validate_against_frame(first.shape, str(video_path))
            if focus_roi is not None:
                focus_roi.validate_against_frame(
                    roi.crop(first).shape, f"{video_path} focus"
                )
        elif focus_roi is not None:
            focus_roi.validate_against_frame(
                np.asarray(raw_frames[0]).shape, f"{video_path} focus"
            )
        return gray(raw_frames)
    except Exception as exc:
        raise SystemExit(
            f"Could not open {video_path}. If this is a video file, make sure "
            "imageio-ffmpeg or the needed PIMS backend is installed."
        ) from exc


def detect_features(
    video_path: Path,
    roi: ROI | None,
    focus_roi: ROI | None,
    diameter: int,
    minmass: float,
    args: argparse.Namespace,
    camera_name: str,
) -> pd.DataFrame:
    try:
        import trackpy as tp
    except ImportError as exc:
        raise SystemExit(
            "Trackpy is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    frames = load_frames(video_path, args.channel, roi=roi, focus_roi=focus_roi)
    selected_frames = frames[: args.max_frames] if args.max_frames else frames

    locate_kwargs: dict[str, Any] = {
        "minmass": minmass,
        "invert": args.invert,
    }
    if args.threshold is not None:
        locate_kwargs["threshold"] = args.threshold
    if args.separation is not None:
        locate_kwargs["separation"] = args.separation

    processes = parse_processes(args.processes)
    print(
        f"Detecting {camera_name} features: video={video_path}, "
        f"roi={roi}, focus_roi={focus_roi}, diameter={diameter}, "
        f"minmass={minmass}, processes={processes}"
    )
    features = tp.batch(
        selected_frames,
        diameter=diameter,
        processes=processes,
        **locate_kwargs,
    )
    if features.empty:
        raise SystemExit(
            f"No features found in {camera_name} video. Try lowering --minmass, "
            "changing --diameter, toggling --invert, or checking the selected channel."
        )
    features = filter_features_by_detection_metrics(features, args, camera_name)
    features = limit_features_per_frame(
        features,
        max_features=args.max_features_per_frame,
        camera_name=camera_name,
    )
    if features.empty:
        raise SystemExit(
            f"All {camera_name} detections were removed by feature filtering. "
            "Increase --max-features-per-frame or lower --minmass/--threshold."
        )
    return features.reset_index(drop=True)


def filter_features_by_detection_metrics(
    features: pd.DataFrame,
    args: argparse.Namespace,
    camera_name: str,
) -> pd.DataFrame:
    filters = [
        ("size", ">=", args.min_feature_size, "--min-feature-size"),
        ("size", "<=", args.max_feature_size, "--max-feature-size"),
        ("mass", "<=", args.max_mass, "--max-mass"),
        ("signal", ">=", args.min_signal, "--min-signal"),
        ("ecc", "<=", args.max_ecc, "--max-ecc"),
    ]
    filtered = features
    for column, operator, value, option_name in filters:
        if value is None:
            continue
        if column not in filtered:
            print(
                f"Warning: {option_name} ignored for {camera_name}; "
                f"Trackpy did not return a {column!r} column."
            )
            continue
        before = len(filtered)
        if operator == ">=":
            filtered = filtered.loc[filtered[column] >= value]
        else:
            filtered = filtered.loc[filtered[column] <= value]
        print(
            f"Filtered {camera_name} with {option_name} {value}: "
            f"{before} -> {len(filtered)} detections"
        )

    if filtered.empty:
        raise SystemExit(
            f"All {camera_name} detections were removed by detection filters. "
            "Relax --max-feature-size/--max-mass/--max-ecc/--min-signal."
        )
    return filtered


def limit_features_per_frame(
    features: pd.DataFrame,
    max_features: int | None,
    camera_name: str,
) -> pd.DataFrame:
    if max_features is None:
        return features

    score_column = first_existing_column(features, ["mass", "raw_mass", "signal", "size"])
    if score_column is None:
        print(
            f"Warning: cannot cap {camera_name} detections because no score column exists."
        )
        return features

    before_count = len(features)
    before_max_per_frame = int(features.groupby("frame").size().max())
    limited = (
        features.sort_values(["frame", score_column], ascending=[True, False])
        .groupby("frame", group_keys=False)
        .head(max_features)
        .sort_values(["frame", score_column], ascending=[True, False])
    )
    after_max_per_frame = int(limited.groupby("frame").size().max())
    print(
        f"Capped {camera_name} detections using {score_column}: "
        f"{before_count} -> {len(limited)} total, "
        f"max/frame {before_max_per_frame} -> {after_max_per_frame}"
    )
    return limited


def first_existing_column(
    frame: pd.DataFrame, candidates: Iterable[str]
) -> str | None:
    for column in candidates:
        if column in frame:
            return column
    return None


def parse_processes(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        processes = int(value)
    except ValueError as exc:
        raise SystemExit('--processes must be an integer or "auto".') from exc
    if processes < 1:
        raise SystemExit("--processes must be at least 1.")
    return processes


def features_to_camera_coordinates(
    features: pd.DataFrame,
    calibration: CameraCalibration,
    geometry: ChamberGeometry,
    axis: Literal["x", "y"],
    camera_name: str,
    source_roi: ROI | None = None,
    detection_roi: ROI | None = None,
) -> pd.DataFrame:
    detected_x = features["x"].to_numpy(dtype=float)
    detected_y = features["y"].to_numpy(dtype=float)
    if detection_roi is None:
        image_x = detected_x
        image_y = detected_y
    else:
        image_x = detection_roi.offset_x(detected_x)
        image_y = detection_roi.offset_y(detected_y)
    z_mm = calibration.z_from_image_y(image_y, geometry)
    horizontal_mm = calibration.horizontal_mm_from_image_x(image_x, z_mm, geometry)

    output = pd.DataFrame(
        {
            "frame": features["frame"].astype(int).to_numpy(),
            f"{camera_name}_detection_id": np.arange(len(features), dtype=int),
            f"{camera_name}_detected_x_px": detected_x,
            f"{camera_name}_detected_y_px": detected_y,
            f"{camera_name}_image_x_px": image_x,
            f"{camera_name}_image_y_px": image_y,
            f"{camera_name}_z_mm": z_mm,
            f"{axis}_mm": horizontal_mm,
        }
    )

    for column in ("mass", "size", "ecc", "signal", "raw_mass", "ep"):
        if column in features:
            output[f"{camera_name}_{column}"] = features[column].to_numpy()

    if source_roi is not None:
        output[f"{camera_name}_full_image_x_px"] = source_roi.offset_x(image_x)
        output[f"{camera_name}_full_image_y_px"] = source_roi.offset_y(image_y)

    return output


def match_orthogonal_detections(
    front_mm: pd.DataFrame,
    side_mm: pd.DataFrame,
    geometry: ChamberGeometry,
    tolerance_mm: float,
    z_source: Literal["average", "front", "side"],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = sorted(set(front_mm["frame"]).intersection(side_mm["frame"]))

    for frame in frames:
        front_frame = front_mm.loc[front_mm["frame"] == frame].reset_index(drop=True)
        side_frame = side_mm.loc[side_mm["frame"] == frame].reset_index(drop=True)
        if front_frame.empty or side_frame.empty:
            continue

        cost = np.abs(
            front_frame["front_z_mm"].to_numpy()[:, np.newaxis]
            - side_frame["side_z_mm"].to_numpy()[np.newaxis, :]
        )
        for front_i, side_i in assignment_pairs(cost):
            z_error = float(cost[front_i, side_i])
            if z_error > tolerance_mm:
                continue

            front_row = front_frame.iloc[front_i]
            side_row = side_frame.iloc[side_i]
            front_z = float(front_row["front_z_mm"])
            side_z = float(side_row["side_z_mm"])
            if z_source == "front":
                z_mm = front_z
            elif z_source == "side":
                z_mm = side_z
            else:
                z_mm = (front_z + side_z) / 2.0

            x_mm = float(front_row["x_mm"])
            y_mm = float(side_row["y_mm"])
            radius_mm = float(geometry.radius_at_z(z_mm))
            radial_mm = math.hypot(x_mm, y_mm)

            rows.append(
                {
                    "frame": int(frame),
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "z_mm": z_mm,
                    "front_z_mm": front_z,
                    "side_z_mm": side_z,
                    "z_error_mm": z_error,
                    "radial_mm": radial_mm,
                    "chamber_radius_mm": radius_mm,
                    "inside_chamber": radial_mm <= radius_mm,
                    "front_detection_id": int(front_row["front_detection_id"]),
                    "side_detection_id": int(side_row["side_detection_id"]),
                    "front_image_x_px": float(front_row["front_image_x_px"]),
                    "front_image_y_px": float(front_row["front_image_y_px"]),
                    "side_image_x_px": float(side_row["side_image_x_px"]),
                    "side_image_y_px": float(side_row["side_image_y_px"]),
                }
            )

    matched = pd.DataFrame(rows)
    if matched.empty:
        raise SystemExit(
            "No front/side detections matched by z. Try increasing "
            "--z-match-tolerance-mm or checking calibration point order."
        )
    return matched.sort_values(["frame", "z_mm", "x_mm", "y_mm"]).reset_index(drop=True)


def assignment_pairs(cost: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    except Exception:
        candidates = [
            (float(cost[row, col]), row, col)
            for row in range(cost.shape[0])
            for col in range(cost.shape[1])
        ]
        candidates.sort()
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for _, row, col in candidates:
            if row in used_rows or col in used_cols:
                continue
            pairs.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
        return pairs


def reset_ambiguous_index_levels(
    frame: pd.DataFrame, sort_columns: Iterable[str]
) -> pd.DataFrame:
    sort_column_set = set(sort_columns)
    has_ambiguous_index = any(
        name in sort_column_set and name in frame.columns for name in frame.index.names
    )
    if has_ambiguous_index:
        return frame.reset_index(drop=True)
    return frame


def apply_seed_detection_filters(
    detections: pd.DataFrame,
    camera_name: Literal["front", "side"],
    seed_points: list[tuple[float, float]],
    seed_frame: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    filtered = detections
    if not args.no_seed_size_filter:
        filtered = apply_seed_metric_filter(
            detections=filtered,
            camera_name=camera_name,
            seed_points=seed_points,
            seed_frame=seed_frame,
            metric="size",
            radius_px=args.seed_filter_radius_px,
            margin=args.seed_size_margin,
            mode="range",
        )
    if args.seed_mass_filter and not args.no_seed_mass_filter:
        filtered = apply_seed_metric_filter(
            detections=filtered,
            camera_name=camera_name,
            seed_points=seed_points,
            seed_frame=seed_frame,
            metric="mass",
            radius_px=args.seed_filter_radius_px,
            margin=args.seed_mass_margin,
            mode="upper",
        )
    return filtered


def apply_seed_metric_filter(
    detections: pd.DataFrame,
    camera_name: Literal["front", "side"],
    seed_points: list[tuple[float, float]],
    seed_frame: int,
    metric: Literal["size", "mass"],
    radius_px: float,
    margin: float,
    mode: Literal["range", "upper"],
) -> pd.DataFrame:
    metric_col = f"{camera_name}_{metric}"
    if metric_col not in detections:
        print(
            f"Warning: cannot apply seed {metric} filter for {camera_name}; "
            f"{metric_col!r} was not found."
        )
        return detections

    seed_values = nearest_seed_metric_values(
        detections=detections,
        camera_name=camera_name,
        seed_points=seed_points,
        seed_frame=seed_frame,
        metric_col=metric_col,
        radius_px=radius_px,
    )
    if mode == "range":
        lower = max(0.0, min(seed_values) * (1.0 - margin))
        upper = max(seed_values) * (1.0 + margin)
        mask = detections[metric_col].between(lower, upper, inclusive="both")
        description = f"{lower:.3g} <= {metric_col} <= {upper:.3g}"
    else:
        lower = None
        upper = max(seed_values) * (1.0 + margin)
        mask = detections[metric_col] <= upper
        description = f"{metric_col} <= {upper:.3g}"

    before = len(detections)
    filtered = detections.loc[mask].copy()
    print(
        f"Seed {camera_name} {metric} filter from clicked particles "
        f"{[round(value, 3) for value in seed_values]}: {description}; "
        f"{before} -> {len(filtered)} detections"
    )
    if filtered.empty:
        raise SystemExit(
            f"The seed {metric} filter removed all {camera_name} detections. "
            f"Increase --seed-{metric}-margin or use --no-seed-{metric}-filter."
        )
    return filtered


def nearest_seed_metric_values(
    detections: pd.DataFrame,
    camera_name: Literal["front", "side"],
    seed_points: list[tuple[float, float]],
    seed_frame: int,
    metric_col: str,
    radius_px: float,
) -> list[float]:
    x_col = f"{camera_name}_image_x_px"
    y_col = f"{camera_name}_image_y_px"
    frame_detections = detections.loc[detections["frame"] == seed_frame].copy()
    if frame_detections.empty:
        available_frame = int(detections["frame"].min())
        print(
            f"No {camera_name} detections exist at seed frame {seed_frame}; "
            f"using first detection frame {available_frame} for seed {metric_col} filter."
        )
        frame_detections = detections.loc[detections["frame"] == available_frame].copy()

    detection_xy = frame_detections[[x_col, y_col]].to_numpy(dtype=float)
    values: list[float] = []
    for seed_i, seed_point in enumerate(seed_points):
        distances = np.linalg.norm(detection_xy - np.asarray(seed_point, dtype=float), axis=1)
        nearest_i = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_i])
        if nearest_distance > radius_px:
            raise SystemExit(
                f"No {camera_name} detection was within {radius_px:g} px of seed "
                f"{seed_i} for the seed filter. Nearest was {nearest_distance:.1f} px. "
                "Increase --seed-filter-radius-px or click closer to the particle center."
            )
        values.append(float(frame_detections.iloc[nearest_i][metric_col]))
    return values


def link_seeded_camera_tracks_to_3d(
    front_mm: pd.DataFrame,
    side_mm: pd.DataFrame,
    front_seed_points: list[tuple[float, float]],
    side_seed_points: list[tuple[float, float]],
    geometry: ChamberGeometry,
    search_range_px: float,
    memory: int,
    stub_length: int,
    seed_frame: int,
    seed_max_distance_px: float,
    z_source: Literal["average", "front", "side"],
) -> pd.DataFrame:
    front_tracks = link_and_select_camera_tracks(
        detections=front_mm,
        camera_name="front",
        seed_points=front_seed_points,
        search_range_px=search_range_px,
        memory=memory,
        stub_length=stub_length,
        seed_frame=seed_frame,
        seed_max_distance_px=seed_max_distance_px,
    )
    side_tracks = link_and_select_camera_tracks(
        detections=side_mm,
        camera_name="side",
        seed_points=side_seed_points,
        search_range_px=search_range_px,
        memory=memory,
        stub_length=stub_length,
        seed_frame=seed_frame,
        seed_max_distance_px=seed_max_distance_px,
    )

    front_keep = front_tracks[
        [
            "seed_id",
            "frame",
            "particle",
            "x_mm",
            "front_z_mm",
            "front_detection_id",
            "front_image_x_px",
            "front_image_y_px",
        ]
    ].rename(columns={"particle": "front_particle"})
    side_keep = side_tracks[
        [
            "seed_id",
            "frame",
            "particle",
            "y_mm",
            "side_z_mm",
            "side_detection_id",
            "side_image_x_px",
            "side_image_y_px",
        ]
    ].rename(columns={"particle": "side_particle"})

    merged = front_keep.merge(side_keep, on=["seed_id", "frame"], how="inner")
    if merged.empty:
        raise SystemExit(
            "The selected front and side 2D tracks have no overlapping frames. "
            "Try increasing --memory or --camera-search-range-px, or click a "
            "later --seed-frame where both particles are visible."
        )

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
    chamber_radius_mm = geometry.radius_at_z(z_mm)

    linked = pd.DataFrame(
        {
            "frame": merged["frame"].astype(int),
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "front_z_mm": front_z,
            "side_z_mm": side_z,
            "z_error_mm": np.abs(front_z - side_z),
            "radial_mm": radial_mm,
            "chamber_radius_mm": chamber_radius_mm,
            "inside_chamber": radial_mm <= chamber_radius_mm,
            "front_detection_id": merged["front_detection_id"].astype(int),
            "side_detection_id": merged["side_detection_id"].astype(int),
            "front_image_x_px": merged["front_image_x_px"].astype(float),
            "front_image_y_px": merged["front_image_y_px"].astype(float),
            "side_image_x_px": merged["side_image_x_px"].astype(float),
            "side_image_y_px": merged["side_image_y_px"].astype(float),
            "front_particle": merged["front_particle"].astype(int),
            "side_particle": merged["side_particle"].astype(int),
            "particle": merged["seed_id"].astype(int),
            "seed_id": merged["seed_id"].astype(int),
        }
    )
    print(
        "Seeded 2D tracking produced "
        f"{len(linked)} paired 3D points across {linked['seed_id'].nunique()} seed(s)."
    )
    return linked.sort_values(["seed_id", "frame"]).reset_index(drop=True)


def link_and_select_camera_tracks(
    detections: pd.DataFrame,
    camera_name: Literal["front", "side"],
    seed_points: list[tuple[float, float]],
    search_range_px: float,
    memory: int,
    stub_length: int,
    seed_frame: int,
    seed_max_distance_px: float,
) -> pd.DataFrame:
    x_col = f"{camera_name}_image_x_px"
    y_col = f"{camera_name}_image_y_px"
    detections = detections.reset_index(drop=True).copy()
    print(
        f"Following {camera_name} seeds frame-by-frame: "
        f"search_range={search_range_px} px, memory={memory} frames"
    )
    selected = follow_seeded_camera_tracks(
        detections=detections,
        camera_name=camera_name,
        seed_points=seed_points,
        search_range_px=search_range_px,
        memory=memory,
        seed_frame=seed_frame,
        seed_max_distance_px=seed_max_distance_px,
    )
    if stub_length > 1:
        before = len(selected)
        counts = selected.groupby("seed_id")["frame"].transform("count")
        selected = selected.loc[counts >= stub_length].copy()
        print(
            f"Filtered {camera_name} seeded tracks by --stub-length {stub_length}: "
            f"{before} -> {len(selected)} points"
        )
    if selected.empty:
        raise SystemExit(
            f"No selected {camera_name} 2D tracks remain. Try lowering --stub-length, "
            "increasing --memory, or increasing --camera-search-range-px."
        )
    return selected


def follow_seeded_camera_tracks(
    detections: pd.DataFrame,
    camera_name: Literal["front", "side"],
    seed_points: list[tuple[float, float]],
    search_range_px: float,
    memory: int,
    seed_frame: int,
    seed_max_distance_px: float,
) -> pd.DataFrame:
    x_col = f"{camera_name}_image_x_px"
    y_col = f"{camera_name}_image_y_px"
    detections = detections.sort_values(["frame", x_col, y_col]).reset_index(drop=True)
    seed_frame_points = detections.loc[detections["frame"] == seed_frame].copy()
    if seed_frame_points.empty:
        available_frame = int(detections["frame"].min())
        print(
            f"No {camera_name} detections exist at seed frame {seed_frame}; "
            f"using first linked frame {available_frame} for seed matching."
        )
        seed_frame = available_frame
        seed_frame_points = detections.loc[detections["frame"] == available_frame].copy()

    seed_positions = np.asarray(seed_points, dtype=float)
    track_positions = seed_frame_points[[x_col, y_col]].to_numpy(dtype=float)
    distances = np.linalg.norm(
        seed_positions[:, np.newaxis, :] - track_positions[np.newaxis, :, :],
        axis=2,
    )

    active: dict[int, dict[str, Any]] = {}
    selected_rows: list[pd.Series] = []
    used_initial_rows: set[int] = set()
    for seed_i, track_i in assignment_pairs(distances):
        distance = float(distances[seed_i, track_i])
        if distance > seed_max_distance_px:
            continue
        row = seed_frame_points.iloc[track_i].copy()
        row["particle"] = seed_i
        row["seed_id"] = seed_i
        selected_rows.append(row)
        used_initial_rows.add(int(row.name))
        active[seed_i] = {
            "x": float(row[x_col]),
            "y": float(row[y_col]),
            "missing": 0,
            "active": True,
        }

    if not active:
        raise SystemExit(
            f"No {camera_name} detection was close enough to the clicked seed(s). "
            "Try increasing --seed-max-distance-px, increasing "
            "--camera-search-range-px, or rerunning with a tighter search region."
        )

    for frame_number in sorted(
        frame for frame in detections["frame"].unique().tolist() if frame > seed_frame
    ):
        frame_detections = detections.loc[detections["frame"] == frame_number]
        active_seed_ids = [
            seed_id for seed_id, state in active.items() if bool(state["active"])
        ]
        if not active_seed_ids:
            break
        if frame_detections.empty:
            for seed_id in active_seed_ids:
                increment_missing(active[seed_id], memory)
            continue

        detection_xy = frame_detections[[x_col, y_col]].to_numpy(dtype=float)
        previous_xy = np.asarray(
            [[active[seed_id]["x"], active[seed_id]["y"]] for seed_id in active_seed_ids],
            dtype=float,
        )
        cost = np.linalg.norm(previous_xy[:, np.newaxis, :] - detection_xy[np.newaxis, :, :], axis=2)
        assigned_seed_ids: set[int] = set()
        assigned_detection_indices: set[int] = set()
        for seed_pos_i, detection_i in assignment_pairs(cost):
            distance = float(cost[seed_pos_i, detection_i])
            if distance > search_range_px:
                continue
            seed_id = active_seed_ids[seed_pos_i]
            row = frame_detections.iloc[detection_i].copy()
            row["particle"] = seed_id
            row["seed_id"] = seed_id
            selected_rows.append(row)
            active[seed_id] = {
                "x": float(row[x_col]),
                "y": float(row[y_col]),
                "missing": 0,
                "active": True,
            }
            assigned_seed_ids.add(seed_id)
            assigned_detection_indices.add(detection_i)

        for seed_id in active_seed_ids:
            if seed_id not in assigned_seed_ids:
                increment_missing(active[seed_id], memory)

    selected = pd.DataFrame(selected_rows)
    selected = selected.sort_values(["seed_id", "frame"]).reset_index(drop=True)
    print(
        f"Kept {camera_name} seeded tracks: "
        + ", ".join(
            f"seed {seed_id}: {len(group)} points, frames {int(group['frame'].min())}-{int(group['frame'].max())}"
            for seed_id, group in selected.groupby("seed_id", sort=True)
        )
    )
    return selected


def increment_missing(state: dict[str, Any], memory: int) -> None:
    state["missing"] = int(state["missing"]) + 1
    if int(state["missing"]) > memory:
        state["active"] = False


def link_3d_tracks(
    matched: pd.DataFrame,
    search_range_mm: float,
    memory: int,
    stub_length: int,
) -> pd.DataFrame:
    try:
        import trackpy as tp
    except ImportError as exc:
        raise SystemExit(
            "Trackpy is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    print(
        f"Linking 3D points: search_range={search_range_mm} mm, memory={memory} frames"
    )
    linked = tp.link_df(
        matched.copy(),
        search_range=search_range_mm,
        memory=memory,
        pos_columns=["x_mm", "y_mm", "z_mm"],
    )
    linked = reset_ambiguous_index_levels(linked, ["particle", "frame"])
    if stub_length > 1:
        linked = tp.filter_stubs(linked, threshold=stub_length)
        linked = reset_ambiguous_index_levels(linked, ["particle", "frame"])
    if linked.empty:
        raise SystemExit(
            "Trackpy linked no trajectories after filtering. Try lowering "
            "--stub-length, increasing --memory, or increasing --search-range-mm."
        )
    return linked.sort_values(["particle", "frame"]).reset_index(drop=True)


def filter_tracks_by_initial_seeds(
    linked: pd.DataFrame,
    seeds: pd.DataFrame,
    seed_frame: int,
    max_distance_mm: float,
) -> pd.DataFrame:
    if linked.empty:
        raise SystemExit("No linked tracks are available to filter by initial seeds.")

    seed_frame_points = linked.loc[linked["frame"] == seed_frame].copy()
    if seed_frame_points.empty:
        available_frame = int(linked["frame"].min())
        print(
            f"No linked points exist at seed frame {seed_frame}; "
            f"using first linked frame {available_frame} for seed matching."
        )
        seed_frame_points = linked.loc[linked["frame"] == available_frame].copy()

    seed_positions = seeds[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    track_positions = seed_frame_points[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    distances = np.linalg.norm(
        seed_positions[:, np.newaxis, :] - track_positions[np.newaxis, :, :],
        axis=2,
    )

    selected_particle_to_seed: dict[int, int] = {}
    for seed_i, track_i in assignment_pairs(distances):
        distance = float(distances[seed_i, track_i])
        if distance > max_distance_mm:
            continue
        particle = int(seed_frame_points.iloc[track_i]["particle"])
        selected_particle_to_seed[particle] = int(seeds.iloc[seed_i]["seed_id"])

    if not selected_particle_to_seed:
        raise SystemExit(
            "No linked tracks were close enough to the clicked initial particles. "
            "Try increasing --seed-max-distance-mm, lowering --stub-length, or "
            "using --select-focus-roi to reduce noise."
        )

    selected_particles = sorted(selected_particle_to_seed)
    filtered = linked.loc[linked["particle"].isin(selected_particles)].copy()
    filtered["seed_id"] = filtered["particle"].map(selected_particle_to_seed)
    filtered = filtered.sort_values(["seed_id", "frame"]).reset_index(drop=True)

    print(
        "Keeping seeded particles: "
        + ", ".join(
            f"seed {seed_id} -> track {particle}"
            for particle, seed_id in sorted(
                selected_particle_to_seed.items(), key=lambda item: item[1]
            )
        )
    )
    return filtered


def write_outputs(
    output_dir: Path,
    geometry: ChamberGeometry,
    front_mm: pd.DataFrame,
    side_mm: pd.DataFrame,
    matched: pd.DataFrame,
    linked: pd.DataFrame,
    seed_points_3d: pd.DataFrame | None,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    front_mm.to_csv(output_dir / "front_detections_mm.csv", index=False)
    side_mm.to_csv(output_dir / "side_detections_mm.csv", index=False)
    matched.to_csv(output_dir / "matched_3d_features.csv", index=False)
    linked.to_csv(output_dir / "tracks_3d.csv", index=False)
    if seed_points_3d is not None:
        seed_points_3d.to_csv(output_dir / "initial_particle_seeds_3d.csv", index=False)

    summary = {
        "chamber": asdict(geometry),
        "front_detections": int(len(front_mm)),
        "side_detections": int(len(side_mm)),
        "matched_3d_points": int(len(matched)),
        "linked_3d_points": int(len(linked)),
        "linked_3d_particles": int(linked["particle"].nunique())
        if "particle" in linked
        else 0,
        "outside_chamber_points": int((~linked["inside_chamber"]).sum())
        if "inside_chamber" in linked
        else 0,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    if make_plots:
        plot_tracks_3d(linked, geometry, output_dir / "tracks_3d.png")
        plot_track_projections(linked, output_dir / "track_projections.png")
        plot_z_match_errors(matched, output_dir / "z_match_errors.png")


def plot_tracks_3d(
    tracks: pd.DataFrame, geometry: ChamberGeometry, output_path: Path
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for _, group in tracks.groupby("particle", sort=False):
        ax.plot(group["x_mm"], group["y_mm"], group["z_mm"], linewidth=1.2, alpha=0.85)
        ax.scatter(
            group["x_mm"],
            group["y_mm"],
            group["z_mm"],
            s=6,
            alpha=0.45,
        )

    draw_chamber_wireframe(ax, geometry)
    ax.set_xlabel("x from front camera [mm]")
    ax.set_ylabel("y from side camera [mm]")
    ax.set_zlabel("z between plates [mm]")
    ax.set_title("3D Particle Tracks")
    span = geometry.bottom_radius_mm
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(0, geometry.gap_mm)
    ax.set_box_aspect(
        (geometry.bottom_diameter_mm, geometry.bottom_diameter_mm, geometry.gap_mm)
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_chamber_wireframe(ax: Any, geometry: ChamberGeometry) -> None:
    theta = np.linspace(0, 2 * np.pi, 160)
    for z, radius, label in (
        (0.0, geometry.bottom_radius_mm, "bottom plate"),
        (geometry.gap_mm, geometry.top_radius_mm, "top plate"),
    ):
        ax.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(theta, z),
            color="black",
            linewidth=0.9,
            alpha=0.35,
            label=label,
        )

    for angle in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot(
            [
                geometry.bottom_radius_mm * math.cos(angle),
                geometry.top_radius_mm * math.cos(angle),
            ],
            [
                geometry.bottom_radius_mm * math.sin(angle),
                geometry.top_radius_mm * math.sin(angle),
            ],
            [0.0, geometry.gap_mm],
            color="black",
            linewidth=0.6,
            alpha=0.25,
        )


def plot_track_projections(tracks: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    projections = [
        ("x_mm", "y_mm", "XY projection"),
        ("x_mm", "z_mm", "XZ projection"),
        ("y_mm", "z_mm", "YZ projection"),
    ]
    for ax, (x_col, y_col, title) in zip(axes, projections):
        for _, group in tracks.groupby("particle", sort=False):
            ax.plot(group[x_col], group[y_col], linewidth=1.0, alpha=0.75)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_z_match_errors(matched: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(matched["z_error_mm"], bins=40, color="#3d6fb6", edgecolor="white")
    ax.set_xlabel("|front z - side z| [mm]")
    ax.set_ylabel("matched detections")
    ax.set_title("Front/Side Z Agreement")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(
    output_dir: Path,
    front_mm: pd.DataFrame,
    side_mm: pd.DataFrame,
    matched: pd.DataFrame,
    linked: pd.DataFrame,
) -> None:
    particle_count = linked["particle"].nunique() if "particle" in linked else 0
    print("\nDone.")
    print(f"Front detections:     {len(front_mm)}")
    print(f"Side detections:      {len(side_mm)}")
    print(f"Matched 3D points:    {len(matched)}")
    print(f"Linked 3D particles:  {particle_count}")
    print(f"Output directory:     {output_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
