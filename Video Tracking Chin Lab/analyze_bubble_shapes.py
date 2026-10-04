"""Launch or batch-run scale-free ceramic glass bubble shape analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from bubble_shape_analysis.core import SUPPORTED_IMAGE_SUFFIXES, analyze_batch
from bubble_shape_analysis.models import AnalysisConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure normalized ceramic glass bubble silhouettes and deviation "
            "magnitudes from still microscopy images."
        )
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="*",
        help="Images to analyze. With no paths, the guided desktop app opens.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Analyze every supported image directly inside this folder.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the guided desktop interface even when image paths are supplied.",
    )
    parser.add_argument(
        "--batch-name",
        default="bubble_shape_batch",
        help="Name used in reports and the output folder.",
    )
    parser.add_argument(
        "--expected-radius-px",
        type=float,
        default=25.0,
        help="Typical apparent radius used only as a detection guide.",
    )
    parser.add_argument(
        "--min-radius-px",
        type=float,
        default=8.0,
        help="Smallest apparent radius eligible for detection.",
    )
    parser.add_argument(
        "--max-radius-px",
        type=float,
        default=70.0,
        help="Largest apparent radius eligible for detection.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Use qualified visible arcs from touching bubbles in the default average.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_bubble_shapes") / "analysis_runs",
        help="Parent folder for timestamped result folders.",
    )
    return parser.parse_args()


def collect_image_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.images]
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise SystemExit(f"Input folder does not exist: {args.input_dir}")
        paths.extend(
            sorted(
                path
                for path in args.input_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def main() -> int:
    args = parse_args()
    paths = collect_image_paths(args)
    if args.gui or not paths:
        from bubble_shape_analysis.gui import launch_gui

        launch_gui()
        return 0

    config = AnalysisConfig(
        expected_radius_px=args.expected_radius_px,
        min_radius_px=args.min_radius_px,
        max_radius_px=args.max_radius_px,
        include_partial_in_average=args.include_partial,
    )
    try:
        config.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    def progress(index: int, total: int, path: Path) -> None:
        print(f"[{index}/{total}] {path}")

    batch = analyze_batch(paths, config, args.batch_name, progress=progress)
    from bubble_shape_analysis.reporting import export_results

    output_dir = export_results(batch, args.output_dir)
    counts = {
        status: sum(
            detection.status == status
            for image in batch.images
            for detection in image.detections
        )
        for status in ("accepted", "partial", "rejected")
    }
    print("\nDone.")
    print(f"Accepted isolated: {counts['accepted']}")
    print(f"Qualified partial: {counts['partial']}")
    print(f"Rejected:          {counts['rejected']}")
    print(f"Output folder:     {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
