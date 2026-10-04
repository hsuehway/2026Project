"""Create a grand average shape profile from multiple dataset outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from bubble_shape_analysis.multi_dataset import (
    PROFILE_KINDS,
    WEIGHTING_MODES,
    GrandAverageConfig,
    build_grand_average,
    discover_dataset_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple reviewed bubble-shape datasets into an equal-dataset "
            "or bubble-count-weighted grand average."
        )
    )
    parser.add_argument(
        "datasets",
        type=Path,
        nargs="*",
        help=(
            "Dataset result folders, session.json files, average_shape_profile.csv "
            "files, or parent folders containing multiple datasets."
        ),
    )
    parser.add_argument(
        "--parent-dir",
        type=Path,
        action="append",
        default=[],
        help="Parent folder to scan recursively. May be passed more than once.",
    )
    parser.add_argument(
        "--name",
        default="grand_average",
        help="Name used in the combined report and output folder.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_KINDS,
        default="isolated-major",
        help="Population and angular alignment to combine.",
    )
    parser.add_argument(
        "--weighting",
        choices=WEIGHTING_MODES,
        default="dataset-equal",
        help=(
            "dataset-equal gives every dataset one vote; bubble-count approximates "
            "a pooled-bubble mean."
        ),
    )
    parser.add_argument(
        "--angular-samples",
        type=int,
        default=180,
        help="Number of angles in the combined normalized contour.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_bubble_shapes") / "grand_averages",
        help="Parent folder for timestamped combined-result folders.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the guided multi-dataset interface.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [*args.datasets, *args.parent_dir]
    if args.gui or not inputs:
        from bubble_shape_analysis.multi_gui import launch_multi_dataset_gui

        launch_multi_dataset_gui()
        return 0

    config = GrandAverageConfig(
        profile_kind=args.profile,
        weighting=args.weighting,
        angular_samples=args.angular_samples,
    )
    sources = discover_dataset_sources(inputs)
    if sources:
        print(f"Found {len(sources)} dataset sources:")
        for source in sources:
            print(f"  {source}")
    try:
        result = build_grand_average(inputs, config, args.name)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    from bubble_shape_analysis.multi_reporting import export_grand_average

    output_dir = export_grand_average(result, args.output_dir)
    print("\nDone.")
    print(f"Datasets combined: {len(result.datasets)}")
    print(f"Source bubbles:    {result.summary()['total_source_bubbles']}")
    print(f"Profile:           {config.profile_kind}")
    print(f"Weighting:         {config.weighting}")
    print(f"Output folder:     {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
