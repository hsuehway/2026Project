"""Project a normalized 2D bubble shape profile onto a 3D sphere."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache")
)

from bubble_shape_analysis.surface_projection import (
    ProjectionConfig,
    load_shape_profile,
    mesh_statistics,
    project_shape_profile,
)
from bubble_shape_analysis.surface_reporting import (
    SUPPORTED_MESH_FORMATS,
    export_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an axisymmetric 3D surface of revolution from an individual "
            "or grand-average normalized bubble shape profile."
        )
    )
    parser.add_argument(
        "profile",
        nargs="?",
        type=Path,
        help=(
            "Optional average_shape_profile.csv or grand_average_shape_profile.csv. "
            "The interactive 3D Surface Studio opens when omitted."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the interactive 3D Surface Studio.",
    )
    parser.add_argument(
        "--profile-column",
        help=(
            "Normalized-radius mean column. Automatically detected when omitted."
        ),
    )
    parser.add_argument(
        "--lower-column",
        help="Optional lower-envelope column. Automatically detected when possible.",
    )
    parser.add_argument(
        "--upper-column",
        help="Optional upper-envelope column. Automatically detected when possible.",
    )
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=1.0,
        help="Radius assigned to the unit-normalized profile.",
    )
    parser.add_argument(
        "--latitude-samples",
        type=int,
        default=91,
        help="Number of meridional samples from north to south pole.",
    )
    parser.add_argument(
        "--longitude-samples",
        type=int,
        default=180,
        help="Number of samples around each latitude ring.",
    )
    parser.add_argument(
        "--symmetry-mode",
        choices=("mean", "side-a", "side-b"),
        default="mean",
        help=(
            "How the two sides of the 2D contour form one revolved meridian. "
            "mean is recommended."
        ),
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Axis around which the meridian is revolved.",
    )
    parser.add_argument(
        "--pole-smoothing-degrees",
        type=float,
        default=6.0,
        help="Angular band used to enforce smooth pole transitions. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-envelope",
        action="store_true",
        help="Do not generate lower and upper variability surfaces.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=SUPPORTED_MESH_FORMATS,
        default=list(SUPPORTED_MESH_FORMATS),
        help="Mesh formats to write. Defaults to obj stl ply.",
    )
    parser.add_argument(
        "--name",
        help="Output/report name. Defaults to the profile result-folder name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_bubble_shapes") / "3d_surfaces",
        help="Parent folder for timestamped 3D surface outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gui or args.profile is None:
        from bubble_shape_analysis.surface_gui import launch_surface_studio

        launch_surface_studio(args.profile, args.output_dir)
        return 0
    if not args.profile.is_file():
        raise SystemExit(f"Shape profile CSV does not exist: {args.profile}")
    config = ProjectionConfig(
        reference_radius=args.reference_radius,
        latitude_samples=args.latitude_samples,
        longitude_samples=args.longitude_samples,
        symmetry_mode=args.symmetry_mode,
        symmetry_axis=args.axis,
        pole_smoothing_degrees=args.pole_smoothing_degrees,
        include_envelope=not args.no_envelope,
    )
    try:
        config.validate()
        source = load_shape_profile(
            args.profile,
            profile_column=args.profile_column,
            lower_column=args.lower_column,
            upper_column=args.upper_column,
        )
        name = args.name or f"{args.profile.parent.name}_3d_surface"
        result = project_shape_profile(source, config, name)
        output_dir = export_projection(result, args.output_dir, args.formats)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    mean_stats = mesh_statistics(result.meshes["mean"])
    print("Done.")
    print(f"Input profile:    {args.profile.resolve()}")
    print(f"Profile column:   {source.mean_column}")
    print(f"Projection:       axisymmetric about {config.symmetry_axis}")
    print(f"Mesh vertices:    {mean_stats['vertices']}")
    print(f"Mesh triangles:   {mean_stats['triangles']}")
    print(f"Surface area:     {mean_stats['surface_area']:.8g}")
    print(f"Enclosed volume:  {mean_stats['volume']:.8g}")
    print(f"Output folder:    {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
