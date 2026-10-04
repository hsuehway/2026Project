"""Swap particle IDs in a tracking CSV over selected frame ranges."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from background_subtraction_tracker import format_frame_ranges, parse_frame_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap particle IDs in a CSV for selected frame ranges."
    )
    parser.add_argument("--csv", type=Path, help="Tracking CSV to correct.")
    parser.add_argument(
        "--frames",
        type=parse_frame_ranges,
        help="Inclusive frame range(s), such as 2200-2350 or 2200-2350,2600-2680.",
    )
    parser.add_argument(
        "--particles",
        type=parse_particle_pair,
        default=(0, 1),
        help="Two particle IDs to swap, as A,B. Default: 0,1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Parent folder for corrected CSV output folders. Defaults to a "
            "corrections folder inside the selected CSV's trial folder."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional exact output CSV path. Defaults to a timestamped folder.",
    )
    return parser.parse_args()


def parse_particle_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use exactly two particle IDs as A,B.")
    try:
        first = int(parts[0])
        second = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Particle IDs must be integers.") from exc
    if first == second:
        raise argparse.ArgumentTypeError("Particle IDs must be different.")
    return first, second


def main() -> int:
    args = parse_args()
    csv_path = args.csv or choose_csv()
    if not csv_path.exists():
        raise SystemExit(f"CSV does not exist: {csv_path}")

    frame_ranges = args.frames or prompt_frame_ranges()
    particle_a, particle_b = args.particles
    corrected = swap_particle_ids(
        pd.read_csv(csv_path),
        frame_ranges,
        particle_a,
        particle_b,
    )
    output_path, log_path = resolve_output_paths(args, csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_path, index=False)

    changed_rows = int(corrected["identity_swap_applied"].sum())
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(csv_path.resolve()),
                "output_csv": str(output_path.resolve()),
                "frame_ranges": format_frame_ranges(frame_ranges),
                "particles_swapped": [particle_a, particle_b],
                "changed_rows": changed_rows,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"Wrote corrected CSV: {output_path.resolve()}")
    print(f"Wrote correction log: {log_path.resolve()}")
    print(f"Rows swapped: {changed_rows}")
    return 0


def choose_csv() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(
            f"Pass --csv because Tk file dialogs are unavailable ({exc})."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose tracking CSV to correct",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No CSV selected.")
    return Path(path)


def prompt_frame_ranges() -> tuple[tuple[int, int], ...]:
    value = input("Frame range(s) to swap, e.g. 2200-2350: ").strip()
    if not value:
        raise SystemExit("No frame range entered.")
    return parse_frame_ranges(value)


def swap_particle_ids(
    df: pd.DataFrame,
    frame_ranges: tuple[tuple[int, int], ...],
    particle_a: int,
    particle_b: int,
) -> pd.DataFrame:
    required = {"frame", "particle"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    corrected = df.copy()
    frames = corrected["frame"].astype(int)
    in_range = frames.map(lambda frame: frame_in_ranges_local(frame, frame_ranges))
    in_pair = corrected["particle"].astype(int).isin([particle_a, particle_b])
    mask = in_range & in_pair

    corrected["identity_swap_applied"] = False
    corrected["identity_swap_ranges"] = format_frame_ranges(frame_ranges)
    corrected["identity_swap_particles"] = f"{particle_a}<->{particle_b}"
    corrected["pre_identity_swap_particle"] = pd.NA
    corrected.loc[mask, "pre_identity_swap_particle"] = corrected.loc[mask, "particle"]

    particle_values = corrected.loc[mask, "particle"].astype(int)
    corrected.loc[mask, "particle"] = particle_values.map(
        {particle_a: particle_b, particle_b: particle_a}
    )

    if "seed_id" in corrected.columns:
        seed_pair = corrected["seed_id"].astype(int).isin([particle_a, particle_b])
        seed_mask = in_range & seed_pair
        corrected["pre_identity_swap_seed_id"] = pd.NA
        corrected.loc[seed_mask, "pre_identity_swap_seed_id"] = corrected.loc[
            seed_mask,
            "seed_id",
        ]
        seed_values = corrected.loc[seed_mask, "seed_id"].astype(int)
        corrected.loc[seed_mask, "seed_id"] = seed_values.map(
            {particle_a: particle_b, particle_b: particle_a}
        )

    corrected.loc[mask, "identity_swap_applied"] = True
    return corrected.sort_values(["particle", "frame"]).reset_index(drop=True)


def frame_in_ranges_local(
    frame_number: int,
    ranges: tuple[tuple[int, int], ...],
) -> bool:
    return any(start <= frame_number <= end for start, end in ranges)


def resolve_output_paths(args: argparse.Namespace, csv_path: Path) -> tuple[Path, Path]:
    if args.output is not None:
        output_path = args.output
        return output_path, output_path.with_suffix(".identity_swap_log.json")

    output_dir = make_run_output_dir(args.output_dir, csv_path)
    output_path = output_dir / f"{csv_path.stem}_ids_swapped.csv"
    log_path = output_dir / "identity_swap_log.json"
    return output_path, log_path


def make_run_output_dir(parent_dir: Path, csv_path: Path) -> Path:
    parent_dir = parent_dir or (infer_trial_dir(csv_path) / "corrections")
    parent_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"manual_identity_swap_{sanitize_output_name(csv_path.stem)}_{timestamp}"
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
    return name or "identity_correction"


def infer_trial_dir(csv_path: Path) -> Path:
    parent = csv_path.resolve().parent
    if parent.name.startswith("identity_correction") and parent.parent.name == "corrections":
        return parent.parent.parent
    if parent.name.startswith("manual_identity_swap") and parent.parent.name == "corrections":
        return parent.parent.parent
    if parent.name == "corrections":
        return parent.parent
    return parent


if __name__ == "__main__":
    raise SystemExit(main())
