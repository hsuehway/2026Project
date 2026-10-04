"""Build individual and, when available, pairwise particle state vectors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

from background_subtraction_tracker import sanitize_output_name

STATE_COLUMNS = (
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
RAW_STATE_COLUMNS = (
    "x_raw_mm",
    "y_raw_mm",
    "z_raw_mm",
    "vx_raw_mm_s",
    "vy_raw_mm_s",
    "vz_raw_mm_s",
    "ax_raw_mm_s2",
    "ay_raw_mm_s2",
    "az_raw_mm_s2",
)
SMOOTH_STATE_COLUMNS = (
    "x_smooth_mm",
    "y_smooth_mm",
    "z_smooth_mm",
    "vx_smooth_mm_s",
    "vy_smooth_mm_s",
    "vz_smooth_mm_s",
    "ax_smooth_mm_s2",
    "ay_smooth_mm_s2",
    "az_smooth_mm_s2",
)
REQUIRED_COLUMNS = ("particle", "frame", "time_s", *STATE_COLUMNS)
RELATIVE_COLUMNS = (
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "dvx_mm_s",
    "dvy_mm_s",
    "dvz_mm_s",
    "dax_mm_s2",
    "day_mm_s2",
    "daz_mm_s2",
)
RAW_RELATIVE_COLUMNS = (
    "dx_raw_mm",
    "dy_raw_mm",
    "dz_raw_mm",
    "dvx_raw_mm_s",
    "dvy_raw_mm_s",
    "dvz_raw_mm_s",
    "dax_raw_mm_s2",
    "day_raw_mm_s2",
    "daz_raw_mm_s2",
)
SMOOTH_RELATIVE_COLUMNS = (
    "dx_smooth_mm",
    "dy_smooth_mm",
    "dz_smooth_mm",
    "dvx_smooth_mm_s",
    "dvy_smooth_mm_s",
    "dvz_smooth_mm_s",
    "dax_smooth_mm_s2",
    "day_smooth_mm_s2",
    "daz_smooth_mm_s2",
)
SCALAR_PAIR_COLUMNS = (
    "distance_3d_mm",
    "distance_xy_mm",
    "distance_xz_mm",
    "distance_yz_mm",
    "relative_speed_mm_s",
    "relative_acceleration_mm_s2",
    "radial_relative_velocity_mm_s",
    "closing_speed_mm_s",
    "tangential_relative_speed_mm_s",
    "position_velocity_dot_mm2_s",
    "velocity_acceleration_dot_mm2_s3",
)
TARGET_SETS = {
    "future_velocity": ("vx_mm_s", "vy_mm_s", "vz_mm_s"),
    "future_acceleration": ("ax_mm_s2", "ay_mm_s2", "az_mm_s2"),
    "future_motion_state": (
        "vx_mm_s",
        "vy_mm_s",
        "vz_mm_s",
        "ax_mm_s2",
        "ay_mm_s2",
        "az_mm_s2",
    ),
    "future_full_state": STATE_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build particle state vectors from velocities_accelerations.csv. "
            "With two or more particles, also build pair/relative vectors and "
            "test whether one particle helps predict another's future motion."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="velocities_accelerations.csv. If omitted, File Explorer opens.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Fallback FPS if time_s is missing or empty.",
    )
    parser.add_argument(
        "--particles",
        type=parse_particle_list,
        help=(
            "Optional comma-separated particle IDs. One ID enables single-particle "
            "mode. Default: all particles in CSV."
        ),
    )
    parser.add_argument(
        "--state-source",
        choices=("smoothed", "raw"),
        default="raw",
        help=(
            "Kinematics used by canonical state-vector columns and downstream "
            "models. Default: raw, preserving the non-smoothed state. Raw and "
            "smoothed representations are always retained."
        ),
    )
    parser.add_argument(
        "--max-prediction-lag-frames",
        type=int,
        default=60,
        help="Predict target state 1..N frames into the future. Default: 60.",
    )
    parser.add_argument(
        "--include-zero-lag",
        action="store_true",
        help="Also include lag 0 in prediction outputs as a non-future control.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength used for prediction models.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Chronological test fraction for prediction models.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum usable rows needed to fit/report a prediction model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_background") / "state_vectors",
        help="Parent folder for timestamped state-vector output folders.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output folder name.",
    )
    parser.add_argument(
        "--no-prediction",
        action="store_true",
        help="Build state-vector CSVs without cross-particle prediction models.",
    )
    parser.add_argument(
        "--no-prediction-dataset",
        action="store_true",
        help="Skip writing lagged_prediction_dataset.csv.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip prediction summary PNG plots.",
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

    csv_path = args.csv or choose_csv_file()
    motion = load_motion_csv(
        pd.read_csv(csv_path),
        csv_path,
        fps=args.fps,
        state_source=args.state_source,
    )
    selected_particles = args.particles or tuple(
        int(particle) for particle in sorted(motion["particle"].unique())
    )
    validate_selected_particles(motion, selected_particles)
    single_particle_mode = len(selected_particles) == 1

    particle_states = build_particle_state_vectors(motion, selected_particles)
    pair_states = pd.DataFrame()
    relative_states = pd.DataFrame()
    if not single_particle_mode:
        pair_states, relative_states = build_pair_and_relative_state_vectors(
            motion,
            selected_particles,
        )

    prediction_dataset = pd.DataFrame()
    prediction_results = pd.DataFrame()
    prediction_comparison = pd.DataFrame()
    prediction_enabled = not args.no_prediction and not single_particle_mode
    if prediction_enabled:
        prediction_dataset = build_lagged_prediction_dataset(
            motion,
            selected_particles,
            max_lag_frames=args.max_prediction_lag_frames,
            include_zero_lag=args.include_zero_lag,
        )
        prediction_results = run_prediction_models(
            prediction_dataset,
            ridge_alpha=args.ridge_alpha,
            test_fraction=args.test_fraction,
            min_samples=args.min_samples,
        )
        prediction_comparison = compare_prediction_models(prediction_results)

    output_dir = make_output_dir(csv_path, args.output_dir, args.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    particle_states.to_csv(output_dir / "particle_state_vectors.csv", index=False)
    if not single_particle_mode:
        pair_states.to_csv(output_dir / "pair_state_vectors.csv", index=False)
        relative_states.to_csv(output_dir / "relative_state_vectors.csv", index=False)
    if prediction_enabled:
        if not args.no_prediction_dataset:
            prediction_dataset.to_csv(output_dir / "lagged_prediction_dataset.csv", index=False)
        prediction_results.to_csv(output_dir / "prediction_model_results.csv", index=False)
        prediction_comparison.to_csv(
            output_dir / "prediction_model_comparison.csv",
            index=False,
        )
    write_metadata(
        output_dir / "metadata.json",
        csv_path,
        args,
        selected_particles,
        particle_states,
        pair_states,
        relative_states,
        prediction_dataset,
        prediction_results,
    )
    if not args.no_plots and prediction_enabled:
        plot_prediction_summary(prediction_comparison, output_dir)

    print("\nDone.")
    print(f"Source CSV:       {csv_path.resolve()}")
    print(f"Mode:             {'single-particle' if single_particle_mode else 'multi-particle'}")
    print(f"Particles:        {','.join(str(p) for p in selected_particles)}")
    print(f"Particle rows:    {len(particle_states)}")
    if not single_particle_mode:
        print(f"Pair rows:        {len(pair_states)}")
        print(f"Relative rows:    {len(relative_states)}")
    if prediction_enabled:
        print(f"Prediction rows:  {len(prediction_dataset)}")
        print(f"Model results:    {len(prediction_results)}")
    print(f"Output folder:    {output_dir.resolve()}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.max_prediction_lag_frames < 0:
        raise SystemExit("--max-prediction-lag-frames must be zero or positive.")
    if args.ridge_alpha < 0:
        raise SystemExit("--ridge-alpha must be non-negative.")
    if not 0 < args.test_fraction < 1:
        raise SystemExit("--test-fraction must be between 0 and 1.")
    if args.min_samples < 4:
        raise SystemExit("--min-samples must be at least 4.")


def choose_csv_file() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        raise SystemExit(f"Pass --csv because File Explorer is unavailable ({exc}).") from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose velocities_accelerations.csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No CSV file selected.")
    return Path(path)


def load_motion_csv(
    source: pd.DataFrame,
    csv_path: Path,
    fps: float,
    state_source: str = "smoothed",
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing required column(s): {', '.join(missing)}. "
            "Run calculate_motion.py first and choose velocities_accelerations.csv."
        )
    if source.empty:
        raise SystemExit(f"{csv_path} is empty.")
    motion = source.copy()
    motion["particle"] = motion["particle"].astype(int)
    motion["frame"] = motion["frame"].astype(int)
    if "time_s" not in motion.columns or motion["time_s"].isna().all():
        motion["time_s"] = motion["frame"].astype(float) / fps
    motion["time_s"] = motion["time_s"].astype(float)
    motion = add_state_source_columns(motion)
    motion = select_canonical_state_source(motion, state_source)
    return motion.sort_values(["particle", "frame"]).reset_index(drop=True)


def add_state_source_columns(motion: pd.DataFrame) -> pd.DataFrame:
    motion = motion.copy()
    aliases = {
        "x_raw_mm": ("x_raw_mm", "x_mm"),
        "y_raw_mm": ("y_raw_mm", "y_mm"),
        "z_raw_mm": ("z_raw_mm", "z_mm"),
        "x_smooth_mm": ("x_smooth_mm", "x_mm"),
        "y_smooth_mm": ("y_smooth_mm", "y_mm"),
        "z_smooth_mm": ("z_smooth_mm", "z_mm"),
        "vx_raw_mm_s": ("vx_raw_mm_s", "vx_mm_s"),
        "vy_raw_mm_s": ("vy_raw_mm_s", "vy_mm_s"),
        "vz_raw_mm_s": ("vz_raw_mm_s", "vz_mm_s"),
        "ax_raw_mm_s2": ("ax_raw_mm_s2", "ax_mm_s2"),
        "ay_raw_mm_s2": ("ay_raw_mm_s2", "ay_mm_s2"),
        "az_raw_mm_s2": ("az_raw_mm_s2", "az_mm_s2"),
        "vx_smooth_mm_s": ("vx_smooth_mm_s", "vx_mm_s"),
        "vy_smooth_mm_s": ("vy_smooth_mm_s", "vy_mm_s"),
        "vz_smooth_mm_s": ("vz_smooth_mm_s", "vz_mm_s"),
        "ax_smooth_mm_s2": ("ax_smooth_mm_s2", "ax_mm_s2"),
        "ay_smooth_mm_s2": ("ay_smooth_mm_s2", "ay_mm_s2"),
        "az_smooth_mm_s2": ("az_smooth_mm_s2", "az_mm_s2"),
    }
    for target, candidates in aliases.items():
        if target in motion.columns:
            continue
        source = next((column for column in candidates if column in motion.columns), None)
        if source is None:
            raise SystemExit(f"Motion CSV cannot provide required state column {target}.")
        motion[target] = motion[source]
    return motion


def select_canonical_state_source(
    motion: pd.DataFrame,
    state_source: str,
) -> pd.DataFrame:
    if state_source not in {"raw", "smoothed"}:
        raise ValueError("state_source must be 'raw' or 'smoothed'.")
    selected_columns = RAW_STATE_COLUMNS if state_source == "raw" else SMOOTH_STATE_COLUMNS
    motion = motion.copy()
    motion.loc[:, STATE_COLUMNS] = motion.loc[:, selected_columns].to_numpy(dtype=float)
    motion["state_source"] = state_source
    return motion


def validate_selected_particles(motion: pd.DataFrame, selected_particles: tuple[int, ...]) -> None:
    if not selected_particles:
        raise SystemExit("Need at least one particle to build state vectors.")
    available = set(motion["particle"].astype(int).unique())
    missing = [particle for particle in selected_particles if particle not in available]
    if missing:
        raise SystemExit(f"Selected particle IDs are not in the CSV: {missing}")


def build_particle_state_vectors(
    motion: pd.DataFrame,
    selected_particles: tuple[int, ...],
) -> pd.DataFrame:
    states = motion.loc[motion["particle"].isin(selected_particles)].copy()
    states = states.sort_values(["particle", "frame"]).reset_index(drop=True)
    states["state_vector_order"] = ",".join(STATE_COLUMNS)
    states["state_vector_json"] = [
        vector_to_json(row)
        for row in states.loc[:, STATE_COLUMNS].to_numpy(dtype=float)
    ]
    add_named_vector_columns(states, "raw", RAW_STATE_COLUMNS)
    add_named_vector_columns(states, "smooth", SMOOTH_STATE_COLUMNS)
    return states


def add_named_vector_columns(
    table: pd.DataFrame,
    name: str,
    columns: tuple[str, ...],
) -> None:
    table[f"{name}_state_vector_order"] = ",".join(columns)
    table[f"{name}_state_vector_json"] = [
        vector_to_json(row) for row in table.loc[:, columns].to_numpy(dtype=float)
    ]


def build_pair_and_relative_state_vectors(
    motion: pd.DataFrame,
    selected_particles: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[pd.DataFrame] = []
    relative_rows: list[pd.DataFrame] = []
    for particle_a, particle_b in combinations(selected_particles, 2):
        merged = merge_pair_motion(motion, particle_a, particle_b)
        if merged.empty:
            continue
        pair, relative = calculate_pair_and_relative_states(
            merged,
            particle_a,
            particle_b,
        )
        pair_rows.append(pair)
        relative_rows.append(relative)
    pair_states = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    relative_states = (
        pd.concat(relative_rows, ignore_index=True) if relative_rows else pd.DataFrame()
    )
    return pair_states, relative_states


def merge_pair_motion(
    motion: pd.DataFrame,
    particle_a: int,
    particle_b: int,
) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
                "frame",
                "time_s",
                *STATE_COLUMNS,
                *RAW_STATE_COLUMNS,
                *SMOOTH_STATE_COLUMNS,
            ]
        )
    )
    a = motion.loc[motion["particle"] == particle_a, columns].copy()
    b = motion.loc[motion["particle"] == particle_b, columns].copy()
    return a.merge(b, on="frame", suffixes=("_a", "_b")).sort_values("frame")


def calculate_pair_and_relative_states(
    merged: pd.DataFrame,
    particle_a: int,
    particle_b: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(
        {
            "particle_a": int(particle_a),
            "particle_b": int(particle_b),
            "frame": merged["frame"].astype(int).to_numpy(),
            "time_s": merged["time_s_a"].to_numpy(dtype=float),
        }
    )
    pair = base.copy()
    for column in STATE_COLUMNS:
        pair[f"a_{column}"] = merged[f"{column}_a"].to_numpy(dtype=float)
        pair[f"b_{column}"] = merged[f"{column}_b"].to_numpy(dtype=float)

    relative = base.copy()
    relative_vectors = []
    for source_column, relative_column in zip(STATE_COLUMNS, RELATIVE_COLUMNS):
        values = (
            merged[f"{source_column}_b"].to_numpy(dtype=float)
            - merged[f"{source_column}_a"].to_numpy(dtype=float)
        )
        relative[relative_column] = values
        relative_vectors.append(values)
    relative_matrix = np.column_stack(relative_vectors)

    add_scalar_pair_metrics(relative, relative_matrix)
    pair["pair_state_vector_order"] = ",".join(
        [f"a_{column}" for column in STATE_COLUMNS]
        + [f"b_{column}" for column in STATE_COLUMNS]
    )
    pair["pair_state_vector_json"] = [
        vector_to_json(row)
        for row in pair[
            [f"a_{column}" for column in STATE_COLUMNS]
            + [f"b_{column}" for column in STATE_COLUMNS]
        ].to_numpy(dtype=float)
    ]
    add_pair_state_representation(
        pair,
        relative,
        merged,
        name="raw",
        state_columns=RAW_STATE_COLUMNS,
        relative_columns=RAW_RELATIVE_COLUMNS,
    )
    add_pair_state_representation(
        pair,
        relative,
        merged,
        name="smooth",
        state_columns=SMOOTH_STATE_COLUMNS,
        relative_columns=SMOOTH_RELATIVE_COLUMNS,
    )
    for column in RELATIVE_COLUMNS:
        pair[column] = relative[column]
    for column in SCALAR_PAIR_COLUMNS:
        pair[column] = relative[column]

    relative["relative_state_definition"] = "particle_b_minus_particle_a"
    relative["relative_state_vector_order"] = ",".join(RELATIVE_COLUMNS)
    relative["relative_state_vector_json"] = [
        vector_to_json(row) for row in relative_matrix
    ]
    return pair, relative


def add_pair_state_representation(
    pair: pd.DataFrame,
    relative: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    name: str,
    state_columns: tuple[str, ...],
    relative_columns: tuple[str, ...],
) -> None:
    relative_vectors: list[np.ndarray] = []
    for state_column, relative_column in zip(state_columns, relative_columns):
        a_column = f"a_{state_column}"
        b_column = f"b_{state_column}"
        a_values = merged[f"{state_column}_a"].to_numpy(dtype=float)
        b_values = merged[f"{state_column}_b"].to_numpy(dtype=float)
        pair[a_column] = a_values
        pair[b_column] = b_values
        relative_values = b_values - a_values
        relative[relative_column] = relative_values
        pair[relative_column] = relative_values
        relative_vectors.append(relative_values)

    ordered_pair_columns = [f"a_{column}" for column in state_columns] + [
        f"b_{column}" for column in state_columns
    ]
    pair[f"{name}_pair_state_vector_order"] = ",".join(ordered_pair_columns)
    pair[f"{name}_pair_state_vector_json"] = [
        vector_to_json(row)
        for row in pair.loc[:, ordered_pair_columns].to_numpy(dtype=float)
    ]
    relative_matrix = np.column_stack(relative_vectors)
    relative[f"{name}_relative_state_vector_order"] = ",".join(relative_columns)
    relative[f"{name}_relative_state_vector_json"] = [
        vector_to_json(row) for row in relative_matrix
    ]


def add_scalar_pair_metrics(relative: pd.DataFrame, relative_matrix: np.ndarray) -> None:
    position = relative_matrix[:, 0:3]
    velocity = relative_matrix[:, 3:6]
    acceleration = relative_matrix[:, 6:9]
    distance_3d = np.linalg.norm(position, axis=1)
    relative_speed = np.linalg.norm(velocity, axis=1)
    radial_velocity = safe_row_dot(position, velocity) / np.where(
        distance_3d > 0,
        distance_3d,
        np.nan,
    )
    tangential_sq = np.maximum(relative_speed**2 - radial_velocity**2, 0)

    relative["distance_3d_mm"] = distance_3d
    relative["distance_xy_mm"] = np.linalg.norm(position[:, [0, 1]], axis=1)
    relative["distance_xz_mm"] = np.linalg.norm(position[:, [0, 2]], axis=1)
    relative["distance_yz_mm"] = np.linalg.norm(position[:, [1, 2]], axis=1)
    relative["relative_speed_mm_s"] = relative_speed
    relative["relative_acceleration_mm_s2"] = np.linalg.norm(acceleration, axis=1)
    relative["radial_relative_velocity_mm_s"] = radial_velocity
    relative["closing_speed_mm_s"] = -radial_velocity
    relative["tangential_relative_speed_mm_s"] = np.sqrt(tangential_sq)
    relative["position_velocity_dot_mm2_s"] = safe_row_dot(position, velocity)
    relative["velocity_acceleration_dot_mm2_s3"] = safe_row_dot(velocity, acceleration)


def safe_row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def build_lagged_prediction_dataset(
    motion: pd.DataFrame,
    selected_particles: tuple[int, ...],
    max_lag_frames: int,
    include_zero_lag: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    start_lag = 0 if include_zero_lag else 1
    for particle_a, particle_b in combinations(selected_particles, 2):
        merged = merge_pair_motion(motion, particle_a, particle_b)
        if merged.empty:
            continue
        for target_particle, other_particle, target_side in (
            (particle_a, particle_b, "a"),
            (particle_b, particle_a, "b"),
        ):
            current_features = build_current_prediction_features(
                merged,
                particle_a,
                particle_b,
                target_particle,
                other_particle,
                target_side,
            )
            future = motion.loc[
                motion["particle"] == target_particle,
                ["frame", "time_s", *STATE_COLUMNS],
            ].copy()
            for lag_frames in range(start_lag, max_lag_frames + 1):
                target = future.copy()
                target["frame"] = target["frame"].astype(int) - int(lag_frames)
                target = target.rename(
                    columns={
                        "time_s": "target_future_time_s",
                        **{
                            column: f"target_future_{column}"
                            for column in STATE_COLUMNS
                        },
                    }
                )
                dataset = current_features.merge(target, on="frame")
                if dataset.empty:
                    continue
                dataset["lag_frames"] = int(lag_frames)
                frame_dt = estimate_frame_dt_s(motion)
                dataset["lag_s"] = float(lag_frames * frame_dt)
                rows.append(dataset)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["particle_a", "particle_b", "target_particle", "lag_frames", "frame"]
    ).reset_index(drop=True)


def build_current_prediction_features(
    merged: pd.DataFrame,
    particle_a: int,
    particle_b: int,
    target_particle: int,
    other_particle: int,
    target_side: str,
) -> pd.DataFrame:
    other_side = "b" if target_side == "a" else "a"
    features = pd.DataFrame(
        {
            "particle_a": int(particle_a),
            "particle_b": int(particle_b),
            "target_particle": int(target_particle),
            "other_particle": int(other_particle),
            "frame": merged["frame"].astype(int).to_numpy(),
            "time_s": merged[f"time_s_{target_side}"].to_numpy(dtype=float),
        }
    )
    for column in STATE_COLUMNS:
        self_values = merged[f"{column}_{target_side}"].to_numpy(dtype=float)
        other_values = merged[f"{column}_{other_side}"].to_numpy(dtype=float)
        features[f"self_{column}"] = self_values
        features[f"other_{column}"] = other_values
        features[f"other_minus_self_{column}"] = other_values - self_values

    rel_matrix = np.column_stack(
        [features[f"other_minus_self_{column}"].to_numpy(dtype=float) for column in STATE_COLUMNS]
    )
    position = rel_matrix[:, 0:3]
    velocity = rel_matrix[:, 3:6]
    acceleration = rel_matrix[:, 6:9]
    distance_3d = np.linalg.norm(position, axis=1)
    relative_speed = np.linalg.norm(velocity, axis=1)
    radial_velocity = safe_row_dot(position, velocity) / np.where(
        distance_3d > 0,
        distance_3d,
        np.nan,
    )
    tangential_sq = np.maximum(relative_speed**2 - radial_velocity**2, 0)
    features["distance_3d_mm"] = distance_3d
    features["relative_speed_mm_s"] = relative_speed
    features["relative_acceleration_mm_s2"] = np.linalg.norm(acceleration, axis=1)
    features["radial_relative_velocity_mm_s"] = radial_velocity
    features["closing_speed_mm_s"] = -radial_velocity
    features["tangential_relative_speed_mm_s"] = np.sqrt(tangential_sq)
    return features


def run_prediction_models(
    prediction_dataset: pd.DataFrame,
    ridge_alpha: float,
    test_fraction: float,
    min_samples: int,
) -> pd.DataFrame:
    if prediction_dataset.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    grouped = prediction_dataset.groupby(
        ["particle_a", "particle_b", "target_particle", "other_particle", "lag_frames"],
        sort=True,
    )
    for keys, group in grouped:
        particle_a, particle_b, target_particle, other_particle, lag_frames = keys
        group = group.sort_values("frame").copy()
        for target_name, target_columns in target_sets_as_dataset_columns().items():
            for model_name, feature_columns in feature_sets().items():
                result = fit_evaluate_ridge(
                    group,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    ridge_alpha=ridge_alpha,
                    test_fraction=test_fraction,
                    min_samples=min_samples,
                )
                if result is None:
                    continue
                rows.append(
                    {
                        "particle_a": int(particle_a),
                        "particle_b": int(particle_b),
                        "target_particle": int(target_particle),
                        "other_particle": int(other_particle),
                        "lag_frames": int(lag_frames),
                        "lag_s": float(group["lag_s"].iloc[0]),
                        "target_vector": target_name,
                        "model": model_name,
                        "feature_count": len(feature_columns),
                        "target_columns": ",".join(target_columns),
                        "feature_columns": ",".join(feature_columns),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def target_sets_as_dataset_columns() -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(f"target_future_{column}" for column in columns)
        for name, columns in TARGET_SETS.items()
    }


def feature_sets() -> dict[str, tuple[str, ...]]:
    self_columns = tuple(f"self_{column}" for column in STATE_COLUMNS)
    other_columns = tuple(f"other_{column}" for column in STATE_COLUMNS)
    relative_columns = tuple(f"other_minus_self_{column}" for column in STATE_COLUMNS)
    scalar_columns = (
        "distance_3d_mm",
        "relative_speed_mm_s",
        "relative_acceleration_mm_s2",
        "radial_relative_velocity_mm_s",
        "closing_speed_mm_s",
        "tangential_relative_speed_mm_s",
    )
    return {
        "self_only": self_columns,
        "self_plus_other_relative": self_columns
        + other_columns
        + relative_columns
        + scalar_columns,
    }


def fit_evaluate_ridge(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    ridge_alpha: float,
    test_fraction: float,
    min_samples: int,
) -> dict[str, float | int | str] | None:
    needed = list(feature_columns) + list(target_columns)
    clean = data[["frame", *needed]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_samples:
        return None
    n_test = max(2, int(np.ceil(len(clean) * test_fraction)))
    n_train = len(clean) - n_test
    if n_train < max(2, min_samples // 2):
        return None

    train = clean.iloc[:n_train]
    test = clean.iloc[n_train:]
    x_train = train.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train.loc[:, target_columns].to_numpy(dtype=float)
    x_test = test.loc[:, feature_columns].to_numpy(dtype=float)
    y_test = test.loc[:, target_columns].to_numpy(dtype=float)

    y_pred = ridge_predict(
        x_train,
        y_train,
        x_test,
        alpha=ridge_alpha,
    )
    metrics = regression_metrics(y_test, y_pred, target_columns)
    return {
        "n_total": int(len(clean)),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_start_frame": int(train["frame"].min()),
        "train_end_frame": int(train["frame"].max()),
        "test_start_frame": int(test["frame"].min()),
        "test_end_frame": int(test["frame"].max()),
        **metrics,
    }


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y_train.mean(axis=0)
    x_train_scaled = (x_train - x_mean) / x_std
    x_test_scaled = (x_test - x_mean) / x_std
    y_centered = y_train - y_mean
    a = x_train_scaled.T @ x_train_scaled
    if alpha > 0:
        a = a + alpha * np.eye(a.shape[0])
    b = x_train_scaled.T @ y_centered
    try:
        beta = linalg.solve(a, b, assume_a="sym")
    except Exception:
        beta = linalg.lstsq(a, b)[0]
    return x_test_scaled @ beta + y_mean


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: tuple[str, ...],
) -> dict[str, float | str]:
    residual = y_pred - y_true
    rmse_by_component = np.sqrt(np.nanmean(residual**2, axis=0))
    mae_by_component = np.nanmean(np.abs(residual), axis=0)
    total_ss_res = float(np.nansum(residual**2))
    total_centered = y_true - np.nanmean(y_true, axis=0)
    total_ss_tot = float(np.nansum(total_centered**2))
    r2_overall = 1.0 - total_ss_res / total_ss_tot if total_ss_tot > 0 else float("nan")

    r2_components = []
    for i in range(y_true.shape[1]):
        component_res = float(np.nansum((y_pred[:, i] - y_true[:, i]) ** 2))
        component_centered = y_true[:, i] - np.nanmean(y_true[:, i])
        component_tot = float(np.nansum(component_centered**2))
        r2_components.append(
            1.0 - component_res / component_tot if component_tot > 0 else float("nan")
        )

    return {
        "rmse": float(np.nanmean(rmse_by_component)),
        "mae": float(np.nanmean(mae_by_component)),
        "r2": float(r2_overall),
        "rmse_by_component_json": vector_to_json(rmse_by_component),
        "mae_by_component_json": vector_to_json(mae_by_component),
        "r2_by_component_json": vector_to_json(np.array(r2_components)),
        "target_component_order": ",".join(target_columns),
    }


def compare_prediction_models(prediction_results: pd.DataFrame) -> pd.DataFrame:
    if prediction_results.empty:
        return pd.DataFrame()
    index_columns = [
        "particle_a",
        "particle_b",
        "target_particle",
        "other_particle",
        "lag_frames",
        "lag_s",
        "target_vector",
    ]
    self_model = prediction_results.loc[
        prediction_results["model"] == "self_only",
        [*index_columns, "rmse", "mae", "r2", "n_train", "n_test"],
    ].rename(
        columns={
            "rmse": "self_only_rmse",
            "mae": "self_only_mae",
            "r2": "self_only_r2",
            "n_train": "n_train",
            "n_test": "n_test",
        }
    )
    full_model = prediction_results.loc[
        prediction_results["model"] == "self_plus_other_relative",
        [*index_columns, "rmse", "mae", "r2"],
    ].rename(
        columns={
            "rmse": "self_plus_other_relative_rmse",
            "mae": "self_plus_other_relative_mae",
            "r2": "self_plus_other_relative_r2",
        }
    )
    comparison = self_model.merge(full_model, on=index_columns)
    if comparison.empty:
        return comparison
    comparison["rmse_reduction"] = (
        comparison["self_only_rmse"] - comparison["self_plus_other_relative_rmse"]
    )
    comparison["rmse_reduction_fraction"] = comparison["rmse_reduction"] / comparison[
        "self_only_rmse"
    ].replace(0, np.nan)
    comparison["r2_gain"] = (
        comparison["self_plus_other_relative_r2"] - comparison["self_only_r2"]
    )
    comparison["other_particle_helped"] = (
        (comparison["rmse_reduction"] > 0) & (comparison["r2_gain"] > 0)
    )
    return comparison


def estimate_frame_dt_s(data: pd.DataFrame) -> float:
    values = data[["frame", "time_s"]].dropna().drop_duplicates("frame")
    if len(values) < 2:
        return float("nan")
    values = values.sort_values("frame")
    frame_delta = values["frame"].diff().dropna()
    time_delta = values["time_s"].diff().dropna()
    valid = frame_delta != 0
    if not valid.any():
        return float("nan")
    return float((time_delta[valid] / frame_delta[valid]).median())


def vector_to_json(values: np.ndarray) -> str:
    clean = []
    for value in np.asarray(values, dtype=float):
        clean.append(float(value) if np.isfinite(value) else None)
    return json.dumps(clean, separators=(",", ":"))


def make_output_dir(
    csv_path: Path,
    output_parent: Path,
    output_name: str | None,
) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_name:
        base_name = sanitize_output_name(output_name)
    else:
        source_name = sanitize_output_name(csv_path.parent.name or csv_path.stem)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"state_vectors_{source_name}_{timestamp}"
    output_dir = output_parent / base_name
    if not output_dir.exists():
        return output_dir
    suffix = 2
    while True:
        candidate = output_parent / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def write_metadata(
    output_path: Path,
    source_csv: Path,
    args: argparse.Namespace,
    selected_particles: tuple[int, ...],
    particle_states: pd.DataFrame,
    pair_states: pd.DataFrame,
    relative_states: pd.DataFrame,
    prediction_dataset: pd.DataFrame,
    prediction_results: pd.DataFrame,
) -> None:
    single_particle_mode = len(selected_particles) == 1
    prediction_enabled = not args.no_prediction and not single_particle_mode
    metadata = {
        "source_csv": str(source_csv.resolve()),
        "selected_particles": [int(particle) for particle in selected_particles],
        "analysis_mode": "single_particle" if single_particle_mode else "multi_particle",
        "pair_outputs_written": not single_particle_mode,
        "canonical_state_source": args.state_source,
        "state_vector_order": list(STATE_COLUMNS),
        "raw_state_vector_order": list(RAW_STATE_COLUMNS),
        "smooth_state_vector_order": list(SMOOTH_STATE_COLUMNS),
        "relative_state_definition": "other/self or particle_b_minus_particle_a as labeled",
        "relative_state_order": list(RELATIVE_COLUMNS),
        "raw_relative_state_order": list(RAW_RELATIVE_COLUMNS),
        "smooth_relative_state_order": list(SMOOTH_RELATIVE_COLUMNS),
        "prediction_enabled": prediction_enabled,
        "prediction_skip_reason": (
            "single-particle mode has no cross-particle predictor"
            if single_particle_mode
            else ("disabled by --no-prediction" if args.no_prediction else None)
        ),
        "prediction_models": [
            "self_only",
            "self_plus_other_relative",
        ],
        "prediction_interpretation": (
            "If self_plus_other_relative has lower RMSE and higher R2 "
            "than self_only, the other particle carried useful "
            "predictive information for that lag/target."
        ),
        "max_prediction_lag_frames": int(args.max_prediction_lag_frames),
        "include_zero_lag": bool(args.include_zero_lag),
        "ridge_alpha": float(args.ridge_alpha),
        "test_fraction": float(args.test_fraction),
        "min_samples": int(args.min_samples),
        "particle_state_rows": int(len(particle_states)),
        "pair_state_rows": int(len(pair_states)),
        "relative_state_rows": int(len(relative_states)),
        "prediction_dataset_rows": int(len(prediction_dataset)),
        "prediction_result_rows": int(len(prediction_results)),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            to_jsonable(metadata),
            f,
            indent=2,
        )
        f.write("\n")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(to_jsonable(key)): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return json_default(value)


def json_default(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def plot_prediction_summary(comparison: pd.DataFrame, output_dir: Path) -> None:
    if comparison.empty:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_data = comparison.loc[
        comparison["target_vector"] == "future_motion_state"
    ].copy()
    if plot_data.empty:
        plot_data = comparison.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for (target_particle, target_vector), group in plot_data.groupby(
        ["target_particle", "target_vector"],
        sort=True,
    ):
        grouped = group.groupby("lag_frames", as_index=False)[
            "rmse_reduction_fraction"
        ].mean()
        ax.plot(
            grouped["lag_frames"],
            grouped["rmse_reduction_fraction"],
            label=f"target {target_particle} {target_vector}",
        )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("prediction lag, frames")
    ax.set_ylabel("RMSE reduction fraction")
    ax.set_title("Positive values mean the other particle improved prediction")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_rmse_improvement_by_lag.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
