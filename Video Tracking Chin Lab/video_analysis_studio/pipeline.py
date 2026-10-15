"""Command planning for the additive video-analysis studio.

The original scripts remain the source of truth.  This module only constructs
explicit command lines and predicts the artifacts those scripts will write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Iterable
from uuid import uuid4


TRIAL_PATTERN = re.compile(r"(?<![A-Z0-9])(MI|SP)\s*[-_ ]?(\d+)", re.IGNORECASE)


def infer_trial_id(value: str | Path) -> str:
    """Return a normalized MI/SP trial identifier when one is present."""

    match = TRIAL_PATTERN.search(Path(value).stem if isinstance(value, Path) else str(value))
    if not match:
        return ""
    return f"{match.group(1).upper()}{int(match.group(2))}"


def slugify(value: str, fallback: str = "trial") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or fallback


def new_run_token(trial_name: str, stage: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return slugify(f"{trial_name}_{stage}_{stamp}_{uuid4().hex[:6]}")


@dataclass(slots=True)
class CommandStep:
    step_id: str
    label: str
    command: list[str]
    artifacts: dict[str, str | list[str]] = field(default_factory=dict)
    interaction_note: str = ""


@dataclass(slots=True)
class PipelinePlan:
    action: str
    label: str
    steps: list[CommandStep]


class PipelinePlanner:
    """Build subprocess plans without changing any original script behavior."""

    def __init__(self, project_root: Path, python_executable: Path) -> None:
        self.root = project_root.resolve()
        # Do not resolve a virtual-environment interpreter symlink to its base
        # Python executable: doing so drops the venv's installed dependencies.
        self.python = python_executable.absolute()

    def build(self, action: str, session: dict[str, Any]) -> PipelinePlan:
        trial = _mapping(session.get("trial"), "trial")
        artifacts = _mapping(session.get("artifacts", {}), "artifacts")
        results = _mapping(session.get("results", {}), "results")
        name = str(trial.get("name") or infer_trial_id(str(trial.get("video_path", ""))) or "trial")
        token = new_run_token(name, action)

        if action in {"preview", "preview_full"}:
            full = action == "preview_full"
            label = "Full background-subtraction preview" if full else "Background-subtraction preview"
            return PipelinePlan(action, label, [self._preview(trial, token, full=full)])
        if action == "track":
            track_step = self._track(trial, token)
            tracks_path = Path(str(track_step.artifacts["tracks_csv"]))
            verify_step = self._verify_tracking(
                trial,
                artifacts,
                f"{token}_overlay",
                tracks_override=tracks_path,
            )
            return PipelinePlan(
                action,
                "Track particles and render video overlays",
                [track_step, verify_step],
            )
        if action == "verify_tracking":
            return PipelinePlan(
                action,
                "Tracking verification overlays",
                [self._verify_tracking(trial, artifacts, token)],
            )
        if action == "review_identities":
            return PipelinePlan(
                action,
                "Particle identity review",
                [self._review_identities(trial, artifacts, token)],
            )
        if action in {"translational", "rotational", "both"}:
            return self._finish_plan(action, trial, artifacts, token)
        if action == "orbit_analysis":
            return PipelinePlan(
                action,
                "Detect circular-motion episodes",
                [self._orbit_analysis(trial, results, token)],
            )
        raise ValueError(f"Unknown studio action: {action}")

    def _orbit_analysis(
        self,
        trial: dict[str, Any],
        settings: dict[str, Any],
        token: str,
    ) -> CommandStep:
        source_value = str(settings.get("source_csv") or "").strip()
        if not source_value:
            raise FileNotFoundError("Choose or import a position-versus-time CSV first.")
        source = Path(source_value).expanduser()
        if not source.is_absolute():
            source = self.root / source
        source = source.resolve()
        if not source.is_file() or source.suffix.lower() != ".csv":
            raise FileNotFoundError(f"Position CSV does not exist: {source}")

        detector = (
            self.root.parent
            / "analysis"
            / "orbit-detection"
            / "code"
            / "analyze_orbits_from_csv.py"
        )
        if not detector.is_file():
            raise FileNotFoundError(f"Circular-motion detector was not found: {detector}")

        output_dir = self.root / "outputs_background" / "orbit_results" / token
        label = str(settings.get("label") or trial.get("name") or source.stem).strip()
        command = [
            str(self.python),
            str(detector),
            str(source),
            "--output-dir",
            str(output_dir),
            "--label",
            label,
        ]
        video_value = str(settings.get("video_path") or "").strip()
        if video_value:
            video = Path(video_value).expanduser()
            if not video.is_absolute():
                video = self.root / video
            video = video.resolve()
            if not video.is_file():
                raise FileNotFoundError(f"Companion video does not exist: {video}")
            command += ["--video", str(video)]
        for flag, key in (
            ("--fps", "fps"),
            ("--particle", "particle"),
            ("--time-col", "time_col"),
            ("--frame-col", "frame_col"),
            ("--x-col", "x_col"),
            ("--y-col", "y_col"),
            ("--z-col", "z_col"),
            ("--position-unit", "position_unit"),
            ("--band-low-hz", "band_low_hz"),
            ("--band-high-hz", "band_high_hz"),
            ("--threshold", "threshold"),
            ("--minimum-episode-s", "minimum_episode_s"),
            ("--merge-gap-s", "merge_gap_s"),
            ("--smooth-window-s", "smooth_window_s"),
            ("--radius-window-s", "radius_window_s"),
            ("--fit-window-s", "fit_window_s"),
            ("--fit-step-s", "fit_step_s"),
            ("--fit-minimum-turns", "fit_minimum_turns"),
            ("--fit-minimum-axis-ratio", "fit_minimum_axis_ratio"),
            (
                "--fit-maximum-normalized-radial-rmse",
                "fit_maximum_normalized_radial_rmse",
            ),
            (
                "--fit-minimum-direction-consistency",
                "fit_minimum_direction_consistency",
            ),
            (
                "--fit-minimum-accepted-fraction",
                "fit_minimum_accepted_fraction",
            ),
            ("--max-gap-samples", "max_gap_samples"),
        ):
            command += _optional_arg(flag, settings.get(key))
        if _truthy(settings.get("rms_only")):
            command.append("--rms-only")

        output_artifacts: dict[str, str | list[str]] = {
            "orbit_intervals_csv": str(output_dir / "orbit_intervals.csv"),
            "orbit_timeseries_csv": str(output_dir / "orbit_timeseries.csv"),
            "orbit_summary_json": str(output_dir / "orbit_summary.json"),
            "orbit_diagnostic_plot": str(output_dir / "orbit_diagnostic.png"),
            "orbit_xy_plot": str(output_dir / "orbit_xy.png"),
            "orbit_report_html": str(output_dir / "orbit_report.html"),
        }
        if video_value and Path(video_value).suffix.lower() not in {".mp4", ".m4v"}:
            output_artifacts["orbit_source_video"] = str(output_dir / "orbit_source_video.mp4")
        return CommandStep(
            "orbit_analysis",
            "Analyze circular motion",
            command,
            output_artifacts,
        )

    def _finish_plan(
        self,
        action: str,
        trial: dict[str, Any],
        artifacts: dict[str, Any],
        token: str,
    ) -> PipelinePlan:
        steps: list[CommandStep] = []
        tracks = self._current_tracks(artifacts)
        preview = self._required_artifact(artifacts, "preview_video", "Run or select a preview first.")

        motion_path = _existing_path(artifacts.get("motion_csv"))
        if motion_path is None:
            motion_step = self._motion(trial, tracks, f"{token}_motion")
            steps.append(motion_step)
            motion_path = Path(str(motion_step.artifacts["motion_csv"]))

        if action in {"translational", "both"}:
            state_step = self._state_vectors(trial, motion_path, f"{token}_state")
            steps.append(state_step)
            state_path = Path(str(state_step.artifacts["translational_state_csv"]))
            steps.append(
                self._translational_overlay(
                    trial,
                    preview,
                    tracks,
                    state_path,
                    f"{token}_translational",
                )
            )

        if action in {"rotational", "both"}:
            rotation_step = self._rotation(trial, tracks, motion_path, preview, f"{token}_rotation")
            steps.append(rotation_step)
            rotation_path = Path(str(rotation_step.artifacts["rotation_state_csv"]))
            steps.append(
                self._rotation_overlay(
                    trial,
                    preview,
                    motion_path,
                    rotation_path,
                    f"{token}_rotational",
                )
            )

        labels = {
            "translational": "Finish translational overlays",
            "rotational": "Finish rotational overlay",
            "both": "Finish all state-vector overlays",
        }
        return PipelinePlan(action, labels[action], steps)

    def _preview(self, trial: dict[str, Any], token: str, *, full: bool = False) -> CommandStep:
        video = self._required_trial_file(trial, "video_path", "Choose a source video.")
        background = self._required_trial_file(trial, "background_path", "Choose a background image or video.")
        calibration = self._trial_path(trial, "calibration_path")
        settings = _mapping(trial.get("preview", {}), "preview settings")
        output_dir = self.root / "outputs_background" / "previews" / token
        output = output_dir / f"{slugify(video.stem)}_preview.mp4"
        command = self._script("background_subtraction_preview.py", "--video", video, "--background-file", background, "--output", output)
        if calibration:
            command += ["--calibration", str(calibration)]
        if _truthy(settings.get("force_calibration")):
            command.append("--force-calibration")
        command += self._common_subtraction_args(trial, settings)
        start_frame = settings.get("start_frame")
        seed_frame = settings.get("seed_frame")
        if str(settings.get("focus_box_mode") or "follow") == "follow":
            try:
                if seed_frame is not None and int(start_frame) < int(seed_frame):
                    start_frame = seed_frame
            except (TypeError, ValueError):
                start_frame = seed_frame
        command += _optional_arg("--start-frame", start_frame)
        command += _optional_arg("--max-frames", 0 if full else settings.get("max_frames"))
        command += _optional_arg("--particle-window-radius-px", settings.get("particle_window_radius_px"))
        command += _optional_arg("--outside-window-display-threshold", settings.get("outside_window_display_threshold"))
        command += _optional_arg("--display-threshold", settings.get("display_threshold"))
        command += _optional_arg("--cleanup-pixels", settings.get("cleanup_pixels"))
        command += _optional_arg("--fixed-scale", settings.get("fixed_scale"))
        return CommandStep(
            "preview",
            "Create preview",
            command,
            {
                "preview_video": str(output),
                "preview_metadata": str(output.with_name(f"{output.stem}_metadata.json")),
            },
            "The original calibration, focus-box, and seed-click windows appear when their coordinates are not supplied.",
        )

    def _track(self, trial: dict[str, Any], token: str) -> CommandStep:
        video = self._required_trial_file(trial, "video_path", "Choose a source video.")
        background = self._required_trial_file(trial, "background_path", "Choose a background image or video.")
        calibration = self._required_trial_file(trial, "calibration_path", "Choose or create a calibration first.")
        settings = _mapping(trial.get("tracking", {}), "tracking settings")
        parent = self.root / "outputs_background" / "tracks"
        output_dir = parent / token
        command = self._script(
            "background_subtraction_tracker.py",
            "--video",
            video,
            "--background-file",
            background,
            "--calibration",
            calibration,
            "--output-dir",
            parent,
            "--output-name",
            token,
        )
        command += self._common_subtraction_args(trial, settings)
        command += ["--fps", str(self._required_fps(trial))]
        command += _optional_arg("--max-frames", settings.get("max_frames"))
        command += _optional_arg(
            "--missing-search-expansion-px",
            settings.get("missing_search_expansion_px"),
        )
        command += _optional_arg("--manual-correction-after-misses", settings.get("manual_correction_after_misses"))
        command += _optional_arg("--manual-correction-close-px", settings.get("manual_correction_close_px"))
        command += _optional_arg("--front-overlap-frames", settings.get("front_overlap_frames"))
        command += _optional_arg("--side-overlap-frames", settings.get("side_overlap_frames"))
        command += _optional_arg("--identity-authority", settings.get("identity_authority"))
        command += _optional_arg("--z-source", settings.get("z_source"))
        command += _optional_arg("--max-jump-mm", settings.get("max_jump_mm"))
        command += _optional_arg("--smooth-window", settings.get("smooth_window"))
        command += _optional_arg("--smooth-polyorder", settings.get("smooth_polyorder"))
        if _truthy(settings.get("z_pairing_check")):
            command.append("--z-pairing-check")
        if _truthy(settings.get("no_manual_correction")):
            command.append("--no-manual-correction")
        return CommandStep(
            "track",
            "Track particles",
            command,
            {
                "tracks_csv": str(output_dir / "tracks_3d.csv"),
                "track_summary": str(output_dir / "summary.json"),
                "track_plot": str(output_dir / "track_projections.png"),
            },
            "The original seed-click and lost-particle correction windows remain available while this job runs.",
        )

    def _verify_tracking(
        self,
        trial: dict[str, Any],
        artifacts: dict[str, Any],
        token: str,
        tracks_override: Path | None = None,
    ) -> CommandStep:
        source_video = self._required_trial_file(trial, "video_path", "Choose a source video.")
        tracks = tracks_override or self._current_tracks(artifacts)
        calibration = self._required_trial_file(trial, "calibration_path", "Choose a calibration.")
        parent = self.root / "outputs_background" / "verification"
        output_dir = parent / token
        base = slugify(source_video.stem)
        settings = _mapping(trial.get("verification", {}), "verification settings")
        tracking_settings = _mapping(trial.get("tracking", {}), "tracking settings")
        command = self._script(
            "verify_tracking_overlay.py",
            "--preview",
            source_video,
            "--csv",
            tracks,
            "--calibration",
            calibration,
            "--output-dir",
            parent,
            "--output-name",
            token,
            "--source-layout",
            "combined",
        )
        command += _optional_arg("--preview-start-frame", tracking_settings.get("seed_frame", 1))
        command += _optional_arg("--trail-length", settings.get("trail_length"))
        frame_step = settings.get("frame_step")
        if frame_step in (None, ""):
            fps = self._required_fps(trial)
            frame_step = max(1, math.ceil(fps / 60.0))
        command += _optional_arg("--preview-frame-step", frame_step)
        position_source = str(settings.get("position_source") or "smoothed")
        command += ["--position-source", position_source]
        command += _optional_arg(
            "--max-frames",
            settings.get("max_frames") or tracking_settings.get("max_frames"),
        )
        overlay_artifacts: dict[str, str | list[str]] = {}
        if position_source in {"raw", "both"}:
            overlay_artifacts["track_verification_raw"] = str(
                output_dir / f"{base}_verification_raw.mp4"
            )
        if position_source in {"smoothed", "both"}:
            overlay_artifacts["track_verification_smoothed"] = str(
                output_dir / f"{base}_verification_smoothed.mp4"
            )
        return CommandStep(
            "verify_tracking",
            "Render tracking overlays",
            command,
            overlay_artifacts,
        )

    def _review_identities(
        self,
        trial: dict[str, Any],
        artifacts: dict[str, Any],
        token: str,
    ) -> CommandStep:
        tracks = self._current_tracks(artifacts)
        settings = _mapping(trial.get("identity_review", {}), "identity review settings")
        mode = str(settings.get("mode") or "manual")
        command = self._script(
            "detect_identity_switches.py",
            "--csv",
            tracks,
            "--mode",
            mode,
            "--output-name",
            token,
        )
        if _truthy(settings.get("include_auto_candidates")):
            command.append("--include-auto-candidates")
        command += _optional_arg("--review-frames", settings.get("review_frames"))
        command += _optional_arg("--review-frame-step", settings.get("review_frame_step"))
        command += _optional_arg("--jump-threshold-mm", settings.get("jump_threshold_mm"))
        command += _optional_arg("--improvement-ratio", settings.get("improvement_ratio"))

        trial_id = infer_trial_id(tracks)
        folder = tracks.resolve().parent / "corrections" / (token if not trial_id or token.upper().startswith(f"{trial_id}_") else f"{trial_id}_{token}")
        corrected_stem = f"{tracks.stem}_identity_corrected"
        if trial_id and not corrected_stem.upper().startswith(f"{trial_id}_"):
            corrected_stem = f"{trial_id}_{corrected_stem}"
        return CommandStep(
            "review_identities",
            "Review particle identities",
            command,
            {"corrected_tracks_csv": str(folder / f"{corrected_stem}.csv")},
            "Manual mode uses the original identity-review window. Report mode writes candidates without stopping.",
        )

    def _motion(self, trial: dict[str, Any], tracks: Path, token: str) -> CommandStep:
        parent = self.root / "outputs_background" / "motion"
        output_dir = parent / token
        settings = _mapping(trial.get("motion", {}), "motion settings")
        command = self._script(
            "calculate_motion.py",
            "--csv",
            tracks,
            "--fps",
            self._required_fps(trial),
            "--output-dir",
            parent,
            "--output-name",
            token,
        )
        command += _optional_arg("--smooth-window", settings.get("smooth_window"))
        command += _optional_arg("--smooth-polyorder", settings.get("smooth_polyorder"))
        return CommandStep(
            "motion",
            "Calculate motion",
            command,
            {"motion_csv": str(output_dir / "velocities_accelerations.csv")},
        )

    def _state_vectors(self, trial: dict[str, Any], motion: Path, token: str) -> CommandStep:
        parent = self.root / "outputs_background" / "state_vectors"
        output_dir = parent / token
        settings = _mapping(trial.get("translational", {}), "translational settings")
        command = self._script(
            "build_state_vectors.py",
            "--csv",
            motion,
            "--fps",
            self._required_fps(trial),
            "--state-source",
            settings.get("canonical_source", "raw"),
            "--output-dir",
            parent,
            "--output-name",
            token,
        )
        particles = str(settings.get("particles") or "").strip()
        if particles:
            command += ["--particles", particles]
        if not _truthy(settings.get("prediction", False)):
            command.append("--no-prediction")
        return CommandStep(
            "state_vectors",
            "Build translational state vectors",
            command,
            {"translational_state_csv": str(output_dir / "particle_state_vectors.csv")},
        )

    def _translational_overlay(
        self,
        trial: dict[str, Any],
        preview: Path,
        tracks: Path,
        state_csv: Path,
        token: str,
    ) -> CommandStep:
        calibration = self._required_trial_file(trial, "calibration_path", "Choose a calibration.")
        parent = self.root / "outputs_background" / "state_vector_overlays"
        output_dir = parent / token
        base = slugify(preview.stem)
        settings = _mapping(trial.get("translational", {}), "translational settings")
        source = str(settings.get("overlay_source") or "smoothed")
        command = self._script(
            "verify_state_vectors_overlay.py",
            "--preview",
            preview,
            "--tracks-csv",
            tracks,
            "--state-csv",
            state_csv,
            "--calibration",
            calibration,
            "--output-dir",
            parent,
            "--output-name",
            token,
            "--state-source",
            source,
        )
        command += _optional_arg("--preview-start-frame", settings.get("preview_start_frame", 1))
        command += _optional_arg("--vectors", settings.get("vectors", "both"))
        command += _optional_arg("--max-frames", settings.get("max_frames"))
        sources = ["raw", "smoothed"] if source == "both" else [source]
        videos = [
            str(output_dir / f"{base}_state_vectors_{item}_{mode}.mp4")
            for item in sources
            for mode in ("scaled", "normalized")
        ]
        return CommandStep(
            "translational_overlay",
            "Render translational overlays",
            command,
            {"translational_overlays": videos},
        )

    def _rotation(
        self,
        trial: dict[str, Any],
        tracks: Path,
        motion: Path,
        preview: Path,
        token: str,
    ) -> CommandStep:
        video = self._required_trial_file(trial, "video_path", "Choose a source video.")
        background = self._required_trial_file(trial, "background_path", "Choose a background.")
        calibration = self._required_trial_file(trial, "calibration_path", "Choose a calibration.")
        parent = self.root / "outputs_background" / "rotation"
        output_dir = parent / token
        settings = _mapping(trial.get("rotation", {}), "rotation settings")
        command = self._script(
            "estimate_particle_rotation.py",
            "--video",
            video,
            "--tracks-csv",
            tracks,
            "--motion-csv",
            motion,
            "--calibration",
            calibration,
            "--background-file",
            background,
            "--preview",
            preview,
            "--fps",
            self._required_fps(trial),
            "--output-dir",
            parent,
            "--output-name",
            token,
            "--no-overlay",
            "--no-preview-overlay",
        )
        for flag, key in (
            ("--crop-width-px", "crop_width_px"),
            ("--crop-height-px", "crop_height_px"),
            ("--front-crop-width-px", "front_crop_width_px"),
            ("--front-crop-height-px", "front_crop_height_px"),
            ("--side-crop-width-px", "side_crop_width_px"),
            ("--side-crop-height-px", "side_crop_height_px"),
            ("--threshold-fraction", "threshold_fraction"),
            ("--front-threshold-fraction", "front_threshold_fraction"),
            ("--side-threshold-fraction", "side_threshold_fraction"),
            ("--minimum-anisotropy", "minimum_anisotropy"),
            ("--front-minimum-anisotropy", "front_minimum_anisotropy"),
            ("--side-minimum-anisotropy", "side_minimum_anisotropy"),
            ("--max-frames", "max_frames"),
        ):
            command += _optional_arg(flag, settings.get(key))
        command += _optional_arg("--channel", settings.get("channel"))
        command += _optional_arg("--difference-mode", settings.get("difference_mode"))
        return CommandStep(
            "rotation",
            "Estimate particle rotation",
            command,
            {
                "rotation_state_csv": str(output_dir / "rotation_state_vectors.csv"),
                "reduced_rotation_state_csv": str(output_dir / "particle_reduced_state_vectors.csv"),
                "rotation_diagnostics": str(output_dir / "rotation_diagnostics.png"),
            },
        )

    def _rotation_overlay(
        self,
        trial: dict[str, Any],
        preview: Path,
        motion: Path,
        rotation: Path,
        token: str,
    ) -> CommandStep:
        calibration = self._required_trial_file(trial, "calibration_path", "Choose a calibration.")
        state_parent = self.root / "outputs_background" / "rotation_state_vectors"
        overlay_parent = self.root / "outputs_background" / "rotation_overlays"
        state_dir = state_parent / token
        overlay_dir = overlay_parent / token
        base = slugify(preview.stem)
        settings = _mapping(trial.get("rotation", {}), "rotation settings")
        command = self._script(
            "build_rotation_state_vectors_with_overlay.py",
            "--rotation-csv",
            rotation,
            "--motion-csv",
            motion,
            "--preview",
            preview,
            "--calibration",
            calibration,
            "--output-dir",
            state_parent,
            "--overlay-output-dir",
            overlay_parent,
            "--output-name",
            token,
            "--motion-vector-mode",
            settings.get("motion_vector_mode", "normalized"),
        )
        command += _optional_arg("--preview-start-frame", settings.get("preview_start_frame", 1))
        command += _optional_arg("--axes", settings.get("axes", "reconstructed-3d"))
        command += _optional_arg("--max-frames", settings.get("overlay_max_frames"))
        if not _truthy(settings.get("draw_omega", True)):
            command.append("--no-omega")
        if _truthy(settings.get("no_zoom_panels", False)):
            command.append("--no-zoom-panels")
        return CommandStep(
            "rotation_overlay",
            "Render rotational state-vector overlay",
            command,
            {
                "reduced_rotation_state_csv": str(state_dir / "particle_reduced_state_vectors.csv"),
                "rotation_overlay": str(overlay_dir / f"{base}_rotation_state_vectors_overlay.mp4"),
            },
        )

    def _common_subtraction_args(self, trial: dict[str, Any], settings: dict[str, Any]) -> list[str]:
        args: list[str] = []
        for flag, value in (
            ("--channel", settings.get("channel", "green")),
            ("--difference-mode", settings.get("difference_mode", "abs")),
            ("--background-frame", settings.get("background_frame", 0)),
            ("--seed-frame", settings.get("seed_frame", 1)),
            ("--particle-count", trial.get("particle_count", 2)),
            ("--focus-box-mode", settings.get("focus_box_mode", "follow")),
            ("--search-radius-px", settings.get("search_radius_px", 12)),
            ("--threshold-fraction", settings.get("threshold_fraction", 0.35)),
            ("--min-difference", settings.get("min_difference", 4)),
            ("--memory", settings.get("memory", 120)),
        ):
            args += _optional_arg(flag, value)
        for flag, key in (
            ("--front-focus-box", "front_focus_box"),
            ("--side-focus-box", "side_focus_box"),
            ("--front-seeds", "front_seeds"),
            ("--side-seeds", "side_seeds"),
        ):
            args += _optional_arg(flag, settings.get(key))
        if _truthy(settings.get("no_focus_box")):
            args.append("--no-focus-box")
        return args

    def _required_fps(self, trial: dict[str, Any]) -> float:
        value = trial.get("fps")
        try:
            fps = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Confirm the source frame rate before calculating motion or state vectors."
            ) from exc
        if fps <= 0:
            raise ValueError(
                "Confirm a positive source frame rate before calculating motion or state vectors."
            )
        return fps

    def _script(self, script: str, *parts: Any) -> list[str]:
        command = [str(self.python), str(self.root / script)]
        command.extend(str(part) for part in parts)
        return command

    def _trial_path(self, trial: dict[str, Any], key: str) -> Path | None:
        value = str(trial.get(key) or "").strip()
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _required_trial_file(self, trial: dict[str, Any], key: str, message: str) -> Path:
        path = self._trial_path(trial, key)
        if path is None or not path.is_file():
            raise FileNotFoundError(message)
        return path

    def _required_artifact(self, artifacts: dict[str, Any], key: str, message: str) -> Path:
        path = _existing_path(artifacts.get(key))
        if path is None:
            raise FileNotFoundError(message)
        return path

    def _current_tracks(self, artifacts: dict[str, Any]) -> Path:
        corrected = _existing_path(artifacts.get("corrected_tracks_csv"))
        if corrected is not None:
            return corrected
        return self._required_artifact(artifacts, "tracks_csv", "Run or select tracking output first.")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _optional_arg(flag: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) and not value.strip():
        return []
    return [flag, str(value)]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _existing_path(value: Any) -> Path | None:
    if isinstance(value, list):
        return None
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    return path if path.is_file() else None


def flatten_artifact_paths(artifacts: dict[str, str | list[str]]) -> Iterable[Path]:
    for value in artifacts.values():
        if isinstance(value, list):
            yield from (Path(item) for item in value)
        else:
            yield Path(value)
