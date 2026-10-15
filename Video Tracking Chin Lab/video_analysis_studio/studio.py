"""Local single-tab controller for the state-vector overlay workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from typing import Any
from urllib.parse import parse_qs, quote, urlparse
from uuid import uuid4
import webbrowser

from .pipeline import PipelinePlan, PipelinePlanner, flatten_artifact_paths, infer_trial_id


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "studio_web"
SESSION_FILENAME = ".video_analysis_studio_session.json"
SESSION_VERSION = 2
MAX_LOG_CHARACTERS = 250_000
MAX_UPLOAD_BYTES = 100 * 1024 * 1024 * 1024


def launch_studio() -> int:
    project_root = PACKAGE_DIR.parent
    backend = StudioBackend(project_root)
    server = StudioHTTPServer(("127.0.0.1", 0), StudioRequestHandler, backend)
    host, port = server.server_address
    # Bind to the loopback interface for safety, but show the familiar hostname
    # in the browser. ``localhost`` and ``127.0.0.1`` resolve to the same Mac.
    url = f"http://localhost:{port}/?token={quote(server.access_token)}"
    print(f"Video Analysis Studio: {url}")
    threading.Timer(0.35, lambda: webbrowser.open(url, new=1)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


@dataclass(slots=True)
class StudioJob:
    job_id: str
    action: str
    label: str
    status: str = "queued"
    message: str = "Queued"
    current_step: int = 0
    step_count: int = 0
    current_step_label: str = ""
    started_at: str = field(default_factory=lambda: _now())
    finished_at: str | None = None
    log: str = ""
    artifacts: dict[str, str | list[str]] = field(default_factory=dict)
    command_preview: list[list[str]] = field(default_factory=list)
    cancel_requested: bool = False
    _process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "action": self.action,
            "label": self.label,
            "status": self.status,
            "message": self.message,
            "current_step": self.current_step,
            "step_count": self.step_count,
            "current_step_label": self.current_step_label,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log": self.log,
            "artifacts": self.artifacts,
            "command_preview": self.command_preview,
        }


@dataclass
class StudioBackend:
    project_root: Path
    jobs: dict[str, StudioJob] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    session_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.python_executable = _resolve_python(self.project_root)
        self.planner = PipelinePlanner(self.project_root, self.python_executable)

    @property
    def session_path(self) -> Path:
        return self.project_root / SESSION_FILENAME

    def bootstrap(self) -> dict[str, Any]:
        loaded = self.load_session()
        return {
            "version": SESSION_VERSION,
            "project_root": str(self.project_root),
            "python": str(self.python_executable),
            "session": loaded["session"],
            "session_source": loaded["source"],
            "library": self.scan_library(),
        }

    def default_session(self) -> dict[str, Any]:
        return {
            "version": SESSION_VERSION,
            "saved_at": None,
            "trial": {
                "name": "",
                "video_path": "",
                "background_path": "",
                "calibration_path": "",
                "fps": "",
                "media_info": {},
                "particle_count": 2,
                "preview": {
                    "channel": "green",
                    "difference_mode": "abs",
                    "background_frame": 0,
                    "start_frame": 1,
                    "seed_frame": 1,
                    "focus_box_mode": "follow",
                    "front_focus_box": "",
                    "side_focus_box": "",
                    "search_radius_px": 12,
                    "particle_window_radius_px": 48,
                    "threshold_fraction": 0.35,
                    "min_difference": 4,
                    "memory": 120,
                    "display_threshold": 8,
                    "outside_window_display_threshold": 12,
                    "cleanup_pixels": 0,
                    "fixed_scale": 40,
                    "max_frames": 300,
                    "force_calibration": False,
                    "no_focus_box": True,
                    "front_seeds": "",
                    "side_seeds": "",
                },
                "tracking": {
                    "channel": "green",
                    "difference_mode": "abs",
                    "background_frame": 0,
                    "seed_frame": 1,
                    "focus_box_mode": "follow",
                    "front_focus_box": "",
                    "side_focus_box": "",
                    "search_radius_px": 12,
                    "missing_search_expansion_px": 1.0,
                    "threshold_fraction": 0.35,
                    "min_difference": 4,
                    "memory": 120,
                    "max_frames": "",
                    "manual_correction_after_misses": 3,
                    "manual_correction_close_px": 0,
                    "front_overlap_frames": "",
                    "side_overlap_frames": "",
                    "identity_authority": "auto",
                    "z_pairing_check": False,
                    "z_source": "average",
                    "max_jump_mm": 0.15,
                    "smooth_window": 7,
                    "smooth_polyorder": 2,
                    "no_manual_correction": False,
                    "no_focus_box": True,
                    "front_seeds": "",
                    "side_seeds": "",
                },
                "verification": {
                    "preview_start_frame": 1,
                    "trail_length": 0,
                    "frame_step": "",
                    "max_frames": "",
                    "position_source": "raw",
                },
                "review": {
                    "raw_overlay_reviewed": False,
                    "raw_overlay_note": "",
                },
                "identity_review": {
                    "mode": "manual",
                    "include_auto_candidates": False,
                    "review_frames": "",
                    "review_frame_step": 1,
                    "jump_threshold_mm": 0.25,
                    "improvement_ratio": 1.5,
                },
                "motion": {"smooth_window": 7, "smooth_polyorder": 2},
                "translational": {
                    "canonical_source": "raw",
                    "overlay_source": "smoothed",
                    "vectors": "both",
                    "particles": "",
                    "prediction": False,
                    "preview_start_frame": 1,
                    "max_frames": "",
                },
                "rotation": {
                    "channel": "green",
                    "difference_mode": "abs",
                    "crop_width_px": 41,
                    "crop_height_px": 41,
                    "front_crop_width_px": "",
                    "front_crop_height_px": "",
                    "side_crop_width_px": "",
                    "side_crop_height_px": "",
                    "threshold_fraction": 0.35,
                    "front_threshold_fraction": "",
                    "side_threshold_fraction": "",
                    "minimum_anisotropy": 0.15,
                    "front_minimum_anisotropy": "",
                    "side_minimum_anisotropy": "",
                    "axes": "reconstructed-3d",
                    "motion_vector_mode": "normalized",
                    "draw_omega": True,
                    "no_zoom_panels": False,
                    "preview_start_frame": 1,
                    "max_frames": "",
                    "overlay_max_frames": "",
                },
            },
            "artifacts": {},
            "results": {
                "source_csv": "",
                "video_path": "",
                "label": "",
                "fps": "",
                "particle": "",
                "time_col": "",
                "frame_col": "",
                "x_col": "",
                "y_col": "",
                "z_col": "",
                "position_unit": "",
                "band_low_hz": 1.0,
                "band_high_hz": 4.0,
                "threshold": "auto",
                "minimum_episode_s": 2.0,
                "merge_gap_s": 0.75,
                "smooth_window_s": 0.10,
                "radius_window_s": 0.50,
                "fit_window_s": 2.0,
                "fit_step_s": 0.25,
                "fit_minimum_turns": 1.5,
                "fit_minimum_axis_ratio": 0.70,
                "fit_maximum_normalized_radial_rmse": 0.25,
                "fit_minimum_direction_consistency": 0.80,
                "fit_minimum_accepted_fraction": 0.60,
                "rms_only": False,
                "max_gap_samples": 2,
            },
            "stage_status": {},
            "history": [],
            "active_result": "",
        }

    def load_session(self) -> dict[str, Any]:
        with self.session_lock:
            if not self.session_path.is_file():
                return {"session": self.default_session(), "source": "default"}
            try:
                value = json.loads(self.session_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"session": self.default_session(), "source": "default"}
        if not isinstance(value, dict):
            return {"session": self.default_session(), "source": "default"}
        return {"session": _merge_defaults(self.default_session(), value), "source": "saved"}

    def save_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = _merge_defaults(self.default_session(), payload)
        session["version"] = SESSION_VERSION
        session["saved_at"] = _now()
        encoded = json.dumps(session, indent=2)
        if len(encoded) > 2_000_000:
            raise ValueError("Studio session is unexpectedly large.")
        temporary = self.session_path.with_suffix(".tmp")
        with self.session_lock:
            temporary.write_text(encoded + "\n", encoding="utf-8")
            temporary.replace(self.session_path)
        return {"saved_at": session["saved_at"], "session": session}

    def scan_library(self) -> dict[str, list[dict[str, Any]]]:
        groups = {
            "videos": self._scan_folder(self.project_root / "Videos", {".mp4", ".mkv", ".mov", ".avi"}),
            "backgrounds": self._scan_folder(
                self.project_root / "Backgrounds",
                {".mp4", ".mkv", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".tif", ".tiff"},
            ),
            "calibrations": self._scan_folder(self.project_root / "calibrations", {".json"}),
        }
        return groups

    def _scan_folder(self, folder: Path, suffixes: set[str]) -> list[dict[str, Any]]:
        if not folder.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in folder.rglob("*"):
            try:
                if not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                stat = path.stat()
            except OSError:
                continue
            items.append(
                {
                    "name": str(path.relative_to(folder)),
                    "path": str(path.resolve()),
                    "trial_id": infer_trial_id(path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )
        return sorted(items, key=lambda item: item["name"].lower())

    def inspect_media(self, raw_path: str) -> dict[str, Any]:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("Media inspection requires the installed PyAV package.") from exc
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("The selected file does not contain a video stream.")
            stream = container.streams.video[0]
            rate = float(stream.average_rate) if stream.average_rate else None
            duration_s = None
            if stream.duration is not None and stream.time_base is not None:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration_s = float(container.duration / av.time_base)
            frame_count = int(stream.frames) if stream.frames else None
            if frame_count is None and duration_s is not None and rate is not None:
                frame_count = int(round(duration_s * rate))
            return {
                "path": str(path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "width_px": int(stream.width),
                "height_px": int(stream.height),
                "fps": rate,
                "frame_count": frame_count,
                "duration_s": duration_s,
                "codec": str(stream.codec_context.name or ""),
            }

    def frame_jpeg(
        self,
        raw_path: str,
        frame_number: int,
        calibration_path: str = "",
        view: str = "full",
    ) -> bytes:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if frame_number < 0:
            raise ValueError("Frame number must be zero or positive.")
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("Frame extraction requires the installed PyAV package.") from exc
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("The selected file does not contain a video stream.")
            stream = container.streams.video[0]
            rate = float(stream.average_rate) if stream.average_rate else None
            if frame_number and rate and stream.time_base:
                target_pts = int((frame_number / rate) / float(stream.time_base))
                container.seek(max(0, target_pts), stream=stream, backward=True)
            selected = None
            for decoded_index, frame in enumerate(container.decode(stream)):
                if rate and frame.pts is not None and stream.time_base:
                    estimated_index = int(round(float(frame.pts * stream.time_base) * rate))
                else:
                    estimated_index = decoded_index
                selected = frame
                if estimated_index >= frame_number:
                    break
            if selected is None:
                raise ValueError(f"Frame {frame_number} is outside the video.")
            image = selected.to_image().convert("RGB")

        if view not in {"full", "front", "side"}:
            raise ValueError("View must be full, front, or side.")
        if view != "full":
            calibration = Path(calibration_path).expanduser().resolve()
            if not calibration.is_file():
                raise FileNotFoundError("Choose or create a calibration before selecting particles.")
            data = json.loads(calibration.read_text(encoding="utf-8"))
            roi = data.get(f"{view}_roi")
            if not isinstance(roi, dict):
                raise ValueError(f"Calibration does not contain {view}_roi.")
            x = int(roi["x"])
            y = int(roi["y"])
            width = int(roi["width"])
            height = int(roi["height"])
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"Calibration contains an invalid {view}_roi.")
            image = image.crop((x, y, x + width, y + height))

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
        return buffer.getvalue()

    def csv_preview(self, raw_path: str, limit: int = 200) -> dict[str, Any]:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".csv":
            raise FileNotFoundError(path)
        limit = min(max(int(limit), 1), 1000)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                columns = next(reader)
            except StopIteration:
                return {"path": str(path), "columns": [], "rows": [], "truncated": False}
            rows: list[list[str]] = []
            for index, row in enumerate(reader):
                if index >= limit:
                    return {
                        "path": str(path),
                        "columns": columns,
                        "rows": rows,
                        "truncated": True,
                    }
                rows.append(row)
        return {"path": str(path), "columns": columns, "rows": rows, "truncated": False}

    def track_quality(self, raw_path: str) -> dict[str, Any]:
        """Return descriptive tracking diagnostics without declaring scientific validity."""

        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".csv":
            raise FileNotFoundError(path)
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Track diagnostics require the installed pandas package.") from exc

        tracks = pd.read_csv(path)
        required = {"particle", "frame", "x_mm", "y_mm", "z_mm"}
        missing = sorted(required.difference(tracks.columns))
        if missing:
            raise ValueError(f"Track CSV is missing required columns: {missing}")
        if tracks.empty:
            raise ValueError("Track CSV is empty.")

        numeric_columns = [
            column
            for column in (
                "frame", "x_mm", "y_mm", "z_mm", "z_error_mm",
                "front_confidence", "side_confidence",
            )
            if column in tracks.columns
        ]
        for column in numeric_columns:
            tracks[column] = pd.to_numeric(tracks[column], errors="coerce")

        particle_summaries: list[dict[str, Any]] = []
        all_steps: list[np.ndarray] = []
        total_expected = 0
        total_observed = 0
        total_gaps = 0
        for particle, group in tracks.groupby("particle", sort=True):
            group = group.sort_values("frame")
            valid_frames = group["frame"].dropna().astype(int)
            if valid_frames.empty:
                continue
            unique_frames = np.sort(valid_frames.unique())
            expected = int(unique_frames[-1] - unique_frames[0] + 1)
            observed = int(len(unique_frames))
            gaps = max(0, expected - observed)
            xyz = group[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
            steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            steps = steps[np.isfinite(steps)]
            if steps.size:
                all_steps.append(steps)
            particle_summaries.append(
                {
                    "particle": _json_scalar(particle),
                    "first_frame": int(unique_frames[0]),
                    "last_frame": int(unique_frames[-1]),
                    "observed_frames": observed,
                    "missing_frames": gaps,
                    "coverage_fraction": observed / expected if expected else 1.0,
                }
            )
            total_expected += expected
            total_observed += observed
            total_gaps += gaps

        def finite_summary(column: str) -> dict[str, float | None] | None:
            if column not in tracks.columns:
                return None
            values = tracks[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                return None
            return {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }

        combined_steps = np.concatenate(all_steps) if all_steps else np.array([], dtype=float)
        missing_positions = int(tracks[["x_mm", "y_mm", "z_mm"]].isna().any(axis=1).sum())
        inside_fraction = None
        if "inside_chamber" in tracks.columns:
            inside = tracks["inside_chamber"].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
            if inside.notna().any():
                inside_fraction = float(inside.mean())

        return {
            "path": str(path),
            "row_count": int(len(tracks)),
            "particle_count": int(tracks["particle"].nunique(dropna=True)),
            "first_frame": int(tracks["frame"].min()),
            "last_frame": int(tracks["frame"].max()),
            "coverage_fraction": total_observed / total_expected if total_expected else None,
            "missing_frame_count": total_gaps,
            "missing_position_rows": missing_positions,
            "inside_chamber_fraction": inside_fraction,
            "z_error_mm": finite_summary("z_error_mm"),
            "front_confidence": finite_summary("front_confidence"),
            "side_confidence": finite_summary("side_confidence"),
            "step_mm": {
                "median": float(np.median(combined_steps)) if combined_steps.size else None,
                "p95": float(np.percentile(combined_steps, 95)) if combined_steps.size else None,
                "max": float(np.max(combined_steps)) if combined_steps.size else None,
            },
            "particles": particle_summaries,
            "interpretation": (
                "Descriptive diagnostics only. Review the overlay and establish dataset-specific "
                "acceptance thresholds before treating a track as valid."
            ),
        }

    def receive_upload(
        self,
        kind: str,
        original_name: str,
        size: int,
        source,
    ) -> dict[str, Any]:
        folders = {
            "video": "Videos",
            "background": "Backgrounds",
            "results_csv": "outputs_background/imported_results",
        }
        suffixes = {
            "video": {".mp4", ".mkv", ".mov", ".avi"},
            "background": {".mp4", ".mkv", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".tif", ".tiff"},
            "results_csv": {".csv"},
        }
        if kind not in folders:
            raise ValueError("Upload type must be video or background.")
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            raise ValueError("Upload size is invalid or exceeds 100 GB.")
        clean_name = Path(original_name).name
        supplied_suffix = Path(clean_name).suffix.lower()
        if supplied_suffix and supplied_suffix not in suffixes[kind]:
            raise ValueError(f"Unsupported {kind} file type: {supplied_suffix}")
        if not supplied_suffix and kind != "video":
            raise ValueError(f"Extensionless {kind.replace('_', ' ')} files are not supported.")

        header = source.read(min(size, 16))
        if not header:
            raise ValueError("The uploaded file is empty.")
        suffix = supplied_suffix or _extensionless_video_suffix(header)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(clean_name).stem).strip("._-") or kind
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_dir = self.project_root / folders[kind]
        if kind in {"video", "background"}:
            destination_dir /= "imported"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{stamp}_{uuid4().hex[:6]}_{stem}{suffix}"
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        remaining = size - len(header)
        try:
            with temporary.open("xb") as handle:
                handle.write(header)
                digest.update(header)
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Upload ended before the declared file size.")
                    handle.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        manifest_dir = self.project_root / "outputs_background" / "import_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{destination.name}.json"
        manifest = {
            "created_at": _now(),
            "original_name": clean_name,
            "imported_path": str(destination.resolve()),
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "kind": kind,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {"path": str(destination.resolve()), "manifest": str(manifest_path.resolve()), **manifest}

    def suggest_companions(self, video_path: str) -> dict[str, str]:
        path = Path(video_path).expanduser().resolve()
        trial_id = infer_trial_id(path)
        library = self.scan_library()
        calibration = _best_trial_match(library["calibrations"], trial_id)
        background = _best_trial_match(library["backgrounds"], trial_id)
        provenance = self._find_provenance_companions(trial_id)
        calibration = calibration or provenance.get("calibration_path", "")
        background = background or provenance.get("background_path", "")
        return {
            "trial_id": trial_id,
            "name": trial_id or path.stem,
            "calibration_path": calibration,
            "background_path": background,
            "provenance_source": provenance.get("provenance_source", ""),
            "calibration_status": provenance.get("calibration_status", ""),
        }

    def _find_provenance_companions(self, trial_id: str) -> dict[str, str]:
        if not trial_id:
            return {}
        summaries = self.project_root.glob("outputs*/**/summary.json")
        candidates: list[tuple[float, dict[str, str]]] = []
        for summary_path in summaries:
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or str(data.get("video_id") or "").upper() != trial_id:
                    continue
                background = Path(str(data.get("background") or "")).expanduser()
                calibration = Path(str(data.get("calibration_file_used") or "")).expanduser()
                candidate = {
                    "background_path": str(background.resolve()) if background.is_file() else "",
                    "calibration_path": str(calibration.resolve()) if calibration.is_file() else "",
                    "provenance_source": str(summary_path.resolve()),
                    "calibration_status": str(data.get("calibration_status") or ""),
                }
                if candidate["background_path"] or candidate["calibration_path"]:
                    candidates.append((summary_path.stat().st_mtime, candidate))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        if not candidates:
            return {}
        return max(candidates, key=lambda item: item[0])[1]

    def pick_path(self, kind: str) -> str:
        filters = {
            "video": ("Choose source video", "Video files|*.mp4;*.mkv;*.mov;*.avi|All files|*.*"),
            "background": (
                "Choose background image or video",
                "Media files|*.mp4;*.mkv;*.mov;*.avi;*.png;*.jpg;*.jpeg;*.tif;*.tiff|All files|*.*",
            ),
            "calibration": ("Choose calibration", "JSON files|*.json|All files|*.*"),
            "preview": ("Choose preview video", "Video files|*.mp4;*.mov;*.mkv|All files|*.*"),
            "tracks": ("Choose tracks CSV", "CSV files|*.csv|All files|*.*"),
            "motion": ("Choose motion CSV", "CSV files|*.csv|All files|*.*"),
            "state": ("Choose state-vector CSV", "CSV files|*.csv|All files|*.*"),
            "rotation": ("Choose rotation-state CSV", "CSV files|*.csv|All files|*.*"),
        }
        if kind not in filters:
            raise ValueError("Unknown file picker type.")
        title, filter_text = filters[kind]
        return _windows_open_dialog(title, filter_text, self.project_root)

    def start_job(self, action: str, session: dict[str, Any]) -> StudioJob:
        with self.lock:
            if any(job.status in {"queued", "running"} for job in self.jobs.values()):
                raise RuntimeError("Another studio job is already running.")
        saved = self.save_session(session)["session"]
        plan = self.planner.build(action, saved)
        job = StudioJob(
            job_id=uuid4().hex,
            action=action,
            label=plan.label,
            step_count=len(plan.steps),
            command_preview=[_display_command(step.command, self.project_root) for step in plan.steps],
        )
        with self.lock:
            self.jobs[job.job_id] = job
        threading.Thread(target=self._run_plan, args=(job, plan, saved), daemon=True).start()
        return job

    def _run_plan(self, job: StudioJob, plan: PipelinePlan, session: dict[str, Any]) -> None:
        with self.lock:
            job.status = "running"
            job.message = "Starting workflow"
        try:
            for index, step in enumerate(plan.steps, start=1):
                with self.lock:
                    if job.cancel_requested:
                        raise _Cancelled()
                    job.current_step = index
                    job.current_step_label = step.label
                    job.message = step.interaction_note or step.label
                    self._append_log(job, f"\n[{index}/{len(plan.steps)}] {step.label}\n")
                    self._append_log(job, " ".join(_display_command(step.command, self.project_root)) + "\n")
                self._run_step(job, step.command)
                missing = [str(path) for path in flatten_artifact_paths(step.artifacts) if not path.is_file()]
                if missing:
                    raise RuntimeError(
                        f"{step.label} exited successfully but did not create the expected output: {missing[0]}"
                    )
                with self.lock:
                    job.artifacts.update(step.artifacts)
            self._complete_job(job, session)
        except _Cancelled:
            with self.lock:
                job.status = "cancelled"
                job.message = "Cancelled"
                job.finished_at = _now()
        except Exception as exc:
            with self.lock:
                job.status = "error"
                job.message = str(exc)
                job.finished_at = _now()
                self._append_log(job, f"\nERROR: {exc}\n")

    def _run_step(self, job: StudioJob, command: list[str]) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            creationflags=flags,
        )
        with self.lock:
            job._process = process
        assert process.stdout is not None
        try:
            for line in process.stdout:
                with self.lock:
                    self._append_log(job, line)
                    cancel = job.cancel_requested
                if cancel:
                    process.terminate()
                    raise _Cancelled()
            return_code = process.wait()
        finally:
            process.stdout.close()
            if job.cancel_requested and process.poll() is None:
                process.terminate()
            if process.poll() is None:
                process.wait()
            with self.lock:
                job._process = None
        if job.cancel_requested:
            raise _Cancelled()
        if return_code != 0:
            raise RuntimeError(f"{job.current_step_label} failed with exit code {return_code}.")

    def _complete_job(self, job: StudioJob, session: dict[str, Any]) -> None:
        session = _merge_defaults(self.default_session(), session)
        manifest_path = self._write_run_manifest(job, session)
        job.artifacts["run_manifest"] = str(manifest_path)
        artifacts = session.setdefault("artifacts", {})
        artifacts.update(job.artifacts)
        if job.action in {"preview", "preview_full"}:
            self._apply_preview_metadata(session, job.artifacts)
        session["active_result"] = _preferred_result(job.artifacts) or str(session.get("active_result", ""))
        stage_status = session.setdefault("stage_status", {})
        stage_status[job.action] = {"status": "complete", "finished_at": _now(), "job_id": job.job_id}
        history = session.setdefault("history", [])
        history.append(
            {
                "job_id": job.job_id,
                "action": job.action,
                "label": job.label,
                "finished_at": _now(),
                "artifacts": job.artifacts,
            }
        )
        session["history"] = history[-50:]
        saved = self.save_session(session)["session"]
        with self.lock:
            job.status = "complete"
            job.message = "Workflow complete"
            job.finished_at = _now()
            self._append_log(job, "\nWorkflow complete.\n")

    def _write_run_manifest(self, job: StudioJob, session: dict[str, Any]) -> Path:
        manifest_dir = self.project_root / "outputs_background" / "studio_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{job.started_at[:19].replace(':', '')}_{job.action}_{job.job_id[:8]}.json"
        revision = ""
        dirty = None
        try:
            revision_result = subprocess.run(
                ["git", "-C", str(self.project_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if revision_result.returncode == 0:
                revision = revision_result.stdout.strip()
            dirty_result = subprocess.run(
                ["git", "-C", str(self.project_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            if dirty_result.returncode == 0:
                dirty = bool(dirty_result.stdout.strip())
        except OSError:
            pass
        trial_value = session.get("trial", {})
        trial = trial_value if isinstance(trial_value, dict) else {}
        results_value = session.get("results", {})
        results = results_value if isinstance(results_value, dict) else {}
        manifest = {
            "schema_version": 1,
            "job_id": job.job_id,
            "action": job.action,
            "label": job.label,
            "started_at": job.started_at,
            "finished_at": _now(),
            "source_video": trial.get("video_path", ""),
            "background": trial.get("background_path", ""),
            "calibration": trial.get("calibration_path", ""),
            "fps": trial.get("fps", ""),
            "particle_count": trial.get("particle_count", ""),
            "results_source_csv": results.get("source_csv", ""),
            "results_video": results.get("video_path", ""),
            "settings": {"trial": trial, "results": results},
            "commands": job.command_preview,
            "artifacts": job.artifacts,
            "code_revision": revision,
            "worktree_dirty": dirty,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path.resolve()

    def _apply_preview_metadata(
        self,
        session: dict[str, Any],
        artifacts: dict[str, str | list[str]],
    ) -> None:
        metadata_value = artifacts.get("preview_metadata")
        if not isinstance(metadata_value, str):
            return
        metadata_path = Path(metadata_value)
        if not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(metadata, dict):
            return
        trial = session.setdefault("trial", {})
        calibration = str(metadata.get("calibration_file_used") or "")
        if calibration:
            trial["calibration_path"] = calibration
        preview = trial.setdefault("preview", {})
        tracking = trial.setdefault("tracking", {})
        for key in ("channel", "difference_mode", "background_frame", "seed_frame", "focus_box_mode"):
            if key in metadata:
                preview[key] = metadata[key]
                tracking[key] = metadata[key]
        for metadata_key, setting_key in (
            ("front_focus_roi", "front_focus_box"),
            ("side_focus_roi", "side_focus_box"),
        ):
            value = metadata.get(metadata_key)
            if isinstance(value, dict):
                try:
                    encoded = ",".join(str(int(value[key])) for key in ("x", "y", "width", "height"))
                except (KeyError, TypeError, ValueError):
                    continue
                preview[setting_key] = encoded
                tracking[setting_key] = encoded

    def _append_log(self, job: StudioJob, text: str) -> None:
        job.log = (job.log + text)[-MAX_LOG_CHARACTERS:]

    def get_job(self, job_id: str) -> StudioJob:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("Unknown studio job.")
            return self.jobs[job_id]

    def cancel_job(self, job_id: str) -> StudioJob:
        with self.lock:
            job = self.get_job(job_id)
            if job.status not in {"queued", "running"}:
                return job
            job.cancel_requested = True
            job.message = "Cancelling…"
            process = job._process
        if process is not None and process.poll() is None:
            process.terminate()
        return job

    def open_path(self, raw_path: str) -> str:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return str(path)


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        backend: StudioBackend,
    ) -> None:
        self.backend = backend
        self.access_token = secrets.token_urlsafe(24)
        super().__init__(server_address, handler_class)


class StudioRequestHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/") and not self._authorized(parsed):
            self._json(HTTPStatus.FORBIDDEN, {"error": "Invalid studio access token."})
            return
        if parsed.path == "/api/bootstrap":
            self._json(HTTPStatus.OK, self.server.backend.bootstrap())
            return
        if parsed.path == "/api/session":
            self._json(HTTPStatus.OK, self.server.backend.load_session())
            return
        if parsed.path == "/api/library":
            self._json(HTTPStatus.OK, self.server.backend.scan_library())
            return
        if parsed.path == "/api/inspect":
            try:
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                self._json(HTTPStatus.OK, self.server.backend.inspect_media(raw_path))
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if parsed.path == "/api/frame":
            try:
                query = parse_qs(parsed.query)
                data = self.server.backend.frame_jpeg(
                    query.get("path", [""])[0],
                    int(query.get("frame", ["0"])[0]),
                    query.get("calibration", [""])[0],
                    query.get("view", ["full"])[0],
                )
                self._bytes(HTTPStatus.OK, data, "image/jpeg")
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if parsed.path == "/api/table":
            try:
                query = parse_qs(parsed.query)
                self._json(
                    HTTPStatus.OK,
                    self.server.backend.csv_preview(
                        query.get("path", [""])[0],
                        int(query.get("limit", ["200"])[0]),
                    ),
                )
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if parsed.path == "/api/quality":
            try:
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                self._json(HTTPStatus.OK, self.server.backend.track_quality(raw_path))
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if parsed.path == "/api/job":
            try:
                job_id = parse_qs(parsed.query).get("id", [""])[0]
                self._json(HTTPStatus.OK, self.server.backend.get_job(job_id).public())
            except KeyError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if parsed.path == "/api/media":
            try:
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                self._serve_media(Path(raw_path).expanduser().resolve())
            except Exception as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if parsed.path == "/sp6" or parsed.path.startswith("/sp6/"):
            relative = parsed.path.removeprefix("/sp6/") if parsed.path.startswith("/sp6/") else ""
            relative = relative or "sp6-orbit-viewer.html"
            root = (self.server.backend.project_root.parent / "orbit").resolve()
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                self._serve_media(target)
            except Exception:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/current-orbit" or parsed.path.startswith("/current-orbit/"):
            session = self.server.backend.load_session()["session"]
            report_value = session.get("artifacts", {}).get("orbit_report_html")
            if not report_value:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            report = Path(str(report_value)).expanduser().resolve()
            root = report.parent
            relative = (
                parsed.path.removeprefix("/current-orbit/")
                if parsed.path.startswith("/current-orbit/")
                else ""
            )
            target = report if not relative else (root / relative).resolve()
            try:
                target.relative_to(root)
                self._serve_media(target)
            except Exception:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        relative = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
        self._serve_static(relative)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorized(parsed):
            self._json(HTTPStatus.FORBIDDEN, {"error": "Invalid studio access token."})
            return
        try:
            if parsed.path == "/api/upload":
                query = parse_qs(parsed.query)
                size = int(self.headers.get("Content-Length", "0"))
                result = self.server.backend.receive_upload(
                    query.get("kind", [""])[0],
                    query.get("filename", [""])[0],
                    size,
                    self.rfile,
                )
                self._json(HTTPStatus.CREATED, result)
                return
            payload = self._read_json()
            if parsed.path == "/api/session":
                self._json(HTTPStatus.OK, self.server.backend.save_session(payload))
                return
            if parsed.path == "/api/pick":
                selected = self.server.backend.pick_path(str(payload.get("kind", "")))
                self._json(HTTPStatus.OK, {"path": selected})
                return
            if parsed.path == "/api/suggest":
                self._json(HTTPStatus.OK, self.server.backend.suggest_companions(str(payload.get("video_path", ""))))
                return
            if parsed.path == "/api/run":
                job = self.server.backend.start_job(str(payload.get("action", "")), _object(payload.get("session")))
                self._json(HTTPStatus.ACCEPTED, job.public())
                return
            if parsed.path == "/api/cancel":
                job = self.server.backend.cancel_job(str(payload.get("job_id", "")))
                self._json(HTTPStatus.OK, job.public())
                return
            if parsed.path == "/api/open":
                opened = self.server.backend.open_path(str(payload.get("path", "")))
                self._json(HTTPStatus.OK, {"opened": opened})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
        except Exception as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _authorized(self, parsed) -> bool:
        header = self.headers.get("X-Studio-Token", "")
        query = parse_qs(parsed.query).get("token", [""])[0]
        return secrets.compare_digest(header or query, self.server.access_token)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2_000_000:
            raise ValueError("Invalid request size.")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request body must be an object.")
        return value

    def _serve_static(self, relative: str) -> None:
        target = (STATIC_DIR / relative).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_media(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            first, _, last = range_header[6:].partition("-")
            if first:
                start = max(0, int(first))
            if last:
                end = min(size - 1, int(last))
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers routinely cancel an in-flight range request when
                    # the user seeks or a video element selects a newer range.
                    # That is a successful cancellation, not a media failure.
                    return
                remaining -= len(chunk)

    def _json(self, status: HTTPStatus, value: Any) -> None:
        data = json.dumps(value).encode("utf-8")
        self._bytes(status, data, "application/json; charset=utf-8")

    def _bytes(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


class _Cancelled(Exception):
    pass


def _resolve_python(project_root: Path) -> Path:
    candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv-macos" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    return Path(sys.executable).absolute()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_defaults(defaults: Any, value: Any) -> Any:
    if isinstance(defaults, dict):
        source = value if isinstance(value, dict) else {}
        merged = {key: _merge_defaults(default, source.get(key)) for key, default in defaults.items()}
        for key, item in source.items():
            if key not in merged:
                merged[key] = item
        return merged
    return defaults if value is None else value


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Session must be an object.")
    return value


def _json_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalar values to their plain Python equivalent."""

    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _best_trial_match(items: list[dict[str, Any]], trial_id: str) -> str:
    if not trial_id:
        return ""
    exact = [item for item in items if item.get("trial_id") == trial_id]
    if exact:
        return str(exact[0]["path"])
    # Shared backgrounds such as MI23-24B do not normalize to both IDs, so use
    # a conservative filename token fallback after exact matching.
    number = re.sub(r"^[A-Z]+", "", trial_id)
    prefix = re.sub(r"\d+$", "", trial_id)
    for item in items:
        stem = Path(str(item["name"])).stem.upper()
        if stem.startswith(prefix) and re.search(rf"(?<!\d){re.escape(number)}(?!\d)", stem):
            return str(item["path"])
    return ""


def _display_command(command: list[str], project_root: Path) -> list[str]:
    display: list[str] = []
    for part in command:
        try:
            path = Path(part)
            if path.is_absolute():
                display.append(str(path.relative_to(project_root)))
                continue
        except (OSError, ValueError):
            pass
        display.append(part)
    return display


def _preferred_result(artifacts: dict[str, str | list[str]]) -> str:
    for key in ("rotation_overlay", "translational_overlays", "track_verification_smoothed", "preview_video"):
        value = artifacts.get(key)
        if isinstance(value, list) and value:
            return value[-1]
        if isinstance(value, str) and value:
            return value
    return ""


def _windows_open_dialog(title: str, file_filter: str, initial_directory: Path) -> str:
    if os.name != "nt":
        raise RuntimeError("The studio file picker currently requires Windows.")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
        f"$dialog.Title = '{_powershell_literal(title)}'; "
        f"$dialog.Filter = '{_powershell_literal(file_filter)}'; "
        f"$dialog.InitialDirectory = '{_powershell_literal(str(initial_directory))}'; "
        "$dialog.Multiselect = $false; "
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ [Console]::Out.Write($dialog.FileName) }"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "File picker failed.")
    return completed.stdout.strip()


def _powershell_literal(value: str) -> str:
    return value.replace("'", "''")


def _extensionless_video_suffix(header: bytes) -> str:
    """Infer common extensionless video containers without changing the source."""

    if header.startswith(b"\x1aE\xdf\xa3"):
        return ".mkv"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return ".mp4"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return ".avi"
    raise ValueError(
        "This extensionless file is not a recognized Matroska, MP4/QuickTime, or AVI video."
    )
