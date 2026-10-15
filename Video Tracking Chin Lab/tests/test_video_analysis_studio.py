from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import numpy as np

from background_subtraction_tracker import add_time_seconds, parse_seed_points
from video_analysis_studio.pipeline import PipelinePlanner
from video_analysis_studio.studio import StudioBackend
from verify_tracking_overlay import compose_combined_panels


class SeedPointTests(unittest.TestCase):
    def test_parse_seed_points(self) -> None:
        self.assertEqual(parse_seed_points("12.5,9; 20,30.25"), ((12.5, 9.0), (20.0, 30.25)))

    def test_rejects_incomplete_point(self) -> None:
        with self.assertRaises(Exception):
            parse_seed_points("12.5")

    def test_confirmed_fps_adds_explicit_time_column(self) -> None:
        tracks = pd.DataFrame({"particle": [0, 0], "frame": [120, 121], "x_mm": [1.0, 2.0]})
        timed = add_time_seconds(tracks, 120.0)
        self.assertEqual(list(timed.columns), ["particle", "frame", "time_s", "x_mm"])
        self.assertAlmostEqual(timed.loc[0, "time_s"], 1.0)
        self.assertNotIn("time_s", tracks.columns)


class OverlayTests(unittest.TestCase):
    def test_combined_source_is_cropped_into_front_and_side_panels(self) -> None:
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[:, :4] = (255, 0, 0)
        frame[:, 4:] = (0, 0, 255)
        panel = {
            "front_x": 0, "front_y": 0, "front_width": 4, "front_height": 4,
            "side_x": 4, "side_y": 0, "side_width": 4, "side_height": 4,
            "source_front_x": 0, "source_front_y": 0,
            "source_side_x": 4, "source_side_y": 0,
        }
        image = compose_combined_panels(frame, panel, "side-by-side")
        self.assertEqual(image.size, (8, 4))
        self.assertEqual(image.getpixel((1, 1)), (255, 0, 0))
        self.assertEqual(image.getpixel((6, 1)), (0, 0, 255))


class PlannerTests(unittest.TestCase):
    def test_planner_preserves_virtual_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interpreter = root / "venv-python"
            interpreter.symlink_to(Path("/usr/bin/python3"))
            planner = PipelinePlanner(root, interpreter)
            self.assertEqual(planner.python, interpreter.absolute())
            self.assertNotEqual(planner.python, interpreter.resolve())

    def test_follow_preview_starts_no_earlier_than_seed_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "SP6.mkv"
            background = root / "background.png"
            for path in (video, background):
                path.touch()
            session = {
                "trial": {
                    "name": "SP6",
                    "video_path": str(video),
                    "background_path": str(background),
                    "particle_count": 1,
                    "preview": {
                        "focus_box_mode": "follow",
                        "start_frame": 1,
                        "seed_frame": 7000,
                        "front_seeds": "402,203",
                        "side_seeds": "423,180",
                    },
                },
                "artifacts": {},
            }
            command = PipelinePlanner(root, Path("/usr/bin/python3")).build("preview", session).steps[0].command
            self.assertEqual(command[command.index("--start-frame") + 1], "7000")

    def test_track_plan_includes_browser_seed_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "trial.mp4"
            background = root / "background.png"
            calibration = root / "calibration.json"
            for path in (video, background, calibration):
                path.touch()
            session = {
                "trial": {
                    "name": "trial",
                    "video_path": str(video),
                    "background_path": str(background),
                    "calibration_path": str(calibration),
                    "fps": 120,
                    "particle_count": 2,
                    "tracking": {
                        "front_seeds": "10,20;30,40",
                        "side_seeds": "11,21;31,41",
                        "missing_search_expansion_px": 1.5,
                        "z_source": "side",
                        "no_focus_box": True,
                    },
                },
                "artifacts": {},
            }
            plan = PipelinePlanner(root, Path("/usr/bin/python3")).build("track", session)
            self.assertEqual(len(plan.steps), 2)
            command = plan.steps[0].command
            self.assertIn("--front-seeds", command)
            self.assertIn("10,20;30,40", command)
            self.assertIn("--side-seeds", command)
            self.assertIn("--no-focus-box", command)
            self.assertEqual(command[command.index("--missing-search-expansion-px") + 1], "1.5")
            self.assertEqual(command[command.index("--z-source") + 1], "side")
            self.assertEqual(command[command.index("--fps") + 1], "120.0")
            overlay_command = plan.steps[1].command
            self.assertEqual(
                overlay_command[overlay_command.index("--source-layout") + 1],
                "combined",
            )
            self.assertEqual(
                Path(overlay_command[overlay_command.index("--preview") + 1]),
                video.resolve(),
            )
            self.assertEqual(
                overlay_command[overlay_command.index("--position-source") + 1],
                "smoothed",
            )
            self.assertEqual(
                overlay_command[overlay_command.index("--preview-frame-step") + 1],
                "2",
            )

    def test_motion_requires_confirmed_frame_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracks = root / "tracks_3d.csv"
            preview = root / "preview.mp4"
            calibration = root / "calibration.json"
            for path in (tracks, preview, calibration):
                path.touch()
            session = {
                "trial": {"name": "trial", "fps": "", "calibration_path": str(calibration)},
                "artifacts": {"tracks_csv": str(tracks), "preview_video": str(preview)},
            }
            with self.assertRaisesRegex(ValueError, "frame rate"):
                PipelinePlanner(root, Path("/usr/bin/python3")).build("translational", session)

    def test_orbit_plan_uses_existing_detector_and_declares_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository_root = Path(temporary)
            root = repository_root / "Video Tracking Chin Lab"
            detector = repository_root / "orbit" / "code" / "analyze_orbits_from_csv.py"
            detector.parent.mkdir(parents=True)
            detector.touch()
            root.mkdir()
            tracks = root / "tracks_3d.csv"
            tracks.touch()
            video = root / "source.mkv"
            video.touch()
            session = {
                "trial": {"name": "SP6"},
                "results": {
                    "source_csv": str(tracks),
                    "video_path": str(video),
                    "fps": 120,
                    "particle": 0,
                    "band_low_hz": 1.0,
                    "band_high_hz": 4.0,
                    "threshold": "auto",
                },
                "artifacts": {},
            }
            plan = PipelinePlanner(root, Path("/usr/bin/python3")).build("orbit_analysis", session)
            self.assertEqual(plan.action, "orbit_analysis")
            command = plan.steps[0].command
            self.assertEqual(Path(command[1]).resolve(), detector.resolve())
            self.assertEqual(command[command.index("--fps") + 1], "120")
            self.assertEqual(command[command.index("--particle") + 1], "0")
            self.assertEqual(Path(command[command.index("--video") + 1]), video.resolve())
            self.assertIn("orbit_report_html", plan.steps[0].artifacts)
            self.assertIn("orbit_source_video", plan.steps[0].artifacts)


class BackendTests(unittest.TestCase):
    def test_static_frontend_is_packaged(self) -> None:
        static = Path(__file__).parents[1] / "video_analysis_studio" / "studio_web"
        self.assertTrue((static / "index.html").is_file())
        self.assertTrue((static / "styles.css").is_file())
        self.assertTrue((static / "app.js").is_file())
        html = (static / "index.html").read_text(encoding="utf-8")
        for page in ("home", "setup", "track", "review", "results"):
            self.assertIn(f'data-page-target="{page}"', html)
        self.assertIn("Verify detection before smoothing", html)
        self.assertIn("What each result means", html)
        self.assertIn("Run circular-motion detector", html)
        self.assertIn("Interactive SP6 orbit report", html)

    def test_upload_is_unique_and_writes_provenance_outside_source_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = StudioBackend(root)
            payload = b"small-test-video"
            result = backend.receive_upload("video", "../trial.mp4", len(payload), io.BytesIO(payload))
            imported = Path(result["path"])
            manifest = Path(result["manifest"])
            self.assertEqual(imported.read_bytes(), payload)
            self.assertEqual(imported.parent, (root / "Videos" / "imported").resolve())
            self.assertEqual(manifest.parent, (root / "outputs_background" / "import_manifests").resolve())
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(metadata["original_name"], "trial.mp4")
            self.assertEqual(len(metadata["sha256"]), 64)

    def test_extensionless_matroska_upload_gets_mkv_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = StudioBackend(Path(temporary))
            payload = b"\x1aE\xdf\xa3" + (b"\x00" * 20)
            result = backend.receive_upload("video", "SP6", len(payload), io.BytesIO(payload))
            imported = Path(result["path"])
            self.assertEqual(imported.name.split("_")[-1], "SP6.mkv")
            self.assertEqual(imported.read_bytes(), payload)

    def test_results_csv_upload_is_preserved_as_a_derived_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = StudioBackend(root)
            payload = b"time_s,x_mm,y_mm\n0,1,2\n"
            result = backend.receive_upload(
                "results_csv", "tracks.csv", len(payload), io.BytesIO(payload)
            )
            imported = Path(result["path"])
            self.assertEqual(imported.read_bytes(), payload)
            self.assertEqual(
                imported.parent,
                (root / "outputs_background" / "imported_results").resolve(),
            )

    def test_suggests_background_and_calibration_from_prior_run_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "SP6.mkv"
            background = root / "derived_background.png"
            calibration = root / "calibration.json"
            for path in (video, background, calibration):
                path.touch()
            summary = root / "outputs_background" / "tracks" / "SP6_run" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "video_id": "SP6",
                        "background": str(background),
                        "calibration_file_used": str(calibration),
                        "calibration_status": "provisional; revalidate",
                    }
                ),
                encoding="utf-8",
            )
            suggestion = StudioBackend(root).suggest_companions(str(video))
            self.assertEqual(suggestion["background_path"], str(background.resolve()))
            self.assertEqual(suggestion["calibration_path"], str(calibration.resolve()))
            self.assertIn("provisional", suggestion["calibration_status"])

    def test_track_quality_reports_gaps_without_declaring_pass_or_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracks = root / "tracks_3d.csv"
            pd.DataFrame(
                {
                    "particle": [0, 0, 0],
                    "frame": [10, 11, 13],
                    "x_mm": [0.0, 0.1, 0.4],
                    "y_mm": [0.0, 0.0, 0.0],
                    "z_mm": [1.0, 1.1, 1.2],
                    "z_error_mm": [0.1, 0.2, 0.3],
                    "front_confidence": [1.0, 0.9, 0.8],
                    "side_confidence": [0.8, 0.7, 0.6],
                    "inside_chamber": [True, True, False],
                }
            ).to_csv(tracks, index=False)
            quality = StudioBackend(root).track_quality(str(tracks))
            self.assertEqual(quality["row_count"], 3)
            self.assertEqual(quality["missing_frame_count"], 1)
            self.assertAlmostEqual(quality["coverage_fraction"], 0.75)
            self.assertIn("Descriptive diagnostics only", quality["interpretation"])


if __name__ == "__main__":
    unittest.main()
