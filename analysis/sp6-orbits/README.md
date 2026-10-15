# SP6 orbit analysis

## Simple geometric orbit detector

`code/simple_orbit_statistics.py` contains the compact calculation. In each
rolling X/Y window it removes linear center drift, fits ellipse axes with PCA,
checks for consistent rotation, checks whether the particle revisits earlier
coordinates after one period, and tests whether the radius changes steadily.
The output label is `circle`, `ellipse`, `spiral_in`, `spiral_out`, or
`not_orbit`.

`code/simple_orbit_detector.py` is only the CSV, plotting, and metadata wrapper.

Example for the versioned SP6 position export:

```bash
python analysis/sp6-orbits/code/simple_orbit_detector.py \
  analysis/sp6-orbits/data/SP6_position_time.csv \
  --output-dir analysis/sp6-orbits/results/SP6_simple_geometry
```

The defaults use 3-second windows stepped every 0.25 seconds. Thresholds are
explicit command-line options, not statistically calibrated confidence limits;
classified intervals must still be checked against the source video.
The detector uses the CSV timestamps, so it does not assume a frame rate.

This folder collects the final SP6 tracking, mode analysis, simulator fit, and
interactive visual review. The source video remains unchanged in `Gianni
Levitation File/New Deets/SP_Ceramic/SP6`; `video/SP6.mp4` is a browser-friendly
stream-copy used by the viewer.

Open **`sp6-orbit-viewer.html`** in a browser to review the unobstructed source
video beside the synchronized measured CSV row and reconstructed trajectory.
The path view supports draggable 3D, XY, XZ, and YZ projections; it shows the
selected interval, most recent one-period trail, center-to-particle radius,
and current-position uncertainty without drawing over the particle. The page
also includes orbit/quiet labels, timing-boundary uncertainty, temperature,
nearby raw CSV rows, and an independent CSV audit.

## Main result

- Sustained circular-mode frequency: **1.9432 Hz**
- Circular-mode period: **0.5146 s**
- Direction: **clockwise** in the reconstructed `x-y` plane
- Representative RMS orbit radius: **0.342 mm**
- Circular bursts: **7**
- Mean burst recurrence period: **42.64 s**
- Bottom-temperature oscillation period: approximately **43 s**

Detected circular intervals:

| Episode | Video time (s) |
| ---: | ---: |
| 1 | 57.375-79.700 |
| 2 | 98.925-123.917 |
| 3 | 142.283-167.817 |
| 4 | 186.442-210.383 |
| 5 | 226.300-251.375 |
| 6 | 270.683-296.533 |
| 7 | 312.050-336.708 |

## Folder contents

### `data/`

- `SP6_position_time.csv`: measured and smoothed XYZ position, time, tracking
  confidence, circular-mode flag, orbit-band coordinates, and bottom
  temperature. It contains 33,938 measured rows from 57.0 to 343.45 s.
- `SP6_circular_intervals.csv`: per-episode timing, frequency, period, radius,
  circularity, direction, center, and temperature statistics.
- `SP6_mode_summary.json`: overall mode and threshold-sensitivity summary.
- `SP6_temperature_trace_1Hz.csv`: temperature reconstructed from the NI logger
  graph in the video at 1 Hz.
- `SP6_uncertainty_audit.json`: machine-readable coordinate precision,
  episode-to-episode uncertainty, stereo mismatch, and CSV audit results.
- `SP6_pre60_global_candidates.csv`: early-video particle candidates used to
  determine the reliable tracking start.

### `figures/`

- `SP6_timeline.png`: XYZ position, circular-mode amplitude, detected intervals,
  and temperature versus video time.
- `SP6_xy_circular_episodes.png`: reconstructed XY paths for the seven circular
  episodes.
- `SP6_global_position_psd.png`: exploratory global position spectrum.
- `SP6_orbit_metrics_exploratory.png`: exploratory sliding orbit metrics.

### `tracking_verification/`

- `contact_sheet.jpg`: red-marker verification frames from 60-70 s.
- `SP6_57s_end_2fps_preview_verification_smoothed.mp4`: full-duration smoothed
  tracking overlay sampled at 2 fps.
- `track_projections.png` and `summary.json`: tracking diagnostics.

### `reference/thermophoresis-simulator/`

- `ThermophoreticLevitationSimulator.html`: simulator with an **SP6 fitted
  orbit** preset and editable initial translational velocity.
- `SP6_ORBIT_FIT.md`: fitted initial state, effective tensors, interpretation,
  and fixed-step verification.

### `code/`

- `reference/thermophoresis-simulator/ThermophoreticForce_v22.pdf`: theory
  document supplied with the simulator.
- `analyze_particle_modes.py`: reproducible mode-analysis command-line script.
- `analyze_orbits_from_csv.py`: reusable CSV-to-orbit analyzer. It accepts
  common time/X/Y/Z schemas, validates sampling and units, detects orbit
  intervals, measures radius/period/frequency/direction/circularity, and writes
  CSV, JSON, plots, and an interactive HTML report. Supply `--video` to add
  synchronized playback, live path/readout views, and timeline seeking. Run `python
  analysis/sp6-orbits/code/analyze_orbits_from_csv.py --help` for all options.
- `orbit_statistics.py`: numerical statistics used by the reusable analyzer,
  including automatic thresholding, interval formation, per-episode radius and
  period metrics, and aggregate summary statistics. It has no CSV, plotting,
  HTML, or command-line responsibilities.
- `build_orbit_visualization.py`: rebuilds the compact browser data bundle and
  the CSV audit/uncertainty report.

## Reusable CSV analyzer

From the project root, run:

```bash
python3 analysis/sp6-orbits/code/analyze_orbits_from_csv.py path/to/positions.csv \
  --video path/to/matching-video.mp4 --output-dir path/to/orbit-results
```

The CSV must contain X and Y position plus either time in seconds or frame
numbers. The script recognizes columns such as `time_s`, `x_smooth_mm`,
`y_smooth_mm`, `x_mm`, and `y_mm`. For a frame-only file, provide the measured
frame rate explicitly:

```bash
python3 analysis/sp6-orbits/code/analyze_orbits_from_csv.py path/to/positions.csv \
  --fps 120 --position-unit mm --output-dir path/to/orbit-results
```

Use `--x-col`, `--y-col`, `--z-col`, `--time-col`, or `--frame-col` for other
schemas. The default automatic threshold works well when quiet and orbiting
states are both present; use `--threshold VALUE` when a physically justified
cutoff is known. Each run produces `orbit_intervals.csv`,
`orbit_timeseries.csv`, `orbit_summary.json`, two diagnostic figures, and
`orbit_report.html`. The video is referenced, not copied or modified. The
analyzer creates a lossless MP4 remux inside the output directory when the
supplied container is not directly browser-friendly; this does not re-encode
the video stream. The dependencies are listed in `Video Tracking Chin
Lab/requirements.txt`.

### Interactive-viewer support files

- `sp6-orbit-viewer.html`: self-contained interface (apart from its local data
  and video files).
- `orbit_data.js`: compact 30 Hz plotting data, the full 120 Hz measured CSV
  rows used by the synchronized inspector, and the full temperature trace.
- `video/SP6.mp4`: browser-compatible H.264 remux of the source recording.

## Important limitations

The original `SP6_calibration.json` referenced by the project is missing. A
compatible 1920x1080 chamber calibration already present in the project was
used provisionally. Frequency, period, direction, and video timestamps are
robust, but absolute millimeter coordinates should be treated as provisional.

The SP6 simulator tensor values are an effective trajectory fit, not an
independent measurement of the particle response tensors. With the simulator's
original isotropic force and drag tensors, initial conditions alone cannot
produce a sustained center-of-mass circle.
