# Orbit detection

Reusable tools for detecting circular, elliptical, and spiral motion in
position-versus-time data. Nothing in this directory is tied to SP6.

## Two-stage detector

The general analyzer deliberately separates two questions:

1. **Is there appreciable motion at the expected frequency?** A Butterworth
   bandpass and rolling RMS calculation identify candidate intervals. The RMS
   value is an oscillation amplitude after filtering; it is not, by itself,
   proof of circular motion and is not a direct estimate of a moving center.
2. **Does the candidate follow an orbit around a moving center?** Overlapping
   local windows fit X and Y jointly as a linearly drifting center plus a
   harmonic orbit. The fit estimates the center, center velocity, geometric
   radius, major/minor axes, frequency, angular coverage, winding consistency,
   and radial residual. A candidate becomes a validated episode only when a
   sufficient fraction of its local fits pass the configured checks.

The explicit center model within each fit window is

```text
x(t) = cx0 + vx*t + ax*cos(2*pi*f*t) + bx*sin(2*pi*f*t)
y(t) = cy0 + vy*t + ay*cos(2*pi*f*t) + by*sin(2*pi*f*t)
```

This supports a slowly translating center and elliptical as well as circular
orbits. It does not yet fit a tilted orbit plane from three-dimensional data;
the selected X/Y coordinates define the analyzed plane.

## General CSV analyzer

From the repository root:

```bash
python3 analysis/orbit-detection/code/analyze_orbits_from_csv.py positions.csv \
  --video matching-video.mp4 \
  --output-dir derived-data/example-orbit-run
```

The CSV must contain X and Y position and either time in seconds or frame
numbers. If only frames are present, provide the measured frame rate with
`--fps`; it is never guessed. Column names and position units can be selected
explicitly with `--x-col`, `--y-col`, `--z-col`, `--time-col`, `--frame-col`,
and `--position-unit`.

Each run can produce an orbit time series, detected intervals, a JSON summary,
diagnostic figures, and an interactive HTML report. Source CSV and video files
are read-only.

The most important geometry controls are:

```bash
python3 analysis/orbit-detection/code/analyze_orbits_from_csv.py positions.csv \
  --fit-window-s 2.0 \
  --fit-step-s 0.25 \
  --fit-minimum-turns 1.5 \
  --fit-minimum-axis-ratio 0.70 \
  --fit-maximum-normalized-radial-rmse 0.25 \
  --fit-minimum-direction-consistency 0.80 \
  --fit-minimum-accepted-fraction 0.60 \
  --output-dir derived-data/example-orbit-run
```

Use `--rms-only` only when reproducing the former amplitude-only behavior.
Changing a geometry threshold does not change the original positions or the
RMS candidate calculation.

## Outputs

- `orbit_timeseries.csv` preserves the source position columns and adds the
  filtered/RMS candidate signal, the local moving-center fit, fit-quality
  diagnostics, and the final validated-episode label as separate columns.
- `orbit_intervals.csv` contains one row per validated episode. Legacy RMS
  amplitude fields remain present; `circle_fit_radius_*` fields are the
  explicit geometric-radius estimates.
- `orbit_summary.json` records both candidate and validated counts, all detector
  settings, input metadata, units, file hashes, and code provenance.
- `orbit_report.html` presents candidate motion and accepted geometry
  separately and can synchronize the results with a supplied source video.
- `diagnostic_plot.png` shades RMS candidates in gold and geometrically
  validated episodes in green.

Fits near interval edges or data gaps may be unavailable or rejected. Their
status is retained per sample rather than silently interpolated into a valid
result.

## Geometric classifier

`simple_orbit_detector.py` classifies rolling windows as `circle`, `ellipse`,
`spiral_in`, `spiral_out`, or `not_orbit` using drift removal, PCA axes,
winding consistency, return error, and radial change.

```bash
python3 analysis/orbit-detection/code/simple_orbit_detector.py positions.csv \
  --output-dir derived-data/example-geometric-run
```

## Files

- `analyze_orbits_from_csv.py`: full CSV-to-report workflow.
- `orbit_statistics.py`: numerical detection and summary functions.
- `simple_orbit_detector.py`: geometric-classifier CLI and output writer.
- `simple_orbit_statistics.py`: geometric calculations without file I/O.
- `analyze_particle_modes.py`: analysis for `tracks_3d.csv` inputs.
- `tests/`: synthetic detection, schema, unit, and classification tests.

## Validation

Orbit labels are diagnostics, not proof of a physical mechanism. Review the
source video, raw tracking, calibration, frame rate, gaps, and identity before
interpreting a detected episode. A low radial residual establishes geometric
agreement with the local model, not particle identity, calibration accuracy,
or the cause of the motion.
