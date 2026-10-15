# Orbit detection

Reusable tools for detecting circular, elliptical, and spiral motion in
position-versus-time data. Nothing in this directory is tied to SP6.

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
interpreting a detected episode.
