# SP6 orbit study

This directory contains the SP6-specific application of the reusable tools in
`analysis/orbit-detection/`. It records the dataset assumptions, visualization,
verification material, and derived results for SP6 without putting SP6 logic
inside the general analyzer.

## Review the study

Open `sp6-orbit-viewer.html` to inspect the synchronized video, measured CSV
row, reconstructed trajectory, orbit/quiet labels, temperature, and uncertainty
information. The viewer reads the local `orbit-data.js` and `video/SP6.mp4`
files, which are generated or run-specific and therefore ignored by Git.

## Reproduce the general analysis

```bash
python3 analysis/orbit-detection/code/analyze_orbits_from_csv.py \
  analysis/studies/sp6/data/SP6_position_time.csv \
  --video analysis/studies/sp6/video/SP6.mp4 \
  --output-dir analysis/studies/sp6/results/recomputed
```

The default analysis first uses the 1--4 Hz bandpassed rolling RMS signal to
find high-amplitude candidates, then validates each candidate with overlapping
moving-center harmonic fits. For every local window it estimates the drifting
center, geometric radius, frequency, ellipse axis ratio, radial fit error,
angular coverage, and direction consistency. Use `--rms-only` only to reproduce
the older amplitude-only episode selection.

Rebuild the compact viewer data with:

```bash
python3 analysis/studies/sp6/scripts/build_orbit_visualization.py
```

## Documented result

- Mean moving-center-fit frequency: **1.9387 Hz**
- Corresponding mean period: **0.5159 s**
- Direction: **clockwise** in the reconstructed `x-y` plane
- Representative moving-center circle-fit radius: **0.343 mm**
- Representative bandpassed RMS amplitude: **0.345 mm**
- Mean normalized radial fit error: **0.094**
- Detected circular bursts: **7**
- Mean burst recurrence period: **42.92 s**
- Bottom-temperature oscillation period: approximately **43 s**

All seven RMS candidates pass the moving-center geometry checks with the
current documented defaults. These results use measured rows beginning at
57.0 s. The original
`SP6_calibration.json` is missing; a compatible 1920×1080 calibration was used
provisionally. Frequency, period, direction, and timestamps are more defensible
than the absolute millimeter coordinates, which retain unknown systematic
calibration uncertainty.

## Local folders

- `data/`: SP6 position, interval, temperature, and uncertainty tables.
- `figures/`: derived diagnostic plots.
- `results/`: orbit-classification outputs.
- `tracking-verification/`: overlays and visual checks.
- `video/`: browser-compatible review copy of the source recording.
- `scripts/`: SP6-specific data-bundle builder.

The source recording remains unchanged in the Gianni raw-media archive. The
local study video is only a browser-compatible review copy.
