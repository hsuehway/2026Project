# Project Context

## Purpose

This project studies particles levitated in a low-pressure, thermally driven gas
cell. It joins experimental video analysis with numerical thermophoresis and
rigid-body simulation. The long-term goal is a reproducible path from raw camera
recordings to defensible particle trajectories, translational and rotational
state vectors, force estimates, and comparisons with physical models.

## Repository Areas

### Video tracking and experimental analysis

`Video Tracking Chin Lab/` contains the active tracking and analysis workflow.
It supports combined screen recordings with two camera panels as well as two
separate synchronized camera files. The tracked sample set currently includes
MI10 through MI15 and a trimmed MI10 recording.

The repository root also contains two raw-media archives:

- `Gianni Levitation File/`
- `Gianni Levitation File 2/`

The first archive includes dated levitation recordings, calibration images, and
a `New Deets/` collection labeled for K64 ceramic bubbles, K65 glass bubbles,
ice spheres, and a silk sheet. These names are inventory labels, not sufficient
experimental metadata. Do not infer pressure, temperature, particle size,
density, frame rate, or run relationships from filenames alone.

### Thermophoresis simulation

`ThermophoresisSimulation/` models the gas-cell temperature field,
temperature-dependent gas properties, thermophoretic forces and torques, rigid
bodies, and six-degree-of-freedom Newton-Euler dynamics. Its empirical functions
and interpolation tables are intended for the documented low-pressure,
cryogenic-to-room-temperature regime; their stated domains must be respected.

## Experimental Geometry and Coordinates

The current video workflow models the chamber as a circular frustum. The
documented nominal dimensions are a 33.7 mm bottom plate, a 25.4 mm top plate,
and a 10.0 mm gap. The gap is a default, not a substitute for a run-specific
measurement.

For reconstructed coordinates:

- front-camera horizontal position maps to `x_mm`;
- side-camera horizontal position maps to `y_mm`;
- vertical image position maps to `z_mm`;
- the two views must be synchronized frame-for-frame;
- same-frame front/side matching primarily uses agreement in reconstructed
  vertical position.

Combined recordings contain both views in one frame and are inherently aligned
at the frame level. Separate camera files require independent evidence of
synchronization.

## Analysis Chain

The intended evidence chain is:

1. Preserve the original recording and inspect its media metadata.
2. Identify the run, camera layout, frame rate, chamber geometry, and known
   experimental conditions.
3. Calibrate camera regions and plate geometry for that recording layout.
4. Tune detection on a short frame range before processing the full recording.
5. Reconstruct and link 3-D trajectories while preserving raw detections.
6. Review front/side vertical disagreement, gaps, identity switches, and video
   overlays before accepting trajectories.
7. Calculate smoothed positions, velocities, accelerations, and rotation as
   explicitly labeled derived quantities.
8. Perform force or model inference only after trajectory quality and physical
   inputs have passed their checks.
9. Compare with simulation while retaining model version, parameters, units,
   reference data, and numerical tolerances.

## Canonical Evidence

- Raw videos and calibration captures are primary evidence.
- `calibration.json`-style files are run- and layout-specific working data;
  `calibration.example.json` is only a schema/example.
- `tracks_3d.csv` is the central linked-trajectory artifact. Raw and smoothed
  coordinate columns must remain distinguishable.
- State-vector and force-analysis tables are derived evidence and should remain
  traceable to the exact tracking artifact and settings that produced them.
- Diagnostic plots, mismatch summaries, and overlays are validation evidence,
  not optional decoration.

## Current Priorities

- Build a reliable inventory for each recording before bulk analysis.
- Preserve original media and make every derived result reproducible.
- Establish run-specific calibration and frame-rate metadata.
- Validate particle identity and front/side correspondence before interpreting
  dynamics or forces.
- Keep simulation assumptions aligned with measured experimental conditions.

## Metadata Still Needed for New Recordings

Before drawing scientific conclusions from the Gianni archives, establish when
available:

- which files form one trial or camera pair;
- whether each file is a combined view or a single-camera recording;
- native and effective frame rate;
- chamber dimensions and applicable calibration image;
- pressure and temperature history;
- particle material, size distribution, density, and particle count;
- run start/stop events, collisions, interventions, and known artifacts.

Record confirmed answers in a versioned manifest rather than relying on memory
or filenames.
