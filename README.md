# Levitation Particle Research

This repository combines experimental particle-video analysis with
thermophoresis and rigid-body simulation. It is organized so that source code
and documentation can be versioned without treating large recordings or
generated results as ordinary Git files.

## Start here

### Video Analysis Studio

On macOS, double-click:

`Video Tracking Chin Lab/Launch Video Analysis Studio.command`

Or run:

```bash
cd "Video Tracking Chin Lab"
./Launch\ Video\ Analysis\ Studio.command
```

The browser workflow guides you through importing a video, confirming metadata
and calibration, marking particles, testing the detector, tracking, reviewing
the raw overlay, smoothing, and exporting CSV/video results.

### Thermophoresis simulation

```bash
cd ThermophoresisSimulation
python -m pytest -q
```

See `ThermophoresisSimulation/README.md` for the model components and setup.

## Repository map

| Path | Purpose | Git policy |
| --- | --- | --- |
| `Video Tracking Chin Lab/` | Video analyzer, tracking scripts, and local tracking workspace | Code is versioned; videos, calibration state, sessions, and outputs stay local |
| `ThermophoresisSimulation/` | Gas-cell, thermophoretic-force, rigid-body, and 6-DOF simulation | Source, tests, notebooks, and small reference tables are versioned |
| `analysis/sp6-orbits/` | Maintained SP6 orbit-analysis code and viewer | Code is versioned; run-specific data, video, and figures stay local |
| `reference/thermophoresis-simulator/` | Supplied simulator and theory reference | Small reference files may be versioned |
| `derived-data/` | Reproducible exports from tracking runs | Ignored |
| `generated/` | Generated reports and documents | Ignored |
| `archive/` | Superseded local snapshots retained for comparison | Ignored |
| `Gianni Levitation File*/` | Immutable raw-media archives | Ignored; use research-data storage, not normal Git |

The two historical top-level codebase names remain unchanged because existing
sessions, manifests, launchers, and nested repository metadata refer to them.
Raw recordings are never renamed in place. See
[`docs/repository-layout.md`](docs/repository-layout.md) for the naming rules and
data-storage rationale.

## Scientific safeguards

- Confirm frame rate, camera pairing, calibration, geometry, and units for each
  run; do not infer them from filenames.
- Preserve raw coordinates beside smoothed or inferred values.
- Review tracking overlays and diagnostics before interpreting state vectors or
  forces.
- Store derived outputs with source, parameters, calibration, code revision,
  and creation time.

## Git workflow

The repository already uses Git and has an `origin` remote. Before committing:

```bash
git status --short
git diff --check
```

Do not add virtual environments, session files, raw recordings, calibration
state, or generated output trees. Large research data should use Git LFS or an
external research-data archive after an explicit storage decision.
