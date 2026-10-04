# Video Tracking Instructions

These instructions apply within `Video Tracking Chin Lab/` and supplement the
repository-level `AGENTS.md`.

## Environment and Entry Points

- Install dependencies with `python -m pip install -r requirements.txt` in an
  isolated environment.
- `track.py` is the short entry point for orthogonal 3-D tracking.
- `background_subtraction_tracker.py` is the main background-subtraction
  tracker.
- `python -m video_analysis_studio` launches the integrated local workflow.
- Preserve the existing Python scripts and their command-line behavior when
  changing Studio orchestration.

## Tracking Invariants

- Treat every file under `Videos/` and every externally selected video or
  calibration image as read-only.
- For combined recordings, front and side regions are full-frame crop
  coordinates. Calibration points within each view are crop-local coordinates.
- Front horizontal maps to `x_mm`; side horizontal maps to `y_mm`; vertical maps
  to `z_mm`.
- Separate front and side videos must be synchronized. Do not compensate for an
  unknown offset silently.
- Use the actual media frame rate. The documented SP cohort is 120 fps, but that
  value must not be generalized to MI or newly supplied recordings.
- Do not reuse calibration across a changed resolution, crop, camera layout, or
  visible plate geometry without revalidation.
- Keep particle IDs consistent across both camera views. When identity is
  uncertain, mark or request review instead of silently remapping.
- Preserve `x_mm`, `y_mm`, and `z_mm` as raw measurements. Keep smoothed values
  in the explicit `*_smooth_mm` columns and never smooth across missing-frame
  gaps.

## Output and Provenance

- Write outputs under the established `outputs*` trees or another explicitly
  selected derived-data directory; never next to or over the source video.
- Each accepted run should record the source path or stable source identifier,
  media frame rate, frame range, calibration, chamber geometry, tracker
  parameters, particle seeds, and code revision.
- Keep local `calibration.json` and session JSON files uncommitted. Only commit
  sanitized examples or intentionally versioned run manifests.
- Preserve existing CSV column meanings. Add new columns with units in the name
  where practical and document schema changes in the README.

## Validation

- Start tracker changes with a short, representative frame range before a full
  video run.
- Inspect detection counts, `z`-matching errors, track gaps, identity switches,
  chamber bounds, and front/side overlays.
- For motion changes, compare raw and smooth coordinates and inspect velocity
  and acceleration near gaps and endpoints.
- For rotation changes, inspect projected axes in both camera views and reject
  low-anisotropy or otherwise ambiguous frames explicitly.
- A script completing without an exception is not sufficient scientific
  validation. Record which diagnostics or overlays were reviewed.

There is not yet a comprehensive tracked automated test suite for this area.
At minimum, use `python -m py_compile` on changed modules, run relevant `--help`
commands, and execute a bounded smoke test when suitable non-destructive fixture
data is available. Do not claim interactive windows or visual outputs were
validated unless they were actually inspected.

## Code Review Rules

- Flag any default that can silently apply the wrong frame rate, calibration,
  camera ordering, or chamber geometry.
- Flag smoothing or correction that destroys raw values or crosses data gaps.
- Flag particle matching that allows two identities to claim the same visible
  component without surfacing ambiguity.
- Flag analysis that treats interpolated values as additional measurements or
  extrapolates beyond a measured temperature trace without an explicit model.
