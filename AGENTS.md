# Repository Instructions

## Scope

This repository combines two related research codebases and several raw-media
collections:

- `Video Tracking Chin Lab/`: orthogonal-camera particle tracking, state-vector
  reconstruction, rotation estimation, and force-analysis workflows.
- `ThermophoresisSimulation/`: gas-cell, thermophoretic-force, rigid-body, and
  six-degree-of-freedom simulation components.
- `Gianni Levitation File/` and `Gianni Levitation File 2/`: raw experimental
  media archives. These may be intentionally untracked because of their size.

Read `PROJECT_CONTEXT.md` and the closest directory-level `AGENTS.md` before
changing code. The more specific file adds to or overrides this file.

## Working Agreements

- Check `git status` before editing. Preserve unrelated changes and untracked
  data.
- Treat original videos, images, calibration captures, and measured tables as
  immutable source data. Never edit, rename, transcode, or overwrite them in
  place.
- Write derived artifacts to a dedicated output directory with enough metadata
  to recover the source file, parameters, calibration, frame rate, code
  revision, and creation time.
- Never guess a recording's frame rate, camera pairing, pressure, temperature,
  particle material, or dimensions. Inspect metadata or ask for the missing
  value, and record the answer with the result.
- Keep units explicit in variable names, schemas, plots, and documentation.
  Refuse silent unit conversion or extrapolation outside a model's stated
  domain.
- Preserve raw measurements alongside smoothed, corrected, or inferred values.
  Derived data must not silently replace canonical raw columns.
- Make the smallest compatible change. Preserve established command-line
  interfaces and output schemas unless the task explicitly calls for a breaking
  change.
- Run the narrowest relevant automated checks, then perform output-level or
  visual validation when tracking, calibration, overlays, or scientific results
  change. Report checks that could not be run.

## Data and Git Safety

- Do not commit secrets, local session files, virtual environments, caches,
  machine-specific calibration files, or generated output trees.
- Do not add new large videos to normal Git history without explicit approval.
  First report their sizes and recommend Git LFS or external research-data
  storage when appropriate.
- The top-level repository is the shared project repository. The
  `ThermophoresisSimulation/.git` directory is preserved historical metadata;
  do not change its remote, branch, or history unless explicitly asked.
- Run project-wide Git operations from the repository root. Stage only files
  relevant to the current task.

## Documentation

- Put durable scientific assumptions, dataset meaning, and cross-project
  decisions in `PROJECT_CONTEXT.md`.
- Put setup, command usage, and output-format details in the appropriate
  `README.md`.
- Update the closest `AGENTS.md` only for durable working rules, not temporary
  task notes.

## Code Review Rules

- Flag any path that can overwrite raw media or calibration evidence.
- Flag hard-coded frame rates, geometry, pressure, temperature, material, or
  particle properties unless the value is explicitly scoped to a documented
  dataset.
- Flag conclusions drawn from tracking or force outputs without calibration,
  identity, synchronization, and diagnostic checks.
- Flag generated artifacts or large binary data added to Git without an
  explicit storage decision.
