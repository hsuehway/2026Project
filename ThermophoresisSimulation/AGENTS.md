# Thermophoresis Simulation Instructions

These instructions apply within `ThermophoresisSimulation/` and supplement the
repository-level `AGENTS.md`.

## Model Boundaries

- Use SI units internally. Pressure arguments named `P_Pa` are Pascals; convert
  Torr with `torr_to_pa` at the boundary.
- Keep temperature, pressure, length, mass, force, torque, time, orientation,
  and reference-frame conventions explicit in public APIs and outputs.
- The documented gas-property regime is approximately 77–350 K and 1–100 Torr.
  Do not silently extrapolate beyond a function or table's validated range.
- Preserve citations and provenance for empirical constants, correlations, and
  interpolation tables. Do not alter `data/tables/` values without a traceable
  source and an accompanying validation update.
- Distinguish body-frame and laboratory-frame vectors. Normalize quaternions and
  preserve the declared state ordering in dynamics interfaces.
- Prefer physically meaningful failure for invalid geometry, pressure,
  temperature, mass, or inertia inputs over returning plausible-looking values.

## Component Structure

- `src/gas_properties.py`: temperature- and pressure-dependent gas properties.
- `src/gas_cell.py`: axisymmetric gas-cell temperature field.
- `src/rigid_body.py`: particle geometry, mass properties, and inertia.
- `src/force_models.py`: thermophoretic force and torque models.
- `src/quaternion_utils.py`: orientation algebra.
- `src/dynamics.py`: six-degree-of-freedom Newton-Euler integration.

Avoid circular dependencies between these layers. Keep low-level property and
geometry modules independent of the integrator.

## Verification

Run from this directory:

```bash
python -m pytest -q
```

At present, only `tests/test_gas_properties.py` contains substantive assertions;
the other tracked test modules are placeholders. Add focused tests when changing
their corresponding components rather than treating an empty test file as
coverage.

For numerical or physical changes:

- test unit conversion, limiting behavior, conservation or symmetry properties,
  and invalid inputs;
- compare against the cited table, analytical limit, or published benchmark;
- state tolerances and why they are appropriate;
- check convergence or sensitivity when results depend on grid, mesh, or
  integrator settings;
- do not present a qualitative plot match as quantitative validation.

## Documentation and Compatibility

- Document new dependencies and provide a reproducible installation path.
- Keep examples honest about which components and notebooks are implemented;
  several notebooks and tests are currently empty placeholders.
- When an experimental input is unknown, expose it as a required or clearly
  labeled sensitivity parameter instead of embedding an undocumented constant.

## Code Review Rules

- Flag unit or reference-frame ambiguity.
- Flag empirical values or table edits without provenance.
- Flag unbounded interpolation/extrapolation or solver failure hidden behind a
  default value.
- Flag changes to numerical tolerances that lack a benchmark or sensitivity
  justification.
