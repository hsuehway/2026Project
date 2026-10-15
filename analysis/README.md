# Analysis workflows

This area separates reusable methods from experiment-specific studies.

- `orbit-detection/`: general CSV orbit detection, classification, reporting,
  and tests. New datasets should use this code instead of copying it.
- `studies/`: dataset-specific inputs, assumptions, validation, scripts, and
  viewers. The current study is `studies/sp6/`.

Reusable logic should move upward into a named analysis package. Run-specific
paths, calibration decisions, figures, overlays, and conclusions should remain
inside the corresponding study directory.
