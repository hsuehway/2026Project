# Repository layout and naming

## Naming convention

New top-level groups and reusable project folders use lowercase `kebab-case`.
Python packages and modules continue to use lowercase `snake_case`. Dataset
identifiers such as `SP6` remain uppercase because they are experimental run
IDs, not generic folder names.

## Logical structure

```text
2026Project/
├── Video Tracking Chin Lab/          active video-tracking workspace
├── ThermophoresisSimulation/         active physical simulation codebase
├── analysis/
│   ├── orbit-detection/              reusable orbit-analysis code and tests
│   └── studies/
│       └── sp6/                      SP6-specific viewer and workflow
├── reference/
│   └── thermophoresis-simulator/     supplied simulator/reference material
├── derived-data/                     local tracking exports
├── generated/                        generated reports
├── archive/                          superseded local snapshots
├── Gianni Levitation File/           immutable raw-media archive
└── Gianni Levitation File 2/         immutable raw-media archive
```

## Why several historical names remain

`Video Tracking Chin Lab/`, `ThermophoresisSimulation/`, and the two Gianni
archive directories have existing provenance, saved-session, launcher, or
nested-Git dependencies. Renaming them would invalidate recorded paths or alter
the location of immutable evidence. They therefore keep their historical names
while all new organizational folders use the convention above.

## What moved

| Previous path | Organized path | Meaning |
| --- | --- | --- |
| `orbit/` | `analysis/orbit-detection/` and `analysis/studies/sp6/` | Reusable orbit analysis separated from the SP6 study |
| `sp6 orbits/` | `archive/sp6-orbits-legacy/` | Superseded duplicate snapshot |
| `Simulation/` | `reference/thermophoresis-simulator/` | Supplied HTML simulator, fit notes, and theory PDF |
| `SP6_online-video-cutter.com_20260831_114256/` | `derived-data/sp6-background-tracking/` | Derived SP6 tracking export |
| `output/` | `generated/documents/` | Generated document output |

The moves do not modify the files themselves. Recorded raw-media paths remain
unchanged.

## Data policy

Raw videos and calibration captures are stored outside normal Git history.
Derived artifacts are reproducible and remain local unless a specific dataset
is intentionally versioned with provenance. For collaboration, use Git LFS or
an external research-data repository after documenting file size, ownership,
retention, and access requirements.
