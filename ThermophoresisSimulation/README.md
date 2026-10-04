# ThermophoresisSimulation

Numerical simulation of rigid-body thermophoresis in the University of Chicago gas cell (Fung et al., *Appl. Phys. Lett.* **110**, 034102, 2017).

A particle trapped between a warm copper plate (T_h ≈ 100–300 K) and a liquid-nitrogen-cooled plate (T_c = 77 K) is levitated by the thermophoretic force and evolves under full 6-DOF Newton-Euler dynamics.

## Project Structure

```
ThermophoresisSimulation/
├── src/                    # Core simulation modules
│   ├── gas_properties.py   # Air κ(T), μ(T), mean free path
│   ├── gas_cell.py         # Part B: temperature field PDE solver
│   ├── rigid_body.py       # Part A: geometry, mesh, inertia tensor
│   ├── force_models.py     # Part C: thermophoretic force & torque
│   ├── dynamics.py         # Part D: 6-DOF Newton-Euler integrator
│   └── quaternion_utils.py # Quaternion algebra helpers
├── data/
│   ├── tables/             # f_T(Kn) interpolation tables
│   │   ├── yamamoto_table.csv   # Yamamoto & Ishihara 1988 (Zheng Table 1)
│   │   └── takata_table.csv     # Takata et al. 1994 (Zheng Table 3)
│   └── meshes/             # STL files for non-spherical particles
├── notebooks/              # Jupyter notebooks for validation and analysis
│   ├── 01_temperature_field.ipynb
│   ├── 02_sphere_levitation.ipynb
│   ├── 03_nonspherical_force.ipynb
│   └── 04_6dof_dynamics.ipynb
├── docs/                   # Component specification documents
│   ├── CompA_RigidBody.docx
│   ├── CompB_TemperatureField.docx
│   ├── CompC_ForceAndTorque.docx
│   └── CompD_Dynamics.docx
└── tests/                  # Unit tests and benchmark checks
    ├── test_gas_properties.py
    ├── test_gas_cell.py
    ├── test_rigid_body.py
    ├── test_force_models.py
    └── test_dynamics.py
```

## Four Components

| Component | File | Description |
|-----------|------|-------------|
| A | `rigid_body.py` | Geometry, surface mesh, center of mass, inertia tensor |
| B | `gas_cell.py` | 2D nonlinear PDE solver for T(r,z); interpolators for ∇T |
| C | `force_models.py` | Thermophoretic force/torque: sphere f_T(Kn) + mesh surface integral |
| D | `dynamics.py` | 6-DOF Newton-Euler ODE system with quaternion orientation |

## Key References

- Fung et al. (2017) *Appl. Phys. Lett.* **110**, 034102 — gas cell geometry and experimental benchmarks
- Zheng (2002) *Adv. Colloid Interface Sci.* **97**, 255 — f_T(Kn) theory review and tables
- Garcia-Ybarra & Rosner (1989) *AIChE J.* **35**, 139 — non-spherical free-molecule force

## Dependencies

```
numpy
scipy
trimesh
scikit-image
matplotlib
pyvista
jupyter
```

Install with:
```bash
pip install numpy scipy trimesh scikit-image matplotlib pyvista jupyter
```

## Development Sequence

1. `gas_properties.py` → air property functions
2. `gas_cell.py` → reproduce Fung et al. Fig. 2
3. `force_models.py` (sphere) → reproduce Fung et al. Figs. 5 & 6
4. `rigid_body.py` → sphere and sphero-cylinder meshes
5. `force_models.py` (mesh) → reproduce Garcia-Ybarra & Rosner Figs. 3 & 4
6. `dynamics.py` → full 6-DOF levitation simulation
