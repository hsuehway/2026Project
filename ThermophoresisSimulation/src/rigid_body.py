"""
rigid_body.py  —  Component A
-------------------------------
Defines the RigidBody data structure and factory functions for creating
particles from analytic primitives, implicit surfaces, or STL mesh files.

Provides
--------
- RigidBody dataclass: mass, r_cm, I_body, surface mesh arrays
- make_sphere(R, rho)
- make_spherocylinder(R, L, rho)
- make_from_stl(path, rho)
- make_two_sphere_aggregate(R1, R2, sep, rho)
"""

import numpy as np
from dataclasses import dataclass, field

try:
    import trimesh
    TRIMESH_OK = True
except ImportError:
    TRIMESH_OK = False
    print("Warning: trimesh not installed. Mesh-based shapes unavailable.")


@dataclass
class RigidBody:
    """
    Rigid body geometry and inertial properties.

    All quantities are defined in the body frame (origin at r_cm).

    Attributes
    ----------
    mesh            : trimesh.Trimesh   — surface mesh (body frame)
    rho             : float             — density [kg/m³]
    mass            : float             — total mass M [kg]
    r_cm            : ndarray (3,)      — center of mass [m]
    I_body          : ndarray (3,3)     — inertia tensor [kg·m²]
    r_eq            : float             — equivalent-volume sphere radius [m]
    face_centroids  : ndarray (N_f, 3)  — face centroids [m]
    face_normals    : ndarray (N_f, 3)  — outward unit normals
    face_areas      : ndarray (N_f,)    — face areas [m²]
    """
    mesh:           object              # trimesh.Trimesh
    rho:            float
    mass:           float
    r_cm:           np.ndarray
    I_body:         np.ndarray
    r_eq:           float               # equivalent-volume sphere radius
    face_centroids: np.ndarray
    face_normals:   np.ndarray
    face_areas:     np.ndarray


def _build(mesh, rho):
    """Compute all RigidBody fields from a trimesh object and density."""
    assert TRIMESH_OK, "trimesh is required for mesh-based rigid bodies."
    trimesh.repair.fix_normals(mesh)

    vol    = abs(mesh.volume)
    mass   = rho * vol
    r_cm   = mesh.center_mass
    r_eq   = (3 * vol / (4 * np.pi)) ** (1/3)

    # Inertia tensor: trimesh returns it about the origin; shift to r_cm
    I_origin = mesh.moment_inertia * rho   # trimesh uses unit density
    d  = r_cm
    # Parallel-axis shift: I_cm = I_origin - M*(|d|² δ - d⊗d)
    I_body = I_origin - mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return RigidBody(
        mesh           = mesh,
        rho            = rho,
        mass           = mass,
        r_cm           = r_cm,
        I_body         = I_body,
        r_eq           = r_eq,
        face_centroids = mesh.triangles_center,
        face_normals   = mesh.face_normals.copy(),
        face_areas     = mesh.area_faces.copy(),
    )


def make_sphere(R, rho, subdivisions=4):
    """
    Spherical particle.

    Parameters
    ----------
    R            : radius [m]
    rho          : density [kg/m³]
    subdivisions : icosphere subdivision level (higher = finer mesh)
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=R)
    return _build(mesh, rho)


def make_spherocylinder(R, L, rho, sections=32):
    """
    Sphero-cylinder: cylinder of length L, radius R, with hemispherical caps.
    Symmetry axis is z.

    Parameters
    ----------
    R        : cylinder radius [m]
    L        : cylinder length (cap-to-cap inner length) [m]
    rho      : density [kg/m³]
    sections : angular resolution of the cylinder
    """
    cyl     = trimesh.creation.cylinder(radius=R, height=L, sections=sections)

    # Two hemisphere caps, translated to ±L/2 along z
    cap_top = trimesh.creation.hemisphere(radius=R, count=[sections//2, sections])
    cap_bot = trimesh.creation.hemisphere(radius=R, count=[sections//2, sections])
    cap_top.apply_translation([0, 0,  L/2])
    cap_bot.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    )
    cap_bot.apply_translation([0, 0, -L/2])

    mesh = trimesh.util.concatenate([cyl, cap_top, cap_bot])
    return _build(mesh, rho)


def make_two_sphere_aggregate(R1, R2, sep, rho):
    """
    Aggregate of two spheres with center-to-center separation sep along z.
    Both spheres have the same density rho.

    Parameters
    ----------
    R1, R2 : sphere radii [m]
    sep    : center-to-center distance [m]  (must be >= R1+R2 for non-overlapping)
    rho    : density [kg/m³]
    """
    s1 = trimesh.creation.icosphere(subdivisions=4, radius=R1)
    s2 = trimesh.creation.icosphere(subdivisions=4, radius=R2)
    s1.apply_translation([0, 0,  sep/2])
    s2.apply_translation([0, 0, -sep/2])
    mesh = trimesh.util.concatenate([s1, s2])
    return _build(mesh, rho)


def make_from_stl(path, rho):
    """
    Load an arbitrary particle shape from an STL file.

    Parameters
    ----------
    path : path to .stl or .obj file
    rho  : density [kg/m³]
    """
    mesh = trimesh.load(path, force='mesh')
    return _build(mesh, rho)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    rho = 998.0   # polyethylene [kg/m³]
    R   = 10e-6   # 10 µm radius

    print("=== Sphere ===")
    sphere = make_sphere(R, rho)
    M_analytic = (4/3) * np.pi * R**3 * rho
    I_analytic = 0.4 * M_analytic * R**2
    print(f"  Mass:       {sphere.mass:.4e} kg  (analytic: {M_analytic:.4e})")
    print(f"  I_xx:       {sphere.I_body[0,0]:.4e}  (analytic: {I_analytic:.4e})")
    print(f"  r_eq:       {sphere.r_eq*1e6:.2f} µm  (expect {R*1e6:.2f})")
    print(f"  # faces:    {sphere.mesh.faces.shape[0]}")

    print("\n=== Sphero-cylinder (R=5µm, L=20µm) ===")
    sc = make_spherocylinder(5e-6, 20e-6, rho)
    print(f"  Mass:  {sc.mass:.4e} kg")
    print(f"  r_eq:  {sc.r_eq*1e6:.2f} µm")
    print(f"  I_body diag: {np.diag(sc.I_body)}")
