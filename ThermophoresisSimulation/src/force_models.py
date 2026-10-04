"""
force_models.py  —  Component C
---------------------------------
Thermophoretic force and torque on a rigid body.

Two calculation paths:
  1. Sphere fast path  — uses f_T(Kn) scalar from tabulated Boltzmann solutions
  2. Mesh surface integral — Garcia-Ybarra & Rosner (1989) free-molecule kernel
     with χ(Kn) transition-regime correction

f_T(Kn) models implemented
---------------------------
  'waldmann'  — free-molecule limit, f_T = 16√π/15 ≈ 1.8906
  'yamamoto'  — Yamamoto & Ishihara (1988) BGK Boltzmann [PRIMARY]
  'takata'    — Takata et al. (1994) hard-sphere Boltzmann  (κ₂₁ = 0)
  'talbot'    — Talbot et al. (1980) interpolation formula  (cross-check)

References
----------
Zheng (2002) Adv. Colloid Interface Sci. 97, 255  — Tables 1 & 3
Garcia-Ybarra & Rosner (1989) AIChE J. 35, 139
Fung et al. (2017) Appl. Phys. Lett. 110, 034102
"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

from gas_properties import kappa, mu, v_mp, mfp, KB, M_AIR

# ── Physical constants ─────────────────────────────────────────────────────────
FT_FM = 16 * np.sqrt(np.pi) / 15   # Waldmann free-molecule limit ≈ 1.8906

# ── f_T table loading ──────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data" / "tables"


def _load_yamamoto():
    """
    Load Yamamoto & Ishihara (1988) f_T table.
    CSV columns: Kn, A_o, A_w, H_o, H_w
    (Table 1 of Zheng 2002)
    TODO: populate data/tables/yamamoto_table.csv
    """
    path = _DATA_DIR / "yamamoto_table.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Yamamoto table not found at {path}.\n"
            "Please add data/tables/yamamoto_table.csv from Zheng (2002) Table 1."
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # columns: Kn, A_o, A_w, H_o, H_w
    return data


def _load_takata():
    """
    Load Takata et al. (1994) f_T table (κ₂₁ = 0).
    CSV columns: Kn, f_T
    (Table 3 of Zheng 2002)
    TODO: populate data/tables/takata_table.csv
    """
    path = _DATA_DIR / "takata_table.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Takata table not found at {path}.\n"
            "Please add data/tables/takata_table.csv from Zheng (2002) Table 3."
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return interp1d(data[:, 0], data[:, 1],
                    kind="cubic", bounds_error=False,
                    fill_value=(data[0, 1], FT_FM))


# Cache loaded tables
_takata_interp = None
_yamamoto_data = None


def f_T_sphere(Kn, model="yamamoto", kappa21=0.03):
    """
    Dimensionless thermophoretic parameter f_T for a sphere.

    Parameters
    ----------
    Kn       : Knudsen number ℓ/a
    model    : 'waldmann' | 'yamamoto' | 'takata' | 'talbot'
    kappa21  : κ_gas / κ_particle  (used by yamamoto and talbot)

    Returns
    -------
    f_T : float  (dimensionless, in [0, FT_FM])
    """
    global _takata_interp, _yamamoto_data

    if model == "waldmann":
        return FT_FM

    elif model == "takata":
        if _takata_interp is None:
            _takata_interp = _load_takata()
        return float(_takata_interp(Kn))

    elif model == "yamamoto":
        # TODO: implement 2-D interpolation over (Kn, kappa21)
        # Placeholder: falls back to Takata for now
        if _takata_interp is None:
            _takata_interp = _load_takata()
        return float(_takata_interp(Kn))

    elif model == "talbot":
        return _talbot(Kn, kappa21)

    else:
        raise ValueError(f"Unknown model: {model!r}. "
                         "Choose 'waldmann', 'yamamoto', 'takata', or 'talbot'.")


def _talbot(Kn, kappa21, Cm=1.14, Ct=2.18, Cs=1.17):
    """
    Talbot et al. (1980) interpolation formula.
    Note: known to deviate 5–10 % from Fung et al. data.
    """
    num = 2 * Cs * (kappa21 + Ct * Kn) * Kn
    den = (1 + 3 * Cm * Kn) * (1 + 2 * kappa21 + 2 * Ct * Kn)
    return num / den


# ── Sphere force (fast path) ───────────────────────────────────────────────────

def thermo_force_sphere(pos, a, gas_cell, P_Pa, model="yamamoto", kappa21=0.03):
    """
    Thermophoretic force vector [N] on a sphere.

    F_th = -f_T(Kn) · (a² κ / v) · ∇T

    Parameters
    ----------
    pos    : (3,) Cartesian position [m]
    a      : sphere radius [m]
    gas_cell : GasCell object from gas_cell.py
    P_Pa   : gas pressure [Pa]
    """
    T   = gas_cell.T_at(pos)
    gT  = gas_cell.gradT_3d(pos)
    kap = kappa(T)
    vv  = v_mp(T)
    Kn  = mfp(T, P_Pa) / a
    fT  = f_T_sphere(Kn, model=model, kappa21=kappa21)
    return -fT * (a**2 * kap / vv) * gT


# ── Tangent vectors ────────────────────────────────────────────────────────────

def _make_tangents(n):
    """Two orthonormal vectors tangent to unit normal n."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1  = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2  = np.cross(n, t1)
    return t1, t2


# ── Mesh surface-integral force (free-molecule + χ correction) ─────────────────

def thermo_force_mesh(body, r_cm_lab, q, gas_cell, P_Pa,
                      model="yamamoto", kappa21=0.03, a_diff=0.8):
    """
    Thermophoretic force [N] and torque [N·m] on an arbitrary mesh particle.

    Uses the Garcia-Ybarra & Rosner (1989) free-molecule surface integral
    multiplied by χ(Kn) to interpolate to the transition regime.

    Parameters
    ----------
    body       : RigidBody object
    r_cm_lab   : (3,) center-of-mass position in lab frame [m]
    q          : (4,) orientation quaternion (body → lab)
    gas_cell   : GasCell object
    P_Pa       : gas pressure [Pa]
    a_diff     : diffuse-reflection fraction (accommodation coefficient)

    Returns
    -------
    F_total : (3,) force [N]
    tau     : (3,) torque about r_cm_lab [N·m]
    """
    from quaternion_utils import quat_to_matrix
    Rmat = quat_to_matrix(q)

    # Gas properties at center of mass
    T     = gas_cell.T_at(r_cm_lab)
    gT    = gas_cell.gradT_3d(r_cm_lab)
    glnT  = gT / T
    Kn_eq = mfp(T, P_Pa) / body.r_eq
    chi   = f_T_sphere(Kn_eq, model=model, kappa21=kappa21) / FT_FM

    # Gas kinetic scalars
    N_gas = P_Pa / (KB * T)          # number density [m⁻³]
    h     = M_AIR / (2 * KB * T)     # inverse thermal velocity scale
    nu    = mu(T) / (N_gas * M_AIR)  # kinematic viscosity [m²/s]
    prefac_n = N_gas * M_AIR / np.sqrt(np.pi * h)
    prefac_t = a_diff * N_gas * M_AIR / (2 * np.sqrt(np.pi * h))

    F_total   = np.zeros(3)
    tau_total = np.zeros(3)

    for f in range(body.mesh.faces.shape[0]):
        # Rotate face normal and centroid to lab frame
        n_lab = Rmat @ body.face_normals[f]
        r_lab = r_cm_lab + Rmat @ (body.face_centroids[f] - body.r_cm)
        dA    = body.face_areas[f]
        t1, t2 = _make_tangents(n_lab)

        # Garcia-Ybarra & Rosner Eq. (C.2): normal thermophoretic contribution
        dFn = -prefac_n * (2 - a_diff) * (3 * nu / 4) * np.dot(n_lab, glnT)

        # Garcia-Ybarra & Rosner Eq. (C.3): tangential contributions
        dFt1 = prefac_t * (3 * nu / 4) * np.dot(t1, glnT)
        dFt2 = prefac_t * (3 * nu / 4) * np.dot(t2, glnT)

        # Assemble element force with transition-regime correction χ
        dF = chi * (dFn * n_lab + dFt1 * t1 + dFt2 * t2) * dA

        F_total   += dF
        tau_total += np.cross(r_lab - r_cm_lab, dF)

    return F_total, tau_total


# ── Drag ───────────────────────────────────────────────────────────────────────

def stokes_drag(v_cm, T, P_Pa, a_eq):
    """
    Stokes drag with Cunningham slip correction for a sphere of radius a_eq.

    F_drag = -(6π μ a / C_c) v_cm
    """
    from gas_properties import cunningham, knudsen
    Kn  = knudsen(T, P_Pa, a_eq)
    Cc  = cunningham(Kn)
    xi  = 6 * np.pi * mu(T) * a_eq / Cc
    return -xi * v_cm


def rotational_drag(omega, T, P_Pa, a_eq):
    """
    Rotational drag torque for a sphere:  τ_drag = -(8π μ a³ / C_c) ω
    """
    from gas_properties import cunningham, knudsen
    Kn  = knudsen(T, P_Pa, a_eq)
    Cc  = cunningham(Kn)
    xi_r = 8 * np.pi * mu(T) * a_eq**3 / Cc
    return -xi_r * omega


# ── Equilibrium height (sphere, axis r=0) ──────────────────────────────────────

def equilibrium_height(a, rho_p, gas_cell, P_Pa, model="yamamoto", kappa21=0.03):
    """
    Find levitation height z_eq where F_th,z = Mg, on the symmetry axis.

    Returns z_eq [m] or None if no equilibrium found in [0.05H, 0.95H].
    """
    from scipy.optimize import brentq
    H    = gas_cell.z[-1]
    M    = (4/3) * np.pi * a**3 * rho_p
    g    = 9.8

    def net_fz(z):
        pos  = np.array([1e-10, 0.0, z])
        F_th = thermo_force_sphere(pos, a, gas_cell, P_Pa, model, kappa21)
        return F_th[2] - M * g

    z_lo, z_hi = 0.05 * H, 0.95 * H
    if net_fz(z_lo) * net_fz(z_hi) > 0:
        return None   # No sign change — no equilibrium in range
    return brentq(net_fz, z_lo, z_hi, xtol=1e-7)
