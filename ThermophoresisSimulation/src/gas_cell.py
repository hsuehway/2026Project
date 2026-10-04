"""
gas_cell.py  —  Component B
----------------------------
Solves the steady-state heat equation ∇·(κ(T)∇T) = 0 in the Fung et al.
(2017) gas cell and returns interpolators for T(r,z) and ∇T(r,z).

Gas cell geometry
-----------------
- Cylindrical domain:  r ∈ [0, R_wall],  z ∈ [0, H]
- Bottom plate (z=0):  T = T_h  (warm, controlled)
- Top plate    (z=H, r≤R_cold): T = T_c = 77 K  (liquid nitrogen)
- Top annulus  (z=H, r>R_cold): T = T_wall = 298 K
- Chamber wall (r=R_wall):      T = T_wall = 298 K
- Symmetry axis (r=0):          ∂T/∂r = 0  (Neumann)

Reference
---------
Fung et al. (2017) Appl. Phys. Lett. 110, 034102
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

from gas_properties import kappa


# ── Default geometry (Fung et al. 2017) ──────────────────────────────────────
DEFAULT_H      = 0.010   # plate separation [m]
DEFAULT_R_WALL = 0.020   # chamber radius [m]
DEFAULT_R_COLD = 0.0127  # liquid-N2 bucket outer radius [m]
DEFAULT_T_C    = 77.0    # cold plate temperature [K]
DEFAULT_T_WALL = 298.0   # chamber wall temperature [K]


class GasCell:
    """
    Solved temperature field and derived interpolators for the gas cell.

    Attributes
    ----------
    T_interp    : RegularGridInterpolator  T(r, z)    [K]
    dTdr_interp : RegularGridInterpolator  ∂T/∂r(r,z) [K/m]
    dTdz_interp : RegularGridInterpolator  ∂T/∂z(r,z) [K/m]
    r, z        : 1-D grid arrays [m]
    T_grid      : 2-D solution array (Nr × Nz)
    """

    def __init__(self, T_interp, dTdr_interp, dTdz_interp, r, z, T_grid):
        self.T_interp    = T_interp
        self.dTdr_interp = dTdr_interp
        self.dTdz_interp = dTdz_interp
        self.r = r
        self.z = z
        self.T_grid = T_grid

    def T_at(self, pos):
        """Temperature [K] at Cartesian position pos=(x,y,z)."""
        rr = np.sqrt(pos[0]**2 + pos[1]**2)
        rr = np.clip(rr, 0, self.r[-1])
        zz = np.clip(pos[2], self.z[0], self.z[-1])
        return float(self.T_interp([[rr, zz]]))

    def gradT_3d(self, pos):
        """
        Temperature gradient vector [K/m] at Cartesian position pos=(x,y,z).
        Projects the cylindrical (∂T/∂r, ∂T/∂z) onto 3-D Cartesian axes.
        """
        x, y, z = pos
        rr = np.sqrt(x**2 + y**2)
        rr_safe = max(rr, 1e-12)
        zz = np.clip(z, self.z[0], self.z[-1])
        pt = [[np.clip(rr, 0, self.r[-1]), zz]]
        dTdr = float(self.dTdr_interp(pt))
        dTdz = float(self.dTdz_interp(pt))
        ex = x / rr_safe
        ey = y / rr_safe
        return np.array([dTdr * ex, dTdr * ey, dTdz])


def solve_gas_cell(
    T_h,
    Nr=150, Nz=80,
    T_c=DEFAULT_T_C, T_wall=DEFAULT_T_WALL,
    H=DEFAULT_H, R_wall=DEFAULT_R_WALL, R_cold=DEFAULT_R_COLD,
    tol=1e-4, max_iter=200,
    verbose=False,
):
    """
    Solve ∇·(κ(T)∇T) = 0 on the gas cell domain via Picard iteration.

    Parameters
    ----------
    T_h      : bottom plate temperature [K]
    Nr, Nz   : grid resolution (radial, vertical)
    tol      : convergence criterion on max |ΔT| [K]
    verbose  : print iteration residuals

    Returns
    -------
    GasCell object with interpolators for T, ∂T/∂r, ∂T/∂z
    """
    r  = np.linspace(0, R_wall, Nr)
    z  = np.linspace(0, H, Nz)
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    # Initial guess: linear interpolation between plates
    T = np.zeros((Nr, Nz))
    for j in range(Nz):
        T[:, j] = T_h + (T_c - T_h) * z[j] / H

    def bc(i, j):
        """Return (is_dirichlet, value) for node (i,j)."""
        if j == 0:
            return True, T_h
        if j == Nz - 1:
            return True, (T_c if r[i] <= R_cold else T_wall)
        if i == Nr - 1:
            return True, T_wall
        return False, None

    N = Nr * Nz

    for iteration in range(max_iter):
        K = kappa(T)
        A = lil_matrix((N, N))
        b = np.zeros(N)

        for i in range(Nr):
            for j in range(Nz):
                n = i * Nz + j
                is_bc, val = bc(i, j)
                if is_bc:
                    A[n, n] = 1.0
                    b[n]    = val
                    continue

                # ── r-direction ──────────────────────────────────────────────
                if i == 0:
                    # Axis: use L'Hôpital → 2κ ∂²T/∂r²
                    kr = 0.5 * (K[0, j] + K[1, j])
                    A[n, n]          -= 2 * kr / dr**2
                    A[n, 1 * Nz + j] += 2 * kr / dr**2
                else:
                    kp = 0.5 * (K[i, j] + K[i + 1, j])
                    km = 0.5 * (K[i, j] + K[i - 1, j])
                    rp = r[i] + dr / 2
                    rm = r[i] - dr / 2
                    coeff = 1.0 / (r[i] * dr**2)
                    A[n, n]              -= coeff * (rp * kp + rm * km)
                    A[n, (i + 1) * Nz + j] += coeff * rp * kp
                    A[n, (i - 1) * Nz + j] += coeff * rm * km

                # ── z-direction ──────────────────────────────────────────────
                kzp = 0.5 * (K[i, j] + K[i, j + 1])
                kzm = 0.5 * (K[i, j] + K[i, j - 1])
                A[n, n]          -= (kzp + kzm) / dz**2
                A[n, i * Nz + j + 1] += kzp / dz**2
                A[n, i * Nz + j - 1] += kzm / dz**2

        T_new = spsolve(A.tocsr(), b).reshape(Nr, Nz)
        residual = np.max(np.abs(T_new - T))
        if verbose:
            print(f"  iter {iteration+1:3d}  max|ΔT| = {residual:.2e} K")
        T = T_new
        if residual < tol:
            if verbose:
                print(f"  Converged after {iteration+1} iterations.")
            break

    # ── Build interpolators ───────────────────────────────────────────────────
    dTdr = np.gradient(T, r, axis=0)
    dTdz = np.gradient(T, z, axis=1)

    interp_T    = RegularGridInterpolator((r, z), T,    bounds_error=False, fill_value=None)
    interp_dTdr = RegularGridInterpolator((r, z), dTdr, bounds_error=False, fill_value=None)
    interp_dTdz = RegularGridInterpolator((r, z), dTdz, bounds_error=False, fill_value=None)

    return GasCell(interp_T, interp_dTdr, interp_dTdz, r, z, T)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Solving gas cell  (T_h = 277 K) ...")
    gc = solve_gas_cell(T_h=277.0, Nr=100, Nz=60, verbose=True)
    # Check BCs
    print(f"\nBC checks:")
    print(f"  T(r=0,   z=0)    = {gc.T_at([0,0,0.0]):.1f} K  (expect 277)")
    print(f"  T(r=0,   z=H)    = {gc.T_at([0,0,0.01]):.1f} K  (expect 77)")
    print(f"  T(r=Rw,  z=H/2)  = {gc.T_at([0.02,0,0.005]):.1f} K  (expect ~298)")
    # Gradient at midpoint
    g = gc.gradT_3d([0.0, 0.0, 0.005])
    print(f"\nGradient at axis midpoint: {g} K/m")
