"""
gas_properties.py
-----------------
Empirical property functions for air as a function of temperature and pressure.
Valid range: ~77–350 K, 1–100 Torr.

References
----------
Kadoya et al. (1985) J. Phys. Chem. Ref. Data 14, 947  — κ(T), μ(T)
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
KB     = 1.380649e-23   # Boltzmann constant [J/K]
M_AIR  = 4.81e-26       # Mean molecular mass of air [kg]
TORR   = 133.322        # 1 Torr in Pa

# ── Reference values at T0 = 273 K ───────────────────────────────────────────
T0     = 273.0          # Reference temperature [K]
KAPPA0 = 0.0241         # Thermal conductivity at T0 [W/(m·K)]
MU0    = 1.81e-5        # Dynamic viscosity at T0 [Pa·s]
ALPHA_K = 0.82          # Power-law exponent for κ(T)
ALPHA_M = 0.74          # Power-law exponent for μ(T)


def kappa(T):
    """Thermal conductivity of air [W/(m·K)]."""
    return KAPPA0 * (T / T0) ** ALPHA_K


def mu(T):
    """Dynamic viscosity of air [Pa·s]."""
    return MU0 * (T / T0) ** ALPHA_M


def v_mp(T):
    """Most probable molecular speed [m/s]."""
    return np.sqrt(2.0 * KB * T / M_AIR)


def mfp(T, P_Pa):
    """
    Gas mean free path [m].

    Parameters
    ----------
    T    : temperature [K]
    P_Pa : pressure [Pa]  (use torr_to_pa() to convert from Torr)
    """
    return mu(T) * v_mp(T) / P_Pa


def knudsen(T, P_Pa, a):
    """
    Knudsen number Kn = ℓ / a.

    Parameters
    ----------
    a : particle characteristic radius [m]
    """
    return mfp(T, P_Pa) / a


def cunningham(Kn):
    """
    Cunningham slip correction factor C_c(Kn).
    Accounts for gas rarefaction in the drag formula.
    """
    return 1.0 + Kn * (1.257 + 0.400 * np.exp(-0.55 / Kn))


def torr_to_pa(P_torr):
    """Convert pressure from Torr to Pascal."""
    return P_torr * TORR


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    T = 200.0
    P = torr_to_pa(5.0)
    a = 10e-6
    print(f"T = {T} K,  P = 5 Torr")
    print(f"  kappa = {kappa(T):.4f} W/(m·K)")
    print(f"  mu    = {mu(T):.3e} Pa·s")
    print(f"  v_mp  = {v_mp(T):.1f} m/s")
    print(f"  mfp   = {mfp(T, P)*1e6:.2f} µm")
    print(f"  Kn (a=10µm) = {knudsen(T, P, a):.3f}")
    print(f"  C_c   = {cunningham(knudsen(T, P, a)):.4f}")
