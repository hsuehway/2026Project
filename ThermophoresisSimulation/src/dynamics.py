"""
dynamics.py  —  Component D
-----------------------------
Six-degree-of-freedom Newton-Euler dynamics for a rigid body
levitating in the thermophoretic gas cell.

State vector (13 components)
-----------------------------
s = [ r_cm (3),  v_cm (3),  q (4),  omega_body (3) ]
      position   velocity   orient.  angular velocity (body frame)

Integrator: scipy RK45 (adaptive step, dense output)

Usage
-----
    from dynamics import simulate, equilibrium_height

    gc   = solve_gas_cell(T_h=277.)
    body = make_sphere(R=10e-6, rho=998.)
    s0   = initial_state(pos=[0, 0, 2e-3])
    sol  = simulate(body, gc, P_Pa=torr_to_pa(5.), s0, t_span=(0, 60.))
"""

import numpy as np
from scipy.integrate import solve_ivp

from gas_properties import mu, mfp, knudsen, cunningham, KB, M_AIR
from force_models    import (thermo_force_sphere, thermo_force_mesh,
                              stokes_drag, rotational_drag, FT_FM, f_T_sphere)
from quaternion_utils import quat_mult, quat_to_matrix, quat_normalize, quat_derivative

G = 9.8   # gravitational acceleration [m/s²]


# ── Initial state ──────────────────────────────────────────────────────────────

def initial_state(pos, vel=None, q=None, omega=None):
    """
    Build a 13-component initial state vector.

    Parameters
    ----------
    pos   : (3,) initial position [m]
    vel   : (3,) initial velocity [m/s]  (default zero)
    q     : (4,) initial orientation quaternion (default identity)
    omega : (3,) initial angular velocity in body frame [rad/s] (default zero)
    """
    if vel   is None: vel   = np.zeros(3)
    if q     is None: q     = np.array([1., 0., 0., 0.])
    if omega is None: omega = np.zeros(3)
    return np.r_[pos, vel, q, omega]


# ── ODE right-hand side ────────────────────────────────────────────────────────

def deriv(t, s, body, gas_cell, P_Pa, model, kappa21, use_mesh):
    """
    Time derivative of the state vector.

    Parameters
    ----------
    use_mesh : if True, use surface-integral force (non-spherical);
               if False, use sphere fast path.
    """
    r_cm  = s[0:3]
    v_cm  = s[3:6]
    q     = quat_normalize(s[6:10])
    omega = s[10:13]   # body frame

    Rmat  = quat_to_matrix(q)
    T     = gas_cell.T_at(r_cm)

    # ── Forces ──────────────────────────────────────────────────────────────
    if use_mesh:
        F_th, tau_th = thermo_force_mesh(
            body, r_cm, q, gas_cell, P_Pa, model, kappa21)
    else:
        F_th  = thermo_force_sphere(r_cm, body.r_eq, gas_cell, P_Pa, model, kappa21)
        tau_th = np.zeros(3)

    F_grav = np.array([0.0, 0.0, -body.mass * G])
    F_drag = stokes_drag(v_cm, T, P_Pa, body.r_eq)

    # ── Translational acceleration ───────────────────────────────────────────
    a_cm = (F_th + F_grav + F_drag) / body.mass

    # ── Rotational (Euler's equations in body frame) ─────────────────────────
    tau_drag_body = rotational_drag(omega, T, P_Pa, body.r_eq)
    tau_body      = Rmat.T @ tau_th + tau_drag_body

    # Principal moments (eigenvalues of I_body)
    I_vals = np.linalg.eigvalsh(body.I_body)   # sorted ascending
    I1, I2, I3 = I_vals

    # Euler equations: I ω̇ = τ - ω × (I ω)
    Iw = I_vals * omega
    omega_dot = np.array([
        (tau_body[0] - (I3 - I2) * omega[1] * omega[2]) / I1,
        (tau_body[1] - (I1 - I3) * omega[2] * omega[0]) / I2,
        (tau_body[2] - (I2 - I1) * omega[0] * omega[1]) / I3,
    ])

    # ── Quaternion kinematics ────────────────────────────────────────────────
    q_dot = quat_derivative(q, omega)

    return np.r_[v_cm, a_cm, q_dot, omega_dot]


# ── Main integrator ────────────────────────────────────────────────────────────

def simulate(body, gas_cell, P_Pa, s0, t_span,
             model="yamamoto", kappa21=0.03, use_mesh=False,
             rtol=1e-6, atol=1e-9, max_step=1e-2):
    """
    Integrate the 6-DOF equations of motion.

    Parameters
    ----------
    body      : RigidBody object
    gas_cell  : GasCell object
    P_Pa      : pressure [Pa]
    s0        : (13,) initial state vector (use initial_state())
    t_span    : (t0, tf) integration interval [s]
    use_mesh  : True for non-spherical surface-integral force
    max_step  : maximum allowed time step [s]

    Returns
    -------
    scipy OdeSolution with fields t, y (13 × N_steps), sol (dense output)
    """
    return solve_ivp(
        deriv,
        t_span, s0,
        method="RK45",
        args=(body, gas_cell, P_Pa, model, kappa21, use_mesh),
        rtol=rtol, atol=atol,
        max_step=max_step,
        dense_output=True,
    )


# ── Convenience extractors ─────────────────────────────────────────────────────

def extract_trajectory(sol):
    """
    Extract position, velocity, quaternion, angular velocity from ODE solution.

    Returns
    -------
    t     : (N,) time array [s]
    r_cm  : (N, 3) position [m]
    v_cm  : (N, 3) velocity [m/s]
    q     : (N, 4) quaternion
    omega : (N, 3) angular velocity in body frame [rad/s]
    """
    y = sol.y.T   # (N, 13)
    return sol.t, y[:, 0:3], y[:, 3:6], y[:, 6:10], y[:, 10:13]


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from gas_properties import torr_to_pa
    from gas_cell       import solve_gas_cell
    from rigid_body     import make_sphere
    from force_models   import equilibrium_height

    print("Solving gas cell ...")
    gc   = solve_gas_cell(T_h=277., Nr=100, Nz=60)
    P_Pa = torr_to_pa(5.0)
    a    = 10e-6   # 10 µm sphere
    rho  = 998.0

    print("Finding equilibrium height ...")
    z_eq = equilibrium_height(a, rho, gc, P_Pa, model="yamamoto")
    if z_eq is not None:
        print(f"  z_eq = {z_eq*1e3:.2f} mm")
    else:
        print("  No equilibrium found.")

    print("\nSimulating 5 s of dynamics ...")
    body = make_sphere(a, rho)
    s0   = initial_state(pos=[0., 0., 1e-3])   # start 1 mm above bottom plate
    sol  = simulate(body, gc, P_Pa, s0, t_span=(0, 5.0),
                    model="yamamoto", max_step=0.05)
    t, r, v, q, om = extract_trajectory(sol)
    print(f"  Final position: z = {r[-1, 2]*1e3:.2f} mm  (z_eq = {z_eq*1e3:.2f} mm)")
    print(f"  Steps taken: {len(t)}")
