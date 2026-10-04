"""
quaternion_utils.py
--------------------
Quaternion algebra helpers for 6-DOF rigid-body orientation.

Convention: q = (q0, q1, q2, q3) where q0 is the scalar part.
All quaternions are assumed unit quaternions unless noted.
"""

import numpy as np


def quat_mult(q1, q2):
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_matrix(q):
    """
    Convert unit quaternion to 3×3 rotation matrix (body → lab frame).
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),    2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_from_axis_angle(axis, angle):
    """
    Create unit quaternion from rotation axis and angle [rad].

    Parameters
    ----------
    axis  : (3,) unit vector
    angle : rotation angle [rad]
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return np.r_[np.cos(angle/2), np.sin(angle/2) * axis]


def quat_identity():
    """Identity quaternion (no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q):
    """Normalize quaternion to unit length."""
    return q / np.linalg.norm(q)


def quat_derivative(q, omega_body):
    """
    Time derivative of quaternion given angular velocity in body frame.

    q̇ = ½ q ⊗ [0, ω_body]
    """
    return 0.5 * quat_mult(q, np.r_[0.0, omega_body])


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = quat_identity()
    print("Identity matrix:\n", quat_to_matrix(q))

    q90z = quat_from_axis_angle([0, 0, 1], np.pi/2)
    R = quat_to_matrix(q90z)
    print("\n90° rotation about z:")
    print(R)
    x_rot = R @ np.array([1, 0, 0])
    print("x-hat rotated:", np.round(x_rot, 6), "  (expect [0, 1, 0])")

    # Norm preservation
    q2 = quat_mult(q90z, q90z)   # 180° about z
    print("\n|q²| =", np.linalg.norm(q2), "  (expect 1.0)")
