"""Tests for gas_properties.py"""
import sys; sys.path.insert(0, '../src')
import numpy as np
from gas_properties import kappa, mu, v_mp, mfp, cunningham, torr_to_pa

def test_kappa_at_reference():
    assert abs(kappa(273.0) - 0.0241) < 1e-6

def test_mu_at_reference():
    assert abs(mu(273.0) - 1.81e-5) < 1e-9

def test_cunningham_large_kn():
    # For Kn >> 1, C_c ≈ 1.257*Kn  (dominant term)
    Cc = cunningham(100.)
    assert Cc > 100.

def test_cunningham_small_kn():
    # For Kn → 0, C_c → 1
    assert abs(cunningham(0.001) - 1.0) < 0.01

def test_mfp_pressure_scaling():
    T = 200.
    P1 = torr_to_pa(5.)
    P2 = torr_to_pa(10.)
    # mfp ∝ 1/P
    assert abs(mfp(T, P1) / mfp(T, P2) - 2.0) < 0.01
