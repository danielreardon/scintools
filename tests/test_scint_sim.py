"""
Unit tests for scintools.scint_sim

Uses small grid sizes so that the (otherwise expensive) simulation and ACF
calculations run in well under a second.
"""

import numpy as np
import pytest

from scintools.scint_sim import Simulation, ACF


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_simulation_produces_expected_dynspec_shape():
    sim = Simulation(ns=32, nf=16, seed=42, verbose=False)

    # self.dyn is transposed to (nchan, nsub)
    assert sim.dyn.shape == (sim.nchan, sim.nsub)
    assert sim.dyn.shape == (16, 32)


def test_simulation_output_is_finite_and_real():
    sim = Simulation(ns=32, nf=16, seed=1, verbose=False)
    assert np.all(np.isfinite(sim.dyn))
    assert np.isrealobj(sim.dyn)


def test_simulation_frequency_and_time_axes_lengths():
    sim = Simulation(ns=20, nf=12, seed=7, verbose=False)
    assert len(sim.freqs) == sim.nchan
    assert len(sim.times) == sim.nsub


def test_simulation_is_deterministic_with_seed():
    sim1 = Simulation(ns=16, nf=8, seed=123, verbose=False)
    sim2 = Simulation(ns=16, nf=8, seed=123, verbose=False)
    np.testing.assert_allclose(sim1.dyn, sim2.dyn)


def test_simulation_intensity_is_non_negative():
    # dynamic spectrum intensity (spi = |E|^2) should be non-negative
    sim = Simulation(ns=16, nf=8, seed=5, verbose=False)
    assert np.all(sim.dyn >= 0)


# ---------------------------------------------------------------------------
# ACF - blocked by a real numpy>=2.0 incompatibility, see report
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "scintools/scint_sim.py:589 and :634 (ACF.calc_acf) use "
        "`dtype=np.complex_`, an alias removed in numpy>=2.0 "
        "(NEP 51 / numpy 2.0 release notes); instantiating ACF() raises "
        "AttributeError: `np.complex_` was removed in the NumPy 2.0 "
        "release. Use `np.complex128` instead."
    ),
    strict=True,
)
def test_acf_small_grid_shape_and_peak():
    acf = ACF(nt=11, nf=11, taumax=2, dnumax=2, amp=1.0, wn=0.0,
              auto_sampling=True)

    assert acf.acf.shape == (acf.nf, acf.nt)
    # peak of the ACF should be at the centre and equal to amp
    centre = (acf.acf.shape[0] // 2, acf.acf.shape[1] // 2)
    np.testing.assert_allclose(acf.acf[centre], acf.amp, atol=1e-6)
    assert np.isclose(np.max(acf.acf), acf.amp, atol=1e-6)
