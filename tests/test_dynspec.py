"""
Unit tests for the lightweight wrapper classes in scintools.dynspec.

These cover regressions for bugs found in the code review: BasicDyn's
mutable-default / ``.size``-on-list handling and SimDyn reading
``self.header`` before it was set.
"""

import numpy as np
import pytest

from scintools.dynspec import BasicDyn, SimDyn
from scintools.scint_sim import Simulation


# ---------------------------------------------------------------------------
# BasicDyn
# ---------------------------------------------------------------------------

def test_basicdyn_accepts_list_axes():
    # Regression: previously raised AttributeError because the guard called
    # ``.size`` on a list default.
    dyn = np.ones((4, 3))
    bd = BasicDyn(dyn, times=[0, 1, 2], freqs=[1400, 1401, 1402, 1403])
    assert bd.nsub == 3
    assert bd.nchan == 4
    np.testing.assert_allclose(bd.tobs, 3.0)
    np.testing.assert_allclose(bd.dt, 1.0)


def test_basicdyn_missing_axes_raises_valueerror():
    # Regression: should raise the intended ValueError, not AttributeError.
    with pytest.raises(ValueError):
        BasicDyn(np.ones((1, 1)))


def test_basicdyn_default_header_not_shared():
    # Regression: mutable default header must not be shared between instances.
    a = BasicDyn(np.ones((2, 2)), times=[0, 1], freqs=[1, 2])
    b = BasicDyn(np.ones((2, 2)), times=[0, 1], freqs=[1, 2])
    a.header.append("mutated")
    assert b.header == ["BasicDyn"]


# ---------------------------------------------------------------------------
# SimDyn
# ---------------------------------------------------------------------------

def test_simdyn_from_simulation():
    # Regression: SimDyn.__init__ used to do ``self.header = self.header``,
    # raising AttributeError before header was defined.
    sim = Simulation(ns=32, nf=8, verbose=False)
    sd = SimDyn(sim)
    assert isinstance(sd.header, list)
    assert sd.dyn.ndim == 2
    assert sd.nsub > 0 and sd.nchan > 0
