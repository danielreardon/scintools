"""
Unit tests for scintools.ththmod

Covers small, deterministic helper functions that do not require an actual
conjugate/secondary spectrum data set.
"""

import warnings

import numpy as np
import pytest
import astropy.units as u
from numpy.testing import assert_allclose

from scintools.ththmod import (
    svd_model,
    chi_par,
    len_arc,
    ext_find,
    fft_axis,
    unit_checks,
    arc_edges,
    min_edges,
    chunk_mask,
    mask_func,
)
from scintools.scint_utils import svd_reconstruct


# ---------------------------------------------------------------------------
# svd_model
# ---------------------------------------------------------------------------

def test_svd_model_reconstructs_rank1_matrix():
    u_vec = np.array([1.0, 2.0, 3.0])
    v_vec = np.array([4.0, 5.0, 6.0, 7.0])
    mat = np.outer(u_vec, v_vec).astype(complex)

    model = svd_model(mat, nmodes=1)

    assert model.shape == mat.shape
    assert_allclose(model, mat, atol=1e-8)


def test_svd_model_delegates_to_shared_core():
    # Regression: ththmod.svd_model and scint_utils.svd_reconstruct share
    # one implementation and must agree exactly.
    rng = np.random.default_rng(3)
    mat = rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))
    assert np.array_equal(svd_model(mat, nmodes=2),
                          svd_reconstruct(mat, nmodes=2))


# ---------------------------------------------------------------------------
# chunk_mask (shared mosaic weighting mask)
# ---------------------------------------------------------------------------

def test_chunk_mask_matches_inline_logic():
    # Reference implementation matching the pre-refactor inline block.
    def ref(shape, cf, ct, ncf, nct, cwf, cwt):
        mask = np.ones(shape)
        if cf > 0:
            mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
        if cf < ncf - 1:
            mask[cwf // 2:, :] *= 1 - mask_func(cwf // 2)[:, np.newaxis]
        if ct > 0:
            mask[:, : cwt // 2] *= mask_func(cwt // 2)
        if ct < nct - 1:
            mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
        return mask

    ncf, nct, cwf, cwt = 3, 3, 4, 6
    for cf in range(ncf):
        for ct in range(nct):
            got = chunk_mask((cwf, cwt), cf, ct, ncf, nct, cwf, cwt)
            assert np.array_equal(got, ref((cwf, cwt), cf, ct, ncf, nct,
                                           cwf, cwt))


def test_chunk_mask_single_chunk_is_all_ones():
    # A lone chunk (ncf=nct=1) has no neighbours to cross-fade with.
    mask = chunk_mask((4, 4), 0, 0, 1, 1, 4, 4)
    assert np.array_equal(mask, np.ones((4, 4)))


def test_svd_model_zero_modes_gives_zero_matrix():
    mat = np.eye(3, dtype=complex)
    model = svd_model(mat, nmodes=0)
    assert_allclose(model, np.zeros_like(mat), atol=1e-10)


# ---------------------------------------------------------------------------
# chi_par
# ---------------------------------------------------------------------------

def test_chi_par_matches_parabola_formula():
    x = 3.0
    A, x0, C = 2.0, 1.0, 5.0
    result = chi_par(x, A, x0, C)
    assert_allclose(result, A * (x - x0) ** 2 + C)


def test_chi_par_apex_value():
    A, x0, C = 4.0, 2.0, -1.0
    # at the apex x == x0, the parabola should equal C exactly
    assert_allclose(chi_par(x0, A, x0, C), C)


# ---------------------------------------------------------------------------
# len_arc
# ---------------------------------------------------------------------------

def test_len_arc_zero_at_origin():
    assert_allclose(len_arc(0.0, 2.0), 0.0)


def test_len_arc_approx_linear_for_small_curvature():
    # for very small curvature the arc length of a near-flat parabola
    # should approximately equal the x-offset
    result = len_arc(1.0, 1e-4)
    assert_allclose(result, 1.0, atol=1e-3)


def test_len_arc_is_odd_function():
    eta = 0.5
    assert_allclose(len_arc(2.0, eta), -len_arc(-2.0, eta))


# ---------------------------------------------------------------------------
# ext_find
# ---------------------------------------------------------------------------

def test_ext_find_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0]) * u.mHz
    y = np.array([10.0, 20.0, 30.0]) * u.us

    ext = ext_find(x, y)

    assert_allclose(ext, [0.5, 4.5, 5.0, 35.0])


# ---------------------------------------------------------------------------
# fft_axis
# ---------------------------------------------------------------------------

def test_fft_axis_length_matches_input():
    x = np.arange(10) * u.s
    fx = fft_axis(x, u.Hz, pad=0)
    assert len(fx) == len(x)
    assert fx.unit == u.Hz


def test_fft_axis_padding_increases_length():
    x = np.arange(8) * u.s
    fx_nopad = fft_axis(x, u.Hz, pad=0)
    fx_pad = fft_axis(x, u.Hz, pad=1)
    assert len(fx_pad) == 2 * len(fx_nopad)


# ---------------------------------------------------------------------------
# unit_checks
# ---------------------------------------------------------------------------

def test_unit_checks_assigns_units_when_missing():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = unit_checks(5.0, 'test', u.mHz)
    assert out.unit == u.mHz
    assert_allclose(out.value, 5.0)
    assert len(caught) == 1
    assert 'missing units' in str(caught[0].message)


def test_unit_checks_passes_through_correct_units():
    out = unit_checks(5.0 * u.mHz, 'test', u.mHz)
    assert out.unit == u.mHz
    assert_allclose(out.value, 5.0)


def test_unit_checks_converts_equivalent_units():
    out = unit_checks(5.0 * u.Hz, 'test', u.mHz)
    assert out.unit == u.mHz
    assert_allclose(out.value, 5000.0)


def test_unit_checks_raises_for_incompatible_units():
    with pytest.raises(u.UnitConversionError):
        unit_checks(5.0 * u.s, 'test', u.mHz)


# ---------------------------------------------------------------------------
# arc_edges / min_edges
# ---------------------------------------------------------------------------

def test_arc_edges_returns_symmetric_even_length_array():
    eta = 1.0 * u.us / u.mHz ** 2
    dfd = 1.0 * u.mHz
    dtau = 1.0 * u.us
    fd_max = 10.0 * u.mHz

    edges = arc_edges(eta, dfd, dtau, fd_max, 10)

    assert len(edges) == 10
    # symmetric about zero
    assert_allclose(edges, -edges[::-1])
    # strictly increasing
    assert np.all(np.diff(edges) > 0)


def test_min_edges_spans_requested_range():
    eta = 1.0 * u.us / u.mHz ** 2
    fd = np.linspace(-10, 10, 21) * u.mHz
    tau = np.linspace(-5, 5, 11) * u.us
    fd_lim = 5 * u.mHz

    edges = min_edges(fd_lim, fd, tau, eta, factor=2)

    assert_allclose(edges[0].value, -5.0)
    assert_allclose(edges[-1].value, 5.0)
    # even number of edges as documented
    assert len(edges) % 2 == 0
