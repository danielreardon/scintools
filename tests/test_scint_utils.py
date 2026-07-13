"""
Unit tests for scintools.scint_utils

These tests focus on pure/deterministic helper functions that do not
require external data files, GUIs, or optional heavy dependencies.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scintools.scint_utils import (
    autocorr,
    is_valid,
    cov_to_corr,
    float_array_from_dict,
    difference,
    mjd_to_year,
    find_nearest,
    longest_run_of_zeros,
    centres_to_edges,
    interp_nan_2d,
    svd_model,
    svd_reconstruct,
    get_window,
    scint_velocity,
    read_par,
    write_results,
    read_results,
    slow_FT,
)


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------

def test_is_valid_masks_nan_and_inf():
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    result = is_valid(arr)
    assert_allclose(result, [True, False, False, False, True])


def test_is_valid_all_finite():
    arr = np.array([1.0, 2.0, -3.5])
    assert np.all(is_valid(arr))


# ---------------------------------------------------------------------------
# autocorr
# ---------------------------------------------------------------------------

def test_autocorr_shape_and_normalisation():
    rng = np.random.default_rng(0)
    small = np.ma.array(rng.random((3, 3)))
    ac = autocorr(small)
    # output is (2*nr, 2*nc)
    assert ac.shape == (6, 6)
    # normalised by its own max
    assert_allclose(np.nanmax(ac), 1.0)


def test_autocorr_symmetric_for_constant_array():
    # a constant array has zero variance in the numerator terms away from
    # zero lag, but the autocorrelation should still be point-symmetric:
    # autocorr[x, y] == autocorr[-x, -y]
    rng = np.random.default_rng(1)
    arr = np.ma.array(rng.random((4, 4)))
    ac = autocorr(arr)
    flipped = ac[::-1, ::-1]
    # the peak (lag 0,0) maps to itself under this flip only for even-sized
    # arrays shifted by one; check general point symmetry away from edges
    assert_allclose(ac, np.roll(np.roll(flipped, 1, axis=0), 1, axis=1),
                     atol=1e-10)


# ---------------------------------------------------------------------------
# cov_to_corr
# ---------------------------------------------------------------------------

def test_cov_to_corr_diagonal_is_one():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    corr = cov_to_corr(cov)
    assert_allclose(np.diag(corr), [1.0, 1.0])
    # known off-diagonal value: 2 / sqrt(4*9) = 1/3
    assert_allclose(corr[0, 1], 1.0 / 3.0)
    assert_allclose(corr[1, 0], 1.0 / 3.0)


def test_cov_to_corr_zero_variance_entry_is_zero():
    cov = np.array([[0.0, 0.0], [0.0, 9.0]])
    corr = cov_to_corr(cov)
    assert corr[0, 0] == 0
    assert corr[0, 1] == 0
    assert corr[1, 0] == 0


# ---------------------------------------------------------------------------
# float_array_from_dict
# ---------------------------------------------------------------------------

def test_float_array_from_dict_converts_strings():
    d = {'x': ['1', '2.5', '3']}
    result = float_array_from_dict(d, 'x')
    assert_allclose(result, [1.0, 2.5, 3.0])


def test_float_array_from_dict_handles_none_as_nan():
    d = {'x': ['1', 'None', '3']}
    result = float_array_from_dict(d, 'x')
    assert result[0] == 1.0
    assert np.isnan(result[1])
    assert result[2] == 3.0


# ---------------------------------------------------------------------------
# difference
# ---------------------------------------------------------------------------

def test_difference_matches_central_difference():
    x = np.array([0.0, 1.0, 3.0, 6.0])
    result = difference(x)
    # first: (1-0)/2 = 0.5
    # second: (3-0)/2 = 1.5
    # third: (6-1)/2 = 2.5
    # last: (6-3)/2 = 1.5
    assert_allclose(result, [0.5, 1.5, 2.5, 1.5])


def test_difference_same_length_as_input():
    x = np.linspace(0, 10, 15)
    result = difference(x)
    assert len(result) == len(x)


# ---------------------------------------------------------------------------
# mjd_to_year
# ---------------------------------------------------------------------------

def test_mjd_to_year_known_epoch():
    # MJD 51544.0 corresponds to 2000-01-01 00:00 UTC
    yr = mjd_to_year(51544.0)
    assert_allclose(yr, 2000.0, atol=0.01)


def test_mjd_to_year_another_epoch():
    # MJD 58849.0 corresponds to 2020-01-01 00:00 UTC
    yr = mjd_to_year(58849.0)
    assert_allclose(yr, 2020.0, atol=0.01)


# ---------------------------------------------------------------------------
# find_nearest
# ---------------------------------------------------------------------------

def test_find_nearest_basic():
    arr = [1, 5, 10]
    assert find_nearest(arr, 6) == 1
    assert find_nearest(arr, 0) == 0
    assert find_nearest(arr, 11) == 2


def test_find_nearest_exact_match():
    arr = np.array([2.0, 4.0, 8.0])
    assert find_nearest(arr, 4.0) == 1


# ---------------------------------------------------------------------------
# longest_run_of_zeros
# ---------------------------------------------------------------------------

def test_longest_run_of_zeros_basic():
    arr = [1, 0, 0, 0, 1, 0, 0, 1]
    assert longest_run_of_zeros(arr) == 3


def test_longest_run_of_zeros_no_zeros():
    arr = [1, 2, 3]
    assert longest_run_of_zeros(arr) == 0


def test_longest_run_of_zeros_all_zeros():
    arr = [0, 0, 0, 0]
    assert longest_run_of_zeros(arr) == 4


# ---------------------------------------------------------------------------
# centres_to_edges
# ---------------------------------------------------------------------------

def test_centres_to_edges_length():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    edges = centres_to_edges(arr)
    assert len(edges) == len(arr) + 1


def test_centres_to_edges_values():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    edges = centres_to_edges(arr)
    assert_allclose(edges, [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])


# ---------------------------------------------------------------------------
# interp_nan_2d
# ---------------------------------------------------------------------------

def test_interp_nan_2d_fills_interior_nan():
    ii, jj = np.meshgrid(np.arange(5), np.arange(5))
    data = (ii + jj).astype(float)
    data_with_nan = data.copy()
    data_with_nan[2, 2] = np.nan

    filled = interp_nan_2d(data_with_nan)

    assert not np.isnan(filled[2, 2])
    # since the underlying function is linear (i+j), interpolation at an
    # interior point should recover the original value almost exactly
    assert_allclose(filled[2, 2], data[2, 2], atol=1e-8)
    # points that were not nan should stay (nearly) the same
    mask = ~np.isnan(data_with_nan)
    assert_allclose(filled[mask], data[mask], atol=1e-8)


# ---------------------------------------------------------------------------
# svd_model
# ---------------------------------------------------------------------------

def test_svd_model_reconstructs_rank1_matrix():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0, 7.0])
    mat = np.outer(u, v).astype(complex)

    arr_out, model = svd_model(mat, nmodes=1)

    # a rank-1 matrix should be reconstructed (almost) exactly by 1 SVD mode
    assert_allclose(np.abs(model), np.abs(mat), atol=1e-8)
    # arr_out = mat / |model| should therefore be close to the phase of mat
    assert arr_out.shape == mat.shape


# ---------------------------------------------------------------------------
# get_window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('window', ['hanning', 'hamming', 'blackman',
                                     'bartlett'])
def test_get_window_returns_expected_lengths(window):
    nt, nf = 20, 16
    chan_window, subint_window = get_window(nt, nf, window=window, frac=0.1)
    assert len(chan_window) == nt
    assert len(subint_window) == nf


# ---------------------------------------------------------------------------
# scint_velocity
# ---------------------------------------------------------------------------

def test_scint_velocity_no_params():
    # freq in MHz is divided by 1e3 to get GHz; with freq=1000 -> 1 GHz
    a = 2.53e4
    viss = scint_velocity(None, dnu=1.0, tau=1.0, freq=1000.0, a=a)
    assert_allclose(viss, a)


def test_scint_velocity_with_errors():
    a = 2.53e4
    dnu, tau, freq = 1.0, 1.0, 1000.0
    dnuerr, tauerr = 0.1, 0.1
    viss, viss_err = scint_velocity(None, dnu=dnu, tau=tau, freq=freq,
                                     dnuerr=dnuerr, tauerr=tauerr, a=a)
    assert_allclose(viss, a)
    # manual calculation of expected error (coeff_err = 0 when params=None)
    freq_ghz = freq / 1e3
    expected_err = (1.0 / (freq_ghz * tau)) * np.sqrt(
        a ** 2 * ((dnuerr ** 2 / (4 * dnu)) + (dnu * tauerr ** 2 / tau ** 2))
    )
    assert_allclose(viss_err, expected_err)


# ---------------------------------------------------------------------------
# read_par
# ---------------------------------------------------------------------------

def test_read_par_basic_parsing(tmp_path):
    parfile = tmp_path / "test.par"
    parfile.write_text(
        "PSRJ J0000+0000\n"
        "F0 100.0 1 1e-10\n"
        "DM 20.0\n"
        "# a comment line\n"
        "JUMP -be PDFB 0.0\n"  # should be ignored
    )

    par = read_par(str(parfile))

    assert par['PSRJ'] == 'J0000+0000'
    assert par['PSRJ_TYPE'] == 's'

    assert_allclose(par['F0'], 100.0)
    assert_allclose(par['F0_ERR'], 1e-10)
    assert par['F0_TYPE'] == 'f'

    assert_allclose(par['DM'], 20.0)
    assert 'JUMP' not in par


def test_read_par_ignores_dmmodel_lines(tmp_path):
    parfile = tmp_path / "test2.par"
    parfile.write_text(
        "PX 0.5\n"
        "DMMODEL something 1.0\n"
    )
    par = read_par(str(parfile))
    assert_allclose(par['PX'], 0.5)
    assert 'DMMODEL' not in par


# ---------------------------------------------------------------------------
# write_results / read_results (round trip)
# ---------------------------------------------------------------------------

class _FakeDyn:
    """Minimal stand-in for a Dynspec object with only the attributes
    write_results() looks for."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_write_and_read_results_roundtrip(tmp_path):
    outfile = tmp_path / "results.txt"

    dyn1 = _FakeDyn(name='obs1', mjd=59000.0, freq=1400.0, bw=100.0,
                    tobs=1800.0, dt=10.0, df=1.0, tau=100.0, tauerr=5.0)
    dyn2 = _FakeDyn(name='obs2', mjd=59001.0, freq=1400.0, bw=100.0,
                    tobs=1800.0, dt=10.0, df=1.0, tau=110.0, tauerr=6.0)

    write_results(str(outfile), dyn1)
    write_results(str(outfile), dyn2)

    results = read_results(str(outfile))

    assert results['name'] == ['obs1', 'obs2']
    assert results['tau'] == ['100.0', '110.0']
    assert results['tauerr'] == ['5.0', '6.0']

    # header should only be written once
    lines = outfile.read_text().splitlines()
    assert lines[0] == "name,mjd,freq,bw,tobs,dt,df,tau,tauerr"
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# svd_reconstruct / svd_model (shared SVD core)
# ---------------------------------------------------------------------------

def test_svd_reconstruct_matches_svd_model_core():
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))
    _, model = svd_model(arr.copy(), nmodes=2)
    assert np.array_equal(model, svd_reconstruct(arr.copy(), nmodes=2))


def test_svd_model_normalises_by_abs_model():
    rng = np.random.default_rng(8)
    arr = rng.standard_normal((5, 4)) + 1j * rng.standard_normal((5, 4))
    normed, model = svd_model(arr.copy(), nmodes=1)
    assert_allclose(normed, arr / np.abs(model))


# ---------------------------------------------------------------------------
# slow_FT - regression: previously raised TypeError (fftshift axis kwarg)
# ---------------------------------------------------------------------------

def test_slow_ft_runs_and_has_expected_shape():
    rng = np.random.default_rng(9)
    dynspec = rng.standard_normal((8, 6))
    freqs = np.linspace(1400.0, 1410.0, 6)
    ss = slow_FT(dynspec, freqs)
    assert ss.shape == dynspec.shape
    assert ss.dtype == np.complex128


def test_slow_ft_default_fref_is_midband():
    rng = np.random.default_rng(10)
    dynspec = rng.standard_normal((8, 6))
    freqs = np.linspace(1400.0, 1410.0, 6)
    ss_default = slow_FT(dynspec, freqs)
    ss_explicit = slow_FT(dynspec, freqs, fref=freqs[len(freqs) // 2])
    assert np.array_equal(ss_default, ss_explicit)
