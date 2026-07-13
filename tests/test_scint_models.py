"""
Unit tests for scintools.scint_models

Focuses on pure model/residual functions that can be exercised with small
synthetic data and lmfit Parameters objects, without invoking an actual
fitter (Minimizer) or optional heavy dependencies (emcee, bilby).
"""

import numpy as np
import pytest
from lmfit import Parameters
from numpy.testing import assert_allclose

from scintools.scint_models import (
    fit_parabola,
    fit_log_parabola,
    powerspectrum_model,
    tau_acf_model,
    dnu_acf_model,
    scint_acf_model,
    scint_sspec_model,
    arc_power_curve,
    effective_velocity_annual,
)


# ---------------------------------------------------------------------------
# fit_parabola / fit_log_parabola
# ---------------------------------------------------------------------------

def test_fit_parabola_recovers_known_peak():
    x = np.linspace(-5, 5, 21)
    true_peak = 1.5
    y = -2 * (x - true_peak) ** 2 + 10

    yfit, peak, peak_error = fit_parabola(x, y)

    assert_allclose(peak, true_peak, atol=1e-6)
    assert_allclose(yfit, y, atol=1e-6)
    assert peak_error < 1e-6


def test_fit_log_parabola_recovers_known_peak():
    x = np.linspace(1, 10, 20)
    true_log_peak = 1.0
    y = -3 * (np.log(x) - true_log_peak) ** 2 + 5

    yfit, peak, peak_error = fit_log_parabola(x, y)

    assert_allclose(peak, np.e, rtol=1e-5)
    assert peak_error < 1e-6


# ---------------------------------------------------------------------------
# powerspectrum_model
# ---------------------------------------------------------------------------

def test_powerspectrum_model_zero_residual_at_truth():
    params = Parameters()
    params.add('amp', value=2.0)
    params.add('wn', value=0.5)
    params.add('alpha', value=-1.0)

    xdata = np.array([1.0, 2.0, 4.0])
    ydata = 0.5 + 2.0 * xdata ** (-1.0)

    resid = powerspectrum_model(params, xdata, ydata)
    assert_allclose(resid, np.zeros_like(xdata), atol=1e-12)


def test_powerspectrum_model_nonzero_residual_when_off():
    params = Parameters()
    params.add('amp', value=2.0)
    params.add('wn', value=0.5)
    params.add('alpha', value=-1.0)

    xdata = np.array([1.0, 2.0, 4.0])
    ydata = np.array([100.0, 100.0, 100.0])

    resid = powerspectrum_model(params, xdata, ydata)
    assert np.all(np.abs(resid) > 0)


# ---------------------------------------------------------------------------
# tau_acf_model / dnu_acf_model / scint_acf_model
# ---------------------------------------------------------------------------

def _tau_params():
    p = Parameters()
    p.add('amp', value=1.0)
    p.add('tau', value=5.0)
    p.add('alpha', value=5 / 3)
    return p


def _dnu_params():
    p = Parameters()
    p.add('amp', value=1.0)
    p.add('dnu', value=3.0)
    return p


def test_tau_acf_model_zero_residual_at_truth():
    params = _tau_params()
    xdata = np.linspace(0, 10, 11)
    amp, tau, alpha = 1.0, 5.0, 5 / 3
    model = amp * np.exp(-np.divide(xdata, tau) ** alpha)
    model = model * (1 - xdata / max(xdata))
    ydata = model.copy()
    weights = np.ones_like(xdata)

    resid = tau_acf_model(params, xdata, ydata, weights)
    assert_allclose(resid, np.zeros_like(xdata), atol=1e-10)


def test_tau_acf_model_does_not_mutate_caller_weights():
    # Regression: the model used to zero weights[0] in place, mutating the
    # caller's array.
    params = _tau_params()
    xdata = np.linspace(0, 10, 11)
    ydata = np.ones_like(xdata)
    weights = np.ones_like(xdata)
    tau_acf_model(params, xdata, ydata, weights)
    assert weights[0] == 1.0  # untouched


def test_dnu_acf_model_zero_residual_at_truth():
    params = _dnu_params()
    xdata = np.linspace(0, 6, 7)
    amp, dnu = 1.0, 3.0
    model = amp * np.exp(-np.divide(xdata, dnu / np.log(2)))
    model = model * (1 - xdata / max(xdata))
    ydata = model.copy()
    weights = np.ones_like(xdata)

    resid = dnu_acf_model(params, xdata, ydata, weights)
    assert_allclose(resid, np.zeros_like(xdata), atol=1e-10)


def test_scint_acf_model_concatenates_both_residuals():
    params = _tau_params()
    params.add('dnu', value=3.0)

    xdata_t = np.linspace(0, 10, 11)
    amp, tau, alpha = 1.0, 5.0, 5 / 3
    model_t = amp * np.exp(-np.divide(xdata_t, tau) ** alpha)
    model_t = model_t * (1 - xdata_t / max(xdata_t))
    ydata_t = model_t.copy()

    xdata_f = np.linspace(0, 6, 7)
    dnu = 3.0
    model_f = amp * np.exp(-np.divide(xdata_f, dnu / np.log(2)))
    model_f = model_f * (1 - xdata_f / max(xdata_f))
    ydata_f = model_f.copy()

    weights_t = np.ones_like(xdata_t)
    weights_f = np.ones_like(xdata_f)

    resid = scint_acf_model(params, [xdata_t, xdata_f], [ydata_t, ydata_f],
                             [weights_t, weights_f])

    assert len(resid) == len(xdata_t) + len(xdata_f)
    assert_allclose(resid, np.zeros_like(resid), atol=1e-10)


# ---------------------------------------------------------------------------
# arc_power_curve - flagged as broken (see report)
# ---------------------------------------------------------------------------

# arc_power_curve is an unimplemented template. It now raises an explicit
# NotImplementedError instead of silently producing a broadcasting error.
def test_arc_power_curve_raises_not_implemented():
    xdata = np.array([1.0, 2.0, 3.0])
    ydata = np.array([1.0, 2.0, 3.0])
    with pytest.raises(NotImplementedError):
        arc_power_curve(None, xdata, ydata, None)


# ---------------------------------------------------------------------------
# scint_sspec_model - regression: the _sspec_model callees now accept an
# optional weights argument, so this no longer raises TypeError.
# ---------------------------------------------------------------------------

def test_scint_sspec_model_runs_with_weights():
    params = _tau_params()
    params.add('dnu', value=3.0)
    xdata_t = np.linspace(0, 10, 11)
    xdata_f = np.linspace(0, 6, 7)
    ydata_t = np.ones_like(xdata_t)
    ydata_f = np.ones_like(xdata_f)
    weights_t = np.ones_like(xdata_t)
    weights_f = np.ones_like(xdata_f)

    resid = scint_sspec_model(params,
                              (xdata_t, xdata_f),
                              (ydata_t, ydata_f),
                              (weights_t, weights_f))
    assert len(resid) == len(xdata_t) + len(xdata_f)
    assert np.all(np.isfinite(resid))


# ---------------------------------------------------------------------------
# effective_velocity_annual - regression: a missing inclination parameter
# now raises a clear KeyError instead of a later NameError on INC.
# ---------------------------------------------------------------------------

def test_effective_velocity_annual_missing_inclination_raises():
    params = Parameters()
    params.add('A1', value=1.0)
    params.add('PB', value=10.0)
    params.add('ECC', value=0.0)
    params.add('OM', value=0.0)
    params.add('KOM', value=0.0)
    ta = np.array([0.0, 1.0])
    ve = np.zeros_like(ta)
    with pytest.raises(KeyError):
        effective_velocity_annual(params, ta, ve, ve)
