#!/usr/bin/env python

"""
scint_models.py
----------------------------------
Scintillation models.

A library of scintillation models to use with ``lmfit``, ``emcee``, or
``bilby`` for modelling and fitting pulsar scintillation data. This
includes:

* 1D and 2D autocorrelation function (ACF) models for measuring the
  scintillation timescale and decorrelation bandwidth.
* Secondary spectrum models, including power-law noise spectra and
  scintillation arc power/curvature.
* Effective velocity models (annual and orbital terms, and the "thin
  screen" relation between arc curvature and effective velocity) used
  to infer pulsar distances, orbital inclinations, and other binary
  parameters from scintillation arcs.

Most fitting-model functions share a common calling convention:

    Inputs (at least):
        params
        xdata
        ydata
        weights

    Output:
        residuals = (ydata - model) * weights

Some functions use additional inputs, and a few (documented below,
e.g. ``effective_velocity_annual``, ``arc_weak``, ``arc_weak_2d``) do
not return residuals for a fitter but instead return the model values
directly. The :func:`fitter` function is the common entry point used
to drive these models with ``lmfit``'s ``Minimizer`` (least-squares or
``emcee``/MCMC).
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scintools.scint_sim import ACF
from lmfit import Minimizer


def fitter(model, params, args, mcmc=False, pos=None, nwalkers=100,
           steps=1000, burn=0.2, progress=True, workers=1,
           nan_policy='raise', max_nfev=None, thin=10, is_weighted=True):
    """
    Common entry point for fitting one of the residual-returning
    models in this module using ``lmfit``.

    Wraps the model function and starting parameters in an ``lmfit``
    ``Minimizer`` and either performs a least-squares minimisation, or
    (if ``mcmc=True``) runs the ``emcee`` MCMC sampler via
    ``Minimizer.emcee``.

    Parameters
    ----------
    model : callable
        A model function from this module (e.g. `tau_acf_model`,
        `scint_acf_model`, `scint_acf_model_2d_approx`,
        `powerspectrum_model`) that takes ``params`` followed by the
        contents of `args` and returns the (weighted) residuals
        between data and model.
    params : lmfit.Parameters
        Initial/starting parameters for the fit.
    args : tuple
        Extra positional arguments passed to `model` after `params`
        (typically some combination of ``xdata``, ``ydata``, and
        ``weights``).
    mcmc : bool, optional
        If True, sample the posterior with ``emcee`` instead of doing
        a least-squares fit. Default is False.
    pos : array_like, optional
        Initial walker positions passed to ``Minimizer.emcee`` (only
        used if `mcmc` is True). Default is None (let ``emcee``
        initialise the walkers).
    nwalkers : int, optional
        Number of ``emcee`` walkers (only used if `mcmc` is True).
        Default is 100.
    steps : int, optional
        Number of ``emcee`` steps to run (only used if `mcmc` is
        True). Default is 1000.
    burn : float, optional
        Fraction of `steps` to discard as burn-in (only used if `mcmc`
        is True); converted internally to an integer number of steps
        via ``int(burn * steps)``. Default is 0.2.
    progress : bool, optional
        Whether ``emcee`` prints a progress bar (only used if `mcmc`
        is True). Default is True.
    workers : int, optional
        Number of parallel workers for ``emcee`` (only used if `mcmc`
        is True). Default is 1.
    nan_policy : str, optional
        How ``lmfit`` should handle NaNs in the residuals for the
        least-squares fit (only used if `mcmc` is False). Default is
        'raise'.
    max_nfev : int, optional
        Maximum number of function evaluations for the least-squares
        fit (only used if `mcmc` is False). Default is None (use the
        ``lmfit`` default).
    thin : int, optional
        Only accept every `thin`-th ``emcee`` sample (only used if
        `mcmc` is True). Default is 10.
    is_weighted : bool, optional
        Whether the residuals returned by `model` are already
        weighted, passed through to ``Minimizer.emcee`` (only used if
        `mcmc` is True). Default is True.

    Returns
    -------
    results : lmfit.minimizer.MinimizerResult
        The fit result object returned by ``Minimizer.minimize()``
        (least-squares) or ``Minimizer.emcee()`` (MCMC), containing
        the best-fit (or posterior) parameters and fit statistics.
    """

    # Do fit
    if mcmc:
        func = Minimizer(model, params, fcn_args=args)
        mcmc_results = func.emcee(nwalkers=nwalkers, steps=steps,
                                  burn=int(burn * steps), pos=pos,
                                  is_weighted=is_weighted, progress=progress,
                                  thin=thin, workers=workers)
        results = mcmc_results
    else:
        func = Minimizer(model, params, fcn_args=args, nan_policy=nan_policy,
                         max_nfev=max_nfev)
        results = func.minimize()

    return results


def powerspectrum_model(params, xdata, ydata):
    """
    Model a red-noise power spectrum (e.g. of the frequency-averaged
    secondary spectrum / cross-power vs sqrt(delay)) as a power law
    plus a white-noise floor, and return the residuals.

    model = wn + amp * xdata**alpha

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the white-noise
        level ('wn'), power-law amplitude ('amp'), and power-law index
        ('alpha').
    xdata : numpy.ndarray
        Independent variable (e.g. sqrt of delay/tdel).
    ydata : numpy.ndarray
        Power spectrum values corresponding to `xdata`.

    Returns
    -------
    numpy.ndarray
        Residual of model and data (``ydata - model``). Unlike most
        other model functions in this module, this residual is not
        multiplied by weights.
    """

    parvals = params.valuesdict()

    amp = parvals['amp']
    wn = parvals['wn']
    alpha = parvals['alpha']

    model = wn + amp * xdata**alpha

    return (ydata - model)


def tau_acf_model(params, xdata, ydata, weights):
    """
    Model a 1D cut through the center of the ACF along the time axis
    and return the residuals.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the current
        best-fit amplitude ('amp'), timescale at 1/e width ('tau'),
        and index of the exponential function ('alpha'; 2 is
        Gaussian, 5/3 is Kolmogorov).
    xdata : numpy.ndarray
        Time of sub-integrations from the center of the ACF to the
        maximum time.
    ydata : numpy.ndarray
        ACF pixel values corresponding to `xdata` and running through
        the center of the ACF.
    weights : numpy.ndarray or None
        Weights of the data. If None, uniform weights of ones are
        used, with the white-noise spike (first element) excluded.

    Returns
    -------
    numpy.ndarray
        Weighted residual of model and data.
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    parvals = params.valuesdict()

    amp = parvals['amp']
    tau = parvals['tau']
    alpha = parvals['alpha']

    model = amp*np.exp(-np.divide(xdata, tau)**(alpha))
    weights[0] = 0  # Not fitting for the white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(xdata, max(xdata)))

    return (ydata - model) * weights


def dnu_acf_model(params, xdata, ydata, weights):
    """
    Model a 1D cut through the center of the ACF along the frequency
    axis and return the residuals.

    Default function is exponential with `dnu` measured at half
    power.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the current
        best-fit amplitude ('amp') and decorrelation bandwidth at
        half power ('dnu').
    xdata : numpy.ndarray
        Frequency of the channels from the center of the ACF to the
        maximum frequency.
    ydata : numpy.ndarray
        ACF pixel values corresponding to `xdata` and running through
        the center of the ACF.
    weights : numpy.ndarray or None
        Weights of the data. If None, uniform weights of ones are
        used, with the white-noise spike (first element) excluded.

    Returns
    -------
    numpy.ndarray
        Weighted residual of model and data.
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    parvals = params.valuesdict()

    amp = parvals['amp']
    dnu = parvals['dnu']

    model = amp*np.exp(-np.divide(xdata, dnu/np.log(2)))
    weights[0] = 0  # Not fitting for the white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(xdata, max(xdata)))

    return (ydata - model) * weights


def scint_acf_model(params, xdata, ydata, weights):
    """
    Apply `tau_acf_model` and `dnu_acf_model` simultaneously and
    return the concatenated residuals.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the current
        best-fit amplitude ('amp'), timescale at 1/e width ('tau'),
        decorrelation bandwidth at half power ('dnu'), and index of
        the exponential function ('alpha').
    xdata : tuple of numpy.ndarray
        Two-element sequence ``(xdata_t, xdata_f)`` with the time and
        frequency axes, passed through to `tau_acf_model` and
        `dnu_acf_model` respectively.
    ydata : tuple of numpy.ndarray
        Two-element sequence ``(ydata_t, ydata_f)`` with the ACF cuts
        along time and frequency, passed through to `tau_acf_model`
        and `dnu_acf_model` respectively.
    weights : tuple of numpy.ndarray
        Two-element sequence ``(weights_t, weights_f)`` with the
        weights for the time and frequency cuts respectively.

    Returns
    -------
    numpy.ndarray
        Concatenation of the weighted residuals from `tau_acf_model`
        and `dnu_acf_model`.
    """

    residuals_t = tau_acf_model(params, xdata[0], ydata[0], weights[0])
    residuals_f = dnu_acf_model(params, xdata[1], ydata[1], weights[1])

    return np.concatenate((residuals_t, residuals_f))


def scint_acf_model_2d_approx(params, tdata, fdata, ydata, weights):
    """
    Model an approximate 2D ACF that incorporates a phase gradient,
    and return the residuals.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the current
        best-fit amplitude ('amp'), timescale at 1/e width ('tau'),
        decorrelation bandwidth at half power ('dnu'), index of the
        exponential function ('alpha'), and phase gradient
        ('phasegrad'), as well as the total observation time
        ('tobs') and total bandwidth ('bw').
    tdata : numpy.ndarray
        Times of sub-integrations along the desired range, centered
        on the sub-integration next to that of the white-noise spike.
    fdata : numpy.ndarray
        Frequencies of channels along the desired range, centered on
        the channel next to that of the white-noise spike.
    ydata : numpy.ndarray
        ACF cropped to the range of times and frequencies matching
        `tdata` and `fdata`.
    weights : numpy.ndarray or None
        Weights of the data, in FFT (fftshift) ordering. If None,
        uniform weights of ones are used. The white-noise spike is
        excluded from the fit.

    Returns
    -------
    numpy.ndarray
        Weighted residual of model and data.
    """

    parvals = params.valuesdict()

    amp = parvals['amp']
    dnu = parvals['dnu']
    tau = parvals['tau']
    alpha = parvals['alpha']
    mu = parvals['phasegrad']*60  # min/MHz to s/MHz
    tobs = parvals['tobs']
    bw = parvals['bw']
    nt = len(tdata)
    nf = len(fdata)

    if weights is None:
        weights = np.ones(np.shape(ydata))

    tdata = np.reshape(tdata, (nt, 1))
    fdata = np.reshape(fdata, (1, nf))

    # model = amp * np.exp(-(abs((tdata / tau) + 2 * phasegrad *
    #                           ((dnu / np.log(2)) / freq)**(1 / 6) *
    #                           (fdata / (dnu / np.log(2))))**(3 * alpha / 2) +
    #                     abs(fdata / (dnu / np.log(2)))**(3 / 2))**(2 / 3))
    model = amp * np.exp(-(abs((tdata - mu*fdata)/tau)**(3 * alpha / 2) +
                         abs(fdata / (dnu / np.log(2)))**(3 / 2))**(2 / 3))

    # multiply by triangle function
    model = np.multiply(model, 1-np.divide(abs(tdata), tobs))
    model = np.multiply(model, 1-np.divide(abs(fdata), bw))
    weights = np.fft.fftshift(weights)
    weights[-1, -1] = 0  # Not fitting for the white noise spike
    weights = np.fft.ifftshift(weights)
    model = np.transpose(model)

    return (ydata - model) * weights


def scint_acf_model_2d(params, ydata, weights):
    """
    Model an analytical 2D ACF using the `scintools.scint_sim.ACF`
    class, and return the residuals.

    This method is significantly slower than
    `scint_acf_model_2d_approx`.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the current
        best-fit timescale at 1/e width ('tau'), decorrelation
        bandwidth at half power ('dnu'), index of the exponential
        function ('alpha'), axial ratio of anisotropy ('ar'),
        orientation of anisotropy ('psi'), phase gradient
        ('phasegrad'), rotation of the phase gradient ('theta'), and
        amplitude ('amp'), as well as the total observation time
        ('tobs'), total bandwidth ('bw'), and number of
        sub-integrations and channels used to build the model ACF
        ('nt', 'nf').
    ydata : numpy.ndarray
        2D ACF cropped symmetrically around its center to a desired
        range, with shape ``(nf_crop, nt_crop)``.
    weights : numpy.ndarray or None
        Weights of the data, in FFT (fftshift) ordering. If None,
        uniform weights of ones are used. The white-noise spike is
        excluded from the fit.

    Returns
    -------
    numpy.ndarray
        Weighted residual of model and data.
    """

    parvals = params.valuesdict()

    tau = np.abs(parvals['tau'])
    dnu = np.abs(parvals['dnu'])
    alpha = parvals['alpha']
    ar = np.abs(parvals['ar'])
    psi = parvals['psi']
    phasegrad = parvals['phasegrad']
    theta = parvals['theta']
    amp = parvals['amp']

    tobs = parvals['tobs']
    bw = parvals['bw']
    nt = parvals['nt']
    nf = parvals['nf']
    nf_crop, nt_crop = np.shape(ydata)

    dt, df = 2 * tobs / nt, 2 * bw / nf
    taumax = nt_crop * dt / tau
    dnumax = nf_crop * df / dnu

    acf = ACF(taumax=taumax, dnumax=dnumax, nt=nt_crop, nf=nf_crop, ar=ar,
              alpha=alpha, phasegrad=phasegrad, theta=theta,
              amp=amp, psi=psi)
    model = acf.acf

    triangle_t = 1 - np.divide(np.tile(np.abs(np.linspace(-taumax*tau,
                                                          taumax*tau,
                                                          nt_crop)),
                                       (nf_crop, 1)), tobs)
    triangle_f = \
        np.transpose(1 - np.divide(np.tile(np.abs(np.linspace(-dnumax*dnu,
                                                              dnumax*dnu,
                                                              nf_crop)),
                                           (nt_crop, 1)), bw))
    triangle = np.multiply(triangle_t, triangle_f)
    model = np.multiply(model, triangle)  # multiply by triangle function

    if weights is None:
        weights = np.ones(np.shape(ydata))
        # weights = 1/model

    weights = np.fft.fftshift(weights)
    weights[-1, -1] = 0  # Not fitting for the white noise spike
    weights = np.fft.ifftshift(weights)

    return (ydata - model) * weights


def tau_sspec_model(params, xdata, ydata):
    """
    Model a 1D cut through the center of the ACF along the time axis
    and apply a Fourier transform, returning the residuals against
    the secondary-spectrum profile.

    Parameters
    ----------
    params : lmfit.Parameters or dict-like
        Object containing the current best-fit amplitude ('amp'),
        timescale at 1/e width ('tau'), and index of the exponential
        function ('alpha'; 2 is Gaussian, 5/3 is Kolmogorov).
    xdata : numpy.ndarray
        Time of sub-integrations from the center of the ACF to the
        maximum time.
    ydata : numpy.ndarray
        Profile from the secondary spectrum corresponding to the ACF
        to model, summed along all columns (all f_t).

    Returns
    -------
    numpy.ndarray
        Residual of model and data, weighted by the model itself
        (used as an approximation of the noise).
    """

    amp = params['amp']
    tau = params['tau']
    alpha = params['alpha']

    model = amp * np.exp(-np.divide(xdata, tau)**alpha)
    model[0] = 0  # Not fitting for the white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1 - np.divide(xdata, max(xdata)))

    model_flipped = model[::-1]
    model = np.concatenate((model, model_flipped))
    model = model[0:2 * len(xdata) - 1]
    # Get Fourier model
    model = np.fft.fft(model)
    model = np.real(model)
    model = model[0:len(xdata)]

    # Use the model for the weights
    return (ydata - model) * model


def dnu_sspec_model(params, xdata, ydata):
    """
    Model a 1D cut through the center of the ACF along the frequency
    axis and apply a Fourier transform, returning the residuals
    against the secondary-spectrum profile.

    Default function is exponential with `dnu` measured at half
    power.

    Parameters
    ----------
    params : lmfit.Parameters or dict-like
        Object containing the current best-fit amplitude ('amp') and
        decorrelation bandwidth at half power ('dnu').
    xdata : numpy.ndarray
        Frequency of the channels from the center of the ACF to the
        maximum frequency.
    ydata : numpy.ndarray
        Profile from the secondary spectrum corresponding to the ACF
        to model, summed along all rows (all f_tau or f_lambda).

    Returns
    -------
    numpy.ndarray
        Residual of model and data, weighted by the model itself
        (used as an approximation of the noise).
    """

    amp = params['amp']
    dnu = params['dnu']

    model = amp * np.exp(-np.divide(xdata, dnu / np.log(2)))
    model[0] = 0  # Not fitting for the white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1 - np.divide(xdata, max(xdata)))

    model_flipped = model[::-1]
    model = np.concatenate((model, model_flipped))
    model = model[0:2 * len(xdata) - 1]
    # Get Fourier model
    model = np.fft.fft(model)
    model = np.real(model)
    model = model[0:len(xdata)]

    # Use the model for the weights
    return (ydata - model) * model


def scint_sspec_model(params, xdata, ydata, weights):
    """
    Apply `tau_sspec_model` and `dnu_sspec_model` simultaneously and
    return the concatenated residuals.

    Parameters
    ----------
    params : lmfit.Parameters or dict-like
        Object containing the current best-fit amplitude ('amp'),
        timescale at 1/e width ('tau'), decorrelation bandwidth at
        half power ('dnu'), and index of the exponential function
        ('alpha').
    xdata : tuple of numpy.ndarray
        Two-element sequence ``(xdata_t, xdata_f)`` with the time and
        frequency axes, passed through to `tau_sspec_model` and
        `dnu_sspec_model` respectively.
    ydata : tuple of numpy.ndarray
        Two-element sequence ``(ydata_t, ydata_f)`` with the
        secondary-spectrum profiles along time and frequency, passed
        through to `tau_sspec_model` and `dnu_sspec_model`
        respectively.
    weights : sequence
        Two-element sequence ``(weights_t, weights_f)``, passed as an
        extra positional argument to `tau_sspec_model` and
        `dnu_sspec_model`.

    Returns
    -------
    numpy.ndarray
        Concatenation of the residuals from `tau_sspec_model` and
        `dnu_sspec_model`.

    Raises
    ------
    TypeError
        `tau_sspec_model` and `dnu_sspec_model` only accept
        ``(params, xdata, ydata)``, so calling them here with an
        extra `weights` element currently raises a TypeError. This
        function is not presently called elsewhere in the package.
    """

    residuals_t = tau_sspec_model(params, xdata[0], ydata[0], weights[0])
    residuals_f = dnu_sspec_model(params, xdata[1], ydata[1], weights[1])

    return np.concatenate((residuals_t, residuals_f))


def arc_power_curve(params, xdata, ydata, weights):
    """
    Return a template for the power curve in the secondary spectrum
    against sqrt(curvature) or normalised f_t.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object (currently unused by the
        model itself).
    xdata : numpy.ndarray
        Square-root arc curvatures.
    ydata : numpy.ndarray
        Secondary-spectrum power profile.
    weights : numpy.ndarray or None
        Weights of the data. If None, uniform weights of ones are
        used.

    Returns
    -------
    numpy.ndarray
        Weighted residual of model and data.

    Notes
    -----
    The model template is not yet implemented: `model` is currently
    an empty list, so ``ydata - model`` will raise a
    ``ValueError`` (mismatched shapes) for any non-empty `ydata`.
    This function is not presently called elsewhere in the package.
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    model = []
    return (ydata - model) * weights


def fit_parabola(x, y):
    """
    Fit a parabola and return the value and error for the peak.

    Parameters
    ----------
    x : numpy.ndarray
        x values of the peak profile.
    y : numpy.ndarray
        y values of the peak profile.

    Returns
    -------
    yfit : numpy.ndarray
        y values of the fit, evaluated at `x`.
    peak : float
        Fit peak value (x position of the parabola's turning point).
    peak_error : float
        Uncertainty on `peak`, propagated from the covariance of the
        fitted parabola coefficients.
    """

    # increase range to help fitter
    ptp = np.ptp(x)
    x = x*(1000/ptp)

    # Do the fit
    params, pcov = np.polyfit(x, y, 2, cov=True)
    yfit = params[0]*np.power(x, 2) + params[1]*x + params[2]  # y values

    # Get parameter errors
    errors = []
    for i in range(len(params)):  # for each parameter
        errors.append(np.absolute(pcov[i][i])**0.5)

    # Get parabola peak and error
    peak = -params[1]/(2*params[0])  # Parabola max (or min)
    peak_error = np.sqrt((errors[1]**2)*((1/(2*params[0]))**2) +
                         (errors[0]**2)*((params[1]/2)**2))  # Error on peak

    peak = peak*(ptp/1000)
    peak_error = peak_error*(ptp/1000)

    return yfit, peak, peak_error


def fit_log_parabola(x, y):
    """
    Fit a log-parabola and return the value and error for the peak.

    Takes the natural log of `x`, fits a parabola in log-x space via
    `fit_parabola`, and converts the resulting peak position and
    error back to linear `x` units.

    Parameters
    ----------
    x : numpy.ndarray
        x values of the peak profile (must be positive, since the
        log is taken).
    y : numpy.ndarray
        y values of the peak profile.

    Returns
    -------
    yfit : numpy.ndarray
        y values of the fit, evaluated at (rescaled) log(`x`).
    peak : float
        Fit peak value, in linear `x` units.
    peak_error : float
        Uncertainty on `peak`, in linear `x` units.
    """

    # Take the log of x
    logx = np.log(x)
    ptp = np.ptp(logx)
    x = logx*(1000/ptp)  # increase range to help fitter

    # Do the fit
    yfit, peak, peak_error = fit_parabola(x, y)
    frac_error = peak_error/peak

    peak = np.e**(peak*ptp/1000)
    # Average the error asymmetries
    peak_error = frac_error*peak

    return yfit, peak, peak_error


def arc_curvature(params, ydata, weights, true_anomaly,
                  vearth_ra, vearth_dec, mjd=None, model_only=False,
                  return_veff=False):
    """
    Model the arc curvature and return the residuals (or, optionally,
    the model curvature and effective velocity components directly).

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the current
        best-fit distance to the pulsar in kpc ('d') and fractional
        screen distance ('s'), plus the parameters required by
        `effective_velocity_annual`. Optionally also: an anisotropy
        model flag ('nmodel'), anisotropy angle ('zeta', required if
        anisotropic), ISM velocity in RA/Dec ('vism_ra', 'vism_dec')
        or, for the anisotropic case, ISM velocity along `zeta`
        ('vism_zeta'). The deprecated parameter names 'psi' and
        'vism_psi' are no longer supported (use 'zeta' and
        'vism_zeta' instead).
    ydata : numpy.ndarray
        Arc curvature data.
    weights : numpy.ndarray or None
        Weights of the data. If None (and `model_only` is False),
        uniform weights of ones are used.
    true_anomaly : numpy.ndarray
        True anomalies corresponding to the data.
    vearth_ra : numpy.ndarray
        Earth velocity in RA corresponding to the data.
    vearth_dec : numpy.ndarray
        Earth velocity in Dec corresponding to the data.
    mjd : numpy.ndarray or float, optional
        MJDs corresponding to the data, used for the 'OMDOT'
        correction in `effective_velocity_annual`. Default is None.
    model_only : bool, optional
        If True, return the model curvature (and, if `return_veff` is
        also True, the effective velocity components) instead of the
        residuals. Default is False.
    return_veff : bool, optional
        If True (and `model_only` is also True), also return the
        ISM-subtracted effective velocity components in RA and Dec.
        Default is False.

    Returns
    -------
    numpy.ndarray
        If `model_only` is False: the weighted residual of model and
        data, ``(ydata - model) * weights``.
    model : numpy.ndarray
        If `model_only` is True: the model arc curvature, in units of
        1/(m mHz**2).
    veff_ra, veff_dec : numpy.ndarray
        If `model_only` and `return_veff` are both True: the
        ISM-subtracted effective velocity in RA and Dec, returned in
        addition to `model`.

    Raises
    ------
    KeyError
        If the deprecated parameter 'psi' or 'vism_psi' is present in
        `params`.
    """

    # ensure dimensionality of arrays makes sense
    if hasattr(ydata,  "__len__"):
        ydata = ydata.squeeze()
        weights = weights.squeeze()
        true_anomaly = true_anomaly.squeeze()
        vearth_ra = vearth_ra.squeeze()
        vearth_dec = vearth_dec.squeeze()

    kmpkpc = 3.085677581e16

    # Other parameters in lower-case
    d = params['d']  # pulsar distance in kpc
    dkm = d * kmpkpc  # kms
    s = params['s']  # fractional screen distance

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    if 'psi' in params.keys():
        raise KeyError("parameter psi is no longer supported. Please use zeta")
    if 'vism_psi' in params.keys():
        raise KeyError("parameter vism_psi is no longer supported. " +
                       "Please use vism_zeta")

    if 'nmodel' in params.keys():
        nmodel = params['nmodel']
    else:
        if 'zeta' in params.keys():
            nmodel = 1
        else:
            nmodel = 0

    if 'vism_ra' in params.keys():
        vism_ra = params['vism_ra']
        vism_dec = params['vism_dec']
    else:
        vism_ra = 0
        vism_dec = 0

    if nmodel > 0.5:  # anisotropic
        zeta = params['zeta'] * np.pi / 180  # anisotropy angle
        if 'vism_zeta' in params.keys():  # anisotropic case
            vism_zeta = params['vism_zeta']  # vism in direction of anisotropy
            veff2 = (veff_ra*np.sin(zeta) + veff_dec*np.cos(zeta) -
                     vism_zeta)**2
        else:
            veff2 = ((veff_ra - vism_ra) * np.sin(zeta) +
                     (veff_dec - vism_dec) * np.cos(zeta)) ** 2
    else:  # isotropic
        veff2 = (veff_ra - vism_ra)**2 + (veff_dec - vism_dec)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9

    if weights is None:
        weights = np.ones(np.shape(ydata))

    if model_only:
        if return_veff:
            return model, (veff_ra - vism_ra), (veff_dec - vism_dec)
        else:
            return model
    else:
        return (ydata - model) * weights


def veff_thin_screen(params, ydata, weights, true_anomaly,
                     vearth_ra, vearth_dec, mjd=None):
    """
    Model the effective velocity implied by a thin-screen scattering
    geometry and return the residuals.

    Uses Eq. 4 from Rickett et al. (2014) for the anisotropy
    coefficients.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing the current
        best-fit fractional screen distance ('s') and pulsar distance
        in kpc ('d'), plus the parameters required by
        `effective_velocity_annual`. Optionally also: a scaling
        factor ('kappa', default 1), an anisotropy model flag
        ('nmodel'), and for the anisotropic case an axial ratio
        parameter ('R') and anisotropy angle in degrees ('psi'). ISM
        velocity in RA/Dec ('vism_ra', 'vism_dec') may also be given
        and is subtracted from the effective velocity.
    ydata : numpy.ndarray
        Measured effective velocity (or, equivalently, quantity
        proportional to it via the arc curvature) data.
    weights : numpy.ndarray
        Weights of the data.
    true_anomaly : numpy.ndarray
        True anomalies corresponding to the data.
    vearth_ra : numpy.ndarray
        Earth velocity in RA corresponding to the data.
    vearth_dec : numpy.ndarray
        Earth velocity in Dec corresponding to the data.
    mjd : numpy.ndarray or float, optional
        MJDs corresponding to the data, used for the 'OMDOT'
        correction in `effective_velocity_annual`. Default is None.

    Returns
    -------
    numpy.ndarray
        Weighted residual of model and data.
    """

    # ensure dimensionality of arrays makes sense
    if hasattr(ydata, "__len__"):
        ydata = ydata.squeeze()
        weights = weights.squeeze()
        true_anomaly = true_anomaly.squeeze()
        vearth_ra = vearth_ra.squeeze()
        vearth_dec = vearth_dec.squeeze()

    s = params['s']  # fractional screen distance
    d = params['d']  # pulsar distance (kpc)
    if 'kappa' in params.keys():
        kappa = params['kappa']
    else:
        kappa = 1

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    if 'nmodel' in params.keys():
        nmodel = params['nmodel']
    else:
        if 'psi' in params.keys():
            nmodel = 1
        else:
            nmodel = 0

    if 'vism_ra' in params.keys():
        vism_ra = params['vism_ra']
        vism_dec = params['vism_dec']
    else:
        vism_ra = 0
        vism_dec = 0

    veff_ra -= vism_ra
    veff_dec -= vism_dec

    if nmodel > 0.5:  # anisotropic
        R = params['R']  # axial ratio parameter
        psi = params['psi'] * np.pi / 180  # anisotropy angle

        cosa = np.cos(2 * psi)
        sina = np.sin(2 * psi)

        # quadratic coefficients
        a = (1 - R * cosa) / np.sqrt(1 - R**2)
        b = (1 + R * cosa) / np.sqrt(1 - R**2)
        c = -2 * R * sina / np.sqrt(1 - R**2)

    else:
        a, b, c = 1, 1, 0

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
                            c*veff_ra*veff_dec))
    model = coeff * veff / s

    return (ydata - model) * weights


"""
Below: Models that do not return residuals for a fitter
"""


def effective_velocity_annual(params, true_anomaly, vearth_ra, vearth_dec,
                              mjd=None):
    """
    Compute the effective velocity including annual (Earth orbital
    and proper motion) and pulsar orbital terms.

    Note: this does NOT include the interstellar medium (IISM)
    velocity, but the returned effective velocity is expressed in the
    IISM frame.

    Parameters
    ----------
    params : lmfit.Parameters
        ``lmfit`` ``Parameters()`` object containing, if the pulsar is
        in a binary: the projected semi-major axis in lt-s ('A1'),
        orbital period in days ('PB'), orbital eccentricity ('ECC'),
        longitude of periastron in degrees ('OM'), inclination via
        one of 'KIN' (degrees), 'COSI', or 'SINI', and optionally an
        inclination-sense flag ('sense') and periastron-advance rate
        in deg/yr ('OMDOT', requires 'T0' and `mjd`). Also,
        optionally, the proper motion in RA and Dec ('PMRA', 'PMDEC',
        mas/yr), and always the longitude of the ascending node in
        degrees ('KOM'), the fractional screen distance ('s'), and
        pulsar distance in kpc ('d'). Note that 'KOM' is required even
        for non-binary pulsars, since it is used unconditionally to
        rotate the pulsar/proper-motion velocity into RA/Dec.
    true_anomaly : numpy.ndarray
        True anomalies to compute over (radians). Ignored if the
        binary parameters are absent from `params`.
    vearth_ra : numpy.ndarray
        Earth velocity in RA to compute over (km/s).
    vearth_dec : numpy.ndarray
        Earth velocity in Dec to compute over (km/s).
    mjd : numpy.ndarray or float, optional
        MJDs corresponding to `true_anomaly`, used to evaluate the
        'OMDOT' correction to the longitude of periastron. Default is
        None; if 'OMDOT' is present in `params` and `mjd` is None, a
        warning is printed and the uncorrected 'OM' is used.

    Returns
    -------
    veff_ra : numpy.ndarray
        Total effective velocity in RA (km/s), combining the Earth
        and pulsar/proper-motion contributions weighted by the
        fractional screen distance.
    veff_dec : numpy.ndarray
        Total effective velocity in Dec (km/s).
    vp_ra : numpy.ndarray or float
        Pulsar orbital velocity in RA (km/s); 0 if no binary
        parameters are present.
    vp_dec : numpy.ndarray or float
        Pulsar orbital velocity in Dec (km/s); 0 if no binary
        parameters are present.
    """
    # Define some constants
    v_c = 299792.458  # km/s
    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    # tempo2 parameters from par file in capitals
    if 'PB' in params.keys():
        A1 = params['A1']  # projected semi-major axis in lt-s
        PB = params['PB']  # orbital period in days
        ECC = params['ECC']  # orbital eccentricity
        OM = params['OM'] * np.pi/180  # longitude of periastron rad
        if 'OMDOT' in params.keys():
            if mjd is None:
                print('Warning, OMDOT present but no mjd for calculation')
                omega = OM
            else:
                omega = OM + \
                    params['OMDOT']*np.pi/180*(mjd-params['T0'])/365.2425
        else:
            omega = OM
        # Note: fifth Keplerian param T0 used in true anomaly calculation
        if 'KIN' in params.keys():
            INC = params['KIN']*np.pi/180  # inclination
        elif 'COSI' in params.keys():
            INC = np.arccos(params['COSI'])
        elif 'SINI' in params.keys():
            INC = np.arcsin(params['SINI'])
        else:
            print('Warning: inclination parameter (KIN, COSI, or SINI) ' +
                  'not found')

        if 'sense' in params.keys():
            sense = params['sense']
            if sense < 0.5:  # KIN < 90
                if INC > np.pi/2:
                    INC = np.pi - INC
            if sense >= 0.5:  # KIN > 90
                if INC < np.pi/2:
                    INC = np.pi - INC

        KOM = params['KOM']*np.pi/180  # longitude ascending node

        # Calculate pulsar velocity aligned with the line of nodes (Vx) and
        #   perpendicular in the plane (Vy)
        vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                         np.sqrt(1 - ECC**2))
        vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))
        vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
                                                                  + omega))
    else:
        vp_x = 0
        vp_y = 0

    if 'PMRA' in params.keys():
        PMRA = params['PMRA']  # proper motion in RA
        PMDEC = params['PMDEC']  # proper motion in DEC
    else:
        PMRA = 0
        PMDEC = 0

    # other parameters in lower-case
    s = params['s']  # fractional screen distance
    d = params['d']  # pulsar distance in kpc
    d = d * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * d / secperyr
    pmdec_v = PMDEC * masrad * d / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC
    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v)
    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v)

    return veff_ra, veff_dec, vp_ra, vp_dec


def arc_weak(ftn, ar=1, psi=0, alpha=11/3):
    """
    Model the 1D weak-scattering scintillation arc Doppler profile
    (power vs normalised Doppler frequency), for a possibly
    anisotropic scattering screen.

    Parameters
    ----------
    ftn : array_like, 1D
        The normalised Doppler frequency (x-axis), where ``ftn = 1``
        is the arc.
    ar : float, optional
        Anisotropy axial ratio. The default is 1 (isotropic).
    psi : float, optional
        Orientation angle of the anisotropy, in degrees. The default
        is 0.
    alpha : float, optional
        Index of the turbulence spectrum. The default is 11/3
        (Kolmogorov).

    Returns
    -------
    p : array_like, 1D
        The model Doppler profile.
    """

    # Begin model
    a = np.cos(psi * np.pi/180)**2 / ar + ar * np.sin(psi*np.pi/180)**2
    b = ar * np.cos(psi * np.pi/180)**2 + (np.sin(psi * np.pi/180)**2)/ar
    c = 2*np.sin(psi * np.pi/180)*np.cos(psi * np.pi/180)*(1/ar - ar)

    p = ((a*ftn**2 + b*(1 - ftn**2) + c*ftn*(1 - ftn**2)**0.5)**(-alpha/2) + \
         (a*ftn**2 + b*(1 - ftn**2) - c*ftn*(1 - ftn**2)**0.5)**(-alpha/2))
    p /= np.sqrt(1 - ftn**2)

    return p


def arc_weak_2d(fdop, tdel, eta=1, ar=1, psi=0, alpha=11/3):
    """
    Model the 2D weak-scattering secondary spectrum along a
    scintillation arc, for a possibly anisotropic scattering screen.

    Parameters
    ----------
    fdop : array_like, 1D
        The Doppler frequency (x-axis) coordinates of the model
        secondary spectrum.
    tdel : array_like, 1D
        The delay (y-axis) coordinates of the model secondary
        spectrum.
    eta : float, optional
        Arc curvature. The default is 1.
    ar : float, optional
        Anisotropy axial ratio. The default is 1 (isotropic).
    psi : float, optional
        Orientation angle of the anisotropy, in degrees. The default
        is 0.
    alpha : float, optional
        Index of the turbulence spectrum. The default is 11/3
        (Kolmogorov). Note that, unlike in `arc_weak`, this parameter
        is currently unused by the model, which always uses a fixed
        exponent of -11/6 (appropriate for Kolmogorov turbulence).

    Returns
    -------
    sspec : array_like, 2D
        The model secondary spectrum, with shape
        ``(len(tdel), len(fdop))``.
    """

    # Begin model
    a = np.cos(psi * np.pi/180)**2 / ar + ar * np.sin(psi*np.pi/180)**2
    b = ar * np.cos(psi * np.pi/180)**2 + (np.sin(psi * np.pi/180)**2)/ar
    c = 2*np.sin(psi * np.pi/180)*np.cos(psi * np.pi/180)*(1/ar - ar)

    fdx, TDEL = np.meshgrid(fdop, tdel)

    f_arc = np.sqrt(TDEL/eta)

    fdy = np.sqrt(TDEL/eta - fdx**2)

    p = (a*fdx**2 + b*fdy**2 + c*fdx*fdy)**(-11/6) + \
        (a*fdx**2 + b*fdy**2 - c*fdx*fdy)**(-11/6)

    arc_frac = np.real(fdx)/np.real(f_arc)
    sspec = p / np.sqrt(1 - arc_frac**2)

    return sspec

