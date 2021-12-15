#!/usr/bin/env python

"""
models.py
----------------------------------
Scintillation models
A library of scintillation models to use with lmfit, emcee, or bilby
    Each model has at least inputs:
        params
        xdata
        ydata
        weights
    And output:
        residuals = (ydata - model) * weights
    Some functions use additional inputs
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scintools.scint_sim import ACF
from lmfit import Minimizer, conf_interval


def fitter(model, params, args, mcmc=False, pos=None, nwalkers=100,
           steps=1000, burn=0.2, progress=True, get_ci=False,
           nan_policy='raise', max_nfev=None, thin=10, is_weighted=True):

    # Do fit
    maxfev = [0 if max_nfev is None else max_nfev]
    maxfev = int(maxfev[0])
    func = Minimizer(model, params, fcn_args=args, nan_policy=nan_policy,
                     max_nfev=maxfev)
    results = func.minimize()
    if mcmc:
        func = Minimizer(model, results.params, fcn_args=args)
        mcmc_results = func.emcee(nwalkers=nwalkers, steps=steps,
                                  burn=int(burn * steps), pos=pos,
                                  is_weighted=is_weighted, progress=progress,
                                  thin=thin)
        results = mcmc_results

    if get_ci:
        if results.errorbars:
            ci = conf_interval(func, results)
        else:
            ci = ''
        return results, ci
    else:
        return results


def powerspectrum_model(params, xdata, ydata):

    parvals = params.valuesdict()

    amp = parvals['amp']
    wn = parvals['wn']
    alpha = parvals['alpha']

    model = wn + amp * xdata**alpha

    return (ydata - model)


def tau_acf_model(params, xdata, ydata, weights):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        alpha = index of exponential function. 2 is Gaussian, 5/3 is Kolmogorov
        wn = white noise spike in ACF cut
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    parvals = params.valuesdict()

    amp = parvals['amp']
    tau = parvals['tau']
    alpha = parvals['alpha']
    wn = parvals['wn']

    model = amp*np.exp(-np.divide(xdata, tau)**(alpha))
    model[0] = model[0] + wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(xdata, max(xdata)))

    return (ydata - model) * weights


def dnu_acf_model(params, xdata, ydata, weights):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
        wn = white noise spike in ACF cut
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    parvals = params.valuesdict()

    amp = parvals['amp']
    dnu = parvals['dnu']
    wn = parvals['wn']

    model = amp*np.exp(-np.divide(xdata, dnu/np.log(2)))
    model[0] = model[0] + wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(xdata, max(xdata)))

    return (ydata - model) * weights


def scint_acf_model(params, xdata, ydata, weights):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    parvals = params.valuesdict()

    nt = parvals['nt']

    # Scintillation timescale model
    xdata_t = xdata[:nt]
    ydata_t = ydata[:nt]
    weights_t = weights[:nt]
    residuals_t = tau_acf_model(params, xdata_t, ydata_t, weights_t)

    # Scintillation bandwidth model
    xdata_f = xdata[nt:]
    ydata_f = ydata[nt:]
    weights_f = weights[nt:]
    residuals_f = dnu_acf_model(params, xdata_f, ydata_f, weights_f)

    return np.concatenate((residuals_t, residuals_f))


def scint_acf_model_2d_approx(params, tdata, fdata, ydata, weights):
    """
    Fit an approximate 2D ACF function
    """

    parvals = params.valuesdict()

    amp = parvals['amp']
    dnu = parvals['dnu']
    tau = parvals['tau']
    alpha = parvals['alpha']
    phasegrad = parvals['phasegrad']
    freq = parvals['freq']
    tobs = parvals['tobs']
    bw = parvals['bw']
    wn = parvals['wn']
    nt = len(tdata)
    nf = len(fdata)

    tdata = np.reshape(tdata, (nt, 1))
    fdata = np.reshape(fdata, (1, nf))

    model = amp * np.exp(-(abs((tdata / tau) + 2 * phasegrad *
                               ((dnu / np.log(2)) / freq)**(1 / 6) *
                               (fdata / (dnu / np.log(2))))**(3 * alpha / 2) +
                         abs(fdata / (dnu / np.log(2)))**(3 / 2))**(2 / 3))

    # multiply by triangle function
    model = np.multiply(model, 1-np.divide(abs(tdata), tobs))
    model = np.multiply(model, 1-np.divide(abs(fdata), bw))
    model = np.fft.fftshift(model)
    model[-1, -1] += wn  # add white noise spike
    model = np.fft.ifftshift(model)
    model = np.transpose(model)

    if weights is None:
        weights = np.ones(np.shape(ydata))

    return (ydata - model) * weights


def scint_acf_model_2d(params, ydata, weights):
    """
    Fit an analytical 2D ACF function
    """

    parvals = params.valuesdict()

    tau = np.abs(parvals['tau'])
    dnu = np.abs(parvals['dnu'])
    alpha = parvals['alpha']
    ar = np.abs(parvals['ar'])
    phasegrad_x = parvals['phasegrad_x']
    phasegrad_y = parvals['phasegrad_y']
    wn = parvals['wn']
    amp = parvals['amp']

    V_x = parvals['v_x']
    V_y = parvals['v_y']
    # psi = parvals['psi']

    tobs = parvals['tobs']
    bw = parvals['bw']
    nt = parvals['nt']
    nf = parvals['nf']

    nt_crop = len(ydata[0])
    nf_crop = len(ydata)

    taumax = (nt_crop / nt) * tobs / tau
    dnumax = (nf_crop / nf) * bw / dnu

    acf = ACF(s_max=taumax, dnu_max=dnumax, ns=nt_crop, nf=nf_crop, ar=ar,
              alpha=alpha, phasegrad_x=phasegrad_x, phasegrad_y=phasegrad_y,
              amp=amp, V_x=V_x, V_y=V_y, psi=None)
    acf.calc_acf()
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

    # add white noise spike
    model = np.fft.fftshift(model)
    model[-1, -1] += wn
    model = np.fft.ifftshift(model)

    return (ydata - model) * weights


def tau_sspec_model(params, xdata, ydata):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        alpha = index of exponential function. 2 is Gaussian, 5/3 is Kolmogorov
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    tau = params['tau']
    alpha = params['alpha']
    wn = params['wn']

    model = amp * np.exp(-np.divide(xdata, tau)**alpha)
    model[0] += wn  # add white noise spike
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
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    dnu = params['dnu']
    wn = params['wn']

    model = amp * np.exp(-np.divide(xdata, dnu / np.log(2)))
    model[0] += wn  # add white noise spike
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


def scint_sspec_model(params, xdata, ydata):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    parvals = params.valuesdict()

    nt = parvals['nt']

    # Scintillation timescale model
    xdata_t = xdata[:nt]
    ydata_t = ydata[:nt]
    residuals_t = tau_sspec_model(params, xdata_t, ydata_t)

    # Scintillation bandwidth model
    xdata_f = xdata[nt:]
    ydata_f = ydata[nt:]
    residuals_f = dnu_sspec_model(params, xdata_f, ydata_f)

    return np.concatenate((residuals_t, residuals_f))


def arc_power_curve(params, xdata, ydata, weights):
    """
    Returns a template for the power curve in secondary spectrum vs
    sqrt(curvature) or normalised fdop
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    model = []
    return (ydata - model) * weights


def fit_parabola(x, y):
    """
    Fit a parabola and return the value and error for the peak
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
    Fit a log-parabola and return the value and error for the peak
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
                  vearth_ra, vearth_dec):
    """
    arc curvature model
        ydata: arc curvature
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
                                  vearth_ra, vearth_dec)

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

    if nmodel > 0.5:  # anisotropic
        psi = params['psi'] * np.pi / 180  # anisotropy angle
        if 'vism_psi' in params.keys():  # anisotropic case
            vism_psi = params['vism_psi']  # vism in direction of anisotropy
            veff2 = (veff_ra*np.sin(psi) + veff_dec*np.cos(psi) - vism_psi)**2
        else:
            veff2 = ((veff_ra - vism_ra) * np.sin(psi) +
                     (veff_dec - vism_dec) * np.cos(psi)) ** 2
    else:  # isotropic
        veff2 = (veff_ra - vism_ra)**2 + (veff_dec - vism_dec)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9

    if weights is None:
        weights = np.ones(np.shape(ydata))

    return (ydata - model) * weights


def veff_thin_screen(params, ydata, weights, true_anomaly,
                     vearth_ra, vearth_dec, mjd=None):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.
        ydata: arc curvature
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

        gamma = psi
        cosa = np.cos(2 * gamma)
        sina = np.sin(2 * gamma)

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
    Effective velocity with annual and pulsar terms
        Note: Does NOT include IISM velocity, but returns veff in IISM frame
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
            omega = OM + params['OMDOT']*np.pi/180*(mjd-params['T0'])/365.2425
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
