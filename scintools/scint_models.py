#!/usr/bin/env python

"""
models.py
----------------------------------
Scintillation models

A library of scintillation models to use with lmfit

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


def tau_sspec_model(params, xdata, ydata, weights):
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

    amp = params['amp']
    tau = params['tau']
    alpha = params['alpha']
    wn = params['wn']

    model = amp*np.exp(-np.divide(xdata, tau)**(alpha))
    model[0] = model[0] + wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(xdata, max(xdata)))

    model_flipped = model[::-1]
    model = np.concatenate((model, model_flipped))
    model = model[0:2*len(xdata)-1]
    # Get Fourier model
    model = np.fft(model)
    model = np.real(model)
    model = model[0:len(xdata)]

    return (ydata - model) * weights


def dnu_sspec_model(params, xdata, ydata, weights):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
        wn = white noise spike in ACF cut
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    y = dnu_acf_model(xdata, ydata, weights, params)
    # From ACF model, construct Fourier-domain model
    y_flipped = y[::-1]
    y = list(y) + list(y_flipped)  # concatenate
    y = y[0:2*len(f)-1]
    # Get Fourier model
    yf = np.fft(y)
    yf = np.real(yf)
    yf = yf[0:len(f)]
    return (data - model) * weights


def scint_sspec_model(params, xdata, ydata, weights):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    y_t = tau_sspec_model(t=t, tau=tau, amp=amp, wn=wn, alpha=alpha)
    y_f = dnu_sspec_model(f=f, dnu=dnu, amp=amp, wn=wn)
    # return list(y_t) + list(y_f)  # concatenate t and f models

    return (ydata - model) * weights


def arc_power_curve(params, xdata, ydata, weights):
    """
    Returns a template for the power curve in secondary spectrum vs
    sqrt(curvature) or normalised fdop
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    model = []
    return (ydata - model) * weights


def thin_screen(params, xdata, ydata, weights):
    """
    Thin screen effective velocity
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    model = []
    return (ydata - model) * weights



#def arc_curvature(params, xdata, ydata, weights, freqs, true_anomaly,
#                  vearth_ra, vearth_dec):
#    """
#    arc curvature model
#
#        xdata: MJDs at barycentre
#        ydata: arc curvature
#    """
#
#    # Other parameters in lower-case
#    d = params['d']  # pulsar distance in kpc
#    s = params['s']  # fractional screen distance
#    vism_ra = params['vism_ra']  # ism velocity in RA
#    vism_dec = params['vism_dec']  # ism velocity in DEC
#
#    veff_ra, veff_dec = effective_velocity_annual(params, true_anomaly,
#                                                  vearth_ra, vearth_dec)
#
#
#    if 'vism_ra' in params.keys():
#        vism_ra = params['vism_ra']
#        vism_dec = params['vism_dec']
#    else:
#        vism_ra = 0
#        vism_dec = 0
#
#    if 'psi' in params.keys():
#        psi = params['psi']  # anisotropy angle
#        vism_psi = params['vism_psi']  # anisotropy angle
#        veff_ra = veff_ra - vism_psi * np.sin(psi)
#        veff_dec = veff_dec - vism_psi * np.cos(psi)
#        # angle between scattered image and velocity vector
#        cosa = np.cos(psi - np.arctan2(veff_ra, veff_ dec)))
#    else:
#        veff_ra = veff_ra - vism_ra
#        veff_dec = veff_dec - vism_dec
#        cosa = 1
#
#    #Now calculate veff
#    veff = np.sqrt(veffra**2 + veffdec**2)
#
#    # Calculate eta
#    etamodel= D*x(1)*(1-x(1)).*lambda.^2./(2*v_c*veff**2 * cosa**2)
#
#    if weights is None:
#        weights = np.ones(np.shape(ydata))
#
#    model = []
#    return (ydata - model) * weights
#
#
#"""
#Below: Models that do not return residuals for a fitter
#"""
#
#
#def effective_velocity_annual(params, true_anomaly, vearth_ra, vearth_dec):
#    """
#    Effective velocity with annual and pulsar terms
#        Note: Does NOT include IISM velocity, but returns veff in IISM frame
#    """
#
#    # Define some constants
#    v_c = 299792.458  # km/s
#    kmpkpc = 3.085677581e16
#    secperyr = 86400*365.2425
#    masrad = np.pi/(3600*180*1000)
#
#    # tempo2 parameters from par file in capitals
#    if 'PB' in params.keys():
#        A1 = params['A1']  # projected semi-major axis in lt-s
#        PB = params['PB']  # orbital period in days
#        ECC = params['ECC']  # orbital eccentricity
#        OM = params['OM']*np.pi/180  # longitude of periastron rad
#        # Note: fifth Keplerian param T0 used in true anomaly calculation
#
#        # Calculate pulsar velocity aligned with the line of nodes (Vx) and
#        #   perpendicular in the plane (Vy)
#        vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(KIN) * PB * 86400 *
#                                         np.sqrt(1 - ECC**2))
#        vp_x = -vp_0 * (ECC * np.sin(OM) + np.sin(true_anomaly + OM))
#        vp_y = vp_0 * cos(KIN) * (ECC * cos(OM) + np.cos(true_anomaly + OM))
#    else:
#        vp_x = 0
#        vp_y = 0
#
#    PMRA = params['PMRA']  # orbital eccentricity
#    PMDEC = params['PMDEC']  # longitude of periastron
#    # other parameters in lower-case
#    s = params['s']  # fractional screen distance
#    d = params['d']  # pulsar distance in kpc
#    d = d * kmpkpc  # distance in km
#    pmra_v = PMRA * masrad * d / secperyr
#    pmdec_v = PMDEC * masrad * d / secperyr
#
#    # Rotate pulsar velocity into RA/DEC
#    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
#    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y
#
#    # find total effective velocity in RA and DEC
#    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v)
#    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v)
#
#    return veff_ra, veff_dec


def fit_parabola(x, y):
    """
    Fit a parabola and return the value and error for the peak
    """

    # Do the fit
    params, pcov = np.polyfit(x, y, 2, cov=True)
    yfit = params[0]*np.power(x, 2) + params[1]*x + params[2]  # y values

    # Get parameter errors
    errors = []
    for i in range(len(params)):  # for each parameter
        errors.append(np.absolute(pcov[i][i])**0.5)

    # Get parabola peak and error
    #   y = a(x – h)**2 + k
    peak = -params[1]/(2*params[0])  # Parabola max (or min)
    peak_error = np.sqrt((errors[1]**2)*((1/(2*params[0]))**2) +
                         (errors[0]**2)*((params[1]/2)**2))  # Error on peak

    return yfit, peak, peak_error
