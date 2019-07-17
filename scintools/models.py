#!/usr/bin/env python

"""
models.py
----------------------------------
Scintillation models
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np


def tau_acf_model(t, tau, amp, wn, alpha):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """

    y = amp*np.exp(-np.divide(t, tau)**(alpha))
    y[0] = y[0] + wn  # add white noise spike
    y = np.multiply(y, 1-np.divide(t, max(t)))  # triangle function
    return y


def dnu_acf_model(f, dnu, amp, wn):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """

    y = amp*np.exp(-np.divide(f, dnu/np.log(2)))
    y[0] = y[0] + wn  # add white noise spike
    y = np.multiply(y, 1-np.divide(f, max(f)))  # triangle function
    return y


def scint_acf_model(tf, tau, dnu, amp, wn, alpha, nt):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    y_t = tau_acf_model(t=tf[:nt], tau=tau, amp=amp, wn=wn, alpha=alpha)
    y_f = dnu_acf_model(f=tf[nt:], dnu=dnu, amp=amp, wn=wn)
    return list(y_t) + list(y_f)  # concatenate t and f models


def tau_sspec_model(t, tau, amp, wn, alpha):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """

    y = tau_acf_model(t=t, tau=tau, amp=amp, wn=wn, alpha=alpha)
    # From ACF model, construct Fourier-domain model
    y_flipped = y[::-1]
    y = list(y) + list(y_flipped)  # concatenate
    y = y[0:2*len(t)-1]
    # Get Fourier model
    yf = np.fft(y)
    yf = np.real(yf)
    yf = yf[0:len(t)]
    return yf


def dnu_sspec_model(f, dnu, amp, wn):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """

    y = dnu_acf_model(f=f, dnu=dnu, amp=amp, wn=wn)
    # From ACF model, construct Fourier-domain model
    y_flipped = y[::-1]
    y = list(y) + list(y_flipped)  # concatenate
    y = y[0:2*len(f)-1]
    # Get Fourier model
    yf = np.fft(y)
    yf = np.real(yf)
    yf = yf[0:len(f)]
    return yf


def scint_sspec_model(t, f, tau, dnu, amp, wn, alpha):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    y_t = tau_sspec_model(t=t, tau=tau, amp=amp, wn=wn, alpha=alpha)
    y_f = dnu_sspec_model(f=f, dnu=dnu, amp=amp, wn=wn)
    return list(y_t) + list(y_f)  # concatenate t and f models


def arc_power_curve(x, x_peak, amp, norm=False):
    """
    Returns a template for the power curve in secondary spectrum vs
    sqrt(curvature) or normalised fdop
    """

    y = x
    return y


def fit_parabola(x, x_peak, amp, const, sign=-1):
    """
    Fit a parabola
        y = amp*(x - x_peak)**2 + const
        Default is an inverted (sign=-1) parabola.
    """

    y = sign*amp*np.power((x - x_peak), 2) + const
    return y
