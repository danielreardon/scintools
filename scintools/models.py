#!/usr/bin/env python

"""
models.py
----------------------------------
Scintillation models
"""

import numpy as np


def tauModel(t, tau, amp, wn, alpha):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """
    y = amp*np.exp(-np.divide(t, tau)**(alpha))
    y[0] = y[0] + wn  # add white noise spike
    y = np.multiply(y, 1-np.divide(t, max(t)))
    return y


def dnuModel(f, dnu, amp, wn):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """
    y = amp*np.exp(-np.divide(f, dnu/np.log(2)))
    y[0] = y[0] + wn  # add white noise spike
    y = np.multiply(y, 1-np.divide(f, max(f)))
    return y


def arc_power_curve(x, x_peak, amp, norm=False):
    """
    Returns a template for the power curve in secondary spectrum vs
    sqrt(survature) or normalised fdop
    """

    return y


def parabola(x, x_peak, amp, sign=1):
    """
    Inverse parabola
    """
    y = sign*np.power((x_peak - eta), 2) + amp
    return y
