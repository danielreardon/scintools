#!/usr/bin/env python

"""
models.py
----------------------------------
Scintillation models
"""
import numpy as np


def tauModel(x, tau, amp, wn, alpha):
    """
    Fit 1D function to cut through ACF for scintillation timescale. 
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """
    y = amp*np.exp(-abs(np.divide(x,tau)**(alpha)))
    y[0] = y[0] + wn #add white noise spike
    y = np.multiply(y,1-np.divide(x,max(x)))
    return y


def dnuModel(x, dnu, amp, wn):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth. 
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        tau = timescale at 1/e
        wn = white noise spike in ACF cut
    """
    y = amp*np.exp(-abs(np.divide(x,dnu/np.log(2))))
    y[0] = y[0] + wn #add white noise spike
    y = np.multiply(y,1-np.divide(x,max(x)))
    return y

