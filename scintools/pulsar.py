#!/usr/bin/env python

"""
pulsar.py
----------------------------------
Pulsar class for reading parameters and toas
"""

import libstempo as T


class Pulsar:

    def __init__(self, parfile, timfile):
        psr = T.tempopulsar(parfile=parfile, timfile=timfile)
        return

    def get_ecc_anomaly(self, mjd, barycentric=True):
        return

    def get_true_anomaly(self, mjd, barycentric=True):
        return

    def get_binary_phase(self, mjd, barycentric=True):
        return

    def next_superior_conjunction(self, mjd, barycentric=True):
        return


def calc_romer(mjd):
    return
