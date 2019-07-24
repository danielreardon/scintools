#!/usr/bin/env python

"""
pulsar.py
----------------------------------
Pulsar class for reading parameters and toas
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import libstempo as T
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set('jpl')


class Pulsar:

    def __init__(self, parfile, timfile):
        psr = T.tempopulsar(parfile=parfile, timfile=timfile)
        return




def calc_ssb_romer(mjd_array):
    """
    Get Romer delay to Solar System Barycentre (SSB) for correction of site
    arrival times to barycentric
    """

    times = Time(mjd_array, format='mjd')
    d = []
    for t in times:
        x, y, z = get_body_barycentric('earth', t)
        d.append(np.sqrt(x**2 + y**2 + z**2))

    return d
