#!/usr/bin/env python

"""
velocity.py
----------------------------------
Scintillation effective velocity models
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, SkyCoord
from astropy import units as u
from astropy.constants import au, c


def thin_screen(pulsar, mjd, anisotropy=False):
    """
    Thin screen effective velocity
    """

    v_eff = []
    return v_eff


def arc_curvature(v_eff, lamsteps=False):
    """
    arc curvature model
    """

    curvature = []
    return curvature


def get_ssb_delay(mjd_array, raj, decj):
    """
    Get Romer delay to Solar System Barycentre (SSB) for correction of site
    arrival times to barycentric.
    """

    coord = SkyCoord('{0} {1}'.format(raj, decj), unit=(u.hourangle, u.deg))
    psr_xyz = coord.cartesian.xyz.value

    t = []
    for mjd in mjd_array:
        time = Time(mjd, format='mjd')
        earth_xyz = get_body_barycentric('earth', time)
        e_dot_p = np.dot(earth_xyz.xyz.value, psr_xyz)
        t.append(e_dot_p*au.value/c.value)

    return t
