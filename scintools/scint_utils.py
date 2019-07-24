#!/usr/bin/env python

"""
scint_utils.py
----------------------------------
Useful functions for scintools
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import os
import csv


def make_dynspec(archive, template):
    """
    Creates a psrflux-format dynamic spectrum from an archive
        $ psrflux -s [template] -e dynspec [archive]
    """
    return


def clean_archive(archive, template=None, bandwagon=0.99, channel_threshold=7,
                  subint_threshold=5, output_directory=None):
    """
    Cleans an archive using coast_guard
    """

    # Try importing necessary modules
    try:
        import psrchive as ps
    except ModuleNotFoundError:
        print('Psrchive not found. Cannot load archives. Returning')
        return
    try:
        from coast_guard import cleaners
    except ModuleNotFoundError:
        print('Coast_guard not found. Cannot clean. Returning')
        return

    # Load the archive
    archive = ps.Archive_load(str(archive))
    archive_path, archive_name = os.path.split(archive.get_filename())
    archive_name = archive_name.split('.')[0]
    if output_directory is None:
        output_directory = archive_path

    # Clean the archive with surgical cleaner
    print("Applying surgical cleaner")
    surgical_cleaner = cleaners.load_cleaner('surgical')
    surgical_parameters = 'chan_numpieces=1,subint_numpieces=1,\
                           chanthresh={},subintthresh={}'.format(
                           channel_threshold, subint_threshold)
    surgical_cleaner.parse_config_string(surgical_parameters)
    surgical_cleaner.run(archive)

    # Apply bandwagon cleaner cleaner
    print("Applying bandwagon cleaner")
    bandwagon_cleaner = cleaners.load_cleaner('bandwagon')
    bandwagon_parameters = 'badchantol={},badsubtol=1.0'.format(bandwagon)
    bandwagon_cleaner.parse_config_string(bandwagon_parameters)
    bandwagon_cleaner.run(archive)

    # Unload cleaned archive
    unload_path = os.path.join(output_directory, archive_name + ".clean")
    print('Unloading cleaned archive as {0}'.format(unload_path))
    archive.unload(unload_path)
    return


def is_valid(array):
    """
    Returns boolean array of values that are finite an not nan
    """
    return np.isfinite(array)*(~np.isnan(array))


def read_dynlist(file_path):
    """
    Reads list of dynamic spectra filenames from path
    """
    with open(file_path) as file:
        dynfiles = file.read().splitlines()
    return dynfiles


def write_results(filename, dyn=None):
    """
    Appends dynamic spectrum information and parameters of interest to file
    """

    header = "name,mjd,freq,bw,tobs,dt,df"
    write_string = "{0},{1},{2},{3},{4},{5},{6}".\
                   format(dyn.name, dyn.mjd, dyn.freq, dyn.bw, dyn.tobs,
                          dyn.dt, dyn.df)

    if hasattr(dyn, 'tau'):  # Scintillation timescale
        header += ",tau,tauerr"
        write_string += ",{0},{1}".format(dyn.tau, dyn.tauerr)

    if hasattr(dyn, 'dnu'):  # Scintillation bandwidth
        header += ",dnu,dnuerr"
        write_string += ",{0},{1}".format(dyn.dnu, dyn.dnuerr)

    if hasattr(dyn, 'eta'):  # Arc curvature
        header += ",eta,etaerr"
        write_string += ",{0},{1}".format(dyn.eta, dyn.etaerr)

    if hasattr(dyn, 'betaeta'):  # Beta arc curvature
        header += ",betaeta,betaetaerr"
        write_string += ",{0},{1}".format(dyn.betaeta, dyn.betaetaerr)

    header += "\n"
    write_string += "\n"

    with open(filename, "a") as outfile:
        if os.stat(filename).st_size == 0:  # file is empty, write header
            outfile.write(header)
        outfile.write(write_string)
    return


def read_results(filename):
    """
    Reads a CSV results file written by write_results()
    """

    csv_data = open(filename, 'r')
    data = list(csv.reader(csv_data, delimiter=","))
    keys = data[0]
    param_dict = {k: [] for k in keys}
    for row in data[1:]:
        for ii in range(0, len(row)):
            param_dict[keys[ii]].append(row[ii])

    return param_dict


def float_array_from_dict(dictionary, key):
    """
    Convert an array stored in dictionary to a numpy array
    """
    return np.array(list(map(float, dictionary[key])))


def get_ssb_delay(mjd_array, raj, decj):
    """
    Get Romer delay to Solar System Barycentre (SSB) for correction of site
    arrival times to barycentric.
    """
    try:
        from astropy.time import Time
        from astropy.coordinates import get_body_barycentric, SkyCoord
        from astropy import units as u
        from astropy.constants import au, c
    except ModuleNotFoundError:
        print('Astropy not found')
        return

    coord = SkyCoord('{0} {1}'.format(raj, decj), unit=(u.hourangle, u.deg))
    psr_xyz = coord.cartesian.xyz.value

    t = []
    for mjd in mjd_array:
        time = Time(mjd, format='mjd')
        earth_xyz = get_body_barycentric('earth', time)
        e_dot_p = np.dot(earth_xyz.xyz.value, psr_xyz)
        t.append(e_dot_p*au.value/c.value)

    return t


# Potential future functions


def remove_duplicates(dyn_files):
    """
    Filters out dynamic spectra from simultaneous observations
    """
    return dyn_files


def make_pickle(dyn, process=True, sspec=True, acf=True, lamsteps=True):
    """
    Pickles a dynamic spectra object
    """
    return
