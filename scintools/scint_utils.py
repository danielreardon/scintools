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


def make_dynspec(archive, template):
    """
    Creates a psrflux-format dynamic spectrum from an archive
        $ psrflux -s [template] -e dynspec [archive]
    """
    return


def clean_archive(archive, template=None, bandwagon=0.99, channel_threshold=7,
                  subint_threshold=4, output_directory=None):
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
