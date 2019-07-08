#!/usr/bin/env python

"""
scint_utils.py
----------------------------------
Useful functions for scintools
"""
import numpy as np


def make_dynspec(archive, template):
    """
    Creates a psrflux-format dynamic spectrum from an archive

    psrflux -s [template] -e dynspec [archive]
    """
    try:
        import psrchive
    except ImportError:
        print('No psrchive package, cannot generate dynamic spectrum')
        return


def clean_archive(archive, template, bandwagon=0.99, channel_threshold=4,
                  subint_threshold=4):
    """
    Cleans an archive using coast_guard
    """
    try:
        from coast_guard import cleaners
    except ModuleNotFoundError:
        print('Warning: coast_guard not found. Cannot clean')
        return
    # Clean the archive with surgical cleaner
    print('This does nothing yet')
    return cleaned_archive


def is_valid(array):
    """
    Returns boolean array of values that are finite an not nan
    """
    return np.isfinite(array)*(~np.isnan(array))


def read_dynlist(filepath):
    """
    Reads list of dynamic spectra filenames from path
    """
    with open(filepath) as file:
        dynfiles = file.read().splitlines()
    return dynfiles


def remove_duplicates(dynfiles):
    """
    Filters out dynamic spectra from simultaneous observations
    """
    return dynfiles


def make_pickle(dyn, process=True, sspec=True, acf=True, lamsteps=True):
    """
    Pickles a dynamic spectra object
    """
    return
