#!/usr/bin/env python

"""
scint_utils.py
----------------------------------
Useful functions for scintools
"""
from dynspec import Dynspec
from os.path import split
import numpy as np


def make_dynspec(archive, template):
    """
    Creates a psrflux-format dynamic spectrum from a psrchive or from
    """
    return


def sort_dynspec(datadir, tmin=10, fmin=64):
    """
    Automatically sorts dynspec into "good" or "bad" on some conditions
    """
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


def sort_dyn(dynfiles, outdir=None, min_nsub=10, min_nchan=50, min_tsub=10,
             min_freq=0, max_freq=5000, remove_nan_sspec=False, verbose=True,
             max_frac_bw=2):
    """
    Sorts dynamic spectra into good and bad files based on some conditions
    """
    if verbose:
        print("Sorting dynspec files in {0}".format(split(dynfiles[0])[0]))
        n_files = len(dynfiles)
        file_count = 0
    if outdir is None:
        outdir, dummy = split(dynfiles[0])  # path of first dynspec
    bad_files = open(outdir+'/bad_files.txt', 'w')
    good_files = open(outdir+'/good_files.txt', 'w')
    bad_files.write("FILENAME\t REASON\n")
    for dynfile in dynfiles:
        if verbose:
            file_count += 1
            print("{0}/{1}\t{2}".format(file_count, n_files,
                  split(dynfile)[1]))
        # Read in dynamic spectrum
        dyn = Dynspec(filename=dynfile, verbose=False, process=False)
        if dyn.freq > max_freq or dyn.freq < min_freq:
            # outside of frequency range
            if dyn.freq < min_freq:
                message = 'freq<{0} '.format(min_freq)
            elif dyn.freq > max_freq:
                message = 'freq>{0}'.format(max_freq)
            bad_files.write("{0}\t{1}\n".format(dynfile, message))
            continue
        if dyn.bw/dyn.freq > max_frac_bw:
            # bandwidth too large
            bad_files.write("{0}\t frac_bw>{1}\n".format(dynfile, max_frac_bw))
            continue
        # Start processing
        dyn.trim_edges()  # remove band edges
        if dyn.nchan < min_nchan or dyn.nsub < min_nsub:
            # skip if not enough channels/subints
            message = ''
            if dyn.nchan < min_nchan:
                message += 'nchan<{0} '.format(min_nchan)
            if dyn.nsub < min_nsub:
                message += 'nsub<{0}'.format(min_nsub)
            bad_files.write("{0}\t {1}\n".format(dynfile, message))
            continue
        elif dyn.tobs < 60*min_tsub:
            # skip if observation too short
            bad_files.write("{0}\t tobs<{1}\n".format(dynfile, min_tsub))
            continue
        dyn.refill()  # linearly interpolate zero values
        dyn.correct_band(time=True)  # correct for bandpass and gain variation
        dyn.calc_sspec()  # calculate secondary spectrum
        if np.isnan(dyn.sspec).all():  # skip if secondary spectrum is all nan
            bad_files.write("{0}\t sspec_isnan\n".format(dynfile))
            continue
        # Passed all tests so far - write to good_files.txt!
        good_files.write("{0}\n".format(dynfile))
    bad_files.close()
    good_files.close()
    return


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


def make_json(dyn, process=True, sspec=True, acf=True, lamsteps=True):
    """
    Saves dynamic spectra object into json file
    """
    return
