#!/usr/bin/env python

"""
scint_utils.py
----------------------------------
General utility functions used throughout scintools.

This module collects functions that are not specific to any single
scintools class, including:

- Reading and writing parameter files (``.par``) and scintillation
  results files (``read_par``, ``pars_to_params``, ``write_results``,
  ``read_results``, ``read_dynlist``, ``search_and_replace``).
- Astrophysical/geometric calculations such as barycentric (Roemer)
  delays, Earth's transverse velocity, binary orbit true anomaly and
  binary phase, LSR proper motions, and differential ISM/Sun velocity
  (``get_ssb_delay``, ``get_earth_velocity``, ``make_lsr``,
  ``get_true_anomaly``, ``get_binphase``, ``differential_velocity``,
  ``scint_velocity``).
- Cleaning and creating dynamic spectra from psrchive archives
  (``clean_archive``, ``make_dynspec``).
- Array/statistics helpers, e.g. autocorrelation, SVD-based modelling,
  a slow Fourier transform along scaled axes, NaN interpolation, and
  simple array utilities (``autocorr``, ``acor``, ``svd_model``,
  ``slow_FT``, ``interp_nan_2d``, ``cov_to_corr``, ``difference``,
  ``find_nearest``, ``centres_to_edges``, ``longest_run_of_zeros``,
  ``is_valid``, ``float_array_from_dict``, ``get_window``).
- Curvature likelihood/probability calculations used when fitting
  scintillation arcs (``calculate_curvature_peak_probability``,
  ``curvature_log_likelihood``, ``save_curvature_data``).
- Pickling and FITS I/O helpers for large dynamic spectrum objects
  (``make_pickle``, ``load_pickle``, ``save_fits``).
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import os
import sys
import csv
from decimal import Decimal, InvalidOperation
from scipy.optimize import fsolve
from scipy.signal import correlate
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import pickle
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body_barycentric


def clean_archive(archive, template=None, bandwagon=0.99, channel_threshold=5,
                  subint_threshold=5, output_directory=None):
    """
    Cleans a psrchive archive using the ``coast_guard`` ``surgical`` and
    ``bandwagon`` cleaners, then unloads the cleaned archive to disk with
    a ``.clean`` extension.

    Parameters
    ----------
    archive : str or psrchive Archive
        Path to, or already-loaded psrchive archive object, to be
        cleaned.
    template : str, optional
        Path to a template profile. Currently unused by this function,
        reserved for future use with the cleaners.
    bandwagon : float, optional
        ``badchantol`` value passed to the ``coast_guard`` ``bandwagon``
        cleaner (fraction of subintegrations that must be already
        flagged bad before a channel is flagged). The default is 0.99.
    channel_threshold : float, optional
        ``chanthresh`` value (sigma threshold for flagging channels)
        passed to the ``surgical`` cleaner. The default is 5.
    subint_threshold : float, optional
        ``subintthresh`` value (sigma threshold for flagging
        subintegrations) passed to the ``surgical`` cleaner. The default
        is 5.
    output_directory : str, optional
        Directory to write the cleaned archive to. If None, the
        cleaned archive is written alongside the input archive.

    Returns
    -------
    None
        The cleaned archive is written to disk as
        ``<archive_name>.clean`` in `output_directory`.
    """

    # import necessary modules
    import psrchive as ps
    from coast_guard import cleaners

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


def autocorr(arr):
    """
    Do a slow (direct, nested-loop) calculation of the 2D autocorrelation
    of an input masked array.

    Parameters
    ----------
    arr : numpy.ma.MaskedArray
        2D masked array to autocorrelate.

    Returns
    -------
    numpy.ndarray
        2D autocorrelation of `arr`, with shape twice that of `arr` in
        each dimension, normalised so the maximum value is 1.
    """
    mean = np.ma.mean(arr)
    std = np.ma.std(arr)
    nr, nc = np.shape(arr)
    autocorr = np.zeros((2*nr, 2*nc))
    for x in range(-nr, nr):
        for y in range(-nc, nc):
            segment = (arr[max(0, x):min(x+nr, nr),
                           max(0, y):min(y+nc, nc)] - mean) \
                * (arr[max(0, -x):min(-x+nr, nr),
                       max(0, -y):min(-y+nc, nc)] - mean)
            numerator = np.ma.sum(segment)
            autocorr[x+nr][y+nc] = numerator / (std ** 2)
    autocorr /= np.nanmax(autocorr)
    return autocorr


def is_valid(array):
    """
    Returns boolean array of values that are finite and not NaN.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray of bool
        Boolean array, True where `array` is finite and not NaN.
    """
    return np.isfinite(array)*(~np.isnan(array))


def read_dynlist(file_path):
    """
    Reads list of dynamic spectra filenames from path.

    Parameters
    ----------
    file_path : str
        Path to a text file containing dynamic spectrum file paths, one
        per line.

    Returns
    -------
    list of str
        List of dynamic spectrum file paths.
    """
    with open(file_path) as file:
        dynfiles = file.read().splitlines()
    return dynfiles


def write_results(filename, dyn=None):
    """
    Appends dynamic spectrum information and parameters of interest to
    file, as a CSV row. Writes a header row first if `filename` is
    empty or does not yet exist. The set of columns written depends on
    which scintillation parameter attributes are present on `dyn`
    (e.g. ``tau``, ``dnu``, ``eta``, ``phasegrad``, etc.).

    Parameters
    ----------
    filename : str
        Path of the CSV file to append to (created if it does not
        exist).
    dyn : Dynspec, optional
        Dynspec object whose attributes (name, mjd, freq, bw, tobs, dt,
        df, and any fitted scintillation parameters) are written as a
        row.

    Returns
    -------
    None
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

    if hasattr(dyn, 'fse_tau'):  # Finite scintle error timescale and bandwidth
        header += ",fse_tau,fse_dnu"
        write_string += ",{0},{1}".format(dyn.fse_tau, dyn.fse_dnu)

    if hasattr(dyn, 'scint_param_method'):  # Method of scint measurement
        header += ",scint_param_method"
        write_string += ",{0}".format(dyn.scint_param_method)

    if hasattr(dyn, 'dnu_est'):  # Estimated scintillation bandwidth
        header += ",dnu_est"
        write_string += ",{0}".format(dyn.dnu_est)

    if hasattr(dyn, 'nscint'):  # Estimated number of scintles
        header += ",nscint"
        write_string += ",{0}".format(dyn.nscint)

    if hasattr(dyn, 'ar'):  # Axial ratio
        header += ",ar,arerr"
        write_string += ",{0},{1}".format(dyn.ar, dyn.arerr)

    if hasattr(dyn, 'acf_tilt'):  # Tilt in the ACF (MHz/min)
        header += ",acf_tilt,acf_tilt_err"
        write_string += ",{0},{1}".format(dyn.acf_tilt, dyn.acf_tilt_err)

    if hasattr(dyn, 'fse_tilt'):  # Finite scintle error, tilt
        header += ",fse_tilt"
        write_string += ",{0}".format(dyn.fse_tilt)

    if hasattr(dyn, 'phasegrad'):  # Phase gradient (shear to the ACF)
        header += ",phasegrad,phasegraderr"
        write_string += ",{0},{1}".format(dyn.phasegrad, dyn.phasegraderr)

    if hasattr(dyn, 'fse_phasegrad'):  # Finite scintle error, phase gradient
        header += ",fse_phasegrad"
        write_string += ",{0}".format(dyn.fse_phasegrad)

    if hasattr(dyn, 'theta'):  # Phase gradient angle relative to V
        header += ",theta,thetaerr"
        write_string += ",{0},{1}".format(dyn.theta, dyn.thetaerr)

    if hasattr(dyn, 'psi'):  # Anisotropy angle relative to V
        header += ",psi,psierr"
        write_string += ",{0},{1}".format(dyn.psi, dyn.psierr)

    if hasattr(dyn, 'eta'):  # Arc curvature
        header += ",eta,etaerr"
        write_string += ",{0},{1}".format(dyn.eta, dyn.etaerr)

    if hasattr(dyn, 'betaeta'):  # Beta arc curvature
        header += ",betaeta,betaetaerr"
        write_string += ",{0},{1}".format(dyn.betaeta, dyn.betaetaerr)

    if hasattr(dyn, 'eta_left'):  # Arc curvature
        header += ",eta_left,etaerr_left"
        write_string += ",{0},{1}".format(dyn.eta_left, dyn.etaerr_left)

    if hasattr(dyn, 'betaeta_left'):  # Beta arc curvature
        header += ",betaeta_left,betaetaerr_left"
        write_string += ",{0},{1}".format(dyn.betaeta_left,
                                          dyn.betaetaerr_left)

    if hasattr(dyn, 'eta_right'):  # Arc curvature
        header += ",eta_right,etaerr_right"
        write_string += ",{0},{1}".format(dyn.eta_right, dyn.etaerr_right)

    if hasattr(dyn, 'betaeta_right'):  # Beta arc curvature
        header += ",betaeta_right,betaetaerr_right"
        write_string += ",{0},{1}".format(dyn.betaeta_right,
                                          dyn.betaetaerr_right)

    if hasattr(dyn, 'norm_delmax'):
        header += ",delmax"
        write_string += ",{0}".format(dyn.norm_delmax)

    header += "\n"
    write_string += "\n"

    with open(filename, "a+") as outfile:
        if os.stat(filename).st_size == 0:  # file is empty, write header
            outfile.write(header)
        outfile.write(write_string)
    return


def read_results(filename):
    """
    Reads a CSV results file written by `write_results`.

    Parameters
    ----------
    filename : str
        Path of the CSV results file to read.

    Returns
    -------
    dict
        Dictionary keyed by column name (from the file's header row),
        with each value a list of the (string) entries in that column.
    """

    csv_data = open(filename, 'r')
    data = list(csv.reader(csv_data, delimiter=","))
    keys = data[0]
    param_dict = {k: [] for k in keys}
    for row in data[1:]:
        for ii in range(0, len(row)):
            param_dict[keys[ii]].append(row[ii])

    return param_dict


def search_and_replace(filename, search, replace):
    """
    Reads a text file, replaces all occurrences of one substring with
    another, and overwrites the file with the result.

    Parameters
    ----------
    filename : str
        Path to the text file to edit in place.
    search : str
        Substring to search for.
    replace : str
        Substring to substitute in place of `search`.

    Returns
    -------
    None
    """

    with open(filename, 'r') as file:
        data = file.read()

    data = data.replace(search, replace)

    with open(filename, 'w') as file:
        file.write(data)

    return


def cov_to_corr(cov):
    """
    Calculate correlation matrix from a covariance matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        2D covariance matrix.

    Returns
    -------
    numpy.ndarray
        Correlation matrix corresponding to `cov`. Entries
        corresponding to zero covariance are set to 0.
    """
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    corr = cov / outer_std
    corr[cov == 0] = 0
    return corr


def float_array_from_dict(dictionary, key):
    """
    Convert an array stored in a dictionary (e.g. as returned by
    `read_results`) to a numpy float array, treating the string
    ``'None'`` as NaN.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing the array to convert, keyed by `key`.
    key : str
        Key of the array within `dictionary`.

    Returns
    -------
    numpy.ndarray
        Squeezed numpy float array of the values stored at
        ``dictionary[key]``.
    """
    ind = np.argwhere(np.array(dictionary[key]) == 'None').ravel()

    if ind.size != 0:
        arr = dictionary[key]
        for i in ind:
            arr[i] = 'nan'
        dictionary[key] = arr

    return np.array(list(map(float, dictionary[key]))).squeeze()


def save_fits(filename, dyn):
    """
    Saves a dynamic spectrum to a FITS file as the primary HDU.

    Parameters
    ----------
    filename : str
        Path of the FITS file to write. Raises an error if the file
        already exists.
    dyn : Dynspec
        Dynspec object whose ``dyn`` attribute (2D dynamic spectrum
        array) is written to the FITS file, transposed and flipped so
        that frequency runs along the first axis in the expected FITS
        orientation.

    Returns
    -------
    None
    """

    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.flip(np.transpose(np.flip(dyn.dyn, axis=1)),
                                  axis=0))
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename)


def difference(x):
    """
    Unlike `numpy.diff`, computes differences between the centres of
    neighbouring elements in `x`, returning an array the same size as
    `x`. For interior points this is ``(x[i+1] - x[i-1]) / 2``; at the
    edges it is the one-sided half-difference.

    Parameters
    ----------
    x : array_like
        Input 1D array.

    Returns
    -------
    numpy.ndarray
        Array of centred differences, the same length as `x`.
    """
    dx = []
    for i in range(0, len(x)):
        if i == 0:
            dx.append((x[i+1] - x[i])/2)
        elif i == len(x)-1:
            dx.append((x[i] - x[i-1])/2)
        else:
            dx.append((x[i+1] - x[i-1])/2)
    return np.array(dx).squeeze()


def get_ssb_delay(mjds, raj, decj, message=True):
    """
    Get Romer delay to Solar System Barycentre (SSB) for correction of
    site arrival times to barycentric.

    Parameters
    ----------
    mjds : array_like
        Modified Julian Dates to calculate the delay for.
    raj : str
        Right ascension (J2000) of the pulsar, in hourangle-parsable
        string format (e.g. ``'hh:mm:ss'``).
    decj : str
        Declination (J2000) of the pulsar, in degree-parsable string
        format (e.g. ``'dd:mm:ss'``).
    message : bool, optional
        If True (default), print a reminder that the returned delays
        should be added to the site arrival times.

    Returns
    -------
    numpy.ndarray
        Romer delays to the SSB, in seconds, one per input MJD.
    """

    from astropy.constants import au, c
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

    coord = SkyCoord('{0} {1}'.format(raj, decj),
                     frame=BarycentricTrueEcliptic,
                     unit=(u.hourangle, u.deg))
    psr_xyz = coord.cartesian.xyz.value

    t = []
    for mjd in mjds:
        time = Time(mjd, format='mjd')
        earth_xyz = get_body_barycentric('earth', time)
        e_dot_p = np.dot(earth_xyz.xyz.value, psr_xyz)
        t.append(e_dot_p*au.value/c.value)

    if message:
        print('Returned SSB Roemer delays (in seconds) should be ' +
              'ADDED to site arrival times')

    return np.array(t)


def make_lsr(d, raj, decj, pmra, pmdec, vr=0):
    """
    Converts a pulsar's barycentric proper motion and radial velocity to
    proper motion in the Local Standard of Rest (LSR) frame, using the
    IAU standard solar motion (11.1, 12.24, 7.25 km/s) with respect to
    the LSR.

    Parameters
    ----------
    d : float
        Distance to the pulsar, in kpc.
    raj : str
        Right ascension (J2000) of the pulsar, in hourangle-parsable
        string format (e.g. ``'hh:mm:ss'``).
    decj : str
        Declination (J2000) of the pulsar, in degree-parsable string
        format (e.g. ``'dd:mm:ss'``).
    pmra : float
        Proper motion in right ascension (``pm_ra_cosdec``), in mas/yr.
    pmdec : float
        Proper motion in declination, in mas/yr.
    vr : float, optional
        Radial velocity, in km/s. The default is 0.

    Returns
    -------
    numpy.ndarray
        Proper motion components (RA, Dec) of the pulsar in the LSR
        frame, in mas/yr.
    """
    from astropy.coordinates import BarycentricTrueEcliptic, LSR, SkyCoord
    from astropy import units as u

    coord = SkyCoord('{0} {1}'.format(raj, decj), unit=(u.hourangle, u.deg))
    ra = coord.ra.value
    dec = coord.dec.value

    # Initialise the barycentric coordinates with the LSR class and v_bary=0
    pm = LSR(ra=ra*u.degree, dec=dec*u.deg,
             pm_ra_cosdec=pmra*u.mas/u.yr,
             pm_dec=pmdec*u.mas/u.yr, distance=d*u.kpc,
             radial_velocity=vr*u.km/u.s,
             v_bary=(0.0*u.km/u.s, 0.0*u.km/u.s, 0.0*u.km/u.s))
    pm_ecliptic = pm.transform_to(BarycentricTrueEcliptic)

    # Get barycentric ecliptic coordinates
    elat = coord.barycentrictrueecliptic.lat.value
    elong = coord.barycentrictrueecliptic.lon.value
    pm_lat = pm_ecliptic.pm_lat.value
    pm_lon_coslat = pm_ecliptic.pm_lon_coslat.value

    bte = BarycentricTrueEcliptic(lon=elong*u.degree, lat=elat*u.degree,
                                  distance=d*u.kpc,
                                  pm_lon_coslat=pm_lon_coslat*u.mas/u.yr,
                                  pm_lat=pm_lat*u.mas/u.yr,
                                  radial_velocity=vr*u.km/u.s)

    # Convert barycentric back to LSR
    lsr_coord = bte.transform_to(LSR(v_bary=(11.1*u.km/u.s,
                                             12.24*u.km/u.s, 7.25*u.km/u.s)))

    return lsr_coord.proper_motion.to_value()


def get_earth_velocity(mjds, raj, decj, radial=False):
    """
    Calculates the component of Earth's velocity transverse to the line
    of sight, in RA and Dec. Optionally also returns the radial
    velocity component.

    Parameters
    ----------
    mjds : array_like
        Modified Julian Dates to calculate the velocity for.
    raj : str
        Right ascension (J2000) of the pulsar, in hourangle-parsable
        string format (e.g. ``'hh:mm:ss'``).
    decj : str
        Declination (J2000) of the pulsar, in degree-parsable string
        format (e.g. ``'dd:mm:ss'``).
    radial : bool, optional
        If True, also return the radial (line-of-sight) component of
        Earth's velocity. The default is False.

    Returns
    -------
    vearth_ra : numpy.ndarray
        Earth's velocity component in the RA direction, in km/s.
    vearth_dec : numpy.ndarray
        Earth's velocity component in the Dec direction, in km/s.
    vearth_radial : numpy.ndarray, optional
        Earth's radial velocity component, in km/s. Only returned if
        `radial` is True.
    """

    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric_posvel, SkyCoord
    from astropy import units as u
    from astropy.constants import au

    coord = SkyCoord('{0} {1}'.format(raj, decj), unit=(u.hourangle, u.deg))
    rarad = coord.ra.value * np.pi/180
    decrad = coord.dec.value * np.pi/180

    vearth_ra = []
    vearth_dec = []
    if radial:
        vearth_radial = []
    for mjd in mjds:
        time = Time(mjd, format='mjd')
        pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)

        vx = vel_xyz.x.value
        vy = vel_xyz.y.value
        vz = vel_xyz.z.value

        vearth_ra.append(- vx * np.sin(rarad) + vy * np.cos(rarad))
        vearth_dec.append(- vx * np.sin(decrad) * np.cos(rarad) -
                          vy * np.sin(decrad) * np.sin(rarad) +
                          vz * np.cos(decrad))
        if radial:
            vearth_radial.append(vx * np.cos(decrad) * np.cos(rarad) +
                                 vy * np.cos(decrad) * np.sin(rarad) +
                                 vz * np.sin(decrad))

    # Convert from AU/d to km/s
    vearth_ra = vearth_ra * au/1e3/86400
    vearth_dec = vearth_dec * au/1e3/86400
    if radial:
        vearth_radial = vearth_radial * au/1e3/86400

    if radial:
        return vearth_ra.value.squeeze(), vearth_dec.value.squeeze(), \
            vearth_radial.value.squeeze()
    else:
        return vearth_ra.value.squeeze(), vearth_dec.value.squeeze()


def read_par(parfile):
    """
    Reads a pulsar parameter (``.par``) file and returns a dictionary of
    parameter names and values. Certain timing-model-only parameters
    (e.g. ``DMMODEL``, ``JUMP``, ``TZRMJD``) are ignored. For each
    numeric parameter, an associated ``<PARAM>_TYPE`` entry is added
    (``'d'`` for int, ``'f'``/``'e'`` for float, ``'s'`` for string),
    and if an uncertainty is present, a ``<PARAM>_ERR`` entry is added.

    Parameters
    ----------
    parfile : str
        Path to the parameter file to read.

    Returns
    -------
    dict
        Dictionary of parameter names and values (plus associated
        ``_ERR`` and ``_TYPE`` entries where applicable).
    """

    par = {}
    ignore = ['DMMODEL', 'DMOFF', "DM_", "CM_", 'CONSTRAIN', 'JUMP', 'NITS',
              'NTOA', 'CORRECT_TROPOSPHERE', 'PLANET_SHAPIRO', 'DILATEFREQ',
              'TIMEEPH', 'MODE', 'TZRMJD', 'TZRSITE', 'TZRFRQ', 'EPHVER',
              'T2CMETHOD']

    file = open(parfile, 'r')
    for line in file.readlines():
        err = None
        p_type = None
        sline = line.split()
        if len(sline) == 0 or line[0] == "#" or line[0:2] == "C " \
           or sline[0] in ignore:
            continue

        param = sline[0]
        if param == "E":
            param = "ECC"

        val = sline[1]
        if len(sline) == 3 and sline[2] not in ['0', '1']:
            err = sline[2].replace('D', 'E')
        elif len(sline) == 4:
            err = sline[3].replace('D', 'E')

        try:
            val = int(val)
            p_type = 'd'
        except ValueError:
            try:
                val = float(Decimal(val.replace('D', 'E')))
                if 'e' in sline[1] or 'E' in sline[1].replace('D', 'E'):
                    p_type = 'e'
                else:
                    p_type = 'f'
            except InvalidOperation:
                p_type = 's'

        par[param] = val
        if err:
            par[param+"_ERR"] = float(err)

        if p_type:
            par[param+"_TYPE"] = p_type

    file.close()

    return par


def mjd_to_year(mjd):
    """
    Converts Modified Julian Date(s) to a Besselian/decimal year.

    Parameters
    ----------
    mjd : float or array_like
        Modified Julian Date(s) to convert.

    Returns
    -------
    float or numpy.ndarray
        Decimal year(s) (``astropy.time.Time.byear``) corresponding to
        `mjd`.
    """
    t = Time(mjd, format='mjd')
    yrs = t.byear  # observation year
    return yrs


def find_nearest(arr, val):
    """
    Returns the index of an array (`arr`) that is nearest to value
    (`val`).

    Parameters
    ----------
    arr : array_like
        Array to search.
    val : float
        Value to find the nearest element to.

    Returns
    -------
    int
        Index of the element of `arr` closest to `val`.
    """
    arr = np.asarray(arr)
    ind = np.argmin(np.abs(arr - val))
    return ind


def longest_run_of_zeros(arr):
    """
    Finds the length of the longest run of consecutive zeros in an
    array.

    Parameters
    ----------
    arr : array_like
        Input 1D array (or iterable) of numbers.

    Returns
    -------
    int
        Length of the longest consecutive run of zero-valued elements
        in `arr`.
    """
    count = 0
    max_count = 0
    for num in arr:
        count = count + 1 if num == 0 else 0
        max_count = max(max_count, count)
    return max_count


def pars_to_params(pars, params=None):
    """
    Converts a dictionary of par file parameters from `read_par` to an
    ``lmfit`` ``Parameters()`` object to use in models. Right ascension
    and declination (``RAJ``/``RA`` and ``DECJ``) are converted from
    sexagesimal strings to radians. By default, parameters are not
    varied (``vary=False``); string-valued parameters are skipped.

    Parameters
    ----------
    pars : dict
        Dictionary of parameters, as returned by `read_par`.
    params : lmfit.Parameters, optional
        Existing ``lmfit`` ``Parameters()`` object to append parameters
        to. If None (default), a new object is created.

    Returns
    -------
    lmfit.Parameters
        ``Parameters()`` object with entries added from `pars`.
    """

    from lmfit import Parameters
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    if params is None:  # start new class, otherwise append to existing
        params = Parameters()

    for key, value in pars.items():
        if key in ['RAJ', 'RA']:  # convert position string to radians
            coord = SkyCoord('{0} {1}'.format(pars['RAJ'], pars['DECJ']),
                             unit=(u.hourangle, u.deg))
            params.add('RAJ', value=coord.ra.value*np.pi/180, vary=False)
            params.add('DECJ', value=coord.dec.value*np.pi/180, vary=False)
        try:
            params.add(key, value=value, vary=False)
        except TypeError:  # Don't add strings
            continue

    return params


def get_true_anomaly(mjds, pars):
    """
    Calculates true anomalies for an array of barycentric MJDs and a
    parameter dictionary. Supports either an ``ELL1``-style binary
    model (using ``TASC``, ``EPS1``, ``EPS2``) or a standard Keplerian
    model (using ``T0``, ``ECC``). For near-circular orbits (``ECC`` <
    1e-4), the eccentric anomaly is approximated by the mean anomaly;
    otherwise Kepler's equation is solved numerically.

    Parameters
    ----------
    mjds : array_like
        Barycentric Modified Julian Dates to calculate the true anomaly
        for.
    pars : dict
        Parameter dictionary containing the orbital eccentricity
        (``ECC``, or ``EPS1``/``EPS2`` for ``ELL1``), binary period
        (``PB``, days), reference epoch (``T0`` or ``TASC``, MJD), and
        optionally the binary period derivative (``PBDOT``).

    Returns
    -------
    numpy.ndarray or float
        True anomaly (or anomalies), in radians, in the range
        [0, 2*pi).
    """

    if 'TASC' in pars.keys():
        T0 = pars['TASC']  # MJD
        ECC = np.sqrt(pars['EPS1']**2 + pars['EPS2']**2)
    else:
        T0 = pars['T0']  # MJD
        ECC = pars['ECC']

    PB = pars['PB']  # days
    PBDOT = 0 if 'PBDOT' not in pars.keys() else pars['PBDOT']
    if np.abs(PBDOT) > 1e-10:
        # correct tempo-format
        PBDOT *= 10**-12

    nb = 2*np.pi/PB

    # mean anomaly
    M = nb*((mjds - T0) - 0.5*(PBDOT/PB) * (mjds - T0)**2)
    M = M.squeeze()

    # eccentric anomaly
    if ECC < 1e-4:
        print('Assuming circular orbit for true anomaly calculation')
        E = M
    else:
        M = np.asarray(M, dtype=np.float64)
        E = []
        for m in M:
            E.append(fsolve(lambda E: E - ECC*np.sin(E) - m, m))
        E = np.asarray(E, dtype=np.longdouble)

    # true anomaly
    U = 2*np.arctan2(np.sqrt(1 + ECC) * np.sin(E/2),
                     np.sqrt(1 - ECC) * np.cos(E/2))  # true anomaly
    if hasattr(U,  "__len__"):
        U[np.argwhere(U < 0)] = U[np.argwhere(U < 0)] + 2*np.pi
        U = U.squeeze()
    elif U < 0:
        U += 2*np.pi

    return U


def get_binphase(mjds, pars):
    """
    Calculates binary phase (true anomaly plus argument of periastron)
    for an array of barycentric MJDs and a parameter dictionary. For
    ``ELL1``-style binary models (parameter dictionary contains
    ``TASC``), the argument of periastron is taken to be zero.

    Parameters
    ----------
    mjds : array_like
        Barycentric Modified Julian Dates to calculate the binary phase
        for.
    pars : dict
        Parameter dictionary as used by `get_true_anomaly`, plus
        (for non-``ELL1`` models) the argument of periastron ``OM``
        (degrees) and optionally its time derivative ``OMDOT``
        (deg/yr).

    Returns
    -------
    numpy.ndarray or float
        Binary phase, in radians.
    """
    U = get_true_anomaly(mjds, pars)

    if 'TASC' in pars.keys():
        OM = 0
    else:
        OM = pars['OM'] * np.pi/180
        if 'OMDOT' in pars.keys():
            OM += pars['OMDOT'] * (np.pi/180) / (365.2425) * \
                (mjds - pars['T0'])

    return U + OM


def acor(arr):
    """

    Parameters
    ----------
    arr : Array
         Array of numbers, e.g. a time series

    Returns
    -------
    Int
        Characteristic (50%) autocorrelation length.

    """
    arr -= np.mean(arr)
    auto_correlation = correlate(arr, arr, mode='full')
    auto_correlation = auto_correlation[auto_correlation.size//2:]
    auto_correlation /= auto_correlation[0]
    indices = np.where(auto_correlation < 0.5)[0]
    if len(indices) > 0:
        return indices[0]
    else:
        return 0


def differential_velocity(params, sun_velocity=220, screen_velocity=220,
                          radius=8):
    """
    Approximates the differential velocity between the scattering screen and
    the Sun assuming zero-inclination circular galactic orbits. Useful for
    determining the intrinsic ISM velocity.

    Parameters
    ----------
    params : dict
        Parameters list containing the pulsar RAJ and DECJ, pulsar distance d,
        screen fractional distance s, and anisotropy angle psi.
    sun_velocity : float, optional
        Orbital speed of the Sun in km/s. The default is 220.
    screen_velocity : float, optional
        Orbital speed of the scattering screen in km/s. The default of 220
        assumes a flat galactic rotation curve.
    radius : float, optional
        Radius of the Sun's orbit about the galactic center in kpc. The default
        is 8.

    Returns
    -------
    v_ra, v_dec : float
        The RA and dec components of the differential velocity in km/s.

    """

    c_icrs = SkyCoord('{0} {1}'.format(params['RAJ'].value,
                                       params['DECJ'].value),
                      unit=(u.radian, u.radian), frame='icrs')
    c_gal = c_icrs.galactic
    long = 2 * np.pi - c_gal.l.radian

    dscr = (1 - params['s'].value) * params['d'].value
    # radial position of screen
    rscr = np.sqrt(dscr**2 + radius**2 - (2 * dscr * radius * np.cos(long)))
    costheta = radius / rscr - (dscr * np.cos(long) / rscr)
    # angle between screen orbital velocity and transverse direction
    phi = long + np.arccos(costheta)

    vtrans_scr = screen_velocity * np.cos(phi)  # screen transverse velocity
    vtrans_sun = sun_velocity * np.cos(long)  # sun velocity in same direction
    diff_vel = vtrans_scr - vtrans_sun

    c_new = SkyCoord(l=c_gal.l.degree+1, b=c_gal.b.degree, unit=(u.deg, u.deg),
                     frame='galactic')
    ra_diff = c_new.icrs.ra.radian - c_icrs.ra.radian
    dec_diff = c_new.icrs.dec.radian - c_icrs.dec.radian
    # angle of velocity on the sky as measured east from the dec axis
    angle = np.pi / 2 - np.arctan(dec_diff / ra_diff)

    return diff_vel * np.sin(angle), diff_vel * np.cos(angle)


def slow_FT(dynspec, freqs, fref=None):
    """
    Slow FT of dynamic spectrum along points of
    t*(f / fref), account for phase scaling of f_D.
    Given a uniform t axis, this reduces to a regular FT

    Parameters
    ----------
    dynspec : ndarray, shape (ntime, nfreq)
        Dynamic spectrum to be Fourier Transformed.
    freqs : array_like
        Frequencies of the channels in `dynspec`.
    fref : float, optional
        Reference frequency used to scale the time axis. If None
        (default), the middle of the band is used.

    Returns
    -------
    numpy.ndarray, shape (ntime, nfreq), complex
        Secondary spectrum: `dynspec` Fourier Transformed along the
        scaled time axis and then along frequency, with both axes
        fftshifted.
    """

    # cast dynspec as float 64
    dynspec = dynspec.astype(np.float64)

    ntime = dynspec.shape[0]
    nfreq = dynspec.shape[1]
    src = np.arange(ntime).astype('float64')

    # declare the empty result array:
    SS = np.empty((ntime, nfreq), dtype=np.complex128)

    # Reference freq. defaults to the middle of the band
    if fref is None:
        midf = len(freqs)//2
        fref = freqs[midf]
    fscale = freqs / fref
    fscale = fscale.astype('float64')

    ft = np.fft.fftfreq(ntime, 1)

    # Scaled array of t * f/fref
    tscale = src[:, np.newaxis]*fscale[np.newaxis, :]
    FTphase = -2j*np.pi*tscale[:, np.newaxis, :] * \
        ft[np.newaxis, :, np.newaxis]
    SS = np.sum(dynspec[:, np.newaxis, :]*np.exp(FTphase), axis=0)
    SS = np.fft.fftshift(SS, axes=0)

    # Still need to FFT y axis, should change to pyfftw for memory and
    #   speed improvement
    SS = np.fft.fft(SS, axis=1)
    SS = np.fft.fftshift(SS, axes=1)

    return SS


def svd_reconstruct(arr, nmodes=1):
    """
    Reconstruct a matrix from the leading `nmodes` modes of its
    singular value decomposition.

    This is the shared SVD core used by `svd_model` (here) and by
    ``ththmod.svd_model``.

    Parameters
    ----------
    arr : array_like
        Matrix to model with the SVD.
    nmodes : int, optional
        Number of leading singular values (modes) to keep; all
        higher-order modes are zeroed. The default is 1.

    Returns
    -------
    model : numpy.ndarray
        Reconstruction of `arr` using only the leading `nmodes`
        singular values.
    """

    u, s, w = np.linalg.svd(arr)
    s[nmodes:] = 0.0
    S = np.zeros([len(u), len(w)], np.complex128)
    S[:len(s), :len(s)] = np.diag(s)

    return np.dot(np.dot(u, S), w)


def svd_model(arr, nmodes=1):
    """
    Take SVD of a dynamic spectrum, divide by the largest N modes

    Parameters
    ----------
    arr : array_like
        Time/freq visibility matrix.
    nmodes : int, optional
        Number of leading singular values (modes) to keep when
        constructing the model; all higher-order modes are zeroed.
        The default is 1.

    Returns
    -------
    arr : numpy.ndarray
        Input array divided by the absolute value of the SVD `model`.
    model : numpy.ndarray
        Reconstruction of `arr` using only the leading `nmodes`
        singular values.
    """

    model = svd_reconstruct(arr, nmodes=nmodes)
    arr = arr / np.abs(model)

    return arr, model


def scint_velocity(params, dnu, tau, freq, dnuerr=None, tauerr=None, a=2.53e4):
    """
    Calculate the scintillation velocity (:math:`V_{ISS}`) from the ACF
    scintillation bandwidth and timescale, using the thin-screen
    scintillation velocity equation. If `params` is provided, the
    thin-screen coefficient is scaled by the pulsar distance ``d`` and
    the screen fractional distance ``s``; otherwise a fixed coefficient
    `a` is used (e.g. for fitting).

    Parameters
    ----------
    params : dict or lmfit.Parameters or None
        Parameters containing the pulsar distance ``d`` (and its
        uncertainty ``derr``/``d.stderr``) and screen fractional
        distance ``s`` (and its uncertainty ``serr``/``s.stderr``). If
        None, `a` is used directly as the thin-screen coefficient.
    dnu : float or array_like
        Scintillation bandwidth, in MHz.
    tau : float or array_like
        Scintillation timescale, in seconds.
    freq : float or array_like
        Observing frequency, in MHz.
    dnuerr : float or array_like, optional
        Uncertainty on `dnu`. If provided along with `tauerr`, the
        uncertainty on the scintillation velocity is also returned.
    tauerr : float or array_like, optional
        Uncertainty on `tau`.
    a : float, optional
        Fixed thin-screen coefficient to use when `params` is None.
        The default is 2.53e4.

    Returns
    -------
    viss : float or numpy.ndarray
        Scintillation velocity, in km/s.
    viss_err : float or numpy.ndarray, optional
        Uncertainty on `viss`. Only returned if both `dnuerr` and
        `tauerr` are provided.
    """

    freq = freq / 1e3   # convert to GHz
    if params is not None:
        try:
            d = params['d']
            d_err = params['derr']
        except KeyError:
            d = params['d'].value
            d_err = params['d'].stderr
        try:
            s = params['s']
            s_err = params['serr']
        except KeyError:
            s = params['s'].value
            s_err = params['s'].stderr

        coeff = a * np.sqrt(2 * d * (1 - s) / s)  # thin screen coefficient
        coeff_err = (dnu / s) * ((1 - s) * d_err**2 / (2 * d) +
                                 (d * s_err**2 / (2 * s**2 * (1 - s))))
    else:
        coeff, coeff_err = a, 0  # thin screen coefficient for fitting

    viss = coeff * np.sqrt(dnu) / (freq * tau)

    if (dnuerr is not None) and (tauerr is not None):
        viss_err = (1 / (freq * tau)) * \
            np.sqrt(coeff**2 * ((dnuerr**2 / (4 * dnu)) +
                                (dnu * tauerr**2 / tau**2)) + coeff_err)
        return viss, viss_err
    else:
        return viss


def interp_nan_2d(array, method='linear'):
    """
    Fill in NaN (and other invalid) values of a 2D array by
    interpolating from the surrounding valid pixels.

    Parameters
    ----------
    array : array_like
        2D input array possibly containing NaN or otherwise invalid
        values.
    method : str, optional
        Interpolation method passed to `scipy.interpolate.griddata`
        (one of ``'linear'``, ``'nearest'``, ``'cubic'``). The default
        is ``'linear'``.

    Returns
    -------
    numpy.ndarray
        2D array with invalid values filled in by interpolation.
    """
    array = np.array(array).squeeze()
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = np.ravel(array[~array.mask])
    array = griddata((x1, y1), newarr, (xx, yy), method=method)
    return array


def centres_to_edges(arr):
    """
    Take an array of pixel-centres, and return an array of pixel-edges.
    Assumes the pixel-centres are evenly spaced.

    Parameters
    ----------
    arr : array_like
        1D array of evenly-spaced pixel-centre values.

    Returns
    -------
    numpy.ndarray
        Array of pixel-edge values, one longer than `arr`.
    """
    darr = np.abs(arr[1] - arr[0])
    arr_edges = arr - darr/2
    return np.append(arr_edges, arr_edges[-1] + darr)


def make_pickle(obj, filepath):
    """
    Pickle `obj` to `filepath`, writing in chunks so that this also
    works for objects whose pickled representation is larger than 2GB.

    Parameters
    ----------
    obj : object
        Python object to pickle (e.g. a Dynspec object).
    filepath : str
        Path of the file to write the pickle to.

    Returns
    -------
    None
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
    return


def get_window(nt, nf, window='hanning', frac=0.1):
    """
    Returns tapering windows to apply along the time and frequency axes
    before computing an FFT. Each window is a standard numpy taper
    (e.g. Hanning) covering a fraction `frac` of the axis length,
    padded with ones in the middle so the returned window matches the
    full axis length.

    Parameters
    ----------
    nt : int
        Length of the time axis.
    nf : int
        Length of the frequency axis.
    window : str, optional
        Window type: one of ``'hanning'``, ``'hamming'``,
        ``'blackman'``, or ``'bartlett'``. The default is
        ``'hanning'``.
    frac : float, optional
        Fraction of each axis length to taper at the edges (via the
        selected window function). The default is 0.1.

    Returns
    -------
    chan_window : numpy.ndarray
        Tapering window of length `nt`, for the time axis.
    subint_window : numpy.ndarray
        Tapering window of length `nf`, for the frequency axis.
    """
    if window.lower() == 'hanning':
        cw = np.hanning(np.floor(frac*nt))
        sw = np.hanning(np.floor(frac*nf))
    elif window.lower() == 'hamming':
        cw = np.hamming(np.floor(frac*nt))
        sw = np.hamming(np.floor(frac*nf))
    elif window.lower() == 'blackman':
        cw = np.blackman(np.floor(frac*nt))
        sw = np.blackman(np.floor(frac*nf))
    elif window.lower() == 'bartlett':
        cw = np.bartlett(np.floor(frac*nt))
        sw = np.bartlett(np.floor(frac*nf))
    else:
        print('Window unknown.. Please add it!')
    chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
                            np.ones([nt-len(cw)]))
    subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),
                              np.ones([nf-len(sw)]))
    return chan_window, subint_window


def calculate_curvature_peak_probability(power_data, noise_level, smooth=True,
                                         curvatures=None, log=False):
    """
    Calculates the (unnormalised) Gaussian probability distribution of
    the arc curvature "power vs curvature" profile, modelling the
    profile as a Gaussian peak at its maximum with standard deviation
    given by `noise_level`.

    Parameters
    ----------
    power_data : array_like
        Power (Doppler profile) as a function of curvature, for one or
        more observations.
    noise_level : float or array_like
        Noise standard deviation of `power_data`, used as the Gaussian
        width. If `smooth` is True, also used as the Gaussian smoothing
        sigma.
    smooth : bool, optional
        If True (default), smooth `power_data` with a Gaussian filter
        of sigma `noise_level` before computing the probability.
    curvatures : array_like, optional
        Curvature values corresponding to `power_data`. Currently
        unused (normalisation by `curvatures` is not yet implemented).
    log : bool, optional
        If True, return the log-probability instead of the probability.
        The default is False.

    Returns
    -------
    numpy.ndarray
        (Log-)probability distribution, same shape as `power_data`.
    """
    if smooth:
        power_data = gaussian_filter1d(power_data, noise_level)
    if np.shape(noise_level) == ():
        max_power = np.max(power_data)
    else:
        max_power = np.max(power_data, axis=1).reshape((len(power_data), 1))
        noise_level = noise_level.reshape((len(noise_level), 1))
    if log:
        prob = np.log(1/(noise_level * np.sqrt(2*np.pi))) + \
            -0.5 * ((power_data - max_power) / noise_level)**2
    else:
        prob = 1/(noise_level * np.sqrt(2*np.pi)) * \
            np.exp(-0.5 * ((power_data - max_power) / noise_level)**2)
    # Note: currently doesn't normalise using "curvatures"
    return prob


def save_curvature_data(dyn, filename=None):
    """
    Saves the "power vs curvature" arc-fitting data and noise level to
    a ``.npz`` file, for later use with
    `calculate_curvature_peak_probability` or
    `curvature_log_likelihood`. The exact arrays saved depend on which
    attributes are present on `dyn` (single or double sideband,
    normalised secondary spectrum averaging method).

    Parameters
    ----------
    dyn : Dynspec
        Dynspec object with arc-curvature fitting results (e.g.
        ``eta_array``, ``norm_sspec_avg`` or ``norm_sspec_avg1``/
        ``norm_sspec_avg2`` or ``normsspec_fdop``/``normsspecavg``, and
        ``noise``).
    filename : str, optional
        Output file path (passed to `numpy.savez`, ``.npz`` is
        appended automatically). If None, defaults to
        ``dyn.name + 'curvature_data'``.

    Returns
    -------
    None
    """
    if filename is None:
        filename = dyn.name + 'curvature_data'

    sup_data = np.array([dyn.name, dyn.mjd])

    if hasattr(dyn, 'normsspecavg'):
        np.savez(filename, sup_data, dyn.normsspec_fdop, dyn.normsspecavg,
                 dyn.noise)
    elif hasattr(dyn, 'norm_sspec_avg1'):
        np.savez(filename, sup_data, dyn.eta_array, dyn.norm_sspec_avg1,
                 dyn.norm_sspec_avg2, dyn.noise)
    else:
        np.savez(filename, sup_data, dyn.eta_array, dyn.norm_sspec_avg,
                 dyn.noise)
    return


def load_pickle(filepath):
    """
    Load a pickled object from `filepath`, reading in chunks so that
    this also works for pickle files larger than 2GB. Counterpart to
    `make_pickle`.

    Parameters
    ----------
    filepath : str
        Path of the pickle file to read.

    Returns
    -------
    object
        The unpickled Python object.
    """
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    return obj


# Potential future functions

def make_dynspec(archive, template=None, phasebin=1):
    """
    Creates a ``psrflux``-format dynamic spectrum from an archive,
    equivalent to running ``psrflux -s [template] -e dynspec
    [archive]``.

    Notes
    -----
    This function is a placeholder for future functionality and is
    not yet implemented.

    Parameters
    ----------
    archive : str or psrchive Archive
        Path to, or already-loaded psrchive archive object, to create
        the dynamic spectrum from.
    template : str, optional
        Path to a template profile, passed to ``psrflux -s``.
    phasebin : int, optional
        Number of pulse phase bins to integrate over when forming the
        dynamic spectrum. The default is 1.

    Returns
    -------
    None
    """
    return


def curvature_log_likelihood(power, nfdop, noise, model_nfdop):
    """
    Calculates the log likelihood of a model prediction for nfdop by taking
    the likelihood function for each observation to be a probability density
    calculated from the doppler profile.

    Parameters
    ----------
    power : array_like
        doppler profile(s)
    nfdop : array_like
        nfdop values for doppler profile(s)
    noise : float or array_like
        noise value for each profile
    model_nfdop : float or array_like
        model preiction for nfdop for each profile

    Returns
    -------
    float
        log likelihood of the input data

    """
    # calculate probability from doppler profile and normalize
    dim = len(np.shape(nfdop))
    eta_prob = calculate_curvature_peak_probability(power, noise, log=True)
    integral = np.sum(np.exp(eta_prob[..., :-1]) * np.diff(nfdop, axis=dim-1),
                      axis=dim-1)
    if dim == 2:
        integral = integral.reshape((len(integral), 1))
    eta_prob_norm = eta_prob - np.log(integral)

    if dim == 2:
        like = np.zeros(len(nfdop))  # initialize likelihood list
        outside = np.argwhere((model_nfdop > np.max(nfdop, axis=1)) |
                              (model_nfdop < np.min(nfdop, axis=1))).flatten()
        inside = np.argwhere((model_nfdop < np.max(nfdop, axis=1)) &
                             (model_nfdop > np.min(nfdop, axis=1))).flatten()
        like[outside] = -200  # for model nfdop outside profile nfdop ranges

        # determine likelihoods at model nfdop
        model_nfdop = model_nfdop[inside].reshape((len(model_nfdop[inside]),
                                                   1))
        inds = np.argmin(np.abs(nfdop[inside] - model_nfdop), axis=1)
        like[inside] = eta_prob_norm[inside, inds]

        return np.sum(like)

    elif dim == 1:
        if np.min(nfdop) < model_nfdop < np.max(nfdop):
            return eta_prob_norm[np.argmin(np.abs(nfdop - model_nfdop))]
        else:
            return -200
    else:
        raise ValueError("Invalid input array dimension. Must be either 1D "
                         "(single observation) or 2D (multiple observations)")
