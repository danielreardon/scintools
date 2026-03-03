#!/usr/bin/env python

"""
scint_utils.py
----------------------------------
Useful functions for scintools
"""

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
    Cleans an archive using coast_guard
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
    Compute the 2D autocorrelation of a masked array.

    Uses a slow direct calculation that properly handles masked (missing) data.

    Parameters
    ----------
    arr : numpy.ma.MaskedArray
        2D masked array to autocorrelate, shape (nr, nc).

    Returns
    -------
    autocorr : numpy.ndarray
        Normalised 2D autocorrelation of shape (2*nr, 2*nc). The zero-lag
        peak is located at index (nr, nc) and the array is normalised so
        that its maximum value is 1.

    Notes
    -----
    This is a direct O(n^4) implementation intended for small arrays where
    masked pixels must be excluded from each lag sum individually. For large
    unmasked arrays, FFT-based methods are much faster.
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
    Return a boolean array of elements that are finite and not NaN.

    Parameters
    ----------
    array : array_like
        Input array to test.

    Returns
    -------
    mask : numpy.ndarray of bool
        Boolean array of the same shape as ``array``, where ``True`` indicates
        that the corresponding element is both finite (not inf) and not NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from scintools.scint_utils import is_valid
    >>> is_valid(np.array([1.0, np.nan, np.inf, -np.inf, 2.0]))
    array([ True, False, False, False,  True])
    """
    return np.isfinite(array)*(~np.isnan(array))


def read_dynlist(file_path):
    """
    Read a list of dynamic spectrum filenames from a plain-text file.

    Parameters
    ----------
    file_path : str
        Path to the text file containing one filename per line.

    Returns
    -------
    dynfiles : list of str
        List of filenames read from ``file_path``.
    """
    with open(file_path) as file:
        dynfiles = file.read().splitlines()
    return dynfiles


def write_results(filename, dyn=None):
    """
    Append dynamic spectrum parameters to a CSV results file.

    Writes a header line on the first call (when the file is empty), then
    appends one data row per call. The set of columns grows dynamically
    depending on which optional attributes exist on ``dyn``.

    Parameters
    ----------
    filename : str
        Path to the output CSV file. Created if it does not exist.
    dyn : Dynspec, optional
        Dynamic spectrum object whose attributes are written. Expected to
        have at minimum: ``name``, ``mjd``, ``freq``, ``bw``, ``tobs``,
        ``dt``, ``df``. Optional attributes (``tau``, ``dnu``, ``eta``, …)
        are included when present.
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
    Read a CSV results file written by :func:`write_results`.

    Parameters
    ----------
    filename : str
        Path to the CSV file to read.

    Returns
    -------
    param_dict : dict
        Dictionary mapping each column header to a list of string values.
        Use :func:`float_array_from_dict` to convert individual columns to
        numeric arrays.
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

    with open(filename, 'r') as file:
        data = file.read()

    data = data.replace(search, replace)

    with open(filename, 'w') as file:
        file.write(data)

    return


def cov_to_corr(cov):
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : numpy.ndarray, shape (n, n)
        Symmetric positive-semidefinite covariance matrix.

    Returns
    -------
    corr : numpy.ndarray, shape (n, n)
        Correlation matrix derived from ``cov``. Elements where the
        corresponding covariance is zero are set to zero.
    """
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    corr = cov / outer_std
    corr[cov == 0] = 0
    return corr


def float_array_from_dict(dictionary, key):
    """
    Extract a numeric array from a string-valued results dictionary.

    Replaces ``'None'`` string entries with ``'nan'`` before converting, so
    that missing values become ``numpy.nan`` rather than raising an error.

    Parameters
    ----------
    dictionary : dict
        Dictionary as returned by :func:`read_results`, where values are
        lists of strings.
    key : str
        Column name to extract and convert.

    Returns
    -------
    arr : numpy.ndarray of float
        1-D float array of the requested column, squeezed to remove
        length-1 dimensions.
    """
    ind = np.argwhere(np.array(dictionary[key]) == 'None').ravel()

    if ind.size != 0:
        arr = dictionary[key]
        for i in ind:
            arr[i] = 'nan'
        dictionary[key] = arr

    return np.array(list(map(float, dictionary[key]))).squeeze()


def save_fits(filename, dyn):

    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.flip(np.transpose(np.flip(dyn.dyn, axis=1)),
                                  axis=0))
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename)


def difference(x):
    """
    Compute centre-to-centre differences for an array, returning the same size.

    Unlike :func:`numpy.diff`, this function returns an array of the same
    length as the input by using one-sided differences at the boundaries.

    Parameters
    ----------
    x : array_like
        1-D input array of values.

    Returns
    -------
    dx : numpy.ndarray
        Array of differences the same length as ``x``. Interior elements
        are ``(x[i+1] - x[i-1]) / 2``; boundary elements use the adjacent
        pair only.
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
    Compute the Solar System Barycentre (SSB) Römer delay for each epoch.

    Calculates the light-travel-time delay between a ground-based observatory
    and the SSB along the line of sight to the pulsar.

    Parameters
    ----------
    mjds : array_like
        Modified Julian Dates (MJD) of the observations (barycentric).
    raj : str
        Right ascension in ``HH:MM:SS.S`` format.
    decj : str
        Declination in ``±DD:MM:SS.S`` format.
    message : bool, optional
        If ``True`` (default), print a reminder that the returned delays
        should be *added* to site arrival times.

    Returns
    -------
    delays : numpy.ndarray
        Römer delays in seconds, one per entry in ``mjds``.
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
    Compute the transverse Earth velocity projected onto the sky.

    Returns the component of Earth's barycentric velocity projected onto the
    RA and Dec axes (and optionally the radial component) at each epoch.

    Parameters
    ----------
    mjds : array_like
        Modified Julian Dates (MJD) at which to evaluate the velocity.
    raj : str
        Right ascension in ``HH:MM:SS.S`` format.
    decj : str
        Declination in ``±DD:MM:SS.S`` format.
    radial : bool, optional
        If ``True``, also return the velocity component along the line of
        sight. Default is ``False``.

    Returns
    -------
    vearth_ra : numpy.ndarray
        Earth velocity component in the RA direction (km/s).
    vearth_dec : numpy.ndarray
        Earth velocity component in the Dec direction (km/s).
    vearth_radial : numpy.ndarray
        Earth velocity along the line of sight (km/s). Only returned when
        ``radial=True``.
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
    Read a TEMPO/TEMPO2 ``.par`` file and return a parameter dictionary.

    Parameters
    ----------
    parfile : str
        Path to the pulsar parameter file.

    Returns
    -------
    par : dict
        Dictionary of parameter names to values. Numeric parameters are
        stored as ``int`` or ``float``; string parameters remain as ``str``.
        Fit errors are stored under ``<param>_ERR`` and the numeric type
        under ``<param>_TYPE`` (``'d'`` for int, ``'f'`` or ``'e'`` for
        float, ``'s'`` for string).
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
    Convert a Modified Julian Date (MJD) to a Besselian year.

    Parameters
    ----------
    mjd : float or array_like
        Modified Julian Date(s) to convert.

    Returns
    -------
    year : float or numpy.ndarray
        Besselian year(s) corresponding to ``mjd``.
    """
    t = Time(mjd, format='mjd')
    yrs = t.byear  # observation year
    return yrs


def find_nearest(arr, val):
    """
    Return the index of the element in ``arr`` closest to ``val``.

    Parameters
    ----------
    arr : array_like
        1-D array to search.
    val : float
        Target value.

    Returns
    -------
    ind : int
        Index of the element in ``arr`` with the smallest absolute difference
        from ``val``.
    """
    arr = np.asarray(arr)
    ind = np.argmin(np.abs(arr - val))
    return ind


def longest_run_of_zeros(arr):
    count = 0
    max_count = 0
    for num in arr:
        count = count + 1 if num == 0 else 0
        max_count = max(max_count, count)
    return max_count


def pars_to_params(pars, params=None):
    """
    Convert a par-file dictionary to an :class:`lmfit.Parameters` object.

    Numeric parameters are added with ``vary=False``. String parameters and
    the special ``RAJ``/``RA`` and ``DECJ`` entries are handled separately.

    Parameters
    ----------
    pars : dict
        Parameter dictionary as returned by :func:`read_par`.
    params : lmfit.Parameters, optional
        Existing :class:`lmfit.Parameters` instance to append to. If
        ``None`` (default), a new instance is created.

    Returns
    -------
    params : lmfit.Parameters
        Updated (or newly created) :class:`lmfit.Parameters` object
        containing all numeric parameters from ``pars``.

    Notes
    -----
    By default, all parameters are fixed (``vary=False``). Enable fitting
    by setting individual parameter ``vary`` attributes after calling this
    function.
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
    Calculate orbital true anomalies from barycentric MJDs.

    Solves Kepler's equation to obtain the eccentric anomaly and converts to
    true anomaly. Supports both T0/ECC and TASC/EPS1/EPS2 parameterisations.

    Parameters
    ----------
    mjds : array_like
        Barycentric Modified Julian Dates at which to evaluate the anomaly.
    pars : dict
        Pulsar timing parameter dictionary (from :func:`read_par`) containing
        at minimum ``PB`` and either ``T0``+``ECC`` or ``TASC``+``EPS1``+
        ``EPS2``. Optional keys: ``PBDOT``.

    Returns
    -------
    U : numpy.ndarray
        True anomaly in radians, in the range ``[0, 2π)``.
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
        E = np.asarray(E, dtype=np.float128)

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
    Calculate the orbital binary phase from barycentric MJDs.

    The binary phase is defined as the true anomaly plus the argument of
    periastron, ``U + OM``.

    Parameters
    ----------
    mjds : array_like
        Barycentric Modified Julian Dates.
    pars : dict
        Pulsar timing parameter dictionary (from :func:`read_par`). If
        ``TASC`` is present the orbit is assumed circular (``OM = 0``);
        otherwise ``OM`` (and optionally ``OMDOT``) are used.

    Returns
    -------
    phase : numpy.ndarray
        Binary phase in radians.
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


def slow_FT(dynspec, freqs):
    """
    Slow FT of dynamic spectrum along points of
    t*(f / fref), account for phase scaling of f_D.
    Given a uniform t axis, this reduces to a regular FT

    Reference freq is currently hardcoded to the middle of the band

    Parameters
    ----------

    dynspec: [time, frequency] ndarray
        Dynamic spectrum to be Fourier Transformed
    f: array of floats
        Frequencies of the channels in dynspec
    """

    # cast dynspec as float 64
    dynspec = dynspec.astype(np.float64)

    ntime = dynspec.shape[0]
    nfreq = dynspec.shape[1]
    src = np.arange(ntime).astype('float64')

    # declare the empty result array:
    SS = np.empty((ntime, nfreq), dtype=np.complex128)

    # Reference freq. to middle of band, should change this
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
    SS = np.fft.fftshift(SS, axis=0)

    # Still need to FFT y axis, should change to pyfftw for memory and
    #   speed improvement
    SS = np.fft.fft(SS, axis=1)
    SS = np.fft.fftshift(SS, axes=1)

    return SS


def svd_model(arr, nmodes=1):
    """
    Take SVD of a dynamic spectrum, divide by the largest N modes

    Parameters
    ----------
    arr : array_like
      Time/freq visiblity matrix
    nmodes :

    Returns
    -------
    Original data array multiplied by the largest SVD mode conjugate,
    and the model
    """

    u, s, w = np.linalg.svd(arr)
    s[nmodes:] = 0.0
    S = np.zeros([len(u), len(w)], np.complex128)
    S[:len(s), :len(s)] = np.diag(s)

    model = np.dot(np.dot(u, S), w)
    arr = arr / np.abs(model)

    return arr, model


def scint_velocity(params, dnu, tau, freq, dnuerr=None, tauerr=None, a=2.53e4):
    """
    Calculate scintillation velocity from ACF frequency and time scales
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
    Fill in NaN values of a 2D array using linear interpolation
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
    Take an array of pixel-centres, and return an array of pixel-edges
        assumes the pixel-centres are evenly spaced
    """
    darr = np.abs(arr[1] - arr[0])
    arr_edges = arr - darr/2
    return np.append(arr_edges, arr_edges[-1] + darr)


def make_pickle(obj, filepath):
    """
    pickle.write, which works for very large files
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
    Returns a window to apply before computing an FFT
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
    Calculates the probability distribution
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
    Saves the "power vs curvature" and noise level to file
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
    pickle.load, which works for very large files
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
    Creates a psrflux-format dynamic spectrum from an archive
        $ psrflux -s [template] -e dynspec [archive]
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
