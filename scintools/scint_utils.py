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
import sys
import csv
from decimal import Decimal, InvalidOperation
from scipy.optimize import fsolve
from scipy.interpolate import griddata
import pickle
from astropy.time import Time


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
        
    if hasattr(dyn, 'dnu_est'):  # Estimated scintillation bandwidth
        header += ",dnu_est"
        write_string += ",{0}".format(dyn.dnu_est)

    if hasattr(dyn, 'ar'):  # Axial ratio
        header += ",ar,arerr"
        write_string += ",{0},{1}".format(dyn.ar, dyn.arerr)

    if hasattr(dyn, 'sigma_x'):  # Phase gradient in x,y
        header += ",sigma_x,sigma_xerr,sigma_y,sigma_yerr"
        write_string += ",{0},{1},{2},{3}".format(dyn.sigma_x,
                                                  dyn.sigma_xerr,
                                                  dyn.sigma_y,
                                                  dyn.sigma_yerr)

    if hasattr(dyn, 'v_x'):  # Velocity in x,y
        header += ",v_x,v_xerr,v_y,v_yerr"
        write_string += ",{0},{1},{2},{3}".format(dyn.v_x,
                                                  dyn.v_xerr,
                                                  dyn.v_y,
                                                  dyn.v_yerr)

    if hasattr(dyn, 'acf_tilt'):  # Tilt in the ACF (MHz/min)
        header += ",acf_tilt,acf_tilt_err"
        write_string += ",{0},{1}".format(dyn.acf_tilt, dyn.acf_tilt_err)

    if hasattr(dyn, 'phasegrad'):  # Phase gradient (shear to the ACF)
        header += ",phasegrad,phasegraderr"
        write_string += ",{0},{1}".format(dyn.phasegrad, dyn.phasegraderr)

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

    if hasattr(dyn, 'norm_delmax'):  # Phase gradient (shear to the ACF)
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


def search_and_replace(filename, search, replace):

    with open(filename, 'r') as file:
        data = file.read()

    data = data.replace(search, replace)

    with open(filename, 'w') as file:
        file.write(data)

    return


def cov_to_corr(cov):
    """
    Calculate correlation matrix from covariance
    """
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    corr = cov / outer_std
    corr[cov == 0] = 0
    return corr


def float_array_from_dict(dictionary, key):
    """
    Convert an array stored in dictionary to a numpy array
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
    unlike np.diff, computes differences between centres of elements in x,
        returns numpy array same size as x
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


def get_ssb_delay(mjds, raj, decj):
    """
    Get Romer delay to Solar System Barycentre (SSB) for correction of site
    arrival times to barycentric.
    """

    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric, SkyCoord
    from astropy import units as u
    from astropy.constants import au, c

    coord = SkyCoord('{0} {1}'.format(raj, decj), unit=(u.hourangle, u.deg))
    psr_xyz = coord.cartesian.xyz.value

    t = []
    for mjd in mjds:
        time = Time(mjd, format='mjd')
        earth_xyz = get_body_barycentric('earth', time)
        e_dot_p = np.dot(earth_xyz.xyz.value, psr_xyz)
        t.append(e_dot_p*au.value/c.value)

    print('WARNING! Understand sign of SSB correction before applying to MJDs')

    return t


def get_earth_velocity(mjds, raj, decj):
    """
    Calculates the component of Earth's velocity transverse to the line of
    sight, in RA and DEC
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

    # Convert from AU/d to km/s
    vearth_ra = vearth_ra * au/1e3/86400
    vearth_dec = vearth_dec * au/1e3/86400

    return vearth_ra.value.squeeze(), vearth_dec.value.squeeze()


def read_par(parfile):
    """
    Reads a par file and return a dictionary of parameter names and values
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
        if len(sline) == 0 or line[0] == "#" or line[0:1] == "C " \
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
    converts mjd to year
    """
    t = Time(mjd, format='mjd')
    yrs = t.byear  # observation year
    return yrs


def pars_to_params(pars, params=None):
    """
    Converts a dictionary of par file parameters from read_par() to an
    lmfit Parameters() class to use in models
    By default, parameters are not varied
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
    Calculates true anomalies for an array of barycentric MJDs and a parameter
    dictionary
    """

    PB = pars['PB']
    T0 = pars['T0']
    ECC = pars['ECC']
    PBDOT = 0 if 'PBDOT' not in pars.keys() else pars['PBDOT']

    nb = 2*np.pi/PB

    # mean anomaly
    M = nb*((mjds - T0) - 0.5*(PBDOT/PB) * (mjds - T0)**2)
    M = M.squeeze()

    # eccentric anomaly
    if ECC < 1e-4:
        print('Assuming circular orbit for true anomaly calculation')
        E = M
    else:
        E = fsolve(lambda E: E - ECC*np.sin(E) - M, M)

    # true anomaly
    U = 2*np.arctan2(np.sqrt(1 + ECC) * np.sin(E/2),
                     np.sqrt(1 - ECC) * np.cos(E/2))  # true anomaly
    if hasattr(U,  "__len__"):
        U[np.argwhere(U < 0)] = U[np.argwhere(U < 0)] + 2*np.pi
        U = U.squeeze()
    elif U < 0:
        U += 2*np.pi

    return U


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


def interp_nan_2d(array):
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
    array = griddata((x1, y1), newarr, (xx, yy), method='linear')
    return array


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
            

def calculate_curvature_peak_probability(power_data, noise_level, 
                                         curvatures=None, log=False):
    """
    Calculates the probability distribution 
    """
    if log:
        prob = np.log(1/(noise_level * np.sqrt(2*np.pi))) + \
            -0.5 * ((power_data - np.max(power_data)) / noise_level)**2
    else:
        prob = 1/(noise_level * np.sqrt(2*np.pi)) * \
            np.exp(-0.5 * ((power_data - np.max(power_data)) / noise_level)**2)
    # Note: currently doesn't normalise using "curvatures"
    return prob


def save_curvature_data(dyn, filename=None):
    """
    Saves the "power vs curvature" and noise level to file
    """
    if filename is None:
        filename = dyn.name + 'curvature_data'
        
    if hasattr(dyn, 'norm_sspec_avg1'):
        np.savez(filename, dyn.eta_array, dyn.norm_sspec_avg1, 
                 dyn.norm_sspec_avg2, dyn.noise)
    else:
        np.savez(filename, dyn.eta_array, dyn.norm_sspec_avg, dyn.noise)
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


def remove_duplicates(dyn_files):
    """
    Filters out dynamic spectra from simultaneous observations
    """
    return dyn_files
