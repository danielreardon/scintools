#!/usr/bin/env python

"""
models.py
----------------------------------
Scintillation models

A library of scintillation models to use with lmfit, emcee, or bilby

    Each model has at least inputs:
        params
        xdata
        ydata
        weights

    And output:
        residuals = (ydata - model) * weights

    Some functions use additional inputs
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scintools.scint_sim import ACF
from lmfit import Minimizer, conf_interval
from scipy.ndimage import gaussian_filter


def fitter(model, params, args, mcmc=False, pos=None, nwalkers=100,
           steps=1000, burn=0.2, progress=True, get_ci=False, workers=1,
           nan_policy='raise', max_nfev=None, thin=10, is_weighted=True):

    # Do fit
    if mcmc:
        func = Minimizer(model, params, fcn_args=args)
        mcmc_results = func.emcee(nwalkers=nwalkers, steps=steps,
                                  burn=int(burn * steps), pos=pos,
                                  is_weighted=is_weighted, progress=progress,
                                  thin=thin, workers=workers)
        results = mcmc_results
    else:
        func = Minimizer(model, params, fcn_args=args, nan_policy=nan_policy,
                         max_nfev=max_nfev)
        results = func.minimize()
    if get_ci:
        if results.errorbars:
            ci = conf_interval(func, results)
        else:
            ci = ''
        return results, ci
    else:
        return results


def powerspectrum_model(params, xdata, ydata):

    parvals = params.valuesdict()

    amp = parvals['amp']
    wn = parvals['wn']
    alpha = parvals['alpha']

    model = wn + amp * xdata**alpha

    return (ydata - model)


def tau_acf_model(params, xdata, ydata, weights):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        alpha = index of exponential function. 2 is Gaussian, 5/3 is Kolmogorov
        wn = white noise spike in ACF cut
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    parvals = params.valuesdict()

    amp = parvals['amp']
    tau = parvals['tau']
    alpha = parvals['alpha']
    wn = parvals['wn']

    model = amp*np.exp(-np.divide(xdata, tau)**(alpha))
    model[0] = model[0] + wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(xdata, max(xdata)))

    return (ydata - model) * weights


def dnu_acf_model(params, xdata, ydata, weights):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
        wn = white noise spike in ACF cut
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    parvals = params.valuesdict()

    amp = parvals['amp']
    dnu = parvals['dnu']
    wn = parvals['wn']

    model = amp*np.exp(-np.divide(xdata, dnu/np.log(2)))
    model[0] = model[0] + wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(xdata, max(xdata)))

    return (ydata - model) * weights


def scint_acf_model(params, xdata, ydata, weights):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    residuals_t = tau_acf_model(params, xdata[0], ydata[0], weights[0])
    residuals_f = dnu_acf_model(params, xdata[1], ydata[1], weights[1])

    return np.concatenate((residuals_t, residuals_f))


def scint_acf_model_2d_approx(params, tdata, fdata, ydata, weights):
    """
    Fit an approximate 2D ACF function
    """

    parvals = params.valuesdict()

    amp = parvals['amp']
    dnu = parvals['dnu']
    tau = parvals['tau']
    alpha = parvals['alpha']
    mu = parvals['phasegrad']*60  # min/MHz to s/MHz
    tobs = parvals['tobs']
    bw = parvals['bw']
    wn = parvals['wn']
    nt = len(tdata)
    nf = len(fdata)

    tdata = np.reshape(tdata, (nt, 1))
    fdata = np.reshape(fdata, (1, nf))

    # model = amp * np.exp(-(abs((tdata / tau) + 2 * phasegrad *
    #                           ((dnu / np.log(2)) / freq)**(1 / 6) *
    #                           (fdata / (dnu / np.log(2))))**(3 * alpha / 2) +
    #                     abs(fdata / (dnu / np.log(2)))**(3 / 2))**(2 / 3))
    model = amp * np.exp(-(abs((tdata - mu*fdata)/tau)**(3 * alpha / 2) +
                         abs(fdata / (dnu / np.log(2)))**(3 / 2))**(2 / 3))

    # multiply by triangle function
    model = np.multiply(model, 1-np.divide(abs(tdata), tobs))
    model = np.multiply(model, 1-np.divide(abs(fdata), bw))
    model = np.fft.fftshift(model)
    model[-1, -1] += wn  # add white noise spike
    model = np.fft.ifftshift(model)
    model = np.transpose(model)

    if weights is None:
        weights = np.ones(np.shape(ydata))

    return (ydata - model) * weights


def scint_acf_model_2d(params, ydata, weights):
    """
    Fit an analytical 2D ACF function
    """

    parvals = params.valuesdict()

    tau = np.abs(parvals['tau'])
    dnu = np.abs(parvals['dnu'])
    alpha = parvals['alpha']
    ar = np.abs(parvals['ar'])
    psi = parvals['psi']
    phasegrad = parvals['phasegrad']
    theta = parvals['theta']
    wn = parvals['wn']
    amp = parvals['amp']

    tobs = parvals['tobs']
    bw = parvals['bw']
    nt = parvals['nt']
    nf = parvals['nf']
    nf_crop, nt_crop = np.shape(ydata)

    dt, df = 2 * tobs / nt, 2 * bw / nf
    taumax = nt_crop * dt / tau
    dnumax = nf_crop * df / dnu

    acf = ACF(taumax=taumax, dnumax=dnumax, nt=nt_crop, nf=nf_crop, ar=ar,
              alpha=alpha, phasegrad=phasegrad, theta=theta,
              amp=amp, wn=wn, psi=psi)
    model = acf.acf

    triangle_t = 1 - np.divide(np.tile(np.abs(np.linspace(-taumax*tau,
                                                          taumax*tau,
                                                          nt_crop)),
                                       (nf_crop, 1)), tobs)
    triangle_f = \
        np.transpose(1 - np.divide(np.tile(np.abs(np.linspace(-dnumax*dnu,
                                                              dnumax*dnu,
                                                              nf_crop)),
                                           (nt_crop, 1)), bw))
    triangle = np.multiply(triangle_t, triangle_f)
    model = np.multiply(model, triangle)  # multiply by triangle function

    if weights is None:
        weights = np.ones(np.shape(ydata))
        # weights = 1/model

    return (ydata - model) * weights


def tau_sspec_model(params, xdata, ydata):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        alpha = index of exponential function. 2 is Gaussian, 5/3 is Kolmogorov
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    tau = params['tau']
    alpha = params['alpha']
    wn = params['wn']

    model = amp * np.exp(-np.divide(xdata, tau)**alpha)
    model[0] += wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1 - np.divide(xdata, max(xdata)))

    model_flipped = model[::-1]
    model = np.concatenate((model, model_flipped))
    model = model[0:2 * len(xdata) - 1]
    # Get Fourier model
    model = np.fft.fft(model)
    model = np.real(model)
    model = model[0:len(xdata)]

    # Use the model for the weights
    return (ydata - model) * model


def dnu_sspec_model(params, xdata, ydata):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    dnu = params['dnu']
    wn = params['wn']

    model = amp * np.exp(-np.divide(xdata, dnu / np.log(2)))
    model[0] += wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1 - np.divide(xdata, max(xdata)))

    model_flipped = model[::-1]
    model = np.concatenate((model, model_flipped))
    model = model[0:2 * len(xdata) - 1]
    # Get Fourier model
    model = np.fft.fft(model)
    model = np.real(model)
    model = model[0:len(xdata)]

    # Use the model for the weights
    return (ydata - model) * model


def scint_sspec_model(params, xdata, ydata):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    parvals = params.valuesdict()

    nt = parvals['nt']

    # Scintillation timescale model
    xdata_t = xdata[:nt]
    ydata_t = ydata[:nt]
    residuals_t = tau_sspec_model(params, xdata_t, ydata_t)

    # Scintillation bandwidth model
    xdata_f = xdata[nt:]
    ydata_f = ydata[nt:]
    residuals_f = dnu_sspec_model(params, xdata_f, ydata_f)

    return np.concatenate((residuals_t, residuals_f))


def arc_power_curve(params, xdata, ydata, weights):
    """
    Returns a template for the power curve in secondary spectrum vs
    sqrt(curvature) or normalised fdop
    """

    if weights is None:
        weights = np.ones(np.shape(ydata))

    model = []
    return (ydata - model) * weights


def fit_parabola(x, y):
    """
    Fit a parabola and return the value and error for the peak
    """

    # increase range to help fitter
    ptp = np.ptp(x)
    x = x*(1000/ptp)

    # Do the fit
    params, pcov = np.polyfit(x, y, 2, cov=True)
    yfit = params[0]*np.power(x, 2) + params[1]*x + params[2]  # y values

    # Get parameter errors
    errors = []
    for i in range(len(params)):  # for each parameter
        errors.append(np.absolute(pcov[i][i])**0.5)

    # Get parabola peak and error
    peak = -params[1]/(2*params[0])  # Parabola max (or min)
    peak_error = np.sqrt((errors[1]**2)*((1/(2*params[0]))**2) +
                         (errors[0]**2)*((params[1]/2)**2))  # Error on peak

    peak = peak*(ptp/1000)
    peak_error = peak_error*(ptp/1000)

    return yfit, peak, peak_error


def fit_log_parabola(x, y):
    """
    Fit a log-parabola and return the value and error for the peak
    """

    # Take the log of x
    logx = np.log(x)
    ptp = np.ptp(logx)
    x = logx*(1000/ptp)  # increase range to help fitter

    # Do the fit
    yfit, peak, peak_error = fit_parabola(x, y)
    frac_error = peak_error/peak

    peak = np.e**(peak*ptp/1000)
    # Average the error asymmetries
    peak_error = frac_error*peak

    return yfit, peak, peak_error


def arc_curvature(params, ydata, weights, true_anomaly,
                  vearth_ra, vearth_dec, mjd=None):
    """
    arc curvature model

        ydata: arc curvature
    """

    # ensure dimensionality of arrays makes sense
    if hasattr(ydata,  "__len__"):
        ydata = ydata.squeeze()
        weights = weights.squeeze()
        true_anomaly = true_anomaly.squeeze()
        vearth_ra = vearth_ra.squeeze()
        vearth_dec = vearth_dec.squeeze()

    kmpkpc = 3.085677581e16

    # Other parameters in lower-case
    d = params['d']  # pulsar distance in kpc
    dkm = d * kmpkpc  # kms
    s = params['s']  # fractional screen distance

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    if 'psi' in params.keys():
        raise KeyError("parameter psi is no longer supported. Please use zeta")
    if 'vism_psi' in params.keys():
        raise KeyError("parameter vism_psi is no longer supported. " +
                       "Please use vism_zeta")

    if 'nmodel' in params.keys():
        nmodel = params['nmodel']
    else:
        if 'zeta' in params.keys():
            nmodel = 1
        else:
            nmodel = 0

    if 'vism_ra' in params.keys():
        vism_ra = params['vism_ra']
        vism_dec = params['vism_dec']
    else:
        vism_ra = 0
        vism_dec = 0

    if nmodel > 0.5:  # anisotropic
        zeta = params['zeta'] * np.pi / 180  # anisotropy angle
        if 'vism_zeta' in params.keys():  # anisotropic case
            vism_zeta = params['vism_zeta']  # vism in direction of anisotropy
            veff2 = (veff_ra*np.sin(zeta) + veff_dec*np.cos(zeta) -
                     vism_zeta)**2
        else:
            veff2 = ((veff_ra - vism_ra) * np.sin(zeta) +
                     (veff_dec - vism_dec) * np.cos(zeta)) ** 2
    else:  # isotropic
        veff2 = (veff_ra - vism_ra)**2 + (veff_dec - vism_dec)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9

    if weights is None:
        weights = np.ones(np.shape(ydata))

    return (ydata - model) * weights


def veff_thin_screen(params, ydata, weights, true_anomaly,
                     vearth_ra, vearth_dec, mjd=None):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

        ydata: arc curvature
    """

    # ensure dimensionality of arrays makes sense
    if hasattr(ydata, "__len__"):
        ydata = ydata.squeeze()
        weights = weights.squeeze()
        true_anomaly = true_anomaly.squeeze()
        vearth_ra = vearth_ra.squeeze()
        vearth_dec = vearth_dec.squeeze()

    s = params['s']  # fractional screen distance
    d = params['d']  # pulsar distance (kpc)
    if 'kappa' in params.keys():
        kappa = params['kappa']
    else:
        kappa = 1

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    if 'nmodel' in params.keys():
        nmodel = params['nmodel']
    else:
        if 'psi' in params.keys():
            nmodel = 1
        else:
            nmodel = 0

    if 'vism_ra' in params.keys():
        vism_ra = params['vism_ra']
        vism_dec = params['vism_dec']
    else:
        vism_ra = 0
        vism_dec = 0

    veff_ra -= vism_ra
    veff_dec -= vism_dec

    if nmodel > 0.5:  # anisotropic
        R = params['R']  # axial ratio parameter
        psi = params['psi'] * np.pi / 180  # anisotropy angle

        cosa = np.cos(2 * psi)
        sina = np.sin(2 * psi)

        # quadratic coefficients
        a = (1 - R * cosa) / np.sqrt(1 - R**2)
        b = (1 + R * cosa) / np.sqrt(1 - R**2)
        c = -2 * R * sina / np.sqrt(1 - R**2)

    else:
        a, b, c = 1, 1, 0

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
                            c*veff_ra*veff_dec))
    model = coeff * veff / s

    return (ydata - model) * weights


"""
Below: Models that do not return residuals for a fitter
"""


def effective_velocity_annual(params, true_anomaly, vearth_ra, vearth_dec,
                              mjd=None):
    """
    Effective velocity with annual and pulsar terms
        Note: Does NOT include IISM velocity, but returns veff in IISM frame
    """
    # Define some constants
    v_c = 299792.458  # km/s
    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    # tempo2 parameters from par file in capitals
    if 'PB' in params.keys():
        A1 = params['A1']  # projected semi-major axis in lt-s
        PB = params['PB']  # orbital period in days
        ECC = params['ECC']  # orbital eccentricity
        OM = params['OM'] * np.pi/180  # longitude of periastron rad
        if 'OMDOT' in params.keys():
            if mjd is None:
                print('Warning, OMDOT present but no mjd for calculation')
                omega = OM
            else:
                omega = OM + \
                    params['OMDOT']*np.pi/180*(mjd-params['T0'])/365.2425
        else:
            omega = OM
        # Note: fifth Keplerian param T0 used in true anomaly calculation
        if 'KIN' in params.keys():
            INC = params['KIN']*np.pi/180  # inclination
        elif 'COSI' in params.keys():
            INC = np.arccos(params['COSI'])
        elif 'SINI' in params.keys():
            INC = np.arcsin(params['SINI'])
        else:
            print('Warning: inclination parameter (KIN, COSI, or SINI) ' +
                  'not found')

        if 'sense' in params.keys():
            sense = params['sense']
            if sense < 0.5:  # KIN < 90
                if INC > np.pi/2:
                    INC = np.pi - INC
            if sense >= 0.5:  # KIN > 90
                if INC < np.pi/2:
                    INC = np.pi - INC

        KOM = params['KOM']*np.pi/180  # longitude ascending node

        # Calculate pulsar velocity aligned with the line of nodes (Vx) and
        #   perpendicular in the plane (Vy)
        vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                         np.sqrt(1 - ECC**2))
        vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))
        vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
                                                                  + omega))
    else:
        vp_x = 0
        vp_y = 0

    if 'PMRA' in params.keys():
        PMRA = params['PMRA']  # proper motion in RA
        PMDEC = params['PMDEC']  # proper motion in DEC
    else:
        PMRA = 0
        PMDEC = 0

    # other parameters in lower-case
    s = params['s']  # fractional screen distance
    d = params['d']  # pulsar distance in kpc
    d = d * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * d / secperyr
    pmdec_v = PMDEC * masrad * d / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC
    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v)
    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v)

    return veff_ra, veff_dec, vp_ra, vp_dec


def arc_weak(fdop, tdel, eta=1, ar=1, psi=0, alpha=11/3, amp=1, smooth=0):
    """
    Parameters
    ----------
    fdop : Array 1D
        The Doppler frequency (x-axis) coordinates of the model secondary
        spectrum.
    tdel : Array 1D
        The wavenumber (y-axis) coordinates of the model secondary spectrum.
    eta : floar, optional
        Arc curvature. The default is 1.
    ar : float, optional
        Anisotropy axial ratio. The default is 1.
    psi : float, optional
        DESCRIPTION. The default is 0.
    alpha : float, optional
        DESCRIPTION. The default is 11/3.
    amp : float, optional
        DESCRIPTION. The default is 1.
    smooth : float, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    sspec : Array 2D
        The model secondary spectrum.

    """

    # Begin model
    a = np.cos(psi * np.pi/180)**2 / ar + ar * np.sin(psi*np.pi/180)**2
    b = ar * np.cos(psi * np.pi/180)**2 + (np.sin(psi * np.pi/180)**2)/ar
    c = 2*np.sin(psi * np.pi/180)*np.cos(psi * np.pi/180)*(1/ar - ar)

    fdx, TDEL = np.meshgrid(fdop, tdel)

    f_arc = np.sqrt(TDEL/eta)

    fdy = np.sqrt(TDEL/eta - fdx**2)

    p = (a*fdx**2 + b*fdy**2 + c*fdx*fdy)**(-11/6) + \
        (a*fdx**2 + b*fdy**2 - c*fdx*fdy)**(-11/6)

    arc_frac = np.real(fdx)/np.real(f_arc)
    arc_frac[np.abs(arc_frac) > 0.995] = 0.995  # restrict the asymptote
    sspec = p / np.sqrt(1 - arc_frac**2)

    # Make minimum 0
    sspec -= np.nanmin(sspec)
    sspec[np.isnan(sspec)] = 0
    # Set amplitude to amp
    sspec *= (TDEL/np.mean(tdel))**alpha
    sspec *= amp / np.nanmax(sspec)

    # smooth the spectrum
    if smooth > 0:
        sspec = gaussian_filter(sspec, smooth)

    return sspec


def arc_weak_1d(fdop, eta=1, ar=1, psi=0, amp=1, smooth=0):
    """
    Parameters
    ----------
    fdop : Array 1D
        The Doppler frequency (x-axis) coordinates of the model secondary
        spectrum.
    eta : floar, optional
        Arc curvature. The default is 1.
    ar : float, optional
        Anisotropy axial ratio. The default is 1.
    psi : float, optional
        DESCRIPTION. The default is 0.
    amp : float, optional
        DESCRIPTION. The default is 1.
    smooth : float, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    norm_sspec : Array 1D
        The model normalised secondary spectrum.

    """

    # Begin model
    a = np.cos(psi * np.pi/180)**2 / ar + ar * np.sin(psi*np.pi/180)**2
    b = ar * np.cos(psi * np.pi/180)**2 + (np.sin(psi * np.pi/180)**2)/ar
    c = 2*np.sin(psi * np.pi/180)*np.cos(psi * np.pi/180)*(1/ar - ar)

    fdx = fdop
    TDEL = eta * fdop**2

    f_arc = np.sqrt(TDEL/eta)

    fdy = np.sqrt(TDEL/eta - fdx**2)

    p = (a*fdx**2 + b*fdy**2 + c*fdx*fdy)**(-11/6) + \
        (a*fdx**2 + b*fdy**2 - c*fdx*fdy)**(-11/6)

    arc_frac = np.real(fdx)/np.real(f_arc)
    arc_frac[np.abs(arc_frac) > 0.995] = 0.995  # restrict the asymptote
    sspec = p / np.sqrt(1 - arc_frac**2)

    # Make minimum 0
    sspec -= np.nanmin(sspec)
    sspec[np.isnan(sspec)] = 0
    # Set amplitude to amp
    sspec *= amp / np.nanmax(sspec)

    # smooth the spectrum
    if smooth > 0:
        sspec = gaussian_filter(sspec, smooth)

    return sspec
