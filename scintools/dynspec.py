#!/usr/bin/env python

"""
dynspec.py
----------------------------------
Dynamic spectrum class
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import time
import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from copy import deepcopy as cp
from scintools.scint_models import scint_acf_model, scint_acf_model_2d_approx,\
                         scint_acf_model_2d, tau_acf_model, dnu_acf_model,\
                         fit_parabola, fit_log_parabola, fitter, \
                         powerspectrum_model
from scintools.scint_utils import is_valid, svd_model, interp_nan_2d
from scipy.interpolate import griddata, interp1d, RectBivariateSpline
from scipy.signal import convolve2d, medfilt, savgol_filter
from scipy.io import loadmat
from lmfit import Parameters
import corner


class Dynspec:

    def __init__(self, filename=None, dyn=None, verbose=True, process=True,
                 lamsteps=False):
        """"
        Initialise a dynamic spectrum object by either reading from file
            or from existing object
        """

        if filename:
            self.load_file(filename, verbose=verbose, process=process,
                           lamsteps=lamsteps)
        elif dyn:
            self.load_dyn_obj(dyn, verbose=verbose, process=process,
                              lamsteps=lamsteps)
        else:
            print("Error: No dynamic spectrum file or object")

    def __add__(self, other):
        """
        Defines dynamic spectra addition, which is concatination in time,
            with the gaps filled
        """

        print("Now adding {} ...".format(other.name))

        if self.freq != other.freq \
                or self.bw != other.bw or self.df != other.df:
            print("WARNING: Code does not yet account for different \
                  frequency properties")

        # Set constant properties
        bw = self.bw
        df = self.df
        freqs = self.freqs
        freq = self.freq
        nchan = self.nchan
        dt = self.dt

        # Calculate properties for the gap
        timegap = round((other.mjd - self.mjd)*86400
                        - self.tobs, 1)  # time between two dynspecs
        extratimes = np.arange(self.dt/2, timegap, dt)
        if timegap < dt:
            extratimes = [0]
            nextra = 0
        else:
            nextra = len(extratimes)
        dyngap = np.zeros([np.shape(self.dyn)[0], nextra])

        # Set changed properties
        name = self.name.split('.')[0] + "+" + other.name.split('.')[0] \
            + ".dynspec"
        header = self.header + other.header
        # times = np.concatenate((self.times, self.times[-1] + extratimes,
        #                       self.times[-1] + extratimes[-1] + other.times))
        nsub = self.nsub + nextra + other.nsub
        tobs = self.tobs + timegap + other.tobs
        # Note: need to check "times" attribute for added dynspec
        times = np.linspace(0, tobs, nsub)
        mjd = np.min([self.mjd, other.mjd])  # mjd for earliest dynspec
        newdyn = np.concatenate((self.dyn, dyngap, other.dyn), axis=1)

        # Get new dynspec object with these properties
        newdyn = BasicDyn(newdyn, name=name, header=header, times=times,
                          freqs=freqs, nchan=nchan, nsub=nsub, bw=bw,
                          df=df, freq=freq, tobs=tobs, dt=dt, mjd=mjd)

        return Dynspec(dyn=newdyn, verbose=False, process=False)

    def load_file(self, filename, verbose=True, process=True, lamsteps=False):
        """
        Load a dynamic spectrum from psrflux-format file
        """

        start = time.time()
        # Import all data from filename
        if verbose:
            print("LOADING {0}...".format(filename))
        head = []
        with open(filename, "r") as file:
            for line in file:
                if line.startswith("#"):
                    headline = str.strip(line[1:])
                    head.append(headline)
                    if str.split(headline)[0] == 'MJD0:':
                        # MJD of start of obs
                        self.mjd = float(str.split(headline)[1])
        self.name = os.path.basename(filename)
        self.header = head
        rawdata = np.loadtxt(filename).transpose()  # read file
        self.times = np.unique(rawdata[2]*60)  # time since obs start (secs)
        self.freqs = rawdata[3]  # Observing frequency in MHz.
        fluxes = rawdata[4]  # fluxes
        fluxerrs = rawdata[5]  # flux errors
        self.nchan = int(np.unique(rawdata[1])[-1])  # number of channels
        self.bw = self.freqs[-1] - self.freqs[0]  # obs bw
        self.df = round(self.bw/self.nchan, 5)  # channel bw
        self.bw = round(self.bw + self.df, 2)  # correct bw
        self.nchan += 1  # correct nchan
        self.nsub = int(np.unique(rawdata[0])[-1]) + 1
        self.tobs = self.times[-1]+self.times[0]  # initial estimate of tobs
        self.dt = self.tobs/self.nsub
        if self.dt > 1:
            self.dt = round(self.dt)
        else:
            self.times = np.linspace(self.times[0], self.times[-1], self.nsub)
        self.tobs = self.dt * self.nsub  # recalculated tobs
        # Now reshape flux arrays into a 2D matrix
        self.freqs = np.unique(self.freqs)
        self.freq = round(np.mean(self.freqs), 2)
        fluxes = fluxes.reshape([self.nsub, self.nchan]).transpose()
        fluxerrs = fluxerrs.reshape([self.nsub, self.nchan]).transpose()
        if self.df < 0:  # flip things
            self.df = -self.df
            self.bw = -self.bw
            # Flip flux matricies since self.freqs is now in ascending order
            fluxes = np.flip(fluxes, 0)
            fluxerrs = np.flip(fluxerrs, 0)
        # Finished reading, now setup dynamic spectrum
        self.dyn = fluxes  # initialise dynamic spectrum
        self.lamsteps = lamsteps
        if process:
            self.default_processing(lamsteps=lamsteps)  # do default processing
        end = time.time()
        if verbose:
            print("...LOADED in {0} seconds\n".format(round(end-start, 2)))
            self.info()

    def load_dyn_obj(self, dyn, verbose=True, process=True, lamsteps=False):
        """
        Load in a dynamic spectrum object of different type.
        """

        start = time.time()
        # Import all data from filename
        if verbose:
            print("LOADING DYNSPEC OBJECT {0}...".format(dyn.name))
        self.name = dyn.name
        self.header = dyn.header
        self.times = dyn.times  # time since obs start (secs)
        self.freqs = dyn.freqs  # Observing frequency in MHz.
        self.nchan = dyn.nchan  # number of channels
        self.nsub = dyn.nsub
        self.bw = dyn.bw  # obs bw
        self.df = dyn.df  # channel bw
        self.freq = dyn.freq
        self.tobs = dyn.tobs
        self.dt = dyn.dt
        self.mjd = dyn.mjd
        self.dyn = dyn.dyn
        self.lamsteps = lamsteps
        if process:
            self.default_processing(lamsteps=lamsteps)  # do default processing
        end = time.time()
        if verbose:
            print("...LOADED in {0} seconds\n".format(round(end-start, 2)))
            self.info()

    def default_processing(self, lamsteps=False):
        """
        Default processing of a Dynspec object
        """

        self.trim_edges()  # remove zeros on band edges
        # self.refill()  # refill and zeroed regions with linear interpolation
        # self.correct_dyn()  # correct by svd
        self.calc_acf()  # calculate the ACF
        if lamsteps:
            self.scale_dyn()
        self.calc_sspec(lamsteps=lamsteps)  # Calculate secondary spectrum

    def plot_dyn(self, lamsteps=False, input_dyn=None, filename=None,
                 input_x=None, input_y=None, trap=False, display=True,
                 figsize=(9, 9), dpi=200):
        """
        Plot the dynamic spectrum
        """
        if input_dyn is None:
            if lamsteps:
                if not hasattr(self, 'lamdyn'):
                    self.scale_dyn()
                dyn = self.lamdyn
            elif trap:
                if not hasattr(self, 'trapdyn'):
                    self.scale_dyn(scale='trapezoid')
                dyn = self.trapdyn
            else:
                dyn = self.dyn
        else:
            dyn = input_dyn
        medval = np.median(dyn[is_valid(dyn)*np.array(np.abs(
                                                      is_valid(dyn)) > 0)])
        minval = np.min(dyn[is_valid(dyn)*np.array(np.abs(
                                                   is_valid(dyn)) > 0)])
        # standard deviation
        std = np.std(dyn[is_valid(dyn)*np.array(np.abs(
                                                is_valid(dyn)) > 0)])
        vmin = minval + std
        vmax = medval + 4*std
        if display or (filename is not None):
            plt.figure(figsize=figsize)
        if input_dyn is None:
            if lamsteps:
                plt.pcolormesh(self.times/60, self.lam, dyn,
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                plt.ylabel('Wavelength (m)')
            else:
                plt.pcolormesh(self.times/60, self.freqs, dyn,
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                plt.ylabel('Frequency (MHz)')
            plt.xlabel('Time (mins)')
            # plt.colorbar()  # arbitrary units
        else:
            plt.pcolormesh(input_x, input_y, dyn, vmin=vmin, vmax=vmax,
                           linewidth=0, rasterized=True)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                        pad_inches=0.1)
            if display:
                plt.show()
            plt.close()
        elif input_dyn is None and display:
            plt.show()

    def plot_acf(self, method='acf1d', alpha=5/3, contour=False, filename=None,
                 input_acf=None, input_t=None, input_f=None, fit=True,
                 mcmc=False, display=True, crop=None, tlim=None, flim=None,
                 figsize=(9, 9), verbose=False, dpi=200):
        """
        Plot the ACF
        """
        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'tau') and input_acf is None and fit:
            self.get_scint_params(method=method, alpha=alpha, mcmc=mcmc,
                                  verbose=verbose)
        if input_acf is None:
            arr = self.acf
            tspan = self.tobs
            fspan = self.bw
        else:
            arr = input_acf
            tspan = max(input_t) - min(input_t)
            fspan = max(input_f) - min(input_f)
        arr = np.fft.ifftshift(arr)
        wn = arr[0][0] - arr[0][1]  # subtract the white noise spike
        arr[0][0] = arr[0][0] - wn  # Set the noise spike to zero for plotting
        arr = np.fft.fftshift(arr)

        if crop == 'auto' or crop == 'manual':
            if crop == 'auto':
                if not fit:
                    print("Can't auto crop without fitting!")
                tlim = 5 * self.tau / 60
                flim = 5 * self.dnu
                if tlim > self.tobs / 60:
                    tlim = self.tobs / 60
                if flim > self.bw:
                    flim = self.bw

            tfactor = tlim * 60 / tspan
            tspan = tlim * 60
            tmin = int((-tfactor/2) * len(arr[0]))
            tmax = int((tfactor/2) * len(arr[0]))

            ffactor = flim / fspan
            fspan = flim
            fmin = int((-ffactor/2) * len(arr))
            fmax = int((ffactor/2) * len(arr))

            arr = arr[fmin:fmax, tmin:tmax]

        elif crop is not None:
            print('\n', "Unrecognized cropping mode. Please specify 'auto',",
                  "'manual' or None.", "'auto' requires selecting a fitting " +
                  "method. \n")

        t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
        f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

        if input_acf is None:  # Also plot scintillation scales axes
            fig, ax1 = plt.subplots(figsize=figsize)
            if contour:
                im = ax1.contourf(t_delays, f_shifts, arr)
            else:
                im = ax1.pcolormesh(t_delays, f_shifts, arr, linewidth=0,
                                    rasterized=True, shading='auto')
            ax1.set_ylabel('Frequency lag (MHz)')
            ax1.set_xlabel('Time lag (mins)')
            miny, maxy = ax1.get_ylim()
            if fit:
                ax2 = ax1.twinx()
                ax2.set_ylim(miny/self.dnu, maxy/self.dnu)
                ax2.set_ylabel(r'Frequency lag / (dnu$_d = {0}$)'.
                               format(round(self.dnu, 2)))
                ax3 = ax1.twiny()
                minx, maxx = ax1.get_xlim()
                ax3.set_xlim(minx/(self.tau/60), maxx/(self.tau/60))
                ax3.set_xlabel(r'Time lag/(tau$_d={0}$)'.format(round(
                                                             self.tau/60, 2)))
            fig.colorbar(im, pad=0.15)
        else:  # just plot acf without scales
            if contour:
                plt.contourf(t_delays, f_shifts, arr)
            else:
                plt.pcolormesh(t_delays, f_shifts, arr, linewidth=0,
                               rasterized=True, shading='auto')
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                        pad_inches=0.1)
            if display:
                plt.show()
            plt.close()
        elif input_acf is None and display:
            plt.show()

    def plot_sspec(self, lamsteps=False, input_sspec=None, filename=None,
                   input_x=None, input_y=None, trap=False, prewhite=False,
                   plotarc=False, maxfdop=np.inf, delmax=None, ref_freq=1400,
                   cutmid=0, startbin=0, display=True, colorbar=True,
                   title=None, figsize=(9, 9), subtract_artefacts=False,
                   overplot_curvature=None, dpi=200):
        """
        Plot the secondary spectrum
        """
        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.lamsspec)
            elif trap:
                if not hasattr(self, 'trapsspec'):
                    self.calc_sspec(trap=trap, prewhite=prewhite)
                sspec = cp(self.trapsspec)
            else:
                if not hasattr(self, 'sspec'):
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.sspec)
            xplot = cp(self.fdop)
        else:
            sspec = input_sspec
            xplot = input_x

        if subtract_artefacts:
            # Subtract off delay response constant in Doppler
            # Estimate using outer 10% of spectrum
            delay_response = np.nanmean(sspec[:, np.argwhere(
                    np.abs(self.fdop) > 0.9*np.max(self.fdop))], axis=1)
            delay_response -= np.median(delay_response)
            sspec = np.subtract(sspec, delay_response)

        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        # std = np.std(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval - 3
        vmax = maxval - 3

        # Get fdop plotting range
        indicies = np.argwhere(np.abs(xplot) < maxfdop)
        xplot = xplot[indicies].squeeze()
        sspec = sspec[:, indicies].squeeze()
        nr, nc = np.shape(sspec)
        sspec[:, int(nc/2-np.floor(cutmid/2)):int(nc/2 +
                                                  np.ceil(cutmid/2))] = np.nan
        sspec[:startbin, :] = np.nan
        # Maximum value delay axis (us @ ref_freq)
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2
        ind = np.argmin(abs(self.tdel-delmax))
        if display or (filename is not None):
            plt.figure(figsize=figsize)
        if input_sspec is None:
            if lamsteps:
                plt.pcolormesh(xplot, self.beta[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
            else:
                plt.pcolormesh(xplot, self.tdel[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                plt.ylabel(r'$f_\nu$ ($\mu$s)')
            if overplot_curvature is not None:
                plt.plot(xplot, overplot_curvature*xplot**2, 'r--')
            plt.xlabel(r'$f_t$ (mHz)')
            bottom, top = plt.ylim()
            if plotarc:
                if lamsteps:
                    eta = self.betaeta
                else:
                    eta = self.eta
                plt.plot(xplot, eta*np.power(xplot, 2),
                         'r--', alpha=0.5)
            plt.ylim(bottom, top)

        else:
            plt.pcolormesh(xplot, input_y, sspec, vmin=vmin, vmax=vmax,
                           linewidth=0, rasterized=True, shading='auto')
        if colorbar:
            plt.colorbar()
        if title:
            plt.title(title)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                        pad_inches=0.1)
            if display:
                plt.show()
            plt.close()
        elif input_sspec is None and display:
            plt.show()

    def plot_scat_im(self, display=True, plot_log=True, colorbar=True,
                     title=None, input_scat_im=None, input_fdop=None,
                     lamsteps=False, trap=False, clean=True, use_angle=False,
                     use_spatial=False, s=None, veff=None, d=None,
                     filename=None, dpi=200):
        """
        Plot the scattered image
        """
        c = 299792458.0  # m/s
        if input_scat_im is None:
            if not hasattr(self, 'scat_im'):
                self.calc_scat_im(lamsteps=lamsteps, trap=trap,
                                  clean=clean)
            scat_im = self.scat_im
            xyaxes = self.scat_im_ax

        else:
            scat_im = input_scat_im
            xyaxes = input_fdop

        if use_angle:
            # plot in on-sky angle
            thetarad = (xyaxes / (1e9 * self.freq)) * \
                       (c * s / (veff * 1000))
            thetaas = (thetarad * 180 / np.pi) * 3600
            xyaxes = thetaas
        elif use_spatial:
            # plot in spatial coordinates
            thetarad = (xyaxes / (1e9 * self.freq)) * \
                       (c * s / (veff * 1000))
            thetaas = (thetarad * 180 / np.pi) * 3600
            xyaxes = thetaas * (1 - s) * d * 1000

        if plot_log:
            scat_im -= np.min(scat_im)
            scat_im += 1e-10
            scat_im = 10 * np.log10(scat_im)
        medval = np.median(scat_im[is_valid(scat_im) *
                                   np.array(np.abs(scat_im) > 0)])
        maxval = np.max(scat_im[is_valid(scat_im) *
                                np.array(np.abs(scat_im) > 0)])
        vmin = medval - 3
        vmax = maxval - 3

        plt.pcolormesh(xyaxes, xyaxes, scat_im, vmin=vmin, vmax=vmax,
                       linewidth=0, rasterized=True, shading='auto')
        plt.title('Scattered image')
        if use_angle:
            plt.xlabel('Angle parallel to velocity (as)')
            plt.ylabel('Angle perpendicular to velocity (as)')
        elif use_spatial:
            plt.xlabel('Distance parallel to velocity (AU)')
            plt.ylabel('Distance perpendicular to velocity (AU)')
        else:
            plt.xlabel('Angle parallel to velocity')
            plt.ylabel('Angle perpendicular to velocity')
        if title:
            plt.title(title)
        if colorbar:
            plt.colorbar()
        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                        pad_inches=0.1)
            if display:
                plt.show()
            plt.close()
        if display:
            plt.show()
        else:
            plt.close()

        return xyaxes

    def plot_all(self, dyn=1, sspec=3, acf=2, norm_sspec=4, colorbar=True,
                 lamsteps=False, filename=None, display=True):
        """
        Plots multiple figures in one
        """

        # Dynamic Spectrum
        plt.subplot(2, 2, dyn)
        self.plot_dyn(lamsteps=lamsteps)
        plt.title("Dynamic Spectrum")

        # Autocovariance Function
        plt.subplot(2, 2, acf)
        self.plot_acf(subplot=True)
        plt.title("Autocovariance")

        # Secondary Spectrum
        plt.subplot(2, 2, sspec)
        self.plot_sspec(lamsteps=lamsteps)
        plt.title("Secondary Spectrum")

        # Normalised Secondary Spectrum
        plt.subplot(2, 2, norm_sspec)
        self.norm_sspec(plot=True, scrunched=False, lamsteps=lamsteps,
                        plot_fit=False)
        plt.title("Normalised fdop secondary spectrum")

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            if display:
                plt.show()
            plt.close()
        elif display:
            plt.show()

    def fit_arc(self, asymm=False, plot=False, delmax=None, numsteps=1e4,
                startbin=3, cutmid=3, lamsteps=False, etamax=None, etamin=None,
                low_power_diff=-1, figsize=(9, 9), high_power_diff=-0.5,
                ref_freq=1400, constraint=[0, np.inf], nsmooth=5, efac=1,
                filename=None, noise_error=True, display=True, figN=None,
                log_parabola=False, logsteps=False, plot_spec=False,
                fit_spectrum=False, subtract_artefacts=False, dpi=200):
        """
        Find the arc curvature with maximum power along it
            constraint: Only search for peaks between constraint[0] and
                constraint[1]
        """

        if not hasattr(self, 'tdel'):
            self.calc_sspec()
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2  # adjust for frequency

        if lamsteps:
            if not hasattr(self, 'lamsspec'):
                self.calc_sspec(lamsteps=lamsteps)
            sspec = np.array(cp(self.lamsspec))
            yaxis = cp(self.beta)
            ind = np.argmin(abs(self.tdel-delmax))
            ymax = self.beta[ind]  # cut beta at equivalent value to delmax
        else:
            if not hasattr(self, 'sspec'):
                self.calc_sspec()
            sspec = np.array(cp(self.sspec))
            yaxis = cp(self.tdel)
            ymax = delmax

        nr, nc = np.shape(sspec)
        # Estimate noise in secondary spectrum
        a = np.array(sspec[int(nr/2):,
                           int(nc/2 + np.ceil(cutmid/2)):].ravel())
        b = np.array(sspec[int(nr/2):, 0:int(nc/2 -
                                             np.floor(cutmid/2))].ravel())
        noise = np.std(np.concatenate((a, b)))

        # Adjust secondary spectrum
        ind = np.argmin(abs(self.tdel-delmax))
        sspec[0:startbin, :] = np.nan  # mask first N delay bins
        # mask middle N Doppler bins
        sspec[:, int(nc/2 - np.floor(cutmid/2)):int(nc/2 +
              np.ceil(cutmid/2))] = np.nan
        sspec = sspec[0:ind, :]  # cut at delmax
        yaxis = yaxis[0:ind]

        # noise of mean out to delmax.
        noise = np.sqrt(np.sum(np.power(noise, 2)))/np.sqrt(len(yaxis)*2)
        self.noise = noise

        if etamax is None:
            etamax = ymax/((self.fdop[1]-self.fdop[0])*cutmid)**2
        if etamin is None:
            etamin = (yaxis[1]-yaxis[0])*startbin/(max(self.fdop))**2

        try:
            len(etamin)
            etamin_array = np.array(etamin).squeeze()
            etamax_array = np.array(etamax).squeeze()
        except TypeError:
            # Force to be arrays for iteration
            etamin_array = np.array([etamin])
            etamax_array = np.array([etamax])

        # At 1mHz for 1400MHz obs, the maximum arc terminates at delmax
        max_sqrt_eta = np.sqrt(np.max(etamax_array))
        min_sqrt_eta = np.sqrt(np.min(etamin_array))
        # Create an array with equal steps in sqrt(curvature)
        sqrt_eta_all = np.linspace(min_sqrt_eta, max_sqrt_eta, int(numsteps))

        for iarc in range(0, len(etamin_array)):
            if len(etamin_array) == 1:
                etamin = etamin
                etamax = etamax
            else:
                etamin = etamin_array.squeeze()[iarc]
                etamax = etamax_array.squeeze()[iarc]

            if not lamsteps:
                c = 299792458.0  # m/s
                beta_to_eta = c*1e6/((ref_freq*10**6)**2)
                etamax = etamax/(self.freq/ref_freq)**2  # correct for freq
                etamax = etamax*beta_to_eta
                etamin = etamin/(self.freq/ref_freq)**2
                etamin = etamin*beta_to_eta
                constraint = constraint/(self.freq/ref_freq)**2
                constraint = constraint*beta_to_eta

            sqrt_eta = sqrt_eta_all[(sqrt_eta_all <= np.sqrt(etamax)) *
                                    (sqrt_eta_all >= np.sqrt(etamin))]
            numsteps_new = len(sqrt_eta)

            # initiate
            etaArray = []

            # Get the normalised secondary spectrum, set for minimum eta as
            #   normalisation. Then calculate peak as
            self.norm_sspec(eta=etamin, delmax=delmax, plot=plot_spec,
                            startbin=startbin, maxnormfac=1, cutmid=cutmid,
                            lamsteps=lamsteps, scrunched=True,
                            logsteps=logsteps, plot_fit=False,
                            numsteps=numsteps_new, fit_spectrum=fit_spectrum,
                            subtract_artefacts=subtract_artefacts)
            norm_sspec = self.normsspecavg.squeeze()
            etafrac_array = self.normsspec_fdop
            ind1 = np.argwhere(etafrac_array >= 0)
            ind2 = np.argwhere(etafrac_array < 0)

            if asymm:
                norm_sspec_avg1 = np.array(norm_sspec[ind1])
                norm_sspec_avg2 = np.flip(norm_sspec[ind2], axis=0)
                nspec = 2
            else:
                # take the mean
                norm_sspec_avg = np.add(norm_sspec[ind1],
                                        np.flip(norm_sspec[ind2], axis=0))/2
                nspec = 1
            etafrac_array_avg_orig = 1/etafrac_array[ind1].squeeze()

            for dummy in range(0, nspec):
                etafrac_array_avg = etafrac_array_avg_orig
                if asymm and dummy == 0:
                    spec = np.array(norm_sspec_avg1)
                elif asymm and dummy == 1:
                    spec = np.array(norm_sspec_avg2)
                else:
                    spec = np.array(norm_sspec_avg)

                spec = spec.squeeze()

                # Make sure is valid
                filt_ind = is_valid(spec)
                spec = np.flip(spec[filt_ind], axis=0)
                etafrac_array_avg = np.flip(etafrac_array_avg[filt_ind],
                                            axis=0)

                # Form eta array and cut at maximum
                etaArray = etamin*etafrac_array_avg**2
                ind = np.argwhere(etaArray < etamax)
                etaArray = etaArray[ind].squeeze()
                spec = spec[ind].squeeze()

                # Smooth data
                norm_sspec_avg_filt = \
                    savgol_filter(spec, nsmooth, 1)

                # search for peaks within constraint range
                indrange = np.argwhere((etaArray > constraint[0]) *
                                       (etaArray < constraint[1]))

                sumpow_inrange = norm_sspec_avg_filt[indrange]
                ind = np.argmin(np.abs(norm_sspec_avg_filt -
                                       np.max(sumpow_inrange)))

                # Now find eta and estimate error by fitting parabola
                #   Data from -3db on low curvature side to -1.5db on high side
                max_power = norm_sspec_avg_filt[ind]
                power = max_power
                ind1 = 1
                while (power > max_power + low_power_diff and
                       ind + ind1 < len(norm_sspec_avg_filt)-1):  # -3db
                    ind1 += 1
                    power = norm_sspec_avg_filt[ind - ind1]
                power = max_power
                ind2 = 1
                while (power > max_power + high_power_diff and
                       ind + ind2 < len(norm_sspec_avg_filt)-1):  # -1db power
                    ind2 += 1
                    power = norm_sspec_avg_filt[ind + ind2]
                # Now select this region of data for fitting
                xdata = etaArray[int(ind-ind1):int(ind+ind2)]
                ydata = spec[int(ind-ind1):int(ind+ind2)]

                # Do the fit
                # yfit, eta, etaerr = fit_parabola(xdata, ydata)
                if log_parabola:
                    yfit, eta, etaerr = fit_log_parabola(xdata, ydata)
                else:
                    yfit, eta, etaerr = fit_parabola(xdata, ydata)

                if np.mean(np.gradient(np.diff(yfit))) > 0:
                    raise ValueError('Fit returned a forward parabola.')
                eta = eta

                if noise_error:
                    # Now get error from the noise in secondary spectra instead
                    etaerr2 = etaerr  # error from parabola fit
                    power = max_power
                    ind1 = 1
                    while (power > (max_power - noise) and (ind - ind1 > 1)):
                        power = norm_sspec_avg_filt[ind - ind1]
                        ind1 += 1
                    power = max_power
                    ind2 = 1
                    while (power > (max_power - noise) and
                           (ind + ind2 < len(norm_sspec_avg_filt) - 1)):
                        ind2 += 1
                        power = norm_sspec_avg_filt[ind + ind2]

                    etaerr = np.abs(etaArray[int(ind-ind1)] -
                                    etaArray[int(ind+ind2)])/2

                self.eta_array = etaArray
                sigma = self.noise * efac
                if asymm:
                    if dummy == 0:
                        self.norm_sspec_avg1 = spec
                        prob = 1/(sigma * np.sqrt(2*np.pi)) * \
                            np.exp(-0.5 * ((spec - np.max(spec)) / sigma)**2)
                        self.prob_eta_peak1 = prob
                    else:
                        self.norm_sspec_avg2 = spec
                        prob = 1/(sigma * np.sqrt(2*np.pi)) * \
                            np.exp(-0.5 * ((spec - np.max(spec)) / sigma)**2)
                        self.prob_eta_peak2 = prob
                else:
                    self.norm_sspec_avg = spec
                    prob = 1/(sigma * np.sqrt(2*np.pi)) * \
                        np.exp(-0.5 * ((spec - np.max(spec)) / sigma)**2)
                    self.prob_eta_peak = prob

                if iarc == 0:  # save primary
                    if asymm and dummy == 0:
                        if lamsteps:
                            self.betaeta_left = eta
                            self.betaetaerr_left = etaerr / np.sqrt(2)
                            self.betaetaerr2_left = etaerr2 / np.sqrt(2)
                        else:
                            self.eta_left = eta
                            self.etaerr_left = etaerr / np.sqrt(2)
                            self.etaerr2_left = etaerr2 / np.sqrt(2)
                    elif dummy == 1:
                        if lamsteps:
                            self.betaeta_right = eta
                            self.betaetaerr_right = etaerr / np.sqrt(2)
                            self.betaetaerr2_right = etaerr2 / np.sqrt(2)
                        else:
                            self.eta_right = eta
                            self.etaerr_right = etaerr / np.sqrt(2)
                            self.etaerr2_right = etaerr2 / np.sqrt(2)
                    else:
                        if lamsteps:
                            self.betaeta = eta
                            self.betaetaerr = etaerr / np.sqrt(2)
                            self.betaetaerr2 = etaerr2 / np.sqrt(2)
                        else:
                            self.eta = eta
                            self.etaerr = etaerr / np.sqrt(2)
                            self.etaerr2 = etaerr2 / np.sqrt(2)

            self.norm_delmax = delmax

            if plot and iarc == 0:
                if figN is None:
                    plt.figure(figsize=figsize)
                else:
                    plt.figure(figN, figsize=figsize)
                plt.plot(self.eta_array[10:], self.norm_sspec_avg[10:])
                plt.plot(etaArray[10:], norm_sspec_avg_filt[10:])
                plt.plot(xdata, yfit, 'k')
                plt.axvspan(xmin=eta-etaerr, xmax=eta+etaerr,
                            facecolor='C2', alpha=0.5)
                plt.xscale('log')
                if lamsteps:
                    plt.xlabel(r'Arc curvature, '
                               r'$\eta$ (${\rm m}^{-1}\,{\rm mHz}^{-2}$)')
                else:
                    plt.xlabel('eta (tdel)')
                plt.ylabel('Mean power (dB)')
            elif plot:
                plt.plot(xdata, yfit,
                         color='k')
                plt.axvspan(xmin=eta-etaerr, xmax=eta+etaerr,
                            facecolor='C{0}'.format(str(int(3+iarc))),
                            alpha=0.3)
            if plot and iarc == len(etamin_array)-1:
                if filename is not None:
                    plt.savefig(filename, dpi=dpi,
                                bbox_inches='tight', pad_inches=0.1)
                    if display:
                        plt.show()
                    plt.close()
                elif display:
                    plt.show()

    def norm_sspec(self, eta=None, delmax=None, plot=False, startbin=1,
                   maxnormfac=5, minnormfac=0, cutmid=3, lamsteps=True,
                   scrunched=True, plot_fit=True, ref_freq=1400,
                   numsteps=None,  filename=None, display=True,
                   unscrunched=True, logsteps=False, powerspec=True,
                   interp_nan=False, fit_spectrum=False, powerspec_cut=False,
                   figsize=(9, 9), subtract_artefacts=False, dpi=200):
        """
        Normalise fdop axis using arc curvature
        """

        # Maximum value delay axis (us @ ref_freq)
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2

        if lamsteps:
            if not hasattr(self, 'lamsspec'):
                self.calc_sspec(lamsteps=lamsteps)
            sspec = cp(self.lamsspec)
            yaxis = cp(self.beta)
            if not hasattr(self, 'betaeta') and eta is None:
                self.fit_arc(lamsteps=lamsteps, delmax=delmax, plot=plot,
                             startbin=startbin)
        else:
            if not hasattr(self, 'sspec'):
                self.calc_sspec()
            sspec = cp(self.sspec)
            yaxis = cp(self.tdel)
            if not hasattr(self, 'eta') and eta is None:
                self.fit_arc(lamsteps=lamsteps, delmax=delmax, plot=plot,
                             startbin=startbin)
        if eta is None:
            if lamsteps:
                eta = self.betaeta
            else:
                eta = self.eta
        else:  # convert to beta
            if not lamsteps:
                c = 299792458.0  # m/s
                beta_to_eta = c*1e6/((ref_freq*10**6)**2)
                eta = eta/(self.freq/ref_freq)**2  # correct for frequency
                eta = eta*beta_to_eta

        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval - 3
        vmax = maxval - 3

        ind = np.argmin(abs(self.tdel-delmax))
        sspec = sspec[startbin:ind, :]  # cut first N delay bins and at delmax
        # sspec[0:startbin] = np.nan
        nr, nc = np.shape(sspec)
        # mask out centre bins
        sspec[:, int(nc/2 - np.floor(cutmid/2)):int(nc/2 +
              np.floor(cutmid/2))] = np.nan
        tdel = yaxis[startbin:ind]

        if subtract_artefacts:
            # Subtract off delay response constant in Doppler
            # Estimate using outer 10% of spectrum
            delay_response = np.nanmean(sspec[:, np.argwhere(
                    np.abs(self.fdop) > 0.9*np.max(self.fdop))], axis=1)
            delay_response -= np.median(delay_response)
            sspec = np.subtract(sspec, delay_response)

        nr, nc = np.shape(sspec)
        # tdel = yaxis[:ind]
        fdop = self.fdop
        maxfdop = maxnormfac*np.sqrt(tdel[-1]/eta)  # Maximum fdop for plot
        if maxfdop > max(fdop):
            maxfdop = max(fdop)
        # Number of fdop bins to use. Oversample by factor of 2
        nfdop = 2*len(fdop[abs(fdop) <=
                           maxfdop]) if numsteps is None else numsteps
        if nfdop % 2 != 0:
            nfdop += 1

        if logsteps:
            # Set fdopnew in equal steps in log space
            fdoplin = np.abs(np.linspace(-maxnormfac, maxnormfac, nfdop))
            fdop_pos = 10**np.linspace(np.log10(np.min(fdoplin)),
                                       np.log10(np.max(fdoplin)), int(nfdop/2))
            fdop_neg = -np.flip(fdop_pos, axis=0)
            fdopnew = np.concatenate((fdop_neg, fdop_pos))
        else:
            fdopnew = np.linspace(-maxnormfac, maxnormfac,
                                  nfdop)  # norm fdop
        if minnormfac > 0:
            unscrunched = False  # Cannot plot 2D function
            inds = np.argwhere(np.abs(fdopnew) > minnormfac)
            fdopnew = fdopnew[inds]
        if logsteps:
            normSspeclin = []
            masklin = []
        normSspec = []
        mask = []
        isspectot = np.zeros(np.shape(fdopnew))
        for ii in range(0, len(tdel)):
            itdel = tdel[ii]
            imaxfdop = maxnormfac*np.sqrt(itdel/eta)
            ifdop = fdop[abs(fdop) <= imaxfdop]/np.sqrt(itdel/eta)
            isspec = sspec[ii, abs(fdop) <= imaxfdop]  # take the iith row
            if logsteps:
                normlinelin = np.interp(fdoplin, ifdop, isspec)
                masklin.append((np.abs(fdoplin) > np.max(np.abs(ifdop))))
                normSspeclin.append(normlinelin)
            normline = np.interp(fdopnew, ifdop, isspec)
            mask.append((np.abs(fdopnew) > np.max(np.abs(ifdop))))
            normSspec.append(normline)
            isspectot = np.add(isspectot, normline)
        normSspec = np.array(normSspec).squeeze()
        if interp_nan:
            # interpolate NaN values
            normSspec = interp_nan_2d(normSspec)
            if logsteps:
                normSspeclin = np.array(normSspeclin).squeeze()
                normSspeclin = interp_nan_2d(normSspeclin)

        if logsteps:
            masklin += np.isnan(normSspeclin)
            normSspeclin = np.ma.array(normSspeclin, mask=mask)
            mask += np.isnan(normSspec)
            normSspec = np.ma.array(normSspec, mask=mask)
            self.mask = mask
            self.powerspectrum = np.ma.mean(np.power(10, normSspeclin/10),
                                            axis=1)
        else:
            mask += np.isnan(normSspec)
            normSspec = np.ma.array(normSspec, mask=mask)
            self.mask = mask
            self.powerspectrum = np.ma.mean(np.power(10, normSspec/10), axis=1)

        xdata = np.sqrt(tdel)
        ydata = np.sqrt(tdel)*self.powerspectrum
        xdata = xdata[~np.isnan(xdata)]
        ydata = ydata[~np.isnan(ydata)]

        # Initial guesses:
        alpha = -11/3
        index = np.argmin(np.abs(xdata - 10))
        amp = ydata[index] * xdata[index]**-alpha
        # Median of last 10%
        wn = np.min(ydata)
        if fit_spectrum:

            params = Parameters()
            params.add('wn', value=wn, vary=True, min=np.min(ydata),
                       max=np.inf)
            params.add('alpha', value=alpha, vary=True, min=-np.inf, max=0)
            params.add('amp', value=amp, vary=True, min=0.0, max=np.inf)

            results = fitter(powerspectrum_model, params, (xdata, ydata))

            params = results.params

            wn = params['wn'].value
            amp = params['amp'].value
            alpha = params['alpha'].value

            self.ps_wn = wn
            self.ps_amp = amp
            self.ps_alpha = alpha

            self.ps_wn_err = params['wn'].stderr
            self.ps_amp_err = params['amp'].stderr
            self.ps_alpha_err = params['alpha'].stderr

        # Now define weights for the normalised spectrum sum
        arc_spectrum = amp*xdata**alpha
        self.weights = 10*np.log10(arc_spectrum)

        # make average
        if powerspec_cut:
            indices = np.argwhere(arc_spectrum > wn)
            isspecavg = np.ma.average(normSspec[indices, :], axis=0,
                                      weights=self.weights[indices].squeeze()
                                      ).squeeze()
        else:
            isspecavg = np.ma.average(normSspec, axis=0,
                                      weights=self.weights.squeeze()).squeeze()

        self.normsspecavg = isspecavg
        self.normsspec = normSspec
        self.normsspec_tdel = tdel
        self.normsspec_fdop = fdopnew

        if plot:
            # Plot delay-scrunched "power profile"
            if scrunched:
                if display or (filename is not None):
                    plt.figure(figsize=figsize)
                plt.plot(fdopnew, isspecavg)
                bottom, top = plt.ylim()
                plt.xlabel("Normalised $f_t$")
                plt.ylabel("Mean power (dB)")
                if plot_fit:
                    plt.plot([1, 1], [bottom*0.9, top*1.1], 'r--', alpha=0.5)
                    plt.plot([-1, -1], [bottom*0.9, top*1.1], 'r--', alpha=0.5)
                plt.ylim(bottom*0.9, top*1.1)
                plt.xlim(-maxnormfac, maxnormfac)
                if filename is not None:
                    filename_name = filename.split('.')[0]
                    if '+' in filename_name:
                        filename_name = filename_name.split('+')[0]
                    filename_extension = filename.split('.')[-1]
                    plt.savefig(filename_name + '_1d.' + filename_extension,
                                bbox_inches='tight', pad_inches=0.1, dpi=dpi)
                    if display:
                        plt.show()
                    plt.close()
                elif display:
                    plt.show()
            # Plot 2D normalised secondary spectrum
            if unscrunched:
                if display or (filename is not None):
                    plt.figure(figsize=figsize)
                np.ma.set_fill_value(normSspec, np.nan)
                plt.pcolormesh(fdopnew, tdel, np.ma.filled(normSspec),
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                if lamsteps:
                    plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
                else:
                    plt.ylabel(r'$f_\nu$ ($\mu$s)')
                bottom, top = plt.ylim()
                plt.xlabel("Normalised $f_t$")
                if plot_fit:
                    plt.plot([1, 1], [bottom, top], 'r--', alpha=0.5)
                    plt.plot([-1, -1], [bottom, top], 'r--', alpha=0.5)
                plt.ylim(bottom, top)
                plt.colorbar()
                if filename is not None:
                    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1,
                                dpi=dpi)
                    if display:
                        plt.show()
                    plt.close()
                elif display:
                    plt.show()
            # plot power spectrum
            if powerspec:
                if display or (filename is not None):
                    plt.figure(figsize=figsize)
                if fit_spectrum:
                    plt.loglog(xdata, ydata, 'mediumblue')
                    # Overlay theory
                    kf = np.argwhere(np.sqrt(tdel) <= 10)
                    amp = np.mean((1/tdel[kf]) * self.powerspectrum[kf] *
                                  (np.sqrt(tdel[kf]))**(11/3 + 1))
                    plt.loglog(xdata, wn*np.ones(np.shape(xdata)), 'darkcyan')
                    plt.loglog(xdata, arc_spectrum, 'crimson')
                    plt.loglog(xdata, wn + arc_spectrum, 'darkorange')
                    plt.loglog(np.sqrt(tdel),
                               tdel*amp*(np.sqrt(tdel))**-((11/3 + 1)))
                    plt.legend(['Power spectrum', 'White noise',
                                'Arc spectrum', 'Full spectrum model'],
                               loc='upper right')
                else:
                    plt.loglog(xdata, ydata)
                    plt.loglog(xdata, arc_spectrum)
                if lamsteps:
                    plt.xlabel(r'$f_\lambda^{1/2}$ (m$^{-1/2}$)')
                else:
                    plt.xlabel(r'$f_\nu^{1/2}$ ($\mu$s$^{1/2}$)')
                plt.ylabel(r'$f_\lambda^{1/2} D(f_\lambda^{1/2})$ ')
                plt.grid(which='both', axis='both')

                if filename is not None:
                    filename_name = filename.split('.')[0]
                    if '+' in filename_name:
                        filename_name = filename_name.split('+')[0]
                    filename_extension = filename.split('.')[-1]
                    plt.savefig(filename_name + '_power.' + filename_extension,
                                bbox_inches='tight', pad_inches=0.1, dpi=dpi)
                    if display:
                        plt.show()
                    plt.close()
                elif display:
                    plt.show()

        return

    def get_acf_tilt(self, plot=False, tmax=None, fmax=None, display=True,
                     filename=None, nscale=0.5, nscaleplot=2, nmin=5, dpi=200):
        """
        Estimates the tilt in the ACF, which is proportional to the phase
            gradient parallel to Veff
        """
        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'dnu'):
            self.get_scint_params()

        if tmax is None:
            tmax = nscale*self.tau/60
        else:
            tmax = tmax
        if fmax is None:
            fmax = nscale*self.dnu
        else:
            fmax = fmax

        acf = cp(self.acf)
        nr, nc = np.shape(acf)
        t_delays = np.linspace(-self.tobs/60, self.tobs/60, np.shape(acf)[1])
        f_shifts = np.linspace(-self.bw, self.bw, np.shape(acf)[0])

        # just the peak
        xdata_inds = np.argwhere(abs(t_delays) <= tmax)
        if len(xdata_inds) < nmin:
            xdata_inds = np.argwhere(abs(t_delays) <= nmin*self.dt)
        xdata = np.array(t_delays[xdata_inds]).squeeze()

        inds = np.argwhere(abs(f_shifts) <= fmax)
        if len(inds) < nmin:
            inds = np.argwhere(abs(f_shifts) <= nmin*self.df)
        peak_array = []
        peakerr_array = []
        y_array = []

        # Fit parabolas to find the peak in each frequency-slice
        for ii in inds:
            f_shift = f_shifts[ii]
            ind = np.argwhere(f_shifts == f_shift)
            ydata = np.array(acf[ind, xdata_inds]).squeeze()
            yfit, peak, peakerr = fit_parabola(xdata, ydata)
            peak_array.append(peak)
            peakerr_array.append(peakerr)
            y_array.append(f_shift)

        # Now do a weighted fit of a straight line to the peaks
        params, pcov = np.polyfit(peak_array, y_array, 1, cov=True,
                                  w=1/np.array(peakerr_array).squeeze())
        yfit = params[0]*peak_array + params[1]  # y values

        # Get parameter errors
        errors = []
        for i in range(len(params)):  # for each parameter
            errors.append(np.absolute(pcov[i][i])**0.5)

        self.acf_tilt = -1/(float(params[0].squeeze()))  # take -ve
        self.acf_tilt_err = float(errors[0].squeeze()) * \
            1/float(params[0].squeeze())**2

        if plot:
            plt.errorbar(peak_array, y_array,
                         xerr=np.array(peakerr_array).squeeze(),
                         marker='.')
            plt.plot(peak_array, yfit)
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')
            plt.title('Peak measurements, and weighted fit')
            if filename is not None:
                filename_name = filename.split('.')[0]
                filename_extension = filename.split('.')[-1]
                plt.savefig(filename_name + '_tilt_fit.' + filename_extension,
                            dpi=dpi, bbox_inches='tight',
                            pad_inches=0.1)
                if display:
                    plt.show()
                plt.close()
            elif display:
                plt.show()

            plt.pcolormesh(t_delays, f_shifts, acf, linewidth=0,
                           rasterized=True, shading='auto')
            plt.plot(peak_array, y_array, 'r', alpha=0.5)
            plt.plot(peak_array, yfit, 'k', alpha=0.5)
            yl = plt.ylim()
            if yl[1] > nscaleplot*self.dnu:
                plt.ylim([-nscaleplot*self.dnu, nscaleplot*self.dnu])
            xl = plt.xlim()
            if xl[1] > nscaleplot*self.tau:
                plt.xlim([-nscaleplot*self.tau, nscaleplot*self.tau])
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')
            plt.title(r'Tilt = {0} $\pm$ {1} (min/MHz)'.format(
                    round(self.acf_tilt, 2), round(self.acf_tilt_err, 2)))
            if filename is not None:
                filename_name = filename.split('.')[0]
                filename_extension = filename.split('.')[-1]
                plt.savefig(filename_name + '_tilt_acf.' + filename_extension,
                            dpi=dpi, bbox_inches='tight',
                            pad_inches=0.1)
                if display:
                    plt.show()
                plt.close()
            elif display:
                plt.show()

        return

    def get_scint_params(self, method="acf1d", plot=False, alpha=5/3,
                         mcmc=False, full_frame=False, nscale=4,
                         nwalkers=100, steps=1000, burn=0.2, nitr=1,
                         lnsigma=True, verbose=False, progress=True,
                         display=True, filename=None, dpi=200,
                         nan_policy='propagate', flux_estimate=False):
        """
        Measure the scintillation timescale
            Method:
                acf1d - takes a 1D cut through the centre of the ACF for
                sspec - measures timescale from the secondary spectrum
                acf2d_approx - uses an analytic approximation to the ACF
                    including a phase gradient (a shear to the ACF)
        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'sspec') and 'sspec' in method:
            self.calc_sspec()

        ydata_f = self.acf[int(self.nchan):, int(self.nsub)]
        xdata_f = self.df * np.linspace(0, len(ydata_f), len(ydata_f))
        ydata_t = self.acf[int(self.nchan), int(self.nsub):]
        xdata_t = self.dt * np.linspace(0, len(ydata_t), len(ydata_t))

        nt = len(xdata_t)  # number of t-lag samples (along half of acf frame)
        nf = len(xdata_f)

        # concatenate x and y arrays
        xdata = np.array(np.concatenate((xdata_t, xdata_f)))
        ydata = np.array(np.concatenate((ydata_t, ydata_f)))

        # Get initial parameter values from 1d fit
        # Estimate amp and white noise level
        wn = min([ydata_f[0]-ydata_f[1], ydata_t[0]-ydata_t[1]])
        amp = max([ydata_f[1], ydata_t[1]])
        # Estimate tau for initial guess. Closest index to 1/e power
        tau = xdata_t[np.argmin(abs(ydata_t - amp/np.e))]
        # Estimate dnu for initial guess. Closest index to 1/2 power
        dnu = xdata_f[np.argmin(abs(ydata_f - amp/2))]

        # Define fit parameters
        params = Parameters()
        params.add('tau', value=tau, vary=True, min=0, max=np.inf)
        params.add('dnu', value=dnu, vary=True, min=0, max=np.inf)
        params.add('amp', value=amp, vary=True, min=0, max=np.inf)
        params.add('wn', value=wn, vary=True, min=0, max=np.inf)
        if verbose:
            print('Initial guesses:',
                  '\ntau:', tau,
                  '\ndnu:', dnu,
                  '\namp:', amp,
                  '\nwn:', wn)
        if 'sim:mb2=' in self.name:
            # No white noise in simulation. Don't fit or conf. int. will break
            params['wn'].value = 0
            params['wn'].vary = False
        params.add('nt', value=nt, vary=False)
        params.add('nf', value=nf, vary=False)
        if alpha is None:
            params.add('alpha', value=5/3, vary=True,
                       min=-np.inf, max=np.inf)
        else:
            params.add('alpha', value=alpha, vary=False)

        if method == 'acf1d' or method == 'acf2d_approx' or method == 'acf2d':
            if method == 'acf2d_approx' or method == 'acf2d':
                if verbose:
                    print("\nDoing 1D fit to initialize fit values")
            elif verbose:
                print("\nPerforming least-squares fit to 1D ACF model")
            chisqr = np.inf
            for itr in range(nitr):
                results = fitter(scint_acf_model, params,
                                 (xdata, ydata, None), nan_policy=nan_policy,
                                 mcmc=mcmc, is_weighted=(not lnsigma),
                                 burn=burn, nwalkers=nwalkers, steps=steps)
                if results.chisqr < chisqr:
                    chisqr = results.chisqr
                    params = results.params
                    res = results

        if method == 'acf2d_approx' or method == 'acf2d':

            dnu = params['dnu'].value
            tau = params['tau'].value
            if verbose:
                print('1D tau estimate:', tau,
                      '\n1D dnu estimate:', dnu)

            ydata = np.copy(self.acf)
            tticks = np.linspace(-self.tobs, self.tobs,
                                 len(ydata[0, :]) + 1)[:-1]
            fticks = np.linspace(-self.bw, self.bw,
                                 len(ydata[:, 0]) + 1)[:-1]

            wn_loc = np.unravel_index(np.argmax(ydata, axis=None),
                                      ydata.shape)

            fleft = wn_loc[0]
            fright = np.shape(ydata)[0] - wn_loc[0] - 1
            fmin = wn_loc[0] - min(fleft, fright)
            fmax = wn_loc[0] + min(fleft, fright) + 1

            tleft = wn_loc[1]
            tright = np.shape(ydata)[1] - wn_loc[1] - 1
            tmin = wn_loc[1] - min(tleft, tright)
            tmax = wn_loc[1] + min(tleft, tright) + 1

            ydata_centered = ydata[fmin:fmax, tmin:tmax]
            tdata_centered = tticks[tmin:tmax]
            fdata_centered = fticks[fmin:fmax]

            if nscale is not None and not full_frame:
                ntau = nscale
                ndnu = nscale

                if (self.tobs / tau) > 2:
                    while ntau > (self.tobs / tau):
                        ntau = ntau - 1
                        if verbose:
                            print('nscale too large for time axis, ' +
                                  'decreasing to', ntau)

                    tframe = int(round(ntau * (tau / self.dt)))
                    tmin = int(np.floor(
                        np.shape(ydata_centered)[1] / 2)) - tframe
                    tmax = int(np.floor(
                        np.shape(ydata_centered)[1] / 2)) + tframe + 1

                else:
                    tmin = 0
                    tmax = len(tticks)

                if (self.bw / dnu) > 2:
                    while ndnu > (self.bw / dnu):
                        ndnu = ndnu - 1
                        if verbose:
                            print('nscale too large for frequency axis, ' +
                                  'decreasing to', ndnu)

                    fframe = int(round(ndnu * (dnu / self.df)))
                    fmin = int(np.floor(
                        np.shape(ydata_centered)[0] / 2)) - fframe
                    fmax = int(np.floor(
                        np.shape(ydata_centered)[0] / 2)) + fframe + 1

                else:
                    fmin = 0
                    fmax = len(fticks)

                ydata_2d = ydata_centered[fmin:fmax, tmin:tmax]
                tdata = tdata_centered[tmin:tmax]
                fdata = fdata_centered[fmin:fmax]
            else:
                ydata_2d = ydata_centered
                tdata = tdata_centered
                fdata = fdata_centered

            plt.pcolormesh(ydata_2d)
            plt.show()

            params.add('tobs', value=self.tobs, vary=False)
            params.add('bw', value=self.bw, vary=False)
            params.add('freq', value=self.freq, vary=False)
            params.add('phasegrad', value=0.1, vary=True,
                       min=-np.Inf, max=np.Inf)
            if hasattr(self, 'acf_tilt'):  # if have a confident measurement
                if self.acf_tilt_err is not None:
                    params['phasegrad'].value = \
                        self.acf_tilt / (self.tau/60) * self.dnu / 2 / \
                        (self.dnu/self.freq)**(1/6)
            if method == 'acf2d':
                if verbose:
                    print("\nDoing approximate 2D fit to initialize fit",
                          "values")
            elif verbose:
                print("\nPerforming least-squares fit to approximate 2D " +
                      "ACF model")
            chisqr = np.inf

            for itr in range(nitr):
                nfit = 5
                # max_nfev = 2000 * (nfit + 1)  # lmfit default
                max_nfev = 10000 * (nfit + 1)
                results = fitter(scint_acf_model_2d_approx, params,
                                 (tdata, fdata, ydata_2d, None),
                                 mcmc=mcmc, max_nfev=max_nfev,
                                 nan_policy='propagate',
                                 is_weighted=(not lnsigma))
                if results.chisqr < chisqr:
                    chisqr = results.chisqr
                    params = results.params
                    res = results

            if method == 'acf2d':

                dnu = params['dnu'].value
                tau = params['tau'].value
                if verbose:
                    print('2D tau estimate:', tau,
                          '\n2D dnu estimate:', dnu)

                chisqr = np.inf
                for itr in range(nitr):
                    params.add('ar', value=1,
                               vary=True, min=-np.inf, max=np.inf)
                    params.add('phasegrad_x', value=params['phasegrad'].value,
                               vary=True, min=-np.inf, max=np.inf)
                    params.add('phasegrad_y', value=0.1,
                               vary=True, min=-np.inf, max=np.inf)
                    params.add('v_x', value=0.1,
                               vary=True, min=-np.inf, max=np.inf)
                    params.add('v_y', value=0.1,
                               vary=True, min=-np.inf, max=np.inf)

                    from lmfit import Parameter

                    params['nf'] = Parameter(name='nf',
                                             value=np.shape(ydata_centered)[0],
                                             vary=False)
                    params['nt'] = Parameter(name='nt',
                                             value=np.shape(ydata_centered)[1],
                                             vary=False)

                    if mcmc:
                        pos_array = []
                        for i in range(nwalkers):
                            pos_i = []
                            pos_i.append(np.random.normal(
                                            loc=params['tau'].value,
                                            scale=params['tau'].value))  # tau
                            pos_i.append(np.random.normal(
                                            loc=params['dnu'].value,
                                            scale=params['dnu'].value))  # dnu
                            pos_i.append(np.random.normal(
                                            loc=params['amp'].value,
                                            scale=params['amp'].value))  # amp
                            pos_i.append(np.random.normal(
                                            loc=params['wn'].value,
                                            scale=params['wn'].value))  # wn
                            if alpha is None:
                                pos_i.append(np.random.normal(
                                             loc=params['alpha'].value,
                                             scale=params['alpha'].value))
                            pos_i.append(np.random.normal(loc=0,
                                                          scale=1))  # phs
                            pos_i.append(
                                1 + np.abs(np.random.normal(loc=0,
                                                            scale=2)))  # ar
                            pos_i.append(np.random.normal(loc=0,
                                                          scale=1))  # phs_x
                            pos_i.append(np.random.normal(loc=0,
                                                          scale=1))  # phs_y
                            pos_i.append(np.random.normal(loc=0,
                                                          scale=1))  # v_x
                            pos_i.append(np.random.normal(loc=0,
                                                          scale=1))  # v_y
                            if lnsigma:
                                pos_i.append(np.random.uniform(low=0,
                                                               high=10))

                            pos_array.append(pos_i)

                        pos = np.array(pos_array)
                        print(np.shape(pos))
                    else:
                        pos = None

                    if verbose:
                        if mcmc:
                            print("\nPerforming mcmc posterior sample for",
                                  "analytical", "2D ACF model")
                        else:
                            print("\nPerforming least-squares fit to",
                                  "analytical 2D ACF model")
                    nfit = 9
                    # max_nfev = 2000 * (nfit + 1)  # lmfit default
                    max_nfev = 10000 * (nfit + 1)
                    results = fitter(scint_acf_model_2d, params,
                                     (ydata_2d, None), mcmc=mcmc,
                                     pos=pos, nwalkers=nwalkers,
                                     steps=steps, burn=burn,
                                     progress=progress,
                                     max_nfev=max_nfev, nan_policy=nan_policy,
                                     is_weighted=(not lnsigma))
                    if results.chisqr < chisqr:
                        chisqr = results.chisqr
                        params = results.params
                        res = results

        elif method == 'sspec':
            '''
            sspec method
            '''
            print("This method doesn't work yet, do something else")
            # fdyn = np.fft.fft2(self.dyn, (2 * nf, 2 * nt))
            # fdynsq = fdyn * np.conjugate(fdyn)

            # secspec = np.real(fdynsq)
            # secspec = np.fft.fftshift(fdynsq)
            # secspec = secspec[nf:2*nf, :]
            # secspec = np.real(secspec)

            # rowsum = np.sum(secspec[:, :nt], axis=0)
            # ydata_t = rowsum / (2*nf)
            # colsum = np.sum(secspec[:nf, :], axis=1)
            # ydata_f = colsum / (2 * nt)

            # # concatenate x and y arrays
            # xdata = np.array(np.concatenate((xdata_t, xdata_f)))
            # ydata = np.concatenate((ydata_t, ydata_f))

            # if verbose:
            #     print("\nPerforming least-squares fit to secondary spectrum")
            # chisqr = np.inf
            # for itr in range(nitr):
            #     results = fitter(scint_sspec_model, params,
            #                      (xdata, ydata), nan_policy=nan_policy,
            #                       mcmc=mcmc, is_weighted=(not lnsigma),
            #                       burn=burn, nwalkers=nwalkers, steps=steps)
            #     if results.chisqr < chisqr:
            #         chisqr = results.chisqr
            #         params = results.params
            #         res = results

        if flux_estimate:
            flux_var_est = \
                np.mean(self.dyn[is_valid(self.dyn) * (self.dyn != 0)])**2
            flux_var = np.var(self.dyn[is_valid(self.dyn) * (self.dyn != 0)])
            # Estimate of scint bandwidth
            self.dnu_est = self.df * flux_var/flux_var_est

        # Done fitting - now define results
        self.tau = res.params['tau'].value
        self.dnu = res.params['dnu'].value
        if self.dnu < self.df and not flux_estimate:
            print("Warning: Scint bandwidth < channel bandwidth.")
            print("Recommend trying the flux variance estimation method with")
            print("\t get_scint_params(flux_estimate=True).")
        # Compute finite scintle error
        N = (1 + 0.2*self.bw/self.dnu) * (1 + 0.2*self.tobs/self.tau)
        fse_tau = self.tau/(2*np.sqrt(N))
        fit_tau = res.params['tau'].stderr
        if fit_tau is None:
            fit_tau = np.inf
        fse_dnu = self.dnu/(2*np.log(2)*np.sqrt(N))
        fit_dnu = res.params['dnu'].stderr
        if fit_dnu is None:
            fit_dnu = np.inf
        if verbose:
            print("\nFinite scintle errors (tau, dnu):\n", fse_tau, fse_dnu)
            print("\nFit errors (tau, dnu):\n", fit_tau, fit_dnu)
        self.tauerr = np.sqrt(fit_tau**2 + fse_tau**2)
        self.dnuerr = np.sqrt(fit_dnu**2 + fse_dnu**2)
        if 'sim:mb2=' not in self.name:
            self.wn = res.params['wn'].value
            self.wnerr = res.params['wn'].stderr
        else:
            self.wn = 0
        if method == 'acf2d_approx':
            self.phasegrad = res.params['phasegrad'].value
            fit_ph = res.params['phasegrad'].stderr
            fse_ph = self.phasegrad * np.sqrt((fse_dnu/self.dnu)**2 +
                                              (fse_tau/self.tau)**2)
            self.phasegraderr = np.sqrt(fit_ph**2 + fse_ph**2)
        elif method == 'acf2d':
            self.ar = res.params['ar'].value
            self.arerr = res.params['ar'].stderr
            self.phasegrad_x = res.params['phasegrad_x'].value
            self.phasegrad_xerr = res.params['phasegrad_x'].stderr
            self.phasegrad_y = res.params['phasegrad_y'].value
            self.phasegrad_yerr = res.params['phasegrad_y'].stderr
            self.v_x = res.params['v_x'].value
            self.v_xerr = res.params['v_x'].stderr
            self.v_y = res.params['v_y'].value
            self.v_yerr = res.params['v_y'].stderr
            # self.psi = res.params['psi'].value
            # self.psierr = res.params['psi'].stderr
        if alpha is None:
            self.talpha = res.params['alpha'].value
            self.talphaerr = res.params['alpha'].stderr
        else:
            self.talpha = alpha
            self.talphaerr = 0

        if verbose:
            print("\n\t ACF FIT PARAMETERS\n")
            print("tau:\t\t\t{val} +/- {err} s".format(val=self.tau,
                  err=self.tauerr))
            print("dnu:\t\t\t{val} +/- {err} MHz".format(val=self.dnu,
                  err=self.dnuerr))
            if alpha is None:
                print("alpha:\t\t\t{val} +/- {err}".format(val=self.talpha,
                      err=self.talphaerr))
            if method == 'acf2d_approx':
                print("phase grad:\t\t{val} +/- {err}".
                      format(val=self.phasegrad, err=self.phasegraderr))
            elif method == 'acf2d':
                print("ar:\t\t{val} +/- {err}".format(val=self.ar,
                      err=self.arerr))
                print("phase grad x:\t\t{val} +/- {err}".format(
                        val=self.phasegrad_x, err=self.phasegrad_xerr))
                print("phase grad y:\t\t{val} +/- {err}".format(
                        val=self.phasegrad_y, err=self.phasegrad_yerr))
                print("v_x:\t\t{val} +/- {err}".format(val=self.v_x,
                      err=self.v_xerr))
                print("v_y:\t\t{val} +/- {err}".format(val=self.v_y,
                      err=self.v_yerr))

        if plot:
            # get models:
            if method == 'acf1d':
                # Get tau model
                t_residuals = tau_acf_model(res.params, xdata_t, ydata_t,
                                            None)
                tmodel = ydata_t - t_residuals
                # Get dnu model
                f_residuals = dnu_acf_model(res.params, xdata_f, ydata_f,
                                            None)
                fmodel = ydata_f - f_residuals

                if display or filename is not None:
                    plt.figure(figsize=(9, 6))
                plt.subplot(2, 1, 1)
                plt.plot(xdata_t, ydata_t/np.max(ydata_t))
                plt.plot(xdata_t, tmodel/np.max(ydata_t))
                # plot 95% white noise level assuming no correlation
                xl = plt.xlim()
                plt.plot([0, xl[1]], [0, 0], 'k--')
                plt.plot([0, xl[1]],
                         [1/np.sqrt(self.nsub), 1/np.sqrt(self.nsub)],
                         ':', color='crimson')
                plt.plot([0, xl[1]],
                         [-1/np.sqrt(self.nsub), -1/np.sqrt(self.nsub)],
                         ':', color='crimson')
                plt.xlabel('Time lag (s)')
                plt.subplot(2, 1, 2)
                plt.plot(xdata_f, ydata_f/np.max(ydata_f))
                plt.plot(xdata_f, fmodel/np.max(ydata_f))
                # plot 95% white noise level assuming no correlation
                xl = plt.xlim()
                plt.plot([xl[0], xl[1]], [0, 0], 'k--')
                plt.plot([xl[0], xl[1]],
                         [1/np.sqrt(self.nchan), 1/np.sqrt(self.nchan)],
                         ':', color='crimson')
                plt.plot([xl[0], xl[1]],
                         [-1/np.sqrt(self.nchan), -1/np.sqrt(self.nchan)],
                         ':', color='crimson')
                plt.xlabel('Frequency lag (MHz)')
                plt.tight_layout()
                if filename is not None:
                    filename_name = filename.split('.')[0]
                    filename_extension = filename.split('.')[-1]
                    plt.savefig(filename_name + '_1Dfit.' + filename_extension,
                                dpi=dpi, bbox_inches='tight',
                                pad_inches=0.1)
                    if display:
                        plt.show()
                    plt.close()
                elif display:
                    plt.show()

            elif method == 'acf2d_approx' or method == 'acf2d':

                weights = np.ones(np.shape(ydata_centered))
                if method == 'acf2d_approx':
                    model = -scint_acf_model_2d_approx(res.params,
                                                       tdata_centered,
                                                       fdata_centered,
                                                       np.zeros(np.shape(
                                                           ydata_centered)),
                                                       np.ones(np.shape(
                                                           weights)))
                else:
                    model = -scint_acf_model_2d(res.params, np.zeros(
                                                np.shape(ydata_centered)),
                                                np.ones(np.shape(weights)))
                residuals = (ydata_centered - model) * weights
                if nscale is not None and not full_frame:
                    model = model[fmin:fmax, tmin:tmax]
                    residuals = residuals[fmin:fmax, tmin:tmax]

                data = [(ydata_2d, 'data'), (model, 'model'),
                        (residuals, 'residuals')]
                for d in data:

                    if d[1] != 'residuals':
                        # subtract the white noise spike from data and model
                        arr = np.fft.ifftshift(d[0])
                        arr[0][0] -= self.wn
                        arr = np.fft.fftshift(arr)
                    else:
                        arr = d[0]

                    plt.pcolormesh(tdata/60, fdata, arr, linewidth=0,
                                   rasterized=True, shading='auto')
                    if d[1] == 'residuals':
                        plt.clim(vmin=-1, vmax=1)  # fractional error
                    plt.title(d[1])
                    plt.xlabel('Time lag (mins)')
                    plt.ylabel('Frequency lag (MHz)')
                    if filename is not None:
                        filename_name = filename.split('.')[0]
                        filename_extension = filename.split('.')[-1]
                        plt.savefig(filename_name + '_2Dfit_{0}.'.format(d[1])
                                    + filename_extension, dpi=dpi,
                                    bbox_inches='tight', pad_inches=0.1)
                        if display:
                            plt.show()
                        plt.close()
                    elif display:
                        plt.show()

            elif method == 'sspec':
                '''
                sspec plotting routine
                '''

            if mcmc and method == "acf2d" and plot:
                corner.corner(res.flatchain,
                              labels=res.var_names,
                              truths=list(res.params.valuesdict().
                                          values()))
                if filename is not None:
                    filename_name = filename.split('.')[0]
                    filename_extension = filename.split('.')[-1]
                    plt.savefig(filename_name + '_corner.'
                                + filename_extension, dpi=dpi,
                                bbox_inches='tight', pad_inches=0.1)
                    if display:
                        plt.show()
                    plt.close()
                elif display:
                    plt.show()

        return res

    def cut_dyn(self, tcuts=0, fcuts=0, plot=False, filename=None, dpi=200,
                lamsteps=False, maxfdop=np.inf, figsize=(8, 13), display=True):
        """
        Cuts the dynamic spectrum into tcuts+1 segments in time and
                fcuts+1 segments in frequency
        """

        if filename is not None:
            plt.ioff()  # turn off interactive plotting
        nchan = len(self.freqs)  # re-define in case of trimming
        nsub = len(self.times)
        fnum = np.floor(nchan/(fcuts + 1))
        tnum = np.floor(nsub/(tcuts + 1))
        cutdyn = np.empty(shape=(fcuts+1, tcuts+1, int(fnum), int(tnum)))
        # find the right fft lengths for rows and columns
        nrfft = int(2**(np.ceil(np.log2(int(fnum)))+1)/2)
        ncfft = int(2**(np.ceil(np.log2(int(tnum)))+1))
        cutsspec = np.empty(shape=(fcuts+1, tcuts+1, nrfft, ncfft))
        cutacf = np.empty(shape=(fcuts+1, tcuts+1, 2*int(fnum), 2*int(tnum)))
        plotnum = 1
        for ii in reversed(range(0, fcuts+1)):  # plot from high to low
            for jj in range(0, tcuts+1):
                cutdyn[int(ii)][int(jj)][:][:] =\
                    self.dyn[int(ii*fnum):int((ii+1)*fnum),
                             int(jj*tnum):int((jj+1)*tnum)]
                input_dyn_x = self.times[int(jj*tnum):int((jj+1)*tnum)]
                input_dyn_y = self.freqs[int(ii*fnum):int((ii+1)*fnum)]
                input_sspec_x, input_sspec_y, cutsspec[int(ii)][int(jj)][:][:]\
                    = self.calc_sspec(input_dyn=cutdyn[int(ii)][int(jj)][:][:],
                                      lamsteps=lamsteps)
                cutacf[int(ii)][int(jj)][:][:] \
                    = self.calc_acf(input_dyn=cutdyn[int(ii)][int(jj)][:][:])
                if plot:
                    # Plot dynamic spectra
                    plt.figure(1, figsize=figsize)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_dyn(input_dyn=cutdyn[int(ii)][int(jj)][:][:],
                                  input_x=input_dyn_x/60, input_y=input_dyn_y)
                    plt.xlabel('t (mins)')
                    plt.ylabel('f (MHz)')

                    # Plot acf
                    plt.figure(2, figsize=figsize)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_acf(input_acf=cutacf[int(ii)][int(jj)][:][:],
                                  input_t=input_dyn_x,
                                  input_f=input_dyn_y)
                    plt.xlabel('t lag (mins)')
                    plt.ylabel('f lag ')

                    # Plot secondary spectra
                    plt.figure(3, figsize=figsize)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_sspec(input_sspec=cutsspec[int(ii)]
                                                        [int(jj)][:][:],
                                    input_x=input_sspec_x,
                                    input_y=input_sspec_y, lamsteps=lamsteps,
                                    maxfdop=maxfdop)
                    plt.xlabel(r'$f_t$ (mHz)')
                    if lamsteps:
                        plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
                    else:
                        plt.ylabel(r'$f_\nu$ ($\mu$s)')
                    plotnum += 1
        if plot:
            plt.figure(1)
            if filename is not None:
                filename_name = filename.split('.')[0]
                filename_extension = filename.split('.')[1]
                plt.savefig(filename_name + '_dynspec.' + filename_extension,
                            figsize=(9, 15), dpi=dpi, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
            plt.figure(2)
            if filename is not None:
                plt.savefig(filename_name + '_acf.' + filename_extension,
                            figsize=(9, 15), dpi=dpi, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
            plt.figure(3)
            if filename is not None:
                plt.savefig(filename_name + '_sspec.' + filename_extension,
                            figsize=(9, 15), dpi=dpi, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
        self.cutdyn = cutdyn
        self.cutsspec = cutsspec

    def trim_edges(self):
        """
        Find and remove the band edges
        """

        rowsum = sum(abs(self.dyn[0][:]))
        # Trim bottom
        while rowsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (0), axis=0)
            self.freqs = np.delete(self.freqs, (0))
            rowsum = sum(abs(self.dyn[0][:]))
        rowsum = sum(abs(self.dyn[-1][:]))
        # Trim top
        while rowsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (-1), axis=0)
            self.freqs = np.delete(self.freqs, (-1))
            rowsum = sum(abs(self.dyn[-1][:]))
        # Trim left
        colsum = sum(abs(self.dyn[:][0]))
        while colsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (0), axis=1)
            self.times = np.delete(self.times, (0))
            colsum = sum(abs(self.dyn[:][0]))
        colsum = sum(abs(self.dyn[:][-1]))
        # Trim right
        while colsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (-1), axis=1)
            self.times = np.delete(self.times, (-1))
            colsum = sum(abs(self.dyn[:][-1]))
        self.nchan = len(self.freqs)
        self.bw = round(max(self.freqs) - min(self.freqs) + self.df, 2)
        self.freq = round(np.mean(self.freqs), 2)
        self.nsub = len(self.times)
        self.tobs = round(max(self.times) - min(self.times) + self.dt, 2)
        self.mjd = self.mjd + self.times[0]/86400

    def refill(self, linear=True, zeros=True):
        """
        Replaces the nan values in array. Also replaces zeros by default
        """

        if zeros:
            self.dyn[self.dyn == 0] = np.nan

        if linear:  # do linear interpolation
            array = cp(self.dyn)
            self.dyn = interp_nan_2d(array)

        # Fill remainder with the mean
        meanval = np.mean(self.dyn[is_valid(self.dyn)])
        self.dyn[np.isnan(self.dyn)] = meanval

    def correct_dyn(self, svd=True, nmodes=1, frequency=False, time=True,
                    lamsteps=False, nsmooth=5):
        """
        Correct for gain variations in time and frequency
        """

        if lamsteps:
            if not self.lamsteps:
                self.scale_dyn()
            dyn = self.lamdyn
        else:
            dyn = self.dyn
        dyn[np.isnan(dyn)] = 0

        if svd:
            dyn, model = svd_model(dyn, nmodes=nmodes)
        else:
            if frequency:
                self.bandpass = np.mean(dyn, axis=1)
                # Make sure there are no zeros
                self.bandpass[self.bandpass == 0] = np.mean(self.bandpass)
                if nsmooth is not None:
                    bandpass = savgol_filter(self.bandpass, nsmooth, 1)
                else:
                    bandpass = self.bandpass
                dyn = np.divide(dyn, np.reshape(bandpass,
                                                [len(bandpass), 1]))

            if time:
                timestructure = np.mean(dyn, axis=0)
                # Make sure there are no zeros
                timestructure[timestructure == 0] = np.mean(timestructure)
                if nsmooth is not None:
                    timestructure = savgol_filter(timestructure, nsmooth, 1)
                dyn = np.divide(dyn, np.reshape(timestructure,
                                                [1, len(timestructure)]))

        if lamsteps:
            self.lamdyn = dyn
        else:
            self.dyn = dyn

    def calc_scat_im(self, input_sspec=None, input_eta=None, input_fdop=None,
                     input_tdel=None, sampling=64, lamsteps=False, trap=False,
                     ref_freq=1400, clean=True, s=None, veff=None, d=None,
                     fit_arc=True, plotarc=False, plot_fit=False, plot=False,
                     plot_log=True, use_angle=False, use_spatial=False):
        """
        Calculate the scattered image.
        Assumes that the scattering is defined by the primary arc,
        i.e. interference between highly scattered waves and unscattered waves
        (B(tx,ty) vs B(0,0)).
        The x axis of the image is aligned with the velocity.
        """

        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = cp(self.lamsspec)
            elif trap:
                if not hasattr(self, 'trapsspec'):
                    self.calc_sspec(trap=trap)
                sspec = cp(self.trapsspec)
            else:
                if not hasattr(self, 'sspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = cp(self.sspec)
            fdop = cp(self.fdop)
            tdel = cp(self.tdel)
        else:
            sspec = input_sspec
            fdop = input_fdop
            tdel = input_tdel

        nf = len(fdop)
        nt = len(tdel)

        sspec = 10**(sspec / 10)

        if input_eta is None and fit_arc:
            if not hasattr(self, 'betaeta') and not hasattr(self, 'eta'):
                self.fit_arc(lamsteps=lamsteps,
                             log_parabola=True, plot=plot_fit)
            if lamsteps:
                c = 299792458.0  # m/s
                beta_to_eta = c * 1e6 / ((ref_freq * 1e6)**2)
                # correct for freq
                eta = self.betaeta / (self.freq / ref_freq)**2
                eta = eta*beta_to_eta
                eta = eta
            else:
                eta = self.eta
        else:
            if input_eta is None:
                eta = tdel[nt-1] / fdop[nf-1]**2
            else:
                eta = input_eta

        if plotarc:
            self.plot_sspec(lamsteps=lamsteps, plotarc=plotarc)

        # crop sspec to desired region
        flim = next(i for i, delay in enumerate(eta * fdop**2) if
                    delay < np.max(tdel))
        if flim == 0:
            tlim = next(i for i, delay in enumerate(tdel) if
                        delay > eta * fdop[0] ** 2)
            sspec = sspec[:tlim, :]
            tdel = fdop[:tlim]
        else:

            sspec = sspec[:, flim-int(0.02*nf):nf-flim+int(0.02*nf)]
            fdop = fdop[flim-int(0.02*nf):nf-flim+int(0.02*nf)]

        if clean:
            try:
                # fill infs and extremely small pixel values
                array = cp(sspec)
                x = np.arange(0, array.shape[1])
                y = np.arange(0, array.shape[0])

                # mask invalid values
                array = np.ma.masked_where((array < 1e-22), array)
                xx, yy = np.meshgrid(x, y)

                # get only the valid values
                x1 = xx[~array.mask]
                y1 = yy[~array.mask]
                newarr = np.ravel(array[~array.mask])

                sspec = griddata((x1, y1), newarr, (xx, yy),
                                 method='linear')

                # fill nans with the mean
                meanval = np.mean(sspec[is_valid(sspec)])
                sspec[np.isnan(sspec)] = meanval
            except Exception as e:
                print(e)
                print('Cleaning failed. Continuing with uncleaned spectrum.')

        max_fd = max(fdop)

        fdop_x = np.linspace(-max_fd, max_fd, 2*sampling)
        fdop_x = np.append(fdop_x, max_fd)
        nx = len(fdop_x)

        fdop_y = np.linspace(0, max_fd, sampling)
        fdop_y = np.append(fdop_y, max_fd)
        ny = len(fdop_y)

        # equally space square
        fdop_x_est, fdop_y_est = np.meshgrid(fdop_x, fdop_y)
        fdop_est = fdop_x_est
        tdel_est = (fdop_x_est**2 + fdop_y_est**2) * eta
        fd, td = np.meshgrid(fdop, tdel)

        # 2D interpolation
        interp = RectBivariateSpline(td[:, 0], fd[0], sspec)
        # interpolate sspec onto grid for theta
        image = interp.ev(tdel_est, fdop_est)

        image = image * fdop_y_est
        scat_im = np.zeros((nx, nx))
        scat_im[ny-1:nx, :] = image
        scat_im[0:ny-1, :] = image[ny-1:0:-1, :]

        xyaxes = fdop_x

        if plot or plot_log:
            self.plot_scat_im(input_scat_im=scat_im, input_fdop=xyaxes,
                              s=s, veff=veff, d=d, use_angle=use_angle,
                              use_spatial=use_spatial, display=True,
                              plot_log=plot_log)

        self.scat_im = scat_im
        self.scat_im_ax = xyaxes

    def calc_sspec(self, prewhite=True, halve=True, plot=False, lamsteps=False,
                   input_dyn=None, input_x=None, input_y=None, trap=False,
                   window='blackman', window_frac=0.1, return_sspec=False):
        """
        Calculate secondary spectrum
        """

        if input_dyn is None:  # use self dynamic spectrum
            if lamsteps:
                if not self.lamsteps:
                    self.scale_dyn()
                dyn = cp(self.lamdyn)
            elif trap:
                if not hasattr(self, 'trap'):
                    self.scale_dyn(scale='trapezoid')
                dyn = cp(self.trapdyn)
            else:
                dyn = cp(self.dyn)
        else:
            dyn = input_dyn  # use imput dynamic spectrum

        nf = np.shape(dyn)[0]
        nt = np.shape(dyn)[1]
        dyn = dyn - np.mean(dyn)  # subtract mean

        if window is not None:
            # Window the dynamic spectrum
            if window == 'hanning':
                cw = np.hanning(np.floor(window_frac*nt))
                sw = np.hanning(np.floor(window_frac*nf))
            elif window == 'hamming':
                cw = np.hamming(np.floor(window_frac*nt))
                sw = np.hamming(np.floor(window_frac*nf))
            elif window == 'blackman':
                cw = np.blackman(np.floor(window_frac*nt))
                sw = np.blackman(np.floor(window_frac*nf))
            elif window == 'bartlett':
                cw = np.bartlett(np.floor(window_frac*nt))
                sw = np.bartlett(np.floor(window_frac*nf))
            else:
                print('Window unknown.. Please add it!')
            chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
                                    np.ones([nt-len(cw)]))
            subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),
                                      np.ones([nf-len(sw)]))
            dyn = np.multiply(chan_window, dyn)
            dyn = np.transpose(np.multiply(subint_window,
                                           np.transpose(dyn)))

        # find the right fft lengths for rows and columns
        nrfft = int(2**(np.ceil(np.log2(nf))+1))
        ncfft = int(2**(np.ceil(np.log2(nt))+1))
        dyn = dyn - np.mean(dyn)  # subtract mean
        if prewhite:
            simpw = convolve2d([[1, -1], [-1, 1]], dyn, mode='valid')
        else:
            simpw = dyn

        simf = np.fft.fft2(simpw, s=[nrfft, ncfft])
        simf = np.real(np.multiply(simf, np.conj(simf)))  # is real
        sec = np.fft.fftshift(simf)  # fftshift
        if halve:
            sec = sec[int(nrfft/2):][:]  # save just positive delays

        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))
        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),
                          [len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),
                          [len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1

        if prewhite:  # Now post-darken
            if halve:
                vec1 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/ncfft, fd)), 2),
                                  [ncfft, 1])
                vec2 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/nrfft, td)), 2),
                                  [1, int(nrfft/2)])
                postdark = np.transpose(vec1*vec2)
                postdark[:, int(ncfft/2)] = 1
                postdark[0, :] = 1
                sec = np.divide(sec, postdark)
            else:
                raise RuntimeError('Cannot apply prewhite to full frame')

        # Make db
        np.seterr(divide='ignore')
        sec = 10*np.log10(sec)

        if input_dyn is None and not return_sspec:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec

    def calc_acf(self, method='direct', input_dyn=None, normalise=True,
                 window_frac=0.1):
        """
        Calculate autocovariance function
        """

        if method == 'direct':  # simply FFT2 and IFFT2
            if input_dyn is None:
                # mean subtracted dynspec
                arr = cp(self.dyn) - np.mean(self.dyn[is_valid(self.dyn)])
                nf = self.nchan
                nt = self.nsub
            else:
                arr = input_dyn
                nf = np.shape(input_dyn)[0]
                nt = np.shape(input_dyn)[1]
            arr = np.fft.fft2(arr, s=[2*nf, 2*nt])  # zero-padded
            arr = np.abs(arr)  # absolute value
            arr **= 2  # Squared manitude
            arr = np.fft.ifft2(arr)
            arr = np.fft.fftshift(arr)
            arr = np.real(arr)  # real component, just in case
            if normalise:
                arr /= np.max(arr)  # normalise
        elif method == 'sspec':  # Calculate through secondary spectrum
            fdop, yaxis, sspec = self.calc_sspec(prewhite=False, halve=False,
                                                 return_sspec=True,
                                                 window_frac=window_frac)
            sspec = np.fft.fftshift(sspec)
            arr = np.fft.fft2(10**(sspec/10))
            arr = np.fft.fftshift(arr)
            arr = np.real(arr)  # real component, just in case
            if normalise:
                arr /= np.max(arr)  # normalise
        else:
            print('Method not understood. Choose "direct" or "sspec"')

        if input_dyn is None:
            self.acf = arr
        else:
            return arr

    def crop_dyn(self, fmin=0, fmax=np.inf, tmin=0, tmax=np.inf):
        """
        Crops dynamic spectrum in frequency to be between fmin and fmax (MHz)
            and in time between tmin and tmax (mins)
        """

        # Crop frequencies
        crop_array = np.array((self.freqs > fmin)*(self.freqs < fmax))
        self.dyn = self.dyn[crop_array, :]
        self.freqs = self.freqs[crop_array]
        self.nchan = len(self.freqs)
        self.bw = round(max(self.freqs) - min(self.freqs) + self.df, 2)
        self.freq = round(np.mean(self.freqs), 2)

        # Crop times
        tmin = tmin*60  # to seconds
        tmax = tmax*60  # to seconds
        if tmax < self.tobs:
            self.tobs = tmax - tmin
        else:
            self.tobs = self.tobs - tmin
        crop_array = np.array((self.times > tmin)*(self.times < tmax))
        self.dyn = self.dyn[:, crop_array]
        self.nsub = len(self.dyn[0, :])
        self.times = np.linspace(self.dt/2, self.tobs - self.dt/2, self.nsub)
        self.mjd = self.mjd + tmin/86400

    def zap(self, method='median', sigma=7, m=3):
        """
        Basic zapping of dynamic spectrum
        """

        if method == 'median':
            d = np.abs(self.dyn - np.median(self.dyn[~np.isnan(self.dyn)]))
            mdev = np.median(d[~np.isnan(d)])
            s = d/mdev
            self.dyn[s > sigma] = np.nan
        elif method == 'medfilt':
            self.dyn = medfilt(self.dyn, kernel_size=m)

    def scale_dyn(self, scale='lambda', factor=1, window_frac=0.1,
                  window='hanning', spacing='auto'):
        """
        Scales the dynamic spectrum along the frequency axis,
            with an alpha relationship
        """

        if scale == 'factor':
            # scale by some factor
            print("This doesn't do anything yet")
        elif scale == 'lambda':
            # function to convert dyn(feq,t) to dyn(lameq,t)
            # fbw = fractional BW = BW / center frequency
            arin = cp(self.dyn)  # input array
            nf, nt = np.shape(arin)
            freqs = cp(self.freqs)
            lams = np.divide(sc.c, freqs*10**6)
            if spacing == 'max':
                dlam = np.max(np.abs(np.diff(lams)))
            elif spacing == 'median':
                dlam = np.median(np.abs(np.diff(lams)))
            elif spacing == 'mean':
                dlam = np.mean(np.abs(np.diff(lams)))
            elif spacing == 'min':
                dlam = np.min(np.abs(np.diff(lams)))
            elif spacing == 'auto':
                dlam = (np.max(lams) - np.min(lams))/len(freqs)
            lam_eq = np.arange(np.min(lams)+1e-10, np.max(lams)-1e-10, dlam)
            self.dlam = dlam
            feq = np.round(np.divide(sc.c, lam_eq)/10**6, 6)
            arout = np.zeros([len(lam_eq), int(nt)])
            for it in range(0, nt):
                f = interp1d(freqs, arin[:, it], kind='cubic')
                # Make sure the range is valid after rounding
                if max(feq) > max(freqs):
                    feq[np.argmax(feq)] = max(freqs)
                if min(feq) < min(freqs):
                    feq[np.argmin(feq)] = min(freqs)
                arout[:, it] = f(feq)
            self.lamdyn = np.flipud(arout)
            self.lam = np.flipud(lam_eq)
            self.nlam = len(self.lam)
        elif scale == 'trapezoid':
            dyn = cp(self.dyn)
            dyn -= np.mean(dyn)
            nf = np.shape(dyn)[0]
            nt = np.shape(dyn)[1]
            if window is not None:
                # Window the dynamic spectrum
                if window == 'hanning':
                    cw = np.hanning(np.floor(window_frac*nt))
                    sw = np.hanning(np.floor(window_frac*nf))
                elif window == 'hamming':
                    cw = np.hamming(np.floor(window_frac*nt))
                    sw = np.hamming(np.floor(window_frac*nf))
                elif window == 'blackman':
                    cw = np.blackman(np.floor(window_frac*nt))
                    sw = np.blackman(np.floor(window_frac*nf))
                elif window == 'bartlett':
                    cw = np.bartlett(np.floor(window_frac*nt))
                    sw = np.bartlett(np.floor(window_frac*nf))
                else:
                    print('Window unknown.. Please add it!')
                chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
                                        np.ones([nt-len(cw)]))
                subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),
                                          np.ones([nf-len(sw)]))
                dyn = np.multiply(chan_window, dyn)
                dyn = np.transpose(np.multiply(subint_window,
                                               np.transpose(dyn)))
            arin = dyn  # input array
            nf, nt = np.shape(arin)
            scalefrac = 1/(max(self.freqs)/min(self.freqs))
            timestep = max(self.times)*(1 - scalefrac)/(nf + 1)  # time step
            trapdyn = np.empty(shape=np.shape(arin))
            for ii in range(0, nf):
                idyn = arin[ii, :]
                maxtime = max(self.times)-(nf-(ii+1))*timestep
                # How many times to resample to, for a given frequency
                inddata = np.argwhere(self.times <= maxtime)
                # How many trailing zeros to add
                indzeros = np.argwhere(self.times > maxtime)
                # Interpolate line
                newline = np.interp(
                          np.linspace(min(self.times), max(self.times),
                                      len(inddata)), self.times, idyn)

                newline = list(newline) + list(np.zeros(np.shape(indzeros)))
                trapdyn[ii, :] = newline
            self.trapdyn = trapdyn

    def info(self):
        """
        print properties of object
        """

        print("\t OBSERVATION PROPERTIES\n")
        print("filename:\t\t\t{0}".format(self.name))
        print("MJD:\t\t\t\t{0}".format(self.mjd))
        print("Centre frequency (MHz):\t\t{0}".format(self.freq))
        print("Bandwidth (MHz):\t\t{0}".format(self.bw))
        print("Channel bandwidth (MHz):\t{0}".format(self.df))
        print("Integration time (s):\t\t{0}".format(self.tobs))
        print("Subintegration time (s):\t{0}".format(self.dt))
        return


class BasicDyn():
    """
    Define a basic dynamic spectrum object from an array of fluxes
        and other variables, which can then be passed to the dynspec
        class to access its functions with:
    BasicDyn_Object = BasicDyn(dyn)
    Dynspec_Object = Dynspec(BasicDyn_Object)
    """

    def __init__(self, dyn, name="BasicDyn", header=["BasicDyn"], times=[],
                 freqs=[], nchan=None, nsub=None, bw=None, df=None,
                 freq=None, tobs=None, dt=None, mjd=None):

        # Set parameters from input
        if times.size == 0 or freqs.size == 0:
            raise ValueError('must input array of times and frequencies')
        self.name = name
        self.header = header
        self.times = times
        self.freqs = freqs
        self.nchan = nchan if nchan is not None else len(freqs)
        self.nsub = nsub if nsub is not None else len(times)
        self.bw = bw if bw is not None else abs(max(freqs)) - abs(min(freqs))
        self.df = df if df is not None else freqs[1] - freqs[2]
        self.freq = freq if freq is not None else np.mean(np.unique(freqs))
        self.tobs = tobs
        self.dt = dt
        self.mjd = mjd
        self.dyn = dyn
        return


class MatlabDyn():
    """
    Imports simulated dynamic spectra from Matlab code by Coles et al. (2010)
    """

    def __init__(self, matfilename):

        self.matfile = loadmat(matfilename)  # reads matfile to a dictionary
        try:
            self.dyn = self.matfile['spi']
        except NameError:
            raise NameError('No variable named "spi" found in mat file')

        try:
            dlam = float(self.matfile['dlam'])
        except NameError:
            raise NameError('No variable named "dlam" found in mat file')
        # Set parameters from input
        self.name = matfilename.split()[0]
        self.header = [self.matfile['__header__'], ["Dynspec loaded \
                       from Matfile {}".format(matfilename)]]
        self.dt = 2.7*60
        self.freq = 1400
        self.nsub = int(np.shape(self.dyn)[0])
        self.nchan = int(np.shape(self.dyn)[1])
        lams = np.linspace(1, 1+dlam, self.nchan)
        freqs = np.divide(1, lams)
        self.freqs = self.freq*np.linspace(np.min(freqs), np.max(freqs),
                                           self.nchan)
        self.bw = max(self.freqs) - min(self.freqs)
        self.times = self.dt*np.arange(0, self.nsub)
        self.df = self.bw/self.nchan
        self.tobs = float(self.times[-1] - self.times[0])
        self.mjd = 50000.0  # dummy.. Not needed
        self.dyn = np.transpose(self.dyn)

        return


class SimDyn():
    """
    Imports Simulation() object from scint_sim to Dynspec class
    """

    def __init__(self, sim):

        self.name =\
            'sim:mb2={0}_ar={1}_psi={2}_dlam={3}'.format(sim.mb2, sim.ar,
                                                         sim.psi, sim.dlam)
        if sim.lamsteps:
            self.name += ',lamsteps'

        self.header = self.name
        self.dyn = sim.spi
        dlam = sim.dlam

        self.dt = sim.dt
        self.freq = sim.freq
        self.mjd = sim.mjd
        self.nsub = int(np.shape(self.dyn)[0])
        self.nchan = int(np.shape(self.dyn)[1])
        lams = np.linspace(1, 1+dlam, self.nchan)
        freqs = np.divide(1, lams)
        self.freqs = self.freq*np.linspace(np.min(freqs), np.max(freqs),
                                           self.nchan)
        self.bw = max(self.freqs) - min(self.freqs)
        self.times = self.dt*np.arange(0, self.nsub)
        self.df = self.bw/self.nchan
        self.tobs = self.nsub * self.dt
        self.dyn = np.transpose(self.dyn)
        return


class HoloDyn():
    """
    Imports model dynamic spectrum from holography code. Walker et al. (2008).
    """

    def __init__(self, holofile, imholofile=None, df=1, dt=1, fmin=0, mjd=0):

        from astropy.io import fits
        redata_hdu = fits.open(holofile)
        redata = redata_hdu[0].data
        if imholofile is not None:
            imdata_hdu = fits.open(imholofile)
            imdata = imdata_hdu[0].data
        else:
            imdata = np.zeros(np.shape(redata))
        dynt = redata + 1j * imdata
        dynt = np.abs(dynt)

        # Set parameters from input
        self.dyn = np.flip(np.transpose(np.flip(dynt, axis=0)), axis=1)
        self.name = os.path.basename(holofile)
        self.header = self.name
        self.freqs = (np.linspace(0, len(self.dyn), len(self.dyn)) * df) + fmin
        self.times = np.linspace(0, len(self.dyn[0]), len(self.dyn[0])) * dt
        self.nchan = len(self.freqs)
        self.nsub = len(self.times)
        self.bw = abs(max(self.freqs)) - abs(min(self.freqs))
        self.tobs = max(self.times)
        self.df = df
        self.dt = dt
        self.freq = np.mean(np.unique(self.freqs))
        self.mjd = mjd
        return


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
        dyn.correct_dyn()  # correct for bandpass and gain variation
        dyn.calc_sspec()  # calculate secondary spectrum
        # report error and proceed
        if np.isnan(dyn.sspec).all():  # skip if secondary spectrum is all nan
            bad_files.write("{0}\t sspec_isnan\n".format(dynfile))
            continue
        # Passed all tests so far - write to good_files.txt!
        good_files.write("{0}\n".format(dynfile))
    bad_files.close()
    good_files.close()
    return
