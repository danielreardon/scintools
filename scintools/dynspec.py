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
from scintools.scint_models import scint_acf_model, scint_sspec_model, tau_acf_model,\
                         dnu_acf_model, fit_parabola, fit_log_parabola
from scintools.scint_utils import is_valid, svd_model
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata, interp1d
from scipy.signal import convolve2d, medfilt, savgol_filter
from scipy.io import loadmat


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
        self.tobs = dyn.tobs  # initial estimate of tobs
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
        self.refill()  # refill with linear interpolation
        self.correct_dyn()
        self.calc_acf()  # calculate the ACF
        if lamsteps:
            self.scale_dyn()
        self.calc_sspec(lamsteps=lamsteps)  # Calculate secondary spectrum

    def plot_dyn(self, lamsteps=False, input_dyn=None, filename=None,
                 input_x=None, input_y=None, trap=False, display=True):
        """
        Plot the dynamic spectrum
        """
        plt.figure(1, figsize=(12, 6))
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
        vmin = minval
        vmax = medval+5*std
        if input_dyn is None:
            if lamsteps:
                plt.pcolormesh(self.times/60, self.lam, dyn,
                               vmin=vmin, vmax=vmax)
                plt.ylabel('Wavelength (m)')
            else:
                plt.pcolormesh(self.times/60, self.freqs, dyn,
                               vmin=vmin, vmax=vmax)
                plt.ylabel('Frequency (MHz)')
            plt.xlabel('Time (mins)')
            # plt.colorbar()  # arbitrary units
        else:
            plt.pcolormesh(input_x, input_y, dyn, vmin=vmin, vmax=vmax)

        if filename is not None:
            plt.savefig(filename, dpi=200, papertype='a4', bbox_inches='tight',
                        pad_inches=0.1)
            plt.close()
        elif input_dyn is None and display:
            plt.show()

    def plot_acf(self, contour=False, filename=None, input_acf=None,
                 input_t=None, input_f=None, fit=True, display=True):
        """
        Plot the ACF
        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'tau') and input_acf is None and fit:
            self.get_scint_params()
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
        t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
        f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

        if input_acf is None:  # Also plot scintillation scales axes
            fig, ax1 = plt.subplots()
            if contour:
                im = ax1.contourf(t_delays, f_shifts, arr)
            else:
                im = ax1.pcolormesh(t_delays, f_shifts, arr)
            ax1.set_ylabel('Frequency lag (MHz)')
            ax1.set_xlabel('Time lag (mins)')
            miny, maxy = ax1.get_ylim()
            if fit:
                ax2 = ax1.twinx()
                ax2.set_ylim(miny/self.dnu, maxy/self.dnu)
                ax2.set_ylabel('Frequency lag / (dnu_d = {0})'.
                               format(round(self.dnu, 2)))
                ax3 = ax1.twiny()
                minx, maxx = ax1.get_xlim()
                ax3.set_xlim(minx/(self.tau/60), maxx/(self.tau/60))
                ax3.set_xlabel('Time lag/(tau_d={0})'.format(round(
                                                             self.tau/60, 2)))
            fig.colorbar(im, pad=0.15)
        else:  # just plot acf without scales
            if contour:
                plt.contourf(t_delays, f_shifts, arr)
            else:
                plt.pcolormesh(t_delays, f_shifts, arr)
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        elif input_acf is None and display:
            plt.show()

    def plot_sspec(self, lamsteps=False, input_sspec=None, filename=None,
                   input_x=None, input_y=None, trap=False, prewhite=True,
                   plotarc=False, maxfdop=np.inf, delmax=None, ref_freq=1400,
                   cutmid=0, startbin=0, display=True):
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
        if input_sspec is None:
            if lamsteps:
                plt.pcolormesh(xplot, self.beta[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
            else:
                plt.pcolormesh(xplot, self.tdel[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\nu$ ($\mu$s)')
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
            plt.colorbar()
        else:
            plt.pcolormesh(xplot, input_y, sspec, vmin=vmin, vmax=vmax)
            plt.colorbar()

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        elif input_sspec is None and display:
            plt.show()

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
            plt.close()
        elif display:
            plt.show()

    def fit_arc(self, method='norm_sspec', asymm=False, plot=False,
                delmax=None, numsteps=1e4, startbin=3, cutmid=3, lamsteps=True,
                etamax=None, etamin=None, low_power_diff=-3,
                high_power_diff=-1.5, ref_freq=1400, constraint=[0, np.inf],
                nsmooth=5, filename=None, noise_error=True, display=True,
                log_parabola=False):
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

        # noise of mean out to delmax
        noise = np.sqrt(np.sum(np.power(noise, 2)))/np.sqrt(len(yaxis))

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
        sqrt_eta_all = np.linspace(min_sqrt_eta, max_sqrt_eta, numsteps)

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

            # Define data
            x = self.fdop
            y = yaxis
            z = sspec
            # initiate arrays
            sumpowL = []
            sumpowR = []
            etaArray = []

            if method == 'gridmax':
                for ii in range(0, len(sqrt_eta)):
                    ieta = sqrt_eta[ii]**2
                    etaArray.append(ieta)
                    ynew = ieta*np.power(x, 2)  # tdel coordinates to sample
                    # convert to pixel coordinates
                    xpx = ((x-np.min(x))/(max(x) - np.min(x)))*np.shape(z)[1]
                    ynewpx = ((ynew-np.min(ynew)) /
                              (max(y) - np.min(ynew)))*np.shape(z)[0]
                    # left side
                    ind = np.where(x < 0)  # find -ve doppler
                    ynewL = ynew[ind]
                    xnewpxL = xpx[ind]
                    ynewpxL = ynewpx[ind]
                    ind = np.where(ynewL < np.max(y))  # inds below tdel cutoff
                    xnewL = xnewpxL[ind]
                    ynewL = ynewpxL[ind]
                    xynewL = np.array([[ynewL[ii], xnewL[ii]] for ii in
                                      range(0, len(xnewL))]).T
                    znewL = map_coordinates(z, xynewL, order=1, cval=np.nan)
                    sumpowL.append(np.mean(znewL[~np.isnan(znewL)]))

                    # Right side
                    ind = np.where(x > 0)  # find +ve doppler
                    ynewR = ynew[ind]
                    xnewpxR = xpx[ind]
                    ynewpxR = ynewpx[ind]
                    ind = np.where(ynewR < np.max(y))  # inds below tdel cutoff
                    xnewR = xnewpxR[ind]
                    ynewR = ynewpxR[ind]
                    xynewR = np.array([[ynewR[ii], xnewR[ii]] for ii in
                                       range(0, len(xnewR))]).T
                    znewR = map_coordinates(z, xynewR, order=1, cval=np.nan)
                    sumpowR.append(np.mean(znewR[~np.isnan(znewR)]))

                    # Total
                    sumpow = np.add(sumpowL, sumpowR)/2  # average

                # Ignore nan sums and smooth
                indicies = np.argwhere(is_valid(sumpow)).ravel()
                etaArray = np.array(etaArray)[indicies]
                sumpow = np.array(sumpow)[indicies]
                sumpowL = np.array(sumpowL)[indicies]
                sumpowR = np.array(sumpowR)[indicies]
                sumpow_filt = savgol_filter(sumpow, nsmooth, 1)
                sumpowL_filt = savgol_filter(sumpowL, nsmooth, 1)
                sumpowR_filt = savgol_filter(sumpowR, nsmooth, 1)

                indrange = np.argwhere((etaArray > constraint[0]) *
                                       (etaArray < constraint[1]))
                sumpow_inrange = sumpow_filt[indrange]
                sumpowL_inrange = sumpow_filt[indrange]
                sumpowR_inrange = sumpow_filt[indrange]
                ind = np.argmin(np.abs(sumpow_filt - np.max(sumpow_inrange)))
                indL = np.argmin(np.abs(sumpow_filt - np.max(sumpowL_inrange)))
                indR = np.argmin(np.abs(sumpow_filt - np.max(sumpowR_inrange)))
                eta = etaArray[ind]
                etaL = etaArray[indL]
                etaR = etaArray[indR]

                # Now find eta and estimate error by fitting parabola
                #   Data from -3db on low curvature side to -1.5db on high side
                max_power = sumpow_filt[ind]
                power = max_power
                ind1 = 1
                while (power > max_power + low_power_diff and
                       ind + ind1 < len(sumpow_filt)-1):  # -3db, or half power
                    ind1 += 1
                    power = sumpow_filt[ind - ind1]
                power = max_power
                ind2 = 1
                while (power > max_power + high_power_diff and
                       ind + ind2 < len(sumpow_filt)-1):  # -1db power
                    ind2 += 1
                    power = sumpow_filt[ind + ind2]
                # Now select this region of data for fitting
                xdata = etaArray[int(ind-ind1):int(ind+ind2)]
                ydata = sumpow[int(ind-ind1):int(ind+ind2)]

                # Do the fit
                # yfit, eta, etaerr = fit_parabola(xdata, ydata)
                yfit, eta, etaerr = fit_log_parabola(xdata, ydata)
                if np.mean(np.gradient(np.diff(yfit))) > 0:
                    raise ValueError('Fit returned a forward parabola.')
                eta = eta

                if noise_error:
                    # Now get error from the noise in secondary spectra instead
                    etaerr2 = etaerr
                    power = max_power
                    ind1 = 1
                    while (power > max_power - noise and ind - ind1 > 1):
                        power = sumpow_filt[ind - ind1]
                        ind1 += 1
                    power = max_power
                    ind2 = 1
                    while (power > max_power - noise and
                           ind + ind2 < len(sumpow_filt)-1):
                        ind2 += 1
                        power = sumpow_filt[ind + ind2]

                    etaerr = np.ptp(etaArray[int(ind-ind1):int(ind+ind2)])/2

                # Now plot
                if plot and iarc == 0:
                    if asymm:
                        plt.subplot(2, 1, 1)
                        plt.plot(etaArray, sumpowL)
                        plt.plot(etaArray, sumpowL_filt)
                        bottom, top = plt.ylim()
                        plt.plot([etaL, etaL], [bottom, top])
                        plt.ylabel('mean power (db)')
                        plt.xscale('log')
                        plt.subplot(2, 1, 2)
                        plt.plot(etaArray, sumpowR)
                        plt.plot(etaArray, sumpowR_filt)
                        bottom, top = plt.ylim()
                        plt.plot([etaR, etaR], [bottom, top])
                    else:
                        plt.plot(etaArray, sumpow)
                        plt.plot(etaArray, sumpow_filt)
                        plt.plot(xdata, yfit)
                        bottom, top = plt.ylim()
                    plt.axvspan(xmin=eta-etaerr, xmax=eta+etaerr,
                                facecolor='C2', alpha=0.5)
                    if lamsteps:
                        plt.xlabel(r'Arc curvature, $\eta$ (${\rm m}^{-1}\,'
                                   '{\rm mHz}^{-2}$)')
                    else:
                        plt.xlabel('eta (tdel)')
                    plt.ylabel('mean power (dB)')
                    plt.xscale('log')
                elif plot:
                    plt.axvspan(xmin=eta-etaerr, xmax=eta+etaerr,
                                facecolor='C{0}'.format(str(int(3+iarc))),
                                alpha=0.3)
                if plot and iarc == len(etamin_array) - 1:
                    if filename is not None:
                        plt.savefig(filename, figsize=(6, 6), dpi=150,
                                    bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                    elif display:
                        plt.show()

            elif method == 'norm_sspec':
                # Get the normalised secondary spectrum, set for minimum eta as
                #   normalisation. Then calculate peak as
                self.norm_sspec(eta=etamin, delmax=delmax, plot=False,
                                startbin=startbin, maxnormfac=1, cutmid=cutmid,
                                lamsteps=lamsteps, scrunched=True,
                                plot_fit=False, numsteps=numsteps_new)
                norm_sspec = self.normsspecavg.squeeze()
                etafrac_array = np.linspace(-1, 1, len(norm_sspec))
                ind1 = np.argwhere(etafrac_array > 1/(2*len(norm_sspec)))
                ind2 = np.argwhere(etafrac_array < -1/(2*len(norm_sspec)))

                norm_sspec_avg = np.add(norm_sspec[ind1],
                                        np.flip(norm_sspec[ind2], axis=0))/2
                norm_sspec_avg = norm_sspec_avg.squeeze()
                etafrac_array_avg = 1/etafrac_array[ind1].squeeze()
                # Make sure is valid
                filt_ind = is_valid(norm_sspec_avg)
                norm_sspec_avg = np.flip(norm_sspec_avg[filt_ind], axis=0)
                etafrac_array_avg = np.flip(etafrac_array_avg[filt_ind],
                                            axis=0)

                # Form eta array and cut at maximum
                etaArray = etamin*etafrac_array_avg**2
                ind = np.argwhere(etaArray < etamax)
                etaArray = etaArray[ind].squeeze()
                norm_sspec_avg = norm_sspec_avg[ind].squeeze()

                # Smooth data
                norm_sspec_avg_filt = \
                    savgol_filter(norm_sspec_avg, nsmooth, 1)

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
                ydata = norm_sspec_avg[int(ind-ind1):int(ind+ind2)]

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

                if plot and iarc == 0:
                    plt.plot(etaArray, norm_sspec_avg)
                    plt.plot(etaArray, norm_sspec_avg_filt)
                    plt.plot(xdata, yfit)
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
                             color='C{0}'.format(str(int(3+iarc))))
                    plt.axvspan(xmin=eta-etaerr, xmax=eta+etaerr,
                                facecolor='C{0}'.format(str(int(3+iarc))),
                                alpha=0.3)
                if plot and iarc == len(etamin_array)-1:
                    if filename is not None:
                        plt.savefig(filename, figsize=(6, 6), dpi=150,
                                    bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                    elif display:
                        plt.show()

            else:
                raise ValueError('Unknown arc fitting method. Please choose \
                                 from gidmax or norm_sspec')

            if iarc == 0:  # save primary
                if lamsteps:
                    self.betaeta = eta
                    self.betaetaerr = etaerr
                    self.betaetaerr2 = etaerr2
                else:
                    self.eta = eta
                    self.etaerr = etaerr
                    self.etaerr2 = etaerr2

    def norm_sspec(self, eta=None, delmax=None, plot=False, startbin=1,
                   maxnormfac=5, minnormfac=0, cutmid=3, lamsteps=False,
                   scrunched=True, plot_fit=True, ref_freq=1400, numsteps=None,
                   filename=None, display=True, unscrunched=True,
                   powerspec=False):
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
        std = np.std(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval - std
        vmax = maxval - 3

        ind = np.argmin(abs(self.tdel-delmax))
        sspec = sspec[startbin:ind, :]  # cut first N delay bins and at delmax
        # sspec[0:startbin] = np.nan
        nr, nc = np.shape(sspec)
        # mask out centre bins
        sspec[:, int(nc/2 - np.floor(cutmid/2)):int(nc/2 +
              np.floor(cutmid/2))] = np.nan
        tdel = yaxis[startbin:ind]
        # tdel = yaxis[:ind]
        fdop = self.fdop
        maxfdop = maxnormfac*np.sqrt(tdel[-1]/eta)  # Maximum fdop for plot
        if maxfdop > max(fdop):
            maxfdop = max(fdop)
        # Number of fdop bins to use. Oversample by factor of 2
        nfdop = 2*len(fdop[abs(fdop) <=
                           maxfdop]) if numsteps is None else numsteps
        fdopnew = np.linspace(-maxnormfac, maxnormfac,
                              nfdop)  # norm fdop
        if minnormfac > 0:
            unscrunched = False  # Cannot plot 2D function
            inds = np.argwhere(np.abs(fdopnew) > minnormfac)
            fdopnew = fdopnew[inds]
        normSspec = []
        isspectot = np.zeros(np.shape(fdopnew))
        for ii in range(0, len(tdel)):
            itdel = tdel[ii]
            imaxfdop = maxnormfac*np.sqrt(itdel/eta)
            ifdop = fdop[abs(fdop) <= imaxfdop]/np.sqrt(itdel/eta)
            isspec = sspec[ii, abs(fdop) <= imaxfdop]  # take the iith row
            ind = np.argmin(abs(fdopnew))
            normline = np.interp(fdopnew, ifdop, isspec)
            normSspec.append(normline)
            isspectot = np.add(isspectot, normline)
        normSspec = np.array(normSspec).squeeze()
        isspecavg = np.nanmean(normSspec, axis=0)  # make average
        powerspectrum = np.nanmean(np.power(10, normSspec/10), axis=1)
        ind1 = np.argmin(abs(fdopnew-1)-2)
        if isspecavg[ind1] < 0:
            isspecavg = isspecavg + 2  # make 1 instead of -1
        if plot:
            # Plot delay-scrunched "power profile"
            if scrunched:
                plt.plot(fdopnew, isspecavg)
                bottom, top = plt.ylim()
                plt.xlabel("Normalised $f_t$")
                plt.ylabel("Mean power (dB)")
                if plot_fit:
                    plt.plot([1, 1], [bottom*0.9, top*1.1], 'r--', alpha=0.5)
                    plt.plot([-1, -1], [bottom*0.9, top*1.1], 'r--', alpha=0.5)
                plt.ylim(bottom*0.9, top*1.1)  # always plot from zero!
                plt.xlim(-maxnormfac, maxnormfac)
                if filename is not None:
                    filename_name = filename.split('.')[0]
                    filename_extension = filename.split('.')[1]
                    plt.savefig(filename_name + '_1d.' + filename_extension,
                                bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                elif display:
                    plt.show()
            # Plot 2D normalised secondary spectrum
            if unscrunched:
                plt.pcolormesh(fdopnew, tdel, normSspec, vmin=vmin, vmax=vmax)
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
                    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                elif display:
                    plt.show()
            # plot power spectrum
            if powerspec:
                plt.loglog(np.sqrt(tdel), powerspectrum)
                # Overlay theory
                kf = np.argwhere(np.sqrt(tdel) <= 10)
                amp = np.mean(powerspectrum[kf]*(np.sqrt(tdel[kf]))**3.67)
                plt.loglog(np.sqrt(tdel), amp*(np.sqrt(tdel))**(-3.67))
                if lamsteps:
                    plt.xlabel(r'$f_\lambda^{1/2}$ (m$^{-1/2}$)')
                else:
                    plt.xlabel(r'$f_\nu^{1/2}$ ($\mu$s$^{1/2}$)')
                plt.ylabel("Mean PSD")
                plt.grid(which='both', axis='both')
                if filename is not None:
                    filename_name = filename.split('.')[0]
                    filename_extension = filename.split('.')[1]
                    plt.savefig(filename_name + '_power.' + filename_extension,
                                bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                elif display:
                    plt.show()

        self.normsspecavg = isspecavg
        self.normsspec = normSspec
        self.normsspec_tdel = tdel
        return

    def get_scint_params(self, method="acf1d", plot=False, alpha=5/3,
                         mcmc=False, display=True):
        """
        Measure the scintillation timescale
            Method:
                acf1d - takes a 1D cut through the centre of the ACF for
                sspec - measures timescale from the power spectrum
                acf2d - uses an analytic approximation to the ACF including
                    phase gradient
        """

        from lmfit import Minimizer, Parameters
        import corner

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'sspec'):
            self.calc_sspec()

        if method == 'acf1d':
            scint_model = scint_acf_model
            ydata_f = self.acf[int(self.nchan):, int(self.nsub)]
            xdata_f = self.df*np.linspace(0, len(ydata_f), len(ydata_f))
            ydata_t = self.acf[int(self.nchan), int(self.nsub):]
            xdata_t = self.dt*np.linspace(0, len(ydata_t), len(ydata_t))
        elif method == 'sspec':
            scint_model = scint_sspec_model
            arr = cp(self.acf)
            arr = np.fft.ifftshift(arr)
            # sspec = np.fft.fft2(arr)

        # concatenate x and y arrays
        xdata = np.array(np.concatenate((xdata_t, xdata_f)))
        ydata = np.array(np.concatenate((ydata_t, ydata_f)))
        weights = np.ones(np.shape(ydata))

        # Get initial parameter values
        nt = len(xdata_t)  # number of t-lag samples
        # Estimate amp and white noise level
        wn = min([ydata_f[0]-ydata_f[1], ydata_t[0]-ydata_t[1]])
        amp = max([ydata_f[1], ydata_t[1]])
        # Estimate tau for initial guess. Closest index to 1/e power
        tau = xdata_t[np.argmin(abs(ydata_t - amp/np.e))]
        # Estimate dnu for initial guess. Closest index to 1/2 power
        dnu = xdata_f[np.argmin(abs(ydata_f - amp/2))]

        # Define fit parameters
        params = Parameters()
        params.add('tau', value=tau, min=0.0, max=np.inf)
        params.add('dnu', value=dnu, min=0.0, max=np.inf)
        params.add('amp', value=amp, min=0.0, max=np.inf)
        params.add('wn', value=wn, min=0.0, max=np.inf)
        params.add('nt', value=nt, vary=False)
        if alpha is None:
            params.add('alpha', value=5/3, min=0, max=8)
        else:
            params.add('alpha', value=alpha, vary=False)

        # Do fit
        func = Minimizer(scint_model, params, fcn_args=(xdata, ydata, weights))
        results = func.minimize()
        if mcmc:
            print('Doing mcmc posterior sample')
            mcmc_results = func.emcee()
            results = mcmc_results

        self.tau = results.params['tau'].value
        self.tauerr = results.params['tau'].stderr
        self.dnu = results.params['dnu'].value
        self.dnuerr = results.params['dnu'].stderr
        self.talpha = results.params['alpha'].value
        if alpha is None:
            self.talphaerr = results.params['alpha'].stderr

        if plot:
            # get models:
            if method == 'acf1d':
                # Get tau model
                tmodel_res = tau_acf_model(results.params, xdata_t, ydata_t,
                                           weights[:nt])
                tmodel = ydata_t - tmodel_res/weights[:nt]
                # Get dnu model
                fmodel_res = dnu_acf_model(results.params, xdata_f, ydata_f,
                                           weights[nt:])
                fmodel = ydata_f - fmodel_res/weights[nt:]

            plt.subplot(2, 1, 1)
            plt.plot(xdata_t, ydata_t)
            plt.plot(xdata_t, tmodel)
            plt.xlabel('Time lag (s)')
            plt.subplot(2, 1, 2)
            plt.plot(xdata_f, ydata_f)
            plt.plot(xdata_f, fmodel)
            plt.xlabel('Frequency lag (MHz)')
            if display:
                plt.show()

            if mcmc:
                corner.corner(mcmc_results.flatchain,
                              labels=mcmc_results.var_names,
                              truths=list(mcmc_results.params.
                                          valuesdict().values()))
                if display:
                    plt.show()

        return

    def cut_dyn(self, tcuts=0, fcuts=0, plot=False, filename=None,
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
                            figsize=(6, 10), dpi=150, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
            plt.figure(2)
            if filename is not None:
                plt.savefig(filename_name + '_acf.' + filename_extension,
                            figsize=(6, 10), dpi=150, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
            plt.figure(3)
            if filename is not None:
                plt.savefig(filename_name + '_sspec.' + filename_extension,
                            figsize=(6, 10), dpi=150, bbox_inches='tight',
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
        array = cp(self.dyn)
        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        if linear:  # do linear interpolation
            # mask invalid values
            array = np.ma.masked_invalid(array)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = np.ravel(array[~array.mask])
            self.dyn = griddata((x1, y1), newarr, (xx, yy),
                                method='linear')
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

    def calc_sspec(self, prewhite=True, plot=False, lamsteps=False,
                   input_dyn=None, input_x=None, input_y=None, trap=False,
                   window='blackman', window_frac=0.1):
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
        sec = sec[int(nrfft/2):][:]  # crop

        td = np.array(list(range(0, int(nrfft/2))))
        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),
                          [len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),
                          [len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1

        if prewhite:  # Now post-darken
            vec1 = np.reshape(np.power(np.sin(
                              np.multiply(sc.pi/ncfft, fd)), 2), [ncfft, 1])
            vec2 = np.reshape(np.power(np.sin(
                              np.multiply(sc.pi/nrfft, td)), 2),
                              [1, int(nrfft/2)])
            postdark = np.transpose(vec1*vec2)
            postdark[:, int(ncfft/2)] = 1
            postdark[0, :] = 1
            sec = np.divide(sec, postdark)

        # Make db
        sec = 10*np.log10(sec)

        if input_dyn is None:
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

    def calc_acf(self, scale=False, input_dyn=None, plot=True):
        """
        Calculate autocovariance function
        """

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
                  window='hanning'):
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
            dlam = np.max(np.abs(np.diff(lams)))
            lam_eq = np.arange(np.min(lams), np.max(lams), dlam)
            self.dlam = dlam
            feq = np.divide(sc.c, lam_eq)/10**6
            arout = np.zeros([len(lam_eq), int(nt)])
            for it in range(0, nt):
                f = interp1d(freqs, arin[:, it], kind='cubic')
                arout[:, it] = f(feq)
            self.lamdyn = np.flipud(arout)
            self.lam = np.flipud(lam_eq)
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

    def __init__(self, sim, freq=1400, dt=0.5, mjd=50000):

        self.name =\
            'sim:mb2={0},ar={1},psi={2},dlam={3}'.format(sim.mb2, sim.ar,
                                                         sim.psi, sim.dlam)
        if sim.lamsteps:
            self.name += ',lamsteps'

        self.header = self.name
        self.dyn = sim.spi
        dlam = sim.dlam

        self.dt = dt
        self.freq = freq
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
        self.mjd = mjd
        self.dyn = np.transpose(self.dyn)
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
