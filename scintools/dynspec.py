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
from models import scint_acf_model, scint_sspec_model
from scint_utils import is_valid
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata
from scipy.signal import convolve2d, medfilt
from scipy.optimize import curve_fit
from scipy.io import loadmat


class Dynspec:

    def __init__(self, filename=None, dyn=None, verbose=True, process=True,
                 lamsteps=False, backup=True, axes=None):
        """"
        Initialise a dynamic spectrum object by either reading from file
            or from existing object
        """

        if filename:
            self.load_file(filename, verbose=verbose, process=process,
                           lamsteps=lamsteps, axes=axes)
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
                        - self.tobs, 2)  # time between two dynspecs
        extratimes = np.arange(self.dt/2, timegap, dt)
        if timegap < dt:
            extratimes = [0]
        nextra = len(extratimes)
        dyngap = np.zeros([np.shape(self.dyn)[0], nextra])

        # Set changed properties
        name = self.name.split('.')[0] + "+" + other.name.split('.')[0] \
            + ".dynspec"
        header = self.header + other.header
        times = np.concatenate((self.times, self.times[-1] + extratimes,
                                self.times[-1] + extratimes[-1] + other.times))
        nsub = self.nsub + nextra + other.nsub
        tobs = self.tobs + timegap + other.tobs
        mjd = np.min([self.mjd, other.mjd])  # mjd for earliest dynspec
        newdyn = np.concatenate((self.dyn, dyngap, other.dyn), axis=1)

        # Get new dynspec object with these properties
        newdyn = BasicDyn(newdyn, name=name, header=header, times=times,
                          freqs=freqs, nchan=nchan, nsub=nsub, bw=bw,
                          df=df, freq=freq, tobs=tobs, dt=dt, mjd=mjd)

        return Dynspec(dyn=newdyn, verbose=False, process=False)

    def load_file(self, filename, verbose=True, process=True, lamsteps=False,
                  axes=None):
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
        self.tobs = self.times[-1]  # initial estimate of tobs
        self.dt = round(self.tobs/self.nsub)
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
        self.calc_acf()  # calculate the ACF
        if lamsteps:
            self.scale_dyn()
        self.calc_sspec(lamsteps=lamsteps)  # Calculate secondary spectrum

    def plot_dyn(self, lamsteps=False, input_dyn=None, filename=None,
                 input_x=None, input_y=None):
        """
        Plot the dynamic spectrum
        """

        if input_dyn is None:
            if lamsteps:
                if not hasattr(self, 'lamdyn'):
                    self.scale_dyn()
                dyn = self.lamdyn
            else:
                dyn = self.dyn
        else:
            dyn = input_dyn
        medval = np.median(dyn[is_valid(dyn)*np.array(np.abs(dyn) > 0)])
        std = np.std(dyn[is_valid(dyn)])  # standard deviation
        vmin = medval-5*std
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
            plt.colorbar()
        else:
            plt.pcolormesh(input_x, input_y, dyn, vmin=vmin, vmax=vmax)

        if filename is not None:
            plt.savefig(filename, dpi=200, papertype='a4')
        elif input_dyn is None:
            plt.show()

    def plot_acf(self, contour=False, filename=None, input_acf=None,
                 input_t=None, input_f=None):
        """
        Plot the ACF
        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'tau'):
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

        plt.pcolormesh(t_delays, f_shifts, arr)

#        fig, ax1 = plt.subplots()
#        if contour:
#            im = ax1.contourf(t_delays, f_shifts, arr)
#        else:
#            im = ax1.pcolormesh(t_delays, f_shifts, arr)
#        ax1.set_ylabel('Frequency lag (MHz)')
#        ax1.set_xlabel('Time lag (mins)')
#        ax2 = ax1.twinx()
#        miny, maxy = ax1.get_ylim()
#        ax2.set_ylim(miny/self.dnu, maxy/self.dnu)
#        ax2.set_ylabel('Frequency lag / (dnu_d = {0})'.format(round(
#                                                              self.dnu, 2)))
#        ax3 = ax1.twiny()
#        minx, maxx = ax1.get_xlim()
#        ax3.set_xlim(minx/(self.tau/60), maxx/(self.tau/60))
#        ax3.set_xlabel('Time lag / (tau_d = {0})'.format(round(
#                                                         self.tau/60, 2)))
#        fig.colorbar(im, pad=0.15)

        if filename is not None:
            plt.savefig(filename)
        elif input_acf is None:
            plt.show()

    def plot_sspec(self, lamsteps=False, input_sspec=None, filename=None,
                   input_x=None, input_y=None):
        """
        Plot the secondary spectrum
        """

        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = self.lamsspec
            else:
                if not hasattr(self, 'sspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = self.sspec
        else:
            sspec = input_sspec
        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval-20
        vmax = vmin+60
        if input_sspec is None:
            if lamsteps:
                plt.pcolormesh(self.fdop, self.beta, sspec,
                               vmin=vmin, vmax=vmax)
                plt.ylabel('Beta (m$^{-1}$)')
            else:
                plt.pcolormesh(self.fdop, self.tdel, sspec,
                               vmin=vmin, vmax=vmax)
                plt.ylabel('Delay (us)')
            plt.xlabel('Doppler Frequency (mHz)')
            bottom, top = plt.ylim()
            if hasattr(self, 'eta'):
                eta = self.eta
                if lamsteps:  # convert eta to beta equivalent
                    eta = eta
                plt.plot(self.fdop, eta*np.power(self.fdop, 2),
                         'r', alpha=0.5)
            plt.ylim(bottom, top)
            plt.colorbar()
        else:
            plt.pcolormesh(input_x, input_y, sspec, vmin=vmin, vmax=vmax)
            plt.colorbar()

        if filename is not None:
            plt.savefig(filename)
        elif input_sspec is None:
            plt.show()

    def plot_all(self, dyn=1, sspec=3, acf=2, norm_sspec=4, colorbar=True,
                 lamsteps=False, filename=None):
        """
        Plots multiple figures in one
        """
        # Dynamic Spectrum
        plt.subplot(2, 2, dyn)
        self.plot_dyn(lamsteps=lamsteps)
        plt.title("Dynamic Spectrum")

        # Autocovariance Function
        plt.subplot(2, 2, acf)
        self.plot_acf()
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
            plt.savefig(filename)
        else:
            plt.show()

    def fit_arc(self, method='gridmax', asymm=False, plot=False, delmax=0.3,
                sqrt_eta_step=1e-3, startbin=9, etamax=0.2, lamsteps=False,
                cutmid=9):
        """
        Find the arc curvature with maximum power along it
        """

        if lamsteps:
            if not hasattr(self, 'lamsspec'):
                self.calc_sspec(lamsteps=lamsteps)
            temp = self.lamsspec
        else:
            if not hasattr(self, 'sspec'):
                self.calc_sspec()
            temp = self.sspec
        sspec = temp
        delmax = delmax*(1400/self.freq)**2
        ind = np.argmin(abs(self.tdel-delmax))
        sspec[0:startbin, :] = np.nan  # mask first N delay bins
        nr, nc = np.shape(sspec)
        # mask middle N Doppler bins
        sspec[:, int(nc/2 - np.floor(cutmid/2)):int(nc/2 +
              np.floor(cutmid/2))] = np.nan
        sspec = sspec[0:ind, :]  # cut at delmax
        tdel = self.tdel[0:ind]
        fdop = self.fdop
        # At 1mHz for 1400MHz obs, the maximum arc terminates at delmax
        max_sqrt_eta = np.sqrt(etamax)*(1400/self.freq)
        sqrt_eta = np.arange(sqrt_eta_step, max_sqrt_eta, sqrt_eta_step)
        sumpowL = []  # initiate arrays
        sumpowR = []
        etaArray = []
        x = fdop
        if lamsteps:
            y = tdel
        else:
            y = tdel
        z = sspec
        if method == 'gridmax':
            etaArray = []
            for ii in range(0, len(sqrt_eta)):
                ieta = sqrt_eta[ii]**2
                etaArray.append(ieta)
                xnew = x  # fdop coordinates to sample - everything
                ynew = ieta*np.power(xnew, 2)  # tdel coordinates to sample
                # convert to pixel coordinates
                xnewpx = ((xnew-np.min(xnew))/(max(x)
                                               - np.min(xnew)))*np.shape(z)[1]
                ynewpx = ((ynew-np.min(ynew))/(max(y)
                                               - np.min(ynew)))*np.shape(z)[0]
                # left side
                ind = np.where(xnew < 0)  # find -ve doppler
                ynewL = ynew[ind]
                xnewpxL = xnewpx[ind]
                ynewpxL = ynewpx[ind]
                ind = np.where(ynewL < np.max(y))  # indices below tdel cuttof
                xnewL = xnewpxL[ind]
                ynewL = ynewpxL[ind]
                xynewL = np.array([[ynewL[ii], xnewL[ii]] for ii in range(0,
                                   len(xnewL))]).T
                znewL = map_coordinates(z, xynewL, order=1, cval=np.nan)
                sumpowL.append(np.mean(znewL[~np.isnan(znewL)]))
                # right side
                ind = np.where(xnew > 0)  # find +ve doppler
                ynewR = ynew[ind]
                xnewpxR = xnewpx[ind]
                ynewpxR = ynewpx[ind]
                ind = np.where(ynewR < np.max(y))  # indices below tdel cuttof
                xnewR = xnewpxR[ind]
                ynewR = ynewpxR[ind]
                xynewR = np.array([[ynewR[ii], xnewR[ii]] for ii in range(0,
                                   len(xnewR))]).T
                znewR = map_coordinates(z, xynewR, order=1, cval=np.nan)
                sumpowR.append(np.mean(znewR[~np.isnan(znewR)]))
        elif method == 'norm_sspec':
            if not hasattr(self, 'normsspecavg'):
                self.norm_sspec(delmax=delmax, plot=plot, startbin=startbin,
                                maxnormfac=3, cutmid=2, lamsteps=lamsteps)
            print('nothing yet')
        else:
            raise ValueError('Unknown arc fitting method. Please choose \
                             from gidmax or norm_sspec')
        sumpow = np.add(sumpowL, sumpowR)/2  # average
        ind = np.argmax(sumpow[~np.isnan(sumpow)])
        indL = np.argmax(sumpowL)
        indR = np.argmax(sumpowR)
        extra = len(np.where(np.isnan(sumpow)))
        eta = etaArray[ind + extra]
        etaL = etaArray[indL + extra]
        etaR = etaArray[indR + extra]
        if plot:
            if asymm:
                plt.plot(np.sqrt(etaArray), sumpowL)
                plt.plot(np.sqrt(etaArray), sumpowR)
                bottom, top = plt.ylim()
                plt.plot([np.sqrt(etaL), np.sqrt(etaL)], [bottom, top], 'b')
                plt.plot([np.sqrt(etaR), np.sqrt(etaR)], [bottom, top], 'r')
            else:
                plt.plot(np.sqrt(etaArray), sumpow)
                bottom, top = plt.ylim()
                plt.plot([np.sqrt(eta), np.sqrt(eta)], [bottom, top], 'b')
            plt.xlabel('sqrt(eta)')
            plt.show()
        print("Currently takes curvature measurement as maximum -- update")
        self.eta = eta

    def norm_sspec(self, eta=None, delmax=None, plot=False, startbin=9,
                   maxnormfac=2, cutmid=2, lamsteps=False, scrunched=True,
                   plot_fit=True):
        """
        Normalise fdop axis using arc curvature
        """

        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(1400/self.freq)**2

        if not hasattr(self, 'eta') and not eta:
            self.fit_arc(lamsteps=lamsteps, delmax=delmax, plot=plot,
                         startbin=startbin)
        if lamsteps:
            if not hasattr(self, 'lamsspec'):
                self.calc_sspec(lamsteps=lamsteps)
            sspec = self.lamsspec
        else:
            if not hasattr(self, 'sspec'):
                self.calc_sspec()
            sspec = self.sspec
        if not eta:
            eta = self.eta

        ind = np.argmin(abs(self.tdel-delmax))
        sspec = sspec[startbin:ind, :]  # cut first N delay bins and at delmax
        tdel = self.tdel[startbin:ind]
        fdop = self.fdop
        maxfdop = maxnormfac*np.sqrt(tdel[-1]/eta)  # Maximum fdop for plot
        if maxfdop > max(fdop):
            maxfdop = max(fdop)
        nfdop = len(fdop[abs(fdop) <= maxfdop])  # Number of fdop bins to use
        fdopnew = np.linspace(-maxnormfac, maxnormfac, nfdop)  # norm fdop
        normSspec = []
        isspectot = np.zeros(np.shape(fdopnew))
        for ii in range(0, len(tdel)):
            itdel = tdel[ii]
            imaxfdop = maxnormfac*np.sqrt(itdel/eta)
            ifdop = fdop[abs(fdop) <= imaxfdop]/np.sqrt(itdel/eta)
            isspec = sspec[ii, abs(fdop) <= imaxfdop]  # take the iith row
            ind = np.argmin(abs(fdopnew))
            normline = np.interp(fdopnew, ifdop, isspec)
            normline[ind-cutmid:ind+cutmid+1] = np.nan  # ignore centre bins
            normSspec.append(normline)
            isspectot = np.add(isspectot, normline)
        isspecavg = isspectot/len(tdel)  # make average
        ind1 = np.argmin(abs(fdopnew-1)-2)
        # ind2 = np.argmin(abs(fdopnew+1)-2)
        # normfac = (abs(isspecavg[ind1])
        #           + abs(isspecavg[ind2]))/2  # mean power at theoretical arc
        isspecavg = isspecavg  # /normfac
        if isspecavg[ind1] < 0:
            isspecavg = isspecavg + 2  # make 1 instead of -1
        if plot:
            if scrunched:
                plt.plot(fdopnew, isspecavg)
                bottom, top = plt.ylim()
                plt.xlabel("Normalised fdop")
                plt.ylabel("Normalised log10(Power)")
                if plot_fit:
                    plt.plot([1, 1], [bottom, top], 'r', alpha=0.5)
                    plt.plot([-1, -1], [bottom, top], 'r', alpha=0.5)
                plt.ylim(bottom, top)
                plt.xlim(-maxnormfac, maxnormfac)
                plt.show()
                return
            if lamsteps:
                beta = np.divide(tdel, self.dlam)  # in m^-1
                plt.pcolormesh(fdopnew, beta, normSspec)
                plt.ylabel('Beta (m$^{-1}$)')
            else:
                plt.pcolormesh(fdopnew, tdel, normSspec)
                plt.ylabel("tdel (us)")
            bottom, top = plt.ylim()
            plt.xlabel("Normalised fdop")
            if plot_fit:
                plt.plot([1, 1], [bottom, top], 'r', alpha=0.5)
                plt.plot([-1, -1], [bottom, top], 'r', alpha=0.5)
            plt.ylim(bottom, top)
            plt.colorbar()
            plt.show()
        self.normsspecavg = isspecavg
        self.normsspec = np.array(normSspec)

    def get_scint_params(self, method="acf", plot=False, alpha=5/3):
        """
        Measure the scintillation timescale
            Method:
                acf - takes a 1D cut through the centre of the ACF
                sspec - measures timescale from the power spectrum
        """
        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'sspec'):
            self.calc_sspec()

        if method == 'acf':
            scint_model = scint_acf_model
            y_f_data = self.acf[int(self.nchan):, int(self.nsub)]
            x_f_data = self.df*np.linspace(0, len(y_f_data), len(y_f_data))
            y_t_data = self.acf[int(self.nchan), int(self.nsub):]
            x_t_data = self.dt*np.linspace(0, len(y_t_data), len(y_t_data))
        elif method == 'sspec':
            scint_model = scint_sspec_model
            y_f_data = np.sum(self.sspec, axis=0)
            x_f_data = self.df*np.linspace(0, len(y_f_data), len(y_f_data))
            y_t_data = np.sum(self.sspec, axis=1)
            x_t_data = self.dt*np.linspace(0, len(y_t_data), len(y_t_data))
            plt.plot(x_f_data, y_f_data)
            plt.show()

        nt = len(x_t_data)  # number of t-lag samples
        xdata = list(x_t_data) + list(x_f_data)  # concatenate x lists
        ydata = list(y_t_data) + list(y_f_data)  # y lists
        # Estimate amp and white noise level
        arr = self.acf
        arr = np.fft.ifftshift(arr)
        amp = arr[0][1]
        wn = arr[0][0] - amp
        # Estimate tau
        # estimate dnu
        # Set initial guess p0 = [tau, dnu, amp, wn, alpha, nt]
        p0 = [8000, 1000, amp, wn, 5/3, nt]  # educated guess
        lb = [0, 0, 0, 0, 0, nt]  # lower bounds
        ub = [np.inf, np.inf, np.inf, np.inf, np.inf, nt]  # upper bounds
        if alpha is None:  # Fit alpha
            p0 = p0[:-1]
            lb = lb[:-1]
            ub = ub[:-1]
            self.scintmodel, pcov = \
                curve_fit(lambda x, tau, dnu, amp, wn, alph:
                          scint_model(x, tau=tau, dnu=dnu, amp=amp,
                                      wn=wn, alpha=alph, nt=nt),
                          xdata, ydata, p0=p0, bounds=(lb, ub))
        else:  # Fix alpha. Default 5/3 is for Kolmogorov turbulence
            p0 = p0[:-2]
            lb = lb[:-2]
            ub = ub[:-2]
            self.scintmodel, pcov = \
                curve_fit(lambda x, tau, dnu, amp, wn:
                          scint_model(x, tau=tau, dnu=dnu, amp=amp,
                                      wn=wn, alpha=alpha, nt=nt),
                          xdata, ydata, p0=p0, bounds=(lb, ub))
        errors = []
        for i in range(len(self.scintmodel)):  # for each parameter
            errors.append(np.absolute(pcov[i][i])**0.5)
        self.scintmodel_errors = np.array(errors)
        self.tau = self.scintmodel[0]
        self.tauerr = self.scintmodel_errors[0]
        self.dnu = self.scintmodel[1]
        self.dnuerr = self.scintmodel_errors[1]
        if plot:
            plt.subplot(2, 1, 1)
            plt.plot(x_t_data, y_t_data)
            # plt.plot(x_t_data, scint_acf_model(x_t_data, self.scintmodel,
            #                                   nt=nt))
            plt.xlabel('Time lag (s)')
            plt.subplot(2, 1, 2)
            plt.plot(x_f_data, y_f_data)
            plt.xlabel('Frequency lag (MHz)')
            plt.show()

    def cut_dyn(self, tcuts=0, fcuts=0, plot=False, filename=None):
        """
        Cuts the dynamic spectrum into tcuts+1 segments in time and
                fcuts+1 segments in frequency
        """

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
        for ii in reversed(range(0, fcuts+1)):
            for jj in reversed(range(0, tcuts+1)):
                cutdyn[int(ii)][int(jj)][:][:] =\
                    self.dyn[int(ii*fnum):int((ii+1)*fnum),
                             int(jj*tnum):int((jj+1)*tnum)]
                input_dyn_x = self.times[int(jj*tnum):int((jj+1)*tnum)]
                input_dyn_y = self.freqs[int(ii*fnum):int((ii+1)*fnum)]
                input_sspec_x, input_sspec_y, cutsspec[int(ii)][int(jj)][:][:]\
                    = self.calc_sspec(input_dyn=cutdyn[int(ii)][int(jj)][:][:])
                cutacf[int(ii)][int(jj)][:][:] \
                    = self.calc_acf(input_dyn=cutdyn[int(ii)][int(jj)][:][:])
                if plot:
                    # Plot dynamic spectra
                    plt.figure(1)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_dyn(input_dyn=cutdyn[int(ii)][int(jj)][:][:],
                                  input_x=input_dyn_x/60, input_y=input_dyn_y)
                    plt.xlabel('t (mins)')
                    plt.ylabel('f (MHz)')

                    # Plot acf
                    plt.figure(2)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_acf(input_acf=cutacf[int(ii)][int(jj)][:][:],
                                  input_t=input_dyn_x,
                                  input_f=input_dyn_y)
                    plt.xlabel('t lag (mins)')
                    plt.ylabel('f lag (MHz)')

                    # Plot secondary spectra
                    plt.figure(3)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_sspec(input_sspec=cutsspec[int(ii)]
                                                        [int(jj)][:][:],
                                    input_x=input_sspec_x,
                                    input_y=input_sspec_y)
                    plt.xlabel('fdop (mHz)')
                    plt.ylabel('tdel (us)')
                    plotnum += 1
        if plot:
            plt.figure(1)
            filename_name = filename.split('.')[0]
            filename_extension = filename.split('.')[1]
            if filename is not None:
                plt.savefig(filename_name + '_dynspec.' + filename_extension,
                            figsize=(6,10), dpi=150)
            else:
                plt.show()
            plt.figure(2)
            if filename is not None:
                plt.savefig(filename_name + '_acf.' + filename_extension,
                            figsize=(6,10), dpi=150)
            else:
                plt.show()
            plt.figure(3)
            if filename is not None:
                plt.savefig(filename_name + '_sspec.' + filename_extension,
                            figsize=(6,10), dpi=150)
            else:
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
        array = self.dyn
        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        if linear:  # do linear interpolation
            # mask invalid values
            array = np.ma.masked_invalid(array)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]
            self.dyn = griddata((x1, y1), newarr.ravel(), (xx, yy),
                                method='linear')
        # Fill remainder with the mean
        meanval = np.mean(self.dyn[is_valid(self.dyn)])
        self.dyn[np.isnan(self.dyn)] = meanval

    def correct_band(self, time=False, lamsteps=False):
        """
        Correct for the bandpass
        """

        if lamsteps:
            if not self.lamsteps:
                self.scale_dyn()
            dyn = self.lamdyn
        else:
            dyn = self.dyn
        dyn[np.isnan(dyn)] = 0
        self.bandpass = np.mean(dyn, axis=1)
        # Make sure there are no zeros
        self.bandpass[self.bandpass == 0] = np.mean(self.bandpass)
        dyn = np.divide(dyn, np.reshape(self.bandpass,
                                        [len(self.bandpass), 1]))
        if time:
            timestructure = np.mean(dyn, axis=0)
            # Make sure there are no zeros
            timestructure[timestructure == 0] = np.mean(timestructure)
            dyn = np.divide(dyn, np.reshape(timestructure,
                                            [1, len(timestructure)]))
        if lamsteps:
            self.lamdyn = dyn
        else:
            self.dyn = dyn

    def calc_sspec(self, prewhite=True, plot=False, lamsteps=False,
                   input_dyn=None, input_x=None, input_y=None):
        """
        Calculate secondary spectrum
        """

        if input_dyn is None:  # use self dynamic spectrum
            if lamsteps:
                if not self.lamsteps:
                    self.scale_dyn()
                dyn = self.lamdyn
            else:
                dyn = self.dyn
        else:
            dyn = input_dyn  # use imput dynamic spectrum
        nf = np.shape(dyn)[0]
        nt = np.shape(dyn)[1]
        # find the right fft lengths for rows and columns
        nrfft = int(2**(np.ceil(np.log2(nf))+1))
        ncfft = int(2**(np.ceil(np.log2(nt))+1))
        if prewhite:
            simpw = convolve2d([[1, -1], [-1, 1]], dyn, mode='valid')
        else:
            simpw = dyn
        simpw = simpw - np.mean(simpw)
        simf = np.abs(np.fft.fft2(simpw, s=[nrfft, ncfft]))
        simf = np.multiply(simf, np.conj(simf))
        sec = np.fft.fftshift(simf)  # fftshift
        sec = sec[int(nrfft/2):][:]  # crop
        td = list(range(0, int(nrfft/2)))
        fd = list(range(int(-ncfft/2), int(ncfft/2)))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),
                          [len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),
                          [len(td)])  # in us

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
        if input_dyn is None:
            if lamsteps:
                # normalise and make db
                self.lamsspec = 10*np.log10(sec)
            else:
                # normalise and make db
                self.sspec = 10*np.log10(sec)  # /np.max(sec)

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps)
        else:
            return fdop, tdel, 10*np.log10(sec)

    def calc_acf(self, scale=False, input_dyn=None, plot=True):
        """
        Calculate autocovariance function
        """

        if input_dyn is None:
            # mean subtracted dynspec
            arr = self.dyn - np.mean(self.dyn[is_valid(self.dyn)])
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

    def crop_dyn(self, fmin=0, fmax=10000):
        """
        Crops dynamic spectrum in frequency to be between fmin and fmax
        """

        crop_array = np.array((self.freqs > fmin)*(self.freqs < fmax))
        self.dyn = self.dyn[crop_array, :]
        self.freqs = self.freqs[crop_array]
        self.nchan = len(self.freqs)
        self.bw = round(max(self.freqs) - min(self.freqs) + self.df, 2)
        self.freq = round(np.mean(self.freqs), 2)

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

    def scale_dyn(self, scale='lambda', factor=1):
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
            arin = self.dyn  # input array
            fbw = self.bw/self.freq  # fractional bandwidth
            nf, nt = np.shape(arin)
            # equal steps in fractional bandwidth
            feq = np.linspace(1 - fbw/2, 1 + fbw/2, nf)
            # all that matters is the ratio of frequencies
            #   so we normalize for convenience
            feq = feq/feq[0]
            dl = (1/feq[0] - 1/feq[1])  # first guess
            minl = 1/feq[-1]  # no need to multiply by c it will cancel out
            maxl = 1
            nl = ((maxl - minl)/dl)+1   # number of lameq samples
            # now take floor of nl to get integer and inflate dl a bit to match
            nl = np.floor(nl)
            dl = (maxl - minl)/(nl-1)  # a bit larger than our first guess
            lam = minl + np.arange(0, nl+1)*dl  # full lam array
            flam = np.divide(1, lam)  # full array of equal frequency samples
            # we can use linear interpolation provided fbw < 33%
            arout = np.zeros([len(lam), int(nt)])
            if fbw <= 1/3:
                for it in range(0, nt):
                    arout[:, it] = np.interp(flam, feq, arin[:, it])
            else:
                raise ValueError('for fbw > 0.33 need lsq interpolation')
                return
            self.lamdyn = arout
            # maximum lambda is minimum freq
            self.lam = lam*sc.c/np.min(self.freqs*1e6)
            self.dlam = abs(self.lam[1]-self.lam[0])

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
    Imports dynamic spectra in Bill Coles format, including simulated
        dynamic spectra produced with code from Coles et al. (2010)
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
