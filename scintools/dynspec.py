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
from scintools.scint_utils import is_valid, svd_model, interp_nan_2d,\
    centres_to_edges
from scipy.interpolate import griddata, interp1d, RectBivariateSpline
from scipy.signal import convolve2d, medfilt, savgol_filter
from scipy.io import loadmat
from lmfit import Parameters
try:
    from skimage.restoration import inpaint
    biharmonic = True
except Exception as e:
    print(e)
    print("skimage not found: cannot use biharmonic inpainting")
    biharmonic = False
try:
    import corner
except Exception as e:
    print(e)
    print("Corner.py not found: cannot plot mcmc results")


class Dynspec:

    def __init__(self, filename=None, dyn=None, verbose=True, process=False,
                 lamsteps=False):
        """
        Initialise a dynamic spectrum object by either reading from file
            or from existing object

        Parameters
        ----------
        filename : str, optional
            The path of the dynamic spectrum file. The default is None.
        dyn : Dynspec object, optional
            Dynamic spectrum object to load directly. The default is None.
        verbose : bool, optional
            Print all the things. The default is True.
        process : bool, optional
            Perform basic processing. The default is False.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.

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

        Parameters
        ----------
        other : Dynspec object
            Dynamic spectrum to be added.

        Returns
        -------
        Dynspec object
            The combined dynamic spectrum.

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

    def load_file(self, filename, verbose=True, process=False, lamsteps=False):
        """
        Load a dynamic spectrum from psrflux-format file

        Parameters
        ----------
        filename : str
            The path of the dynamic spectrum file.
        verbose : bool, optional
            Print all the things. The default is True.
        process : bool, optional
            Perform basic processing. The default is True.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.

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
                    if str.split(headline) != []:
                        if str.split(headline)[0] == 'MJD0:':
                            # MJD of start of obs
                            self.mjd = float(str.split(headline)[1])
        self.name = os.path.basename(filename)
        self.filename = filename  # full path
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
        # self.dyn_err = fluxerrs
        self.dyn_noise = np.nanmedian(fluxerrs)
        self.lamsteps = lamsteps
        if process:
            self.auto_processing(lamsteps=lamsteps)  # do automatic processing
        end = time.time()
        if verbose:
            print("...LOADED in {0} seconds\n".format(round(end-start, 2)))
            self.info()

    def write_file(self, filename=None, verbose=True, note=None):
        """
        Writes the dynspec object to psrflux-format file

        Parameters
        ----------
        filename : str
            The path of the dynamic spectrum file.
        verbose : bool, optional
            Print all the things. The default is True.
        note : str, optional
            Note to write at top of file. The default is None.

        """

        if filename is None:
            ext = self.filename.split('.')[-1]
            fname = '.'.join(self.filename.split('.')[0:-1]) + \
                '.processed.' + ext
        else:
            fname = filename
        # now write to file
        with open(fname, 'w') as fn:
            fn.write("# Scintools-modified dynamic spectrum " +
                     "in psrflux format\n")
            fn.write("# Created using write_file method in Dynspec class\n")
            if note is not None:
                fn.write("# Note: {0}\n".format(note))
            fn.write("# Original header begins below:\n")
            fn.write("#\n")
            for line in self.header:
                fn.write("# {} \n".format(line))

            for i in range(len(self.times)):
                ti = self.times[i]/60
                for j in range(len(self.freqs)):
                    fi = self.freqs[j]
                    di = self.dyn[j, i]
                    # di_err = self.dyn_err[j, i]
                    fn.write("{0} {1} {2} {3} {4} {5}\n".
                             format(i, j, ti, fi, di, 0)) #, # di_err))
        if verbose:
            print("Wrote dynamic spectrum file as {}".format(fname))

    def load_dyn_obj(self, dyn, verbose=True, process=True, lamsteps=False):
        """
        Load in a dynamic spectrum object of different type.

        Parameters
        ----------
        dyn : Dynspec object
            Dynamic spectrum object to load directly.
        verbose : bool, optional
            Print all the things. The default is True.
        process : bool, optional
            Perform basic processing. The default is True.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.

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

    def auto_processing(self, lamsteps=False):
        """
        Automatic processing of a Dynspec object, using common utilities

        Parameters
        ----------
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.

        """

        self.trim_edges()  # remove zeros on band edges
        self.refill()  # refill and zeroed regions with linear interpolation
        self.correct_dyn()  # correct by svd
        self.calc_acf()  # calculate the ACF
        if lamsteps:
            self.scale_dyn()
        self.calc_sspec(lamsteps=lamsteps)  # Calculate secondary spectrum

    def plot_dyn(self, lamsteps=False, input_dyn=None, filename=None,
                 input_x=None, input_y=None, trap=False, display=True,
                 figsize=(9, 9), dpi=200, title=None, velocity=False):
        """
        Plot the dynamic spectrum

        Parameters
        ----------
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        input_dyn : Dynspec object, optional
            Ignore the class-defined dynamic spectrum and use this input
            dynamic spectrum. The default is None.
        filename : str, optional
            The path at which to save the figure. The default is None.
        input_x : 1D array, optional
            `x`-axis of input dynamic spectrum. The default is None.
        input_y : 1D array, optional
            `y`-axis of input dynamic spectrum. The default is None.
        trap : bool, optional
            Trapezoidal scaling. The default is False.
        display : bool, optional
            Display the plot. The default is True.
        figsize : tuple, optional
            Size of the figure. The default is (9, 9).
        dpi : float, optional
            dpi of the figure. The default is 200.
        title : str, optional
            Figure title. The default is None.
        velocity : bool, optional
            Scale the dynamic spectrum using the velocity. The default is
            False.

        """

        if input_dyn is None:
            if lamsteps:
                if velocity:
                    if not hasattr(self, 'vlamdyn'):
                        self.scale_dyn(scale='lambda-velocity')
                    dyn = self.vlamdyn
                else:
                    if not hasattr(self, 'lamdyn'):
                        self.scale_dyn()
                    dyn = self.lamdyn
            elif velocity:
                if not hasattr(self, 'vdyn'):
                    self.scale_dyn(scale='velocity')
                dyn = self.vdyn
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
                tedges = centres_to_edges(self.times/60)
                lamedges = centres_to_edges(self.lam)
                plt.pcolormesh(tedges, lamedges, dyn,
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                plt.ylabel('Wavelength (m)')
            else:
                tedges = centres_to_edges(self.times/60)
                fedges = centres_to_edges(self.freqs)
                plt.pcolormesh(tedges, fedges, dyn,
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                plt.ylabel('Frequency (MHz)')
            plt.xlabel('Time (mins)')
            # plt.colorbar()  # arbitrary units
        else:
            xedges = centres_to_edges(input_x)
            yedges = centres_to_edges(input_y)
            plt.pcolormesh(xedges, yedges, dyn, vmin=vmin, vmax=vmax,
                           linewidth=0, rasterized=True)

        if title is not None:
            plt.title(title)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                        pad_inches=0.1)
            if display:
                plt.show()
            plt.close()
        elif input_dyn is None and display:
            plt.show()

    def plot_acf(self, method='acf1d', alpha=5/3, contour=False, filename=None,
                 input_acf=None, input_t=None, input_f=None, nscale=4,
                 mcmc=False, display=True, crop=False, tlim=None, flim=None,
                 figsize=(9, 9), verbose=False, dpi=200):
        """
        Plot the autocorrelation function

        Parameters
        ----------
        method : str {'acf1d', 'acf2d_approx', 'acf2d', 'sspec', 'nofit'},
        optional
            Fitting method for determining scintillation scales:

                ``acf1d``
                    Fit to central 1D cuts in time and frequency.
                ``acf2d_approx``
                    Fit an approximate 2D model.
                ``acf2d``
                    Fit an analytical 2D model.
                ``sspec``
                    Secondary spectrum method.
                ``nofit``
                    Don't perform fit.
        alpha : float, optional
            Structure function index. The default is 5/3, corresponding to a
            Kolmogorov spectrum.
        contour : bool, optional
            Plot contours. The default is False.
        filename : str, optional
            The path at which to save the figure. The default is None.
        input_acf : Dynspec object, optional
            Ignore the class-defined ACF and use this input ACF. The default is
            None.
        input_t : 1D array, optional
            `t`-axis of input ACF. The default is None.
        input_f : 1D array, optional
            `f`-axis of input ACF. The default is None.
        nscale : float, optional
            The number of scintillation scales to plot out to. The default is
            4.
        mcmc : bool, optional
            Use MCMC to fit for scintillation scales. The default is False.
        display : bool, optional
            Display the plot. The default is True.
        crop : bool, optional
            Crop the figure to the specified limits ``tlim`` and ``flim``. The
            default is False.
        tlim : float, optional
            Value of time lag to plot out to. The default is None.
        flim : float, optional
            Value of frequency lag to plot out to. The default is None.
        figsize : tuple, optional
            Size of the figure. The default is (9, 9).
        verbose : bool, optional
            Print all the things. The default is True.
        dpi : float, optional
            dpi of the figure. The default is 200.

        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'tau') and input_acf is None:
            try:
                self.get_scint_params(method=method, alpha=alpha, mcmc=mcmc,
                                      verbose=verbose)
            except Exception as e:
                print(e)
                print("Could not determine scintillation scales for plot")

        if input_acf is None:
            arr = self.acf
            tspan = self.tobs
            fspan = self.bw
        else:
            arr = input_acf
            tspan = max(input_t) - min(input_t)
            fspan = max(input_f) - min(input_f)
        arr = np.fft.ifftshift(arr)
        # subtract the white noise spike
        wn = arr[0][0] - max([arr[1][0], arr[0][1]])
        arr[0][0] = arr[0][0] - wn  # Set the noise spike to zero for plotting
        arr = np.fft.fftshift(arr)

        t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
        f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

        if crop or (tlim is not None):
            if tlim is None:
                # Set limits automatically
                tlim = nscale * self.tau / 60
                flim = nscale * self.dnu
            if tlim > self.tobs / 60:
                tlim = self.tobs / 60
            if flim > self.bw:
                flim = self.bw

            t_inds = np.argwhere(np.abs(t_delays) <= tlim).squeeze()
            f_inds = np.argwhere(np.abs(f_shifts) <= flim).squeeze()
            t_delays = t_delays[t_inds]
            f_shifts = f_shifts[f_inds]

            arr = arr[f_inds, :]
            arr = arr[:, t_inds]

        if input_acf is None:  # Also plot scintillation scales axes

            fig, ax1 = plt.subplots(figsize=figsize)
            if contour:
                ax1.contourf(t_delays, f_shifts, arr)
            else:
                ax1.pcolormesh(t_delays, f_shifts, arr, linewidth=0,
                               rasterized=True, shading='auto')
            ax1.set_ylabel(r'Frequency shift, $\Delta\nu$ (MHz)')
            ax1.set_xlabel(r'Time lag, $\tau$ (mins)')
            miny, maxy = ax1.get_ylim()
            if hasattr(self, 'tau'):
                ax2 = ax1.twinx()
                ax2.set_ylim(miny/self.dnu, maxy/self.dnu)
                ax2.set_ylabel(r'$\Delta\nu$ / ($\Delta\nu_d = {0}\,$MHz)'.
                               format(round(self.dnu, 2)))
                ax3 = ax1.twiny()
                minx, maxx = ax1.get_xlim()
                ax3.set_xlim(minx/(self.tau/60), maxx/(self.tau/60))
                ax3.set_xlabel(r'$\tau$/($\tau_d={0}\,$min)'.format(round(
                                                             self.tau/60, 2)))
        else:  # just plot acf without scales
            if contour:
                plt.contourf(t_delays, f_shifts, arr)
            else:
                tedges = centres_to_edges(t_delays)
                fedges = centres_to_edges(f_shifts)
                plt.pcolormesh(tedges, fedges, arr, linewidth=0,
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
                   overplot_curvature=None, dpi=200, velocity=False):
        """
        Plot the secondary spectrum

        Parameters
        ----------
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        input_sspec : Dynspec object, optional
            Ignore the class-defined secondary spectrum and use this input
            spectrum. The default is None.
        filename : str, optional
            The path at which to save the figure. The default is None.
        input_x : 1D array, optional
            `x`-axis of input secondary spectrum. The default is None.
        input_y : 1D array, optional
            `y`-axis of input secondary spectrum. The default is None.
        trap : bool, optional
            Trapezoidal scaling. The default is False.
        prewhite : bool, optional
            Perform pre-whitening using the first-difference method, then
            post-darken. The default is False.
        plotarc : bool, optional
            Plot the arc fit. The default is False.
        maxfdop : float, optional
            Maximum fdop to plot out to. The default is np.inf.
        delmax : float, optional
            Maximum delay to plot out to. The default is None.
        ref_freq : float, optional
            Reference frequency. The default is 1400.
        cutmid : int, optional
            Number of columns around fdop=0 to set to nan. The default is 0.
        startbin : int, optional
            Number of rows from delay=0 to set to nan. The default is 0.
        display : bool, optional
            Display the plot. The default is True.
        colorbar : bool, optional
            Display colorbar. The default is True.
        title : str, optional
            Figure title. The default is None.
        figsize : tuple, optional
            Size of the figure. The default is (9, 9).
        subtract_artefacts : bool, optional
            Subtract delay response to try to remove artefacts. The default is
            False.
        overplot_curvature : float, optional
            Plot parabola with this curvature. The default is None.
        dpi : float, optional
            dpi of the figure. The default is 200.
        velocity : bool, optional
            Scale the dynamic spectrum using the velocity. The default is
            False.

        """

        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                if velocity:
                    if not hasattr(self, 'vlamsspec'):
                        self.calc_sspec(lamsteps=lamsteps, velocity=velocity,
                                        prewhite=prewhite)
                    sspec = cp(self.vlamsspec)
                else:
                    sspec = cp(self.lamsspec)
            elif velocity:
                sspec = cp(self.vsspec)
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
                xedges = centres_to_edges(xplot)
                betaedges = centres_to_edges(self.beta[:ind])
                plt.pcolormesh(xedges, betaedges, sspec[:ind, :],
                               vmin=vmin, vmax=vmax, linewidth=0,
                               rasterized=True, shading='auto')
                plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
            else:
                xedges = centres_to_edges(xplot)
                tdeledges = centres_to_edges(self.tdel[:ind])
                plt.pcolormesh(xedges, tdeledges, sspec[:ind, :],
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
            xedges = centres_to_edges(xplot)
            yedges = centres_to_edges(input_y)
            plt.pcolormesh(xedges, yedges, sspec, vmin=vmin, vmax=vmax,
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

    def plot_scattered_image(self, display=True, plot_log=True, colorbar=True,
                             title=None, input_scattered_image=None,
                             input_fdop=None, lamsteps=False, trap=False,
                             clean=True, use_angle=False, use_spatial=False,
                             s=None, veff=None, d=None, filename=None,
                             dpi=200):
        """
        Plot the scattered image

        Parameters
        ----------
        display : bool, optional
            Display the plot. The default is True.
        plot_log : bool, optional
            Plot brightness on a logarithmic scale. The default is True.
        colorbar : bool, optional
            Display colorbar alongside plot. The default is True.
        title : str, optional
            Figure title. The default is None.
        input_scattered_image : array_like, optional
            Ignore the class-defined scattered image and use this input
            image. The default is None.
        input_fdop : 1D array, optional
            fdop axis of the corresponding secondary spectrum. The default is
            None.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        trap : bool, optional
            Trapezoidal scaling. The default is False.
        clean : bool, optional
            Fill infs and extremely small pixel values. The default is True.
        use_angle : bool, optional
            Use angular axes in plot. The default is False.
        use_spatial : bool, optional
            Use spatial axes in plot. The default is False.
        s : float in range [0,1], optional
            Fractional screen distance. The default is None.
        veff : float, optional
            Magnitude of the effective velocity. The default is None.
        d : float, optional
            Pulsar distance. The default is None.
        filename : str, optional
            The path at which to save the figure. The default is None.
        dpi : float, optional
            dpi of the figure. The default is 200.

        """

        c = 299792458.0  # m/s
        if input_scattered_image is None:
            if not hasattr(self, 'scattered_image'):
                self.calc_scattered_image(lamsteps=lamsteps, trap=trap,
                                          clean=clean)
            scat_im = self.scattered_image
            xyaxes = self.scattered_image_ax

        else:
            scat_im = input_scattered_image
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

        xyedges = centres_to_edges(xyaxes)
        plt.pcolormesh(xyedges, xyedges, scat_im, vmin=vmin, vmax=vmax,
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

        return

    def fit_arc(self, asymm=False, plot=False, delmax=None, numsteps=1e4,
                startbin=3, cutmid=3, lamsteps=False, etamax=None, etamin=None,
                low_power_diff=-1, high_power_diff=-0.5, ref_freq=1400,
                constraint=[0, np.inf], nsmooth=5, efac=1, filename=None,
                noise_error=True, display=True, figN=None, log_parabola=False,
                logsteps=False, plot_spec=False, fit_spectrum=False,
                subtract_artefacts=False, figsize=(9, 9), dpi=200,
                velocity=False, weighted=False):
        """
        Find the arc curvature with maximum power along it
            constraint: Only search for peaks between constraint[0] and
                constraint[1]

        Parameters
        ----------
        asymm : bool, optional
            Fit to each side of the spectrum separately. The default is False.
        plot : bool, optional
            Plot the curvature fit. The default is False.
        delmax : float, optional
            tdel at which to crop the secondary spectrum. The default is None.
        numsteps : int, optional
            Number of steps in eta to use in fit. The default is 1e4.
        startbin : int, optional
            Number of rows from delay=0 to set to nan. The default is 3.
        cutmid : int, optional
            Number of columns around fdop=0 to set to nan. The default is 3.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        etamax : float, optional
            Maximum allowed curvature. The default is None.
        etamin : float, optional
            Minimum allowed curvature. The default is None.
        low_power_diff : float < 0, optional
            Fit parabolic template to the part of the normalized fdop profile
            with power greater than the sum of the peak power and this value on
            the low-curvature side of the peak. The default is -1.
        high_power_diff : float < 0, optional
            Fit parabolic template to the part of the normalized fdop profile
            with power greater than the sum of the peak power and this value on
            the high-curvature side of the peak. The default is -0.5.
        ref_freq : float, optional
            Reference frequency. The default is 1400.
        constraint : array_like, optional
            Search for peaks with curvature between constraint[0] and
            constraint[1]. The default is [0, np.inf].
        nsmooth : int, optional
            The length of the smoothing filter window. Must be a positive odd
            integer. The default is 5.
        efac : float, optional
            Factor by which to multiply the secondary spectrum noise. The
            default is 1.
        filename : str, optional
            The path at which to save the figure. The default is None.
        noise_error : bool, optional
            Determine the noise-based error in the curvature. The default is
            True.
        display : bool, optional
            Display the plot. The default is True.
        figN : int, optional
            Figure identifier. The default is None.
        log_parabola : bool, optional
            Fit parabolic template to profile peak using log curvature. The
            default is False.
        logsteps : bool, optional
            Use equal steps in logspace for normalized fdop. The default is
            False.
        plot_spec : bool, optional
            Plot delay-scrunched power profile. The default is False.
        fit_spectrum : bool, optional
            Fit a model to the power spectrum. The default is False.
        subtract_artefacts : bool, optional
            Subtract delay response to try to remove artefacts. The default is
            False.
        figsize : tuple, optional
            Size of the figure. The default is (9, 9).
        dpi : float, optional
            dpi of the figure. The default is 200.
        velocity : bool, optional
            Scale the dynamic spectrum using the velocity. The default is
            False.
        weighted : bool, optional
            Weighted average when computing the delay-scrunched power profile.
            The default is True.

        Raises
        ------
        ValueError
            If fit returns a forward parabola.

        """

        if not hasattr(self, 'tdel'):
            self.calc_sspec()
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2  # adjust for frequency

        if lamsteps:
            if velocity:
                if not hasattr(self, 'vlamsspec'):
                    self.calc_sspec(lamsteps=lamsteps, velocity=velocity)
                sspec = np.array(cp(self.vlamsspec))

            else:
                if not hasattr(self, 'lamsspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = np.array(cp(self.lamsspec))
            yaxis = cp(self.beta)
            ind = np.argmin(abs(self.tdel-delmax))
            ymax = self.beta[ind]  # cut beta at equivalent value to delmax
        else:
            if velocity:
                if not hasattr(self, 'vsspec'):
                    self.calc_sspec(velocity=velocity)
                sspec = np.array(cp(self.vsspec))
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
                            subtract_artefacts=subtract_artefacts,
                            velocity=velocity, weighted=weighted)
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
                   maxnormfac=5, minnormfac=0, cutmid=0, lamsteps=True,
                   scrunched=True, plot_fit=True, ref_freq=1400, velocity=False,
                   numsteps=None,  filename=None, display=True, weighted=True,
                   unscrunched=True, logsteps=False, powerspec=True,
                   interp_nan=False, fit_spectrum=False, powerspec_cut=False,
                   figsize=(9, 9), subtract_artefacts=False, dpi=200):
        """
        Normalise fdop axis using arc curvature

        Parameters
        ----------
        eta : float, optional
            The arc curvature. The default is None.
        delmax : float, optional
            tdel at which to crop the secondary spectrum. The default is None.
        plot : bool, optional
            Plot delay-scrunched power profile. The default is False.
        startbin : int, optional
            Number of rows from delay=0 to set to nan. The default is 1.
        maxnormfac : float, optional
            Maximum normalized fdop. The default is 5.
        minnormfac : float, optional
            Minimum normalized fdop. The default is 0.
        cutmid : int, optional
            Number of columns around fdop=0 to set to nan. The default is 0.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        scrunched : bool, optional
            Plot delay-scrunched power profile. The default is True.
        plot_fit : bool, optional
            Mark the location of the curvature fit (normalized fdop = 1). The
            default is True.
        ref_freq : float, optional
            Reference frequency. The default is 1400.
        velocity : bool, optional
            Scale the dynamic spectrum using the velocity. The default is True.
        numsteps : int, optional
            Number of steps in eta to use in fit. The default is None.
        filename : str, optional
            The path at which to save the figure. The default is None.
        display : bool, optional
            Display the plot. The default is True.
        weighted : bool, optional
            Weighted average when computing the delay-scrunched power profile.
            The default is True.
        unscrunched : bool, optional
            Plot 2D normalised secondary spectrum. The default is True.
        logsteps : bool, optional
            Use equal steps in logspace for normalized fdop. The default is
            False.
        powerspec : bool, optional
            Plot the power spectrum. The default is True.
        interp_nan : bool, optional
            Interpolate NaN values. The default is False.
        fit_spectrum : bool, optional
            Fit a model to the power spectrum. The default is False.
        powerspec_cut : bool, optional
            Cut the normalized secondary spectrum where the power spectrum
            model exceeds twice the white noise before delay-scrunching. The
            default is False.
        figsize : tuple, optional
            Size of the figure. The default is (9, 9).
        subtract_artefacts : bool, optional
            Subtract delay response to try to remove artefacts. The default is
            False.
        dpi : float, optional
            dpi of the figure. The default is 200.

        """

        # Maximum value delay axis (us @ ref_freq)
        delmax = np.max(self.tdel) if delmax is None else \
            delmax*(ref_freq/self.freq)**2

        # Set up data based on whether we're in lamsteps or not
        if lamsteps:
            yaxis = cp(self.beta)
            if velocity:
                if not hasattr(self, 'vlamsspec'):
                    self.calc_sspec(lamsteps=lamsteps, velocity=velocity)
                sspec = cp(self.vlamsspec)
            else:
                if not hasattr(self, 'lamsspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = cp(self.lamsspec)
            if not hasattr(self, 'betaeta') and eta is None:
                self.fit_arc(lamsteps=lamsteps, delmax=delmax, plot=plot,
                             startbin=startbin, velocity=velocity)
        elif velocity:
            if not hasattr(self, 'vsspec'):
                self.calc_sspec(velocity=velocity)
            sspec = cp(self.vsspec)
            yaxis = cp(self.tdel)
            if not hasattr(self, 'eta') and eta is None:
                self.fit_arc(lamsteps=lamsteps, delmax=delmax, plot=plot,
                             startbin=startbin, velocity=velocity)
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

        # set levels for plotting
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
            fdoplin = np.abs(np.linspace(-maxnormfac, maxnormfac, int(nfdop)))
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
        if weighted:
            self.weights = 10*np.log10(arc_spectrum)
        else:
            self.weights = np.ones(np.shape(arc_spectrum))

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
                    filename_name = ''.join(filename.split('.')[0:-1])
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
                fdopedges = centres_to_edges(fdopnew)
                tdeledges = centres_to_edges(tdel)
                plt.pcolormesh(fdopedges, tdeledges, np.ma.filled(normSspec),
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
                    filename_name = ''.join(filename.split('.')[0:-1])
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
                     filename=None, nscale=0.5, nscaleplot=2, nmin=5, dpi=200,
                     method='acf1d', tmaxplot=None, fmaxplot=None):
        """
        Estimates the tilt in the ACF, which is proportional to the phase
            gradient parallel to Veff

        Parameters
        ----------
        plot : bool, optional
            Plot the fit. The default is False.
        tmax : float, optional
            Maximum time lag. The default is None.
        fmax : float, optional
            Maximum frequency lag. The default is None.
        display : bool, optional
            Display the plot. The default is True.
        filename : str, optional
            The path at which to save the figure. The default is None.
        nscale : float, optional
            Maximum time and frequency lag in units of the scintillation
            timescale and decorrelation bandwidth, respectively. The default is
            0.5.
        nscaleplot : float, optional
            Number of scintillation timescales and decorrelation bandwidths to
            plot out to. The default is 2.
        nmin : int, optional
            Minimum number of sub-integrations for use as the maximum time lag.
            The default is 5.
        dpi : float, optional
            dpi of the figure. The default is 200.
        method : str {'acf1d', 'acf2d_approx', 'acf2d', 'sspec', 'nofit'},
        optional
            Fitting method for determining scintillation scales:

                ``acf1d``
                    Fit to central 1D cuts in time and frequency.
                ``acf2d_approx``
                    Fit an approximate 2D model.
                ``acf2d``
                    Fit an analytical 2D model.
                ``sspec``
                    Secondary spectrum method.
                ``nofit``
                    Don't perform fit.
        tmaxplot : float, optional
            Maximum time lag to plot out to. The default is None.
        fmaxplot : float, optional
            Maximum time lag to plot out to. The default is None.

        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'dnu'):
            self.get_scint_params(method=method)

        if tmax is None:
            tmax = nscale*self.tau/60
        else:
            tmax = tmax
        if fmax is None:
            fmax = nscale*self.dnu
        else:
            fmax = fmax

        if tmaxplot is None:
            tmaxplot = tmax*4
        else:
            tmaxplot = tmax
        if fmaxplot is None:
            fmaxplot = fmax*4
        else:
            fmaxplot = fmax

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
        peak_array = np.array(peak_array).squeeze()
        y_array = np.array(y_array).squeeze()
        peakerr_array = np.array(peakerr_array).squeeze()

        # Now do a weighted fit of a straight line to the peaks
        params, pcov = np.polyfit(peak_array, y_array, 1, cov=True,
                                  w=1/peakerr_array)
        yfit = params[0]*peak_array + params[1]  # y values
        xfit = (y_array - params[1])/params[0]

        # Get parameter errors
        errors = []
        for i in range(len(params)):  # for each parameter
            errors.append(np.absolute(pcov[i][i])**0.5)
        errors = np.array(errors).squeeze()
        res = np.array(peak_array - xfit).squeeze()
        reduced_chi_sq = np.sum(res**2/peakerr_array**2)/(len(xfit) - 2)
        errors *= np.sqrt(reduced_chi_sq)
        peakerr_array *= np.sqrt(reduced_chi_sq)

        self.acf_tilt = 1/(float(params[0].squeeze()))  # make min/MHz
        acf_tilt_err = float(errors[0].squeeze()) * \
            1/float(params[0].squeeze())**2

        # Compute finite scintle error
        N = (1 + 0.2*self.bw/(self.dnu)) * \
            (1 + 0.2*self.tobs/(self.tau*np.log(2)))  # 2*half power
        fse_tau = self.tau/(2*np.sqrt(N))
        fse_dnu = self.dnu/(2*np.sqrt(N))

        fse_tilt = self.acf_tilt * np.sqrt((fse_dnu/self.dnu)**2 +
                                           (fse_tau/self.tau)**2)

        self.fse_tilt = fse_tilt
        self.acf_tilt_err = acf_tilt_err

        if plot:
            plt.errorbar(peak_array, y_array,
                         xerr=np.array(peakerr_array).squeeze(),
                         marker='.')
            plt.plot(peak_array, yfit)
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')
            plt.title('Peak measurements, and weighted fit')
            if filename is not None:
                filename_name = ''.join(filename.split('.')[0:-1])
                filename_extension = filename.split('.')[-1]
                plt.savefig(filename_name + '_tilt_fit.' + filename_extension,
                            dpi=dpi, bbox_inches='tight',
                            pad_inches=0.1)
                if display:
                    plt.show()
                plt.close()
            elif display:
                plt.show()

            tedges = centres_to_edges(t_delays)
            fedges = centres_to_edges(f_shifts)
            plt.pcolormesh(tedges, fedges, acf, linewidth=0,
                           rasterized=True, shading='auto')
            plt.plot(peak_array, y_array, 'r', alpha=0.5)
            plt.plot(peak_array, yfit, 'k', alpha=0.5)
            yl = plt.ylim()
            if yl[1] > nscaleplot*self.dnu:
                plt.ylim([-nscaleplot*self.dnu, nscaleplot*self.dnu])
            if yl[1] > fmaxplot:
                plt.ylim([-fmaxplot, fmaxplot])
            xl = plt.xlim()
            if xl[1] > nscaleplot*self.tau:
                plt.xlim([-nscaleplot*self.tau, nscaleplot*self.tau])
            if xl[1] > tmaxplot:
                plt.xlim([-tmaxplot, tmaxplot])
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')
            err = np.sqrt(self.acf_tilt_err**2 + self.fse_tilt**2)
            plt.title(r'Tilt = {0} $\pm$ {1} (min/MHz)'.format(
                    round(self.acf_tilt, 3), round(err, 3)))
            if filename is not None:
                filename_name = ''.join(filename.split('.')[0:-1])
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
                         mcmc=False, full_frame=False, nscale=5,
                         nwalkers=100, steps=1000, burn=0.2, nitr=1,
                         lnsigma=True, verbose=False, progress=True,
                         display=True, filename=None, dpi=200,
                         nan_policy='raise', weighted=True, workers=1,
                         tau_vary_2d=True, tau_input=None):
        """
        Measure the scintillation timescale

        Method:
            nofit - Using the 0.5 and 1/e levels in the ACF to define the
                scale, and assumes the finite-scintle error dominates
                the uncertainty
            acf1d - takes a 1D cut through the centre of the ACF and fits
                an exponential in frequency and a Gaussian-like curve
                in time
            sspec - (N/A) measures scintillation scales from the secondary
               spectrum, using the Fourier transform of the 1d models
            acf2d_approx - uses an analytic approximation to the ACF
                including a phase gradient (a shear to the ACF)

        Parameters
        ----------
        method : str {'acf1d', 'acf2d_approx', 'acf2d', 'sspec', 'nofit'},
        optional
            Fitting method for determining scintillation scales:

                ``acf1d``
                    Fit to central 1D cuts in time and frequency.
                ``acf2d_approx``
                    Fit an approximate 2D model.
                ``acf2d``
                    Fit an analytical 2D model.
                ``sspec``
                    Secondary spectrum method.
                ``nofit``
                    Don't perform fit.
        plot : bool, optional
            Plot the data and model fit. The default is False.
        alpha : float, optional
            Structure function index. The default is 5/3, corresponding to a
            Kolmogorov spectrum.
        mcmc : bool, optional
            Use MCMC for the fit. The default is False.
        full_frame : bool, optional
            Use the full ACF in the fit. The default is False.
        nscale : float, optional
            Number of approximate scintillation timescales and decorrelation
            bandwidths at which to crop the ACF. The default is 5.
        nwalkers : int, optional
            Number of walkers to use for MCMC fit. The default is 100.
        steps : int, optional
            Number of samples to use for MCMC fit. The default is 1000.
        burn : float in [0,1], optional
            Fraction of samples to discard. The default is 0.2.
        nitr : int, optional
            Number of least-squares fits of the analytical 2D model to perform.
            Samples a new set of initial guesses each time. The default is 1.
        lnsigma : bool, optional
            If not returning weighted residuals during fitting. The default is
            True.
        verbose : bool, optional
            Print all the things. The default is True.
        progress : bool, optional
            Display a progress bar. The default is True.
        display : bool, optional
            Display the plot. The default is True.
        filename : str, optional
            The path at which to save the figure. The default is None.
        dpi : float, optional
            dpi of the figure. The default is 200.
        nan_policy : str, optional
            Response to NaN values being returned by the model. The default is
            'raise'.
        weighted : bool, optional
            Weight the residuals. The default is True.
        workers : Pool-like or int, optional
            For parallelized MCMC sampling using a Pool-like object or a
            multiprocessing pool with the specified number of processes spawned
            internally. The default is 1.
        tau_vary_2d : bool, optional
            Allow tau to vary when performing 2D fit. The default is True.
        tau_input : float, optional
            Value for tau to use in 2D fit. If left unspecified, the value from
            a 1D fit will be used. The default is None.

        Returns
        -------
        results : MinimizerResult
            Results object containing the parameter fit values and
            goodness-of-fit statistics.

        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'sspec') and 'sspec' in method:
            self.calc_sspec()

        nf, nt = np.shape(self.acf)
        ydata_f = self.acf[int(nf/2):, int(nt/2)]
        xdata_f = self.df * np.linspace(0, len(ydata_f)-1, len(ydata_f))
        ydata_t = self.acf[int(nf/2), int(nt/2):]
        xdata_t = self.dt * np.linspace(0, len(ydata_t)-1, len(ydata_t))

        # Get initial parameter values for 1d fit
        # Estimate amp and white noise level
        wn = min([ydata_f[0]-ydata_f[1], ydata_t[0]-ydata_t[1]])
        amp = max([ydata_f[0] - wn, ydata_t[0] - wn])
        # Estimate tau for initial guess. Closest index to 1/e power
        if np.argwhere(ydata_t < amp/np.e).squeeze().size == 0:
            if ydata_t[1] < 0:
                tau = self.dt
            else:
                tau = self.tobs
        else:
            tau = xdata_t[np.argwhere(ydata_t < amp/np.e).squeeze()[0]]
        # Estimate dnu for initial guess. Closest index to 1/2 power
        if np.argwhere(ydata_f < amp/2).squeeze().size == 0:
            if ydata_f[1] < 0:
                dnu = self.df
            else:
                dnu = self.bw
        else:
            dnu = xdata_f[np.argwhere(ydata_f < amp/2).squeeze()[0]]

        # crop arrays to nscale number of scales, or 5 samples
        if not full_frame:
            if nscale*tau <= 5*self.dt or nscale*dnu <= 5*self.df:
                t_inds = np.argwhere(xdata_t <= 5*self.dt).squeeze()
                f_inds = np.argwhere(xdata_f <= 5*self.df).squeeze()
            else:
                t_inds = np.argwhere(xdata_t <= nscale*tau).squeeze()
                f_inds = np.argwhere(xdata_f <= nscale*dnu).squeeze()
            xdata_t = xdata_t[t_inds]
            ydata_t = ydata_t[t_inds]
            xdata_f = xdata_f[f_inds]
            ydata_f = ydata_f[f_inds]

        # Save values determined without fitting: over-write if improved
        self.tau = tau
        self.dnu = dnu
        self.amp = amp
        self.wn = wn
        # Estimated number of scintles
        tau_half = xdata_t[np.argmin(abs(ydata_t - amp/2))]  # half power
        if tau_half < self.dt:
            tau_half = self.dt
        elif tau_half > self.tobs:
            tau_half = self.tobs
        nscint = (1 + 0.2*self.bw/(self.dnu)) * \
            (1 + 0.2*self.tobs/(tau_half))
        # Estimated errors
        self.dnuerr = dnu / np.sqrt(nscint)
        self.tauerr = tau / np.sqrt(nscint)
        self.amperr = amp / np.sqrt(nscint)
        self.wnerr = wn / np.sqrt(nscint)
        self.tscat = 1/(2*np.pi*self.dnu)  # scattering timescale
        self.nscint = nscint
        self.scint_param_method = 'nofit'

        mean = np.mean(self.dyn[is_valid(self.dyn) * (self.dyn != 0)])
        flux_var_est = mean**2
        flux_var = np.var(self.dyn[is_valid(self.dyn) * (self.dyn != 0)])
        # Estimate of scint bandwidth
        self.dnu_est = self.df * (flux_var/flux_var_est - 1)
        if self.dnu_est < 0:
            self.dnu_est = 0
        self.dnu_esterr = self.dnu_est / np.sqrt(nscint)
        if self.dnu_est > 0:
            self.tscat_est = 1/(2*np.pi*self.dnu_est)  # scattering timescale
        else:
            self.tscat_est = 0
        self.modulation_index = np.sqrt(flux_var)/mean

        if method == 'nofit':  # Don't want to try fitting, then just exit
            return

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
        if alpha is None:
            params.add('alpha', value=5/3, vary=True,
                       min=-np.inf, max=np.inf)
        else:
            params.add('alpha', value=alpha, vary=False)
        params.add('nt', value=nt, vary=False)
        params.add('nf', value=nf, vary=False)

        # Create weights array
        t_errors = 2/np.pi * np.arctan(xdata_t / tau) / \
            np.sqrt(self.nsub)
        t_errors[t_errors == 0] = 1e-3
        f_errors = 2/np.pi * np.arctan(xdata_f / dnu) / \
            np.sqrt(self.nchan)
        f_errors[f_errors == 0] = 1e-3
        if weighted:
            weights_t = 1/t_errors
            weights_f = 1/f_errors
        else:
            weights_t = None
            weights_f = None

        if method == 'acf1d' or method == 'acf2d_approx' or method == 'acf2d':
            if verbose:
                if method == 'acf2d_approx' or method == 'acf2d':
                    print("\nInitialising model with 1D fit")
                else:
                    print("\nPerforming least-squares fit to 1D ACF model")
            nfit = 4
            # max_nfev = 2000 * (nfit + 1)  # lmfit default
            max_nfev = 10000 * (nfit + 1)
            results = fitter(scint_acf_model, params,
                             ((xdata_t, xdata_f), (ydata_t, ydata_f),
                              (weights_t, weights_f)), max_nfev=max_nfev,
                             nan_policy=nan_policy)

        # overwrite initial value if successful:
        if results.params['dnu'].stderr is not None:
            params['tau'].value = results.params['tau'].value
            params['dnu'].value = results.params['dnu'].value
            params['wn'].value = results.params['wn'].value
            params['amp'].value = results.params['amp'].value

        if method == 'acf2d_approx' or method == 'acf2d':

            params['tau'].vary = tau_vary_2d
            if tau_input is not None:
                params['tau'].value = tau_input

            tticks = np.linspace(-self.tobs, self.tobs, nt + 1)[:-1]
            fticks = np.linspace(-self.bw, self.bw, nf + 1)[:-1]

            # Create weights array
            t_errors_2d = 2/np.pi * np.arctan(np.abs(tticks) / tau) / \
                np.sqrt(self.nsub)
            t_errors_2d[t_errors_2d == 0] = 1e-3
            f_errors_2d = 2/np.pi * np.arctan(np.abs(fticks) / dnu) / \
                np.sqrt(self.nchan)
            f_errors_2d[f_errors_2d == 0] = 1e-3

            weights_2d = np.ones(np.shape(self.acf))
            if weighted:
                weights_2d = weights_2d / np.transpose([f_errors_2d])
                weights_2d = weights_2d / [t_errors_2d]

            wn_loc = np.unravel_index(np.argmax(self.acf, axis=None),
                                      self.acf.shape)

            fleft = wn_loc[0]
            fright = nf - wn_loc[0] - 1
            fmin = wn_loc[0] - min(fleft, fright)
            fmax = wn_loc[0] + min(fleft, fright) + 1

            tleft = wn_loc[1]
            tright = nt - wn_loc[1] - 1
            tmin = wn_loc[1] - min(tleft, tright)
            tmax = wn_loc[1] + min(tleft, tright) + 1

            ydata_centered = self.acf[fmin:fmax, tmin:tmax]
            weights_centered = weights_2d[fmin:fmax, tmin:tmax]
            tdata_centered = tticks[tmin:tmax]
            fdata_centered = fticks[fmin:fmax]

            if nscale is not None and not full_frame:
                ntau = nscale
                ndnu = nscale

                if ntau > (self.tobs / tau):
                    if verbose:
                        print('WARNING: nscale exceeds range in time lag')
                    tmin = 0
                    tmax = nt
                else:
                    tframe = int(round(ntau * (tau / self.dt)))
                    tmin = int(np.floor(
                        np.shape(ydata_centered)[1] / 2)) - tframe
                    tmax = int(np.floor(
                        np.shape(ydata_centered)[1] / 2)) + tframe + 1

                if ndnu > (self.bw / dnu):
                    if verbose:
                        print('WARNING: nscale exceeds range in frequency lag')
                    tmin = 0
                    tmax = nf
                else:
                    fframe = int(round(ndnu * (dnu / self.df)))
                    fmin = int(np.floor(
                        np.shape(ydata_centered)[0] / 2)) - fframe
                    fmax = int(np.floor(
                        np.shape(ydata_centered)[0] / 2)) + fframe + 1

                ydata_2d = ydata_centered[fmin:fmax, tmin:tmax]
                weights_2d = weights_centered[fmin:fmax, tmin:tmax]
                tdata = tdata_centered[tmin:tmax]
                fdata = fdata_centered[fmin:fmax]
            else:
                ydata_2d = ydata_centered
                tdata = tdata_centered
                fdata = fdata_centered

            params.add('phasegrad', value=0, vary=True,
                       min=-np.inf, max=np.inf)
            if hasattr(self, 'acf_tilt'):  # if have a confident measurement
                if self.acf_tilt_err is not None:
                    params['phasegrad'].value = self.acf_tilt
            params.add('tobs', value=self.tobs, vary=False)
            params.add('bw', value=self.bw, vary=False)
            params.add('freq', value=self.freq, vary=False)

            if method == 'acf2d' and verbose:
                print("\nPerforming approximate 2D fit to initialize fit",
                      "values")
            elif verbose:
                print("\nPerforming least-squares fit to approximate 2D " +
                      "ACF model")

            pos_array = []
            if mcmc:
                for i in range(nwalkers):
                    pos_i = []
                    if tau_vary_2d:
                        pos_i.append(np.random.normal(
                                        loc=self.tau,
                                        scale=2*self.tauerr))
                    pos_i.append(np.random.normal(
                                    loc=self.dnu,
                                    scale=2*self.dnuerr))
                    pos_i.append(np.random.normal(
                                    loc=self.amp,
                                    scale=2*self.amperr))
                    if 'sim:mb2=' not in self.name:
                        pos_i.append(np.random.normal(
                                        loc=self.wn,
                                        scale=self.wnerr))
                    if alpha is None:
                        pos_i.append(np.random.normal(loc=5/3, scale=0.1))
                    pos_i.append(np.random.uniform(low=0,
                                                   high=5))  # phase grad
                    if lnsigma:
                        pos_i.append(np.random.uniform(low=0,
                                                       high=10))

                    pos_array.append(pos_i)
                pos = np.array(pos_array).squeeze()
            else:
                pos = None
            nfit = 5
            # max_nfev = 2000 * (nfit + 1)  # lmfit default
            max_nfev = 10000 * (nfit + 1)
            results = fitter(scint_acf_model_2d_approx, params,
                             (tdata, fdata, ydata_2d, None), mcmc=mcmc,
                             max_nfev=max_nfev, nan_policy=nan_policy,
                             pos=pos, steps=steps, burn=burn,
                             progress=progress, workers=workers,
                             is_weighted=(not lnsigma))

            if method == 'acf2d':

                if verbose:
                    print('2D tau estimate:', results.params['tau'].value,
                          '\n2D dnu estimate:', results.params['dnu'].value)

                params2d = results.params
                params2d.add('ar', value=2,
                             vary=False, min=-np.inf, max=np.inf)
                params2d.add('theta', value=0,
                             vary=False, min=-np.inf, max=np.inf)
                params2d.add('psi', value=60,
                             vary=True, min=-np.inf, max=np.inf)
                chisqr = np.inf
                for itr in range(nitr):
                    if mcmc:
                        pos_array = []
                        for i in range(nwalkers):
                            pos_i = []
                            if tau_vary_2d:
                                pos_i.append(np.random.normal(
                                    loc=results.params['tau'].value,
                                    scale=results.params['tau'].value/2))
                            pos_i.append(np.random.normal(
                                loc=results.params['dnu'].value,
                                scale=results.params['dnu'].value/2))
                            pos_i.append(np.random.normal(
                                loc=results.params['amp'].value,
                                scale=results.params['amp'].value/2))
                            if 'sim:mb2=' not in self.name:
                                pos_i.append(np.random.normal(
                                    loc=results.params['wn'].value,
                                    scale=results.params['wn'].value/2))
                            if alpha is None:
                                pos_i.append(np.random.normal(
                                    loc=results.params['alpha'].value,
                                    scale=results.params['alpha'].value/2))
                            pos_i.append(np.random.normal(
                                loc=results.params['phasegrad'].value,
                                scale=results.params['phasegrad'].value/2))
                            pos_i.append(np.random.uniform(low=0,
                                                           high=90))  # psi
                            if lnsigma:
                                pos_i.append(np.random.uniform(low=0,
                                                               high=10))

                            pos_array.append(pos_i)
                        pos = np.array(pos_array)
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
                    res = fitter(scint_acf_model_2d, params2d,
                                 (ydata_2d, None), mcmc=mcmc, pos=pos,
                                 nwalkers=nwalkers, steps=steps, burn=burn,
                                 progress=progress, workers=workers,
                                 max_nfev=max_nfev, nan_policy=nan_policy,
                                 is_weighted=(not lnsigma))
                    if res.chisqr < chisqr:
                        chisqr = res.chisqr
                        results = res

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

        if results.params['tau'].stderr is None or \
           results.params['dnu'].stderr is None:
            print("\n Warning: Fit failed")
            return
        elif (results.params['tau'].stderr > results.params['tau'].value or
              results.params['dnu'].stderr > results.params['dnu'].value):
            print("\n Warning: Parameters unconstraiend")

        self.scint_param_method = method

        # Done fitting - now define results
        self.tau = results.params['tau'].value
        self.dnu = results.params['dnu'].value
        self.tscat = 1/(2*np.pi*self.dnu)  # scattering timescale
        if self.dnu < self.df:
            print("Warning: Scint bandwidth < channel bandwidth.")
        # Compute finite scintle error
        nscint = (1 + 0.2*self.bw/(self.dnu)) * \
            (1 + 0.2*self.tobs/(self.tau*np.log(2)))
        self.nscint = nscint
        self.fse_tau = self.tau/(2*np.sqrt(nscint))
        fit_tau = results.params['tau'].stderr
        self.fse_dnu = self.dnu/(2*np.sqrt(nscint))
        fit_dnu = results.params['dnu'].stderr

        if verbose:
            print("\nFinite scintle errors (tau, dnu):\n",
                  self.fse_tau, self.fse_dnu)
            print("\nFit errors (tau, dnu):\n",
                  fit_tau, fit_dnu)

        if fit_dnu is None:
            fit_dnu = np.inf
        if fit_tau is None:
            fit_tau = np.inf

        self.tauerr = np.sqrt(fit_tau**2 + self.fse_tau**2)
        self.dnuerr = np.sqrt(fit_dnu**2 + self.fse_dnu**2)

        self.amp = results.params['amp'].value
        self.amperr = results.params['amp'].stderr
        if 'sim:mb2=' not in self.name:
            self.wn = results.params['wn'].value
            self.wnerr = results.params['wn'].stderr
        else:
            self.wn = 0
        if alpha is None:
            self.talpha = results.params['alpha'].value
            self.talphaerr = results.params['alpha'].stderr
        else:
            self.talpha = alpha
            self.talphaerr = 0
        if method[:5] == 'acf2d':
            weights = np.ones(np.shape(ydata_2d))
            if method == 'acf2d_approx':
                model = -scint_acf_model_2d_approx(
                    results.params, tdata, fdata,
                    np.zeros(np.shape(ydata_2d)), None)
            else:
                model = -scint_acf_model_2d(results.params,
                                            np.zeros(np.shape(ydata_2d)),
                                            None)
            self.acf_model = model
            self.phasegrad = results.params['phasegrad'].value
            fit_ph = results.params['phasegrad'].stderr
            if fit_ph is None:
                fit_ph = np.inf
            fse_ph = self.phasegrad * np.sqrt((self.fse_dnu/self.dnu)**2 +
                                              (self.fse_tau/self.tau)**2)
            self.phasegraderr = fit_ph
            self.fse_phasegrad = fse_ph
            if method == 'acf2d':
                self.ar = results.params['ar'].value
                self.arerr = results.params['ar'].stderr
                self.theta = results.params['theta'].value
                self.thetaerr = results.params['theta'].stderr
                self.psi = results.params['psi'].value
                self.psierr = results.params['psi'].stderr

        if verbose:
            print("\n\t ACF FIT PARAMETERS\n")
            print("tau:\t\t\t{val} +/- {err} s".format(val=self.tau,
                  err=self.tauerr))
            print("dnu:\t\t\t{val} +/- {err} MHz".format(val=self.dnu,
                  err=self.dnuerr))
            if alpha is None:
                print("alpha:\t\t\t{val} +/- {err}".format(val=self.talpha,
                      err=self.talphaerr))
            if method[:5] == 'acf2d':
                err = np.sqrt(self.phasegraderr**2 + fse_ph**2)
                print("phase grad:\t\t{val} +/- {err}".
                      format(val=self.phasegrad, err=err))
                if method == 'acf2d':
                    print("ar:\t\t{val} +/- {err}".format(val=self.ar,
                          err=self.arerr))
                    print("theta:\t\t{val} +/- {err}".format(
                            val=self.theta, err=self.thetaerr))
                    print("psi:\t\t{val} +/- {err}".format(val=self.psi,
                          err=self.psierr))

        if plot:
            if method == 'acf1d':
                tmodel = -tau_acf_model(results.params, xdata_t,
                                        np.zeros(len(xdata_t)), None)
                fmodel = -dnu_acf_model(results.params, xdata_f,
                                        np.zeros(len(xdata_f)), None)

                fig = plt.subplots(2, 1, figsize=(8, 6))
                fig[1][0].plot(xdata_t, ydata_t, label='data')
                fig[1][0].fill_between(xdata_t, ydata_t+t_errors,
                                       ydata_t-t_errors, color='C0',
                                       alpha=0.4, label='error')
                fig[1][0].plot(xdata_t, tmodel, label='model')
                # plot 95% white noise level assuming no correlation
                xl = fig[1][0].get_xlim()
                fig[1][0].plot([0, xl[1]], [0, 0], 'k--')
                fig[1][0].plot([0, xl[1]],
                               [1/np.sqrt(self.nsub), 1/np.sqrt(self.nsub)],
                               ':', color='crimson',
                               label=r'$\pm 1/\sqrt{n_\mathrm{sub}}$')
                fig[1][0].plot([0, xl[1]],
                               [-1/np.sqrt(self.nsub), -1/np.sqrt(self.nsub)],
                               ':', color='crimson')
                fig[1][0].set_xlabel(r'$\tau$ (s)')
                fig[1][0].legend()
                fig[0].tight_layout()

                fig[1][1].plot(xdata_f, ydata_f, label='data')
                fig[1][1].fill_between(xdata_f, ydata_f+f_errors,
                                       ydata_f-f_errors, color='C0',
                                       alpha=0.4, label='error')
                fig[1][1].plot(xdata_f, fmodel, label='model')
                # plot 95% white noise level assuming no correlation
                xl = fig[1][1].get_xlim()
                fig[1][1].plot([0, xl[1]], [0, 0], 'k--')
                fig[1][1].plot([0, xl[1]],
                               [1/np.sqrt(self.nchan), 1/np.sqrt(self.nchan)],
                               ':', color='crimson',
                               label=r'$\pm 1/\sqrt{n_\mathrm{chan}}$')
                fig[1][1].plot([0, xl[1]],
                               [-1/np.sqrt(self.nchan),
                                -1/np.sqrt(self.nchan)],
                               ':', color='crimson')
                fig[1][1].set_xlabel(r'$\Delta\nu$ (MHz)')
                fig[1][1].legend()
                fig[0].tight_layout()

                if filename is not None:
                    filename_name = ''.join(filename.split('.')[0:-1])
                    filename_extension = filename.split('.')[-1]
                    fig[0].savefig(
                        filename_name + '_1Dfit.' + filename_extension,
                        dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                if display:
                    plt.show()
                plt.close(fig[0])

            elif method[:5] == 'acf2d':
                weights = np.ones(np.shape(ydata_2d))
                if method == 'acf2d_approx':
                    model = -scint_acf_model_2d_approx(
                        results.params, tdata, fdata,
                        np.zeros(np.shape(ydata_2d)), None)
                else:
                    model = -scint_acf_model_2d(results.params,
                                                np.zeros(np.shape(ydata_2d)),
                                                None)
                residuals = (ydata_2d - model) * weights

                fig = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
                data = [(ydata_2d, 'data'), (model, 'model'),
                        (residuals, 'residuals')]
                for i, d in enumerate(data):
                    if d[1] != 'residuals':
                        # subtract the white noise spike from data and model
                        arr = np.fft.ifftshift(d[0])
                        arr[0][0] -= self.wn
                        arr = np.fft.fftshift(arr)
                    else:
                        arr = d[0]

                    tedges = centres_to_edges(tdata/60)
                    fedges = centres_to_edges(fdata)
                    mesh = fig[1][i].pcolormesh(tedges, fedges, arr,
                                                linewidth=0, rasterized=True,
                                                shading='auto')
                    if d[1] == 'residuals':
                        mesh.set_clim(vmin=-1, vmax=1)  # fractional error
                    fig[1][i].set_title(d[1])
                    fig[1][i].set_xlabel(r'$\tau$ (mins)')
                    if i == 0:
                        fig[1][i].set_ylabel(r'$\Delta\nu$ (MHz)')
                plt.tight_layout()
                if filename is not None:
                    filename_name = ''.join(filename.split('.')[0:-1])
                    filename_extension = filename.split('.')[-1]
                    fig[0].savefig(filename_name + '_2Dfit.'
                                   + filename_extension, dpi=dpi,
                                   bbox_inches='tight', pad_inches=0.1)
                if display:
                    plt.show()
                plt.close(fig[0])

            elif method == 'sspec':
                '''
                sspec plotting routine
                '''

            if mcmc and method == "acf2d":
                corner.corner(results.flatchain,
                              labels=results.var_names,
                              truths=list(results.params.valuesdict().
                                          values()))
                if filename is not None:
                    filename_name = ''.join(filename.split('.')[0:-1])
                    filename_extension = filename.split('.')[-1]
                    plt.savefig(filename_name + '_corner.'
                                + filename_extension, dpi=dpi,
                                bbox_inches='tight', pad_inches=0.1)
                if display:
                    plt.show()
                plt.close()

        return results

    def cut_dyn(self, tcuts=0, fcuts=0, plot=False, filename=None, dpi=200,
                lamsteps=False, maxfdop=np.inf, figsize=(8, 13), display=True):
        """
        Cuts the dynamic spectrum into tcuts+1 segments in time and
                fcuts+1 segments in frequency

        Parameters
        ----------
        tcuts : int, optional
            Number of cuts in time. The default is 0.
        fcuts : int, optional
            Number of cuts in frequency. The default is 0.
        plot : bool, optional
            Plot the individual segments. The default is False.
        filename : str, optional
            The path at which to save the figure. The default is None.
        dpi : float, optional
            dpi of the figure. The default is 200.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        maxfdop : float, optional
            Maximum fdop for secondary spectrum plots. The default is np.inf.
        figsize : tuple, optional
            Size of the figure. The default is (8, 13).
        display : bool, optional
            Display the plot. The default is True.

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
                filename_name = ''.join(filename.split('.')[0:-1])
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

    def trim_edges(self, bandwagon_frac=0.5):
        """
        Find and remove the band edges

        Parameters
        ----------
        bandwagon_frac : float in [0,1], optional
            Set entire edge to zero if more than this fraction of the edge
            pixels is zero or NaN. The default is 0.5.

        """

        self.dyn[np.isnan(self.dyn)] = 0  # fill NaNs with zero

        nc = len(self.dyn[0, :])
        nr = len(self.dyn[:, 0])

        # Trim bottom
        if len(np.argwhere(self.dyn[0, :] == 0)) > bandwagon_frac*nc:
            self.dyn[0, :] = np.zeros(np.shape(self.dyn[0, :]))
            # self.dyn_err[0, :] = np.zeros(np.shape(self.dyn_err[0, :]))
        rowsum = sum(abs(self.dyn[0, :]))
        while rowsum == 0:
            self.dyn = np.delete(self.dyn, (0), axis=0)
            # self.dyn_err = np.delete(self.dyn_err, (0), axis=0)
            self.freqs = np.delete(self.freqs, (0))
            if len(np.argwhere(self.dyn[0, :] == 0)) > bandwagon_frac*nc:
                self.dyn[0, :] = np.zeros(np.shape(self.dyn[0, :]))
                # self.dyn_err[0, :] = np.zeros(np.shape(self.dyn_err[0, :]))
            rowsum = sum(abs(self.dyn[0, :]))

        # Trim top
        if len(np.argwhere(self.dyn[-1, :] == 0)) > bandwagon_frac*nc:
            self.dyn[-1, :] = np.zeros(np.shape(self.dyn[-1, :]))
            # self.dyn_err[-1, :] = np.zeros(np.shape(self.dyn_err[-1, :]))
        rowsum = sum(abs(self.dyn[-1, :]))
        while rowsum == 0:
            self.dyn = np.delete(self.dyn, (-1), axis=0)
            # self.dyn_err = np.delete(self.dyn_err, (-1), axis=0)
            self.freqs = np.delete(self.freqs, (-1))
            if len(np.argwhere(self.dyn[-1, :] == 0)) > bandwagon_frac*nc:
                self.dyn[-1, :] = np.zeros(np.shape(self.dyn[-1, :]))
                # self.dyn_err[-1, :] = np.zeros(np.shape(self.dyn_err[-1, :]))
            rowsum = sum(abs(self.dyn[-1, :]))

        # Trim left
        if len(np.argwhere(self.dyn[:, 0] == 0)) > bandwagon_frac*nr:
            self.dyn[:, 0] = np.zeros(np.shape(self.dyn[:, 0]))
            # self.dyn_err[:, 0] = np.zeros(np.shape(self.dyn_err[:, 0]))
        colsum = sum(abs(self.dyn[:, 0]))
        while colsum == 0:
            self.dyn = np.delete(self.dyn, (0), axis=1)
            # self.dyn_err = np.delete(self.dyn_err, (0), axis=1)
            self.times = np.delete(self.times, (0))
            if len(np.argwhere(self.dyn[:, 0] == 0)) > bandwagon_frac*nr:
                self.dyn[:, 0] = np.zeros(np.shape(self.dyn[:, 0]))
                # self.dyn_err[:, 0] = np.zeros(np.shape(self.dyn_err[:, 0]))
            colsum = sum(abs(self.dyn[:, 0]))

        # Trim right
        if len(np.argwhere(self.dyn[:, -1] == 0)) > bandwagon_frac*nr:
            self.dyn[:, -1] = np.zeros(np.shape(self.dyn[:, -1]))
            # self.dyn_err[:, -1] = np.zeros(np.shape(self.dyn_err[:, -1]))
        colsum = sum(abs(self.dyn[:, -1]))
        while colsum == 0:
            self.dyn = np.delete(self.dyn, (-1), axis=1)
            # self.dyn_err = np.delete(self.dyn_err, (-1), axis=1)
            self.times = np.delete(self.times, (-1))
            if len(np.argwhere(self.dyn[:, -1] == 0)) > bandwagon_frac*nr:
                self.dyn[:, -1] = np.zeros(np.shape(self.dyn[:, -1]))
                # self.dyn_err[:, -1] = np.zeros(np.shape(self.dyn_err[:, -1]))
            colsum = sum(abs(self.dyn[:, -1]))

        self.nchan = len(self.freqs)
        self.bw = round(max(self.freqs) - min(self.freqs) + self.df, 2)
        self.freq = round(np.mean(self.freqs), 2)
        self.nsub = len(self.times)
        self.tobs = round(max(self.times) - min(self.times) + self.dt, 2)
        self.mjd = self.mjd + self.times[0]/86400

    def refill(self, method='biharmonic', zeros=True, kernel_size=5,
               linear=True):
        """
        Replaces the nan values in array. Also replaces zeros by default.

        Parameters
        ----------
        method : str {'biharmonic', 'linear', 'cubic', 'nearest', 'median'},
        optional
            Interpolation method.
        zeros : bool, optional
            Replace zeros. The default is True.
        kernel_size : int or array_like, optional
            Size of the filter window in each dimension when using a median
            filter. The default is 5.
        linear : bool, optional
            Perform interpolation. The default is True.

        """

        if (not biharmonic) and (method == 'biharmonic'):
            print('Warning: biharmonic inpainting not available.' +
                  'Defaulting to linear interpolation.')
            method = 'linear'

        if zeros:
            self.dyn[self.dyn == 0] = np.nan

        if method == 'biharmonic':
            array = cp(self.dyn)
            # Create mask
            mask = np.zeros(np.shape(array))
            mask[np.isnan(array)] = 1
            inpainted = inpaint.inpaint_biharmonic(array, mask)
            self.dyn[np.isnan(self.dyn)] = inpainted[np.isnan(self.dyn)]
        elif method == 'median':
            array = cp(self.dyn)
            if kernel_size == 5:
                print("Warning: kernel size is set to default.")
            array[np.isnan(array)] = np.mean(array[is_valid(array)])
            ds_med = medfilt(array, kernel_size=kernel_size)
            self.dyn[np.isnan(self.dyn)] = ds_med[np.isnan(self.dyn)]
        elif (method == 'linear' or method == 'cubic' or
              method == 'nearest') and linear:
            # do interpolation
            array = cp(self.dyn)
            self.dyn = interp_nan_2d(array, method=method)

        # Fill with the mean
        meanval = np.mean(self.dyn[is_valid(self.dyn)])
        self.dyn[np.isnan(self.dyn)] = meanval

    def correct_dyn(self, svd=True, nmodes=1, frequency=True, time=True,
                    lamsteps=False, nsmooth=None, velocity=False):
        """
        Correct for apparent flux variations in time and frequency

        Parameters
        ----------
        svd : bool, optional
            Perform a singular value decomposition on the dynamic spectrum. The
            default is True.
        nmodes : int, optional
            Use the largest nmodes singular values in the SVD approximation.
            The default is 1.
        frequency : bool, optional
            Perform correction in frequency. The default is True.
        time : bool, optional
            Perform correction in time. The default is True.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        nsmooth : int, optional
            The length of the smoothing filter window. Must be a positive odd
            integer. The default is None.
        velocity : bool, optional
            Scale the dynamic spectrum using the velocity. The default is
            False.

        """

        if hasattr(self, 'svd_model'):
            print('Warning: An svd_model exists. Check before applying twice')

        if lamsteps:
            if velocity:
                if not hasattr(self, 'vlamdyn'):
                    raise ValueError('Need to run scale_dyn with a model')
                dyn = self.vlamdyn
            else:
                if not hasattr(self, 'lamdyn'):
                    self.scale_dyn(lamsteps=lamsteps)
                dyn = self.lamdyn
        elif velocity:
            if not hasattr(self, 'vdyn'):
                raise ValueError('Need to run scale_dyn with a model')
            dyn = self.vdyn
        else:
            dyn = self.dyn

        dyn[np.isnan(dyn)] = 0

        if svd:
            dyn, model = svd_model(dyn, nmodes=nmodes)
            self.svd_model = model
        else:
            if frequency:
                self.dyn[self.dyn == 0] = np.nan
                self.bandpass = np.nanmean(dyn, axis=1)
                # Make sure there are no zeros
                self.bandpass[self.bandpass == 0] = np.mean(self.bandpass)
                if nsmooth is not None:
                    bandpass = savgol_filter(self.bandpass, nsmooth, 1)
                else:
                    bandpass = self.bandpass
                dyn = np.divide(dyn, np.reshape(bandpass,
                                                [len(bandpass), 1]))

            if time:
                self.dyn[self.dyn == 0] = np.nan
                timestructure = np.nanmean(dyn, axis=0)
                # Make sure there are no zeros
                timestructure[timestructure == 0] = np.mean(timestructure)
                if nsmooth is not None:
                    timestructure = savgol_filter(timestructure, nsmooth, 1)
                dyn = np.divide(dyn, np.reshape(timestructure,
                                                [1, len(timestructure)]))
            self.dyn[np.isnan(self.dyn)] = 0

        if lamsteps:
            if velocity:
                self.vlamdyn = dyn
            else:
                self.lamdyn = dyn
        elif velocity:
            self.vdyn = dyn
        else:
            self.dyn = dyn

    def calc_scattered_image(self, input_sspec=None, input_eta=None,
                             input_fdop=None, input_tdel=None, sampling=64,
                             lamsteps=False, trap=False, ref_freq=1400,
                             clean=True, s=None, veff=None, d=None,
                             fit_arc=True, plot_fit=False, plot=False,
                             plot_log=True, use_angle=False,
                             use_spatial=False):
        """
        Calculate the scattered image.

        Assumes that the scattering is defined by the primary arc,
        i.e. interference between highly scattered waves and unscattered waves
        (B(tx,ty) vs B(0,0)).

        The x axis of the image is aligned with the velocity.

        Parameters
        ----------
        input_sspec : Dynspec object, optional
            Ignore the class-defined secondary spectrum and use this input
            spectrum. The default is None.
        input_eta : float, optional
            Input curvature value. The default is None.
        input_fdop : 1D array, optional
            fdop axis of the secondary spectrum. The default is None.
        input_tdel : 1D array, optional
            tdel axis of the secondary spectrum. The default is None.
        sampling : int, optional
            Number of samples in fdop. The default is 64.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        trap : bool, optional
            Trapezoidal scaling. The default is False.
        ref_freq : float, optional
            Reference frequency. The default is 1400.
        clean : bool, optional
            Fill infs and extremely small pixel values. The default is True.
        s : float in range [0,1], optional
            Fractional screen distance. The default is None.
        veff : float, optional
            Magnitude of the effective velocity. The default is None.
        d : float, optional
            Pulsar distance. The default is None.
        fit_arc : bool, optional
            Fit for the arc curvature. The default is True.
        plot_fit : bool, optional
            Plot the arc curvature fit. The default is False.
        plot : bool, optional
            Plot the scattered image. The default is False.
        plot_log : bool, optional
            Plot the scattered image on a logarithmic scale. The default is
            True.
        use_angle : bool, optional
            Use angular axes in plot. The default is False.
        use_spatial : bool, optional
            Use spatial axes in plot. The default is False.

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

        nf, nt = len(fdop), len(tdel)
        linsspec = 10**(sspec / 10)

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

        # crop sspec to desired region
        flim = next(i for i, delay in enumerate(eta * fdop**2) if
                    delay < np.max(tdel))
        if flim == 0:
            tlim = next(i for i, delay in enumerate(tdel) if
                        delay > eta * fdop[0] ** 2)
            linsspec = linsspec[:tlim, :]
            tdel = fdop[:tlim]
        else:

            linsspec = linsspec[:, flim-int(0.02*nf):nf-flim+int(0.02*nf)]
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

        nx, ny = 2*sampling+1, sampling+1
        fdop_x = np.linspace(-max(fdop), max(fdop), nx)
        fdop_y = np.linspace(0, max(fdop), ny)

        # equally space square
        fdop_x_est, fdop_y_est = np.meshgrid(fdop_x, fdop_y)
        fdop_est = fdop_x_est
        tdel_est = (fdop_x_est**2 + fdop_y_est**2) * eta

        # 2D interpolation
        interp = RectBivariateSpline(tdel, fdop, linsspec)
        # interpolate sspec onto grid for theta
        image = interp.ev(tdel_est, fdop_est)

        image = image * fdop_y_est
        scat_im = np.zeros((nx, nx))
        scat_im[ny-1:nx, :] = image
        scat_im[0:ny-1, :] = image[ny-1:0:-1, :]

        xyaxes = fdop_x

        if plot or plot_log:
            self.plot_scattered_image(input_scattered_image=scat_im,
                                      input_fdop=xyaxes, s=s, veff=veff, d=d,
                                      use_angle=use_angle,
                                      use_spatial=use_spatial, display=True,
                                      plot_log=plot_log)

        self.scattered_image = scat_im
        self.scattered_image_ax = xyaxes

    def calc_sspec(self, prewhite=False, halve=True, plot=False,
                   lamsteps=False, input_dyn=None, input_x=None, input_y=None,
                   trap=False, window='hanning', window_frac=0.1,
                   return_sspec=False, velocity=False):
        """
        Calculate secondary spectrum

        Parameters
        ----------
        prewhite : bool, optional
            Perform pre-whitening using the first-difference method, then
            post-darken. The default is False.
        halve : bool, optional
            Consider only positive tdel. The default is True.
        plot : bool, optional
            Plot the secondary spectrum. The default is False.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        input_dyn : Dynspec object, optional
            Ignore the class-defined dynamic spectrum and use this input
            dynamic spectrum. The default is None.
        input_x : 1D array, optional
            `x`-axis of input secondary spectrum. The default is None.
        input_y : 1D array, optional
            `y`-axis of input secondary spectrum. The default is None.
        trap : bool, optional
            Trapezoidal scaling. The default is False.
        window : str {'blackman', 'hanning', 'hamming', 'bartlett'}, optional
            Type of window for the dynamic spectrum.
        window_frac : float in [0,1], optional
            Number of points in the output window as a fraction of the
            pixel dimensions of the dynamic spectrum. The default is 0.1.
        return_sspec : bool, optional
            Return secondary spectrum and axes rather than assigning class
            variables. The default is False.
        velocity : bool, optional
            Scale the dynamic spectrum using the velocity. The default is
            False.

        Raises
        ------
        RuntimeError
            If trying to apply prewhite to the full frame.

        Returns
        -------
        fdop : 1D array
            fdop axis.
        yaxis : 1D array
            The y-axis. If lamsteps=True, this is the conjugate to the
            wavelength, otherwise it is the conjugate to frequency (i.e. the
            differential time delay).
        sec : 2D array
            The secondary spectrum.

        """

        if input_dyn is None:  # use self dynamic spectrum
            if lamsteps:
                if not hasattr(self, 'lamdyn'):
                    self.scale_dyn()
                if velocity:
                    if not hasattr(self, 'vlamdyn'):
                        self.scale_dyn(scale='velocity')
                    dyn = cp(self.vlamdyn)
                else:
                    dyn = cp(self.lamdyn)
            elif velocity:
                if not hasattr(self, 'vdyn'):
                    self.scale_dyn(scale='velocity')
                dyn = cp(self.vdyn)
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
            if window.lower() == 'hanning':
                cw = np.hanning(np.floor(window_frac*nt))
                sw = np.hanning(np.floor(window_frac*nf))
            elif window.lower() == 'hamming':
                cw = np.hamming(np.floor(window_frac*nt))
                sw = np.hamming(np.floor(window_frac*nf))
            elif window.lower() == 'blackman':
                cw = np.blackman(np.floor(window_frac*nt))
                sw = np.blackman(np.floor(window_frac*nf))
            elif window.lower() == 'bartlett':
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
                if velocity:
                    self.vlamsspec = sec
                else:
                    self.lamsspec = sec
            elif velocity:
                self.vsspec = sec
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

        Parameters
        ----------
        method : str {'direct', 'sspec'}, optional
            ACF calculation method.

                ``direct``
                    Apply a fast Fourier transform (FFT) and its inverse.
                ``sspec``
                    Calculate the FFT of the secondary spectrum.
        input_dyn : Dynspec object, optional
            Ignore the class-defined dynamic spectrum and use this input
            dynamic spectrum. The default is None.
        normalise : bool, optional
            Normalise the ACF. The default is True.
        window_frac : float in [0,1], optional
            Number of points in the output window as a fraction of the
            pixel dimensions of the dynamic spectrum. The default is 0.1.

        Returns
        -------
        arr : 2D array
            The ACF.

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

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency. The default is 0.
        fmax : float, optional
            Maximum frequency. The default is np.inf.
        tmin : float, optional
            Minimum time. The default is 0.
        tmax : float, optional
            Maximum time. The default is np.inf.

        """

        # Crop frequencies
        crop_array = np.array((self.freqs > fmin)*(self.freqs < fmax))
        self.dyn = self.dyn[crop_array, :]
        # self.dyn_err = self.dyn_err[crop_array, :]
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
        # self.dyn_err = self.dyn_err[:, crop_array]
        self.nsub = len(self.dyn[0, :])
        self.times = np.linspace(self.dt/2, self.tobs - self.dt/2, self.nsub)
        self.mjd = self.mjd + tmin/86400

    def zap(self, sigma=7):
        """
        Basic zapping (RFI mitigation) of dynamic spectrum

        Parameters
        ----------
        sigma : float, optional
            Level at which to zap. The default is 7.

        """

        d = np.abs(self.dyn - np.median(self.dyn[~np.isnan(self.dyn)]))
        mdev = np.median(d[~np.isnan(d)])
        s = d/mdev
        self.dyn[s > sigma] = np.nan

    def scale_dyn(self, scale='lambda', window_frac=0.1, pars=None,
                  parfile=None, window='hanning', spacing='auto', s=None,
                  d=None, vism_ra=None, vism_dec=None, Omega=None, inc=None,
                  vism_zeta=None, zeta=None, lamsteps=False, velocity=False,
                  trap=False):
        """
        Rescales the dynamic spectrum to specified shape

        Parameters
        ----------
        scale : str {'lambda', 'wavelength', 'velocity', 'orbit', 'trapezoid'},
        optional
            Type of scaling.
        window_frac : float in [0,1], optional
            Number of points in the output window as a fraction of the
            pixel dimensions of the dynamic spectrum. The default is 0.1.
        pars : dict, optional
            Parameters dictionary. The default is None.
        parfile : str, optional
            Path to the parameters file. The default is None.
        window : str {'hanning', 'blackman', 'hamming', 'bartlett'}, optional
            Type of window for the dynamic spectrum.
        spacing : str {'auto', 'max', 'min', 'mean', 'median'}, optional
            Which spacing to select from the wavelength axis.
        s : float in [0,1], optional
            Fractional screen distance. The default is None.
        d : float, optional
            Distance to the pulsar in kpc. The default is None.
        vism_ra : float, optional
            ISM velocity in RA direction. The default is None.
        vism_dec : float, optional
            ISM velocity in dec direction. The default is None.
        Omega : float, optional
            Ascending node of the orbit. The default is None.
        inc : float, optional
            Inclination of the orbit. The default is None.
        vism_zeta : float, optional
            ISM veliocity in the direction of anisotropy. The default is None.
        zeta : float, optional
            Angle of anisotopy on the sky (East of North). The default is None.
        lamsteps : bool, optional
            Use equal steps in wavelength rather than frequency. The default is
            False.
        velocity : bool, optional
            Scale the dynamic spectrum using the velocity. The default is
            False.
        trap : bool, optional
            Use trapezoidal scaling. The default is False.

        Raises
        ------
        ValueError
            If parameter disctionary is unspecified.

        """

        if ('lambda' in scale) or ('wavelength' in scale) or lamsteps:
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

        if ('velocity' in scale) or ('orbit' in scale) or velocity:

            from scintools.scint_utils import get_ssb_delay, \
                get_earth_velocity, get_true_anomaly, read_par
            from scintools.scint_models import effective_velocity_annual

            if pars is None and parfile is None:
                raise ValueError('Requires dictionary of parameters ' +
                                 'or .par file for velocity calculation')
            if parfile is not None:
                pars = read_par(parfile)

            arin = cp(self.dyn)  # input array
            if hasattr(self, 'lamdyn'):
                arin2 = cp(self.lamdyn)  # input array
            nf, nt = np.shape(arin)
            arout = np.zeros([nf, nt])
            if hasattr(self, 'lamdyn'):
                arin2 = cp(self.lamdyn)  # input array
                nf2, nt2 = np.shape(arin2)
                arout2 = np.zeros([nf2, nt2])
            mjd = np.asarray(self.mjd, dtype=np.float128) + \
                np.asarray(self.times, dtype=np.float128)/86400

            print('Getting SSB delays')
            ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
            mjd += np.divide(ssb_delays, 86400)  # add ssb delay
            print('Getting Earth velocity')
            vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                                       pars['DECJ'])
            print('Getting true anomaly')
            true_anomaly = get_true_anomaly(mjd, pars)
            if 's' not in pars.keys():
                if s is None:
                    raise ValueError('Requires screen distance s in ' +
                                     'parameter dictionary, or as input')
                pars['s'] = s
            if 'd' not in pars.keys():
                if d is None:
                    raise ValueError('Requires pulsar distance d in ' +
                                     'parameter dictionary, or as input')
                pars['d'] = d
            if 'KIN' not in pars.keys():
                if inc is None:
                    raise ValueError('Requires inclination angle (KIN) in ' +
                                     'parameter dictionary, or as input inc')
                pars['KIN'] = inc
            if 'KOM' not in pars.keys():
                if Omega is None:
                    raise ValueError('Requires ascending node (KOM) in ' +
                                     'parameter dictionary, or as input Omega')
                pars['KOM'] = Omega

            veff_ra, veff_dec, vp_ra, vp_dec = \
                effective_velocity_annual(pars, true_anomaly, vearth_ra,
                                          vearth_dec, mjd=mjd)

            # anisotropic case
            if ('zeta' in pars.keys()) or (zeta is not None):
                if 'zeta' in pars.keys():
                    zeta = pars['zeta']
                zeta *= np.pi / 180  # make radians
                # vism in direction of anisotropy
                if 'vism_zeta' in pars.keys():
                    vism_zeta = pars['vism_zeta']
                    veff2 = (veff_ra*np.sin(zeta) +
                             veff_dec*np.cos(zeta) - vism_zeta)**2
                elif vism_zeta is not None:
                    veff2 = (veff_ra*np.sin(zeta) +
                             veff_dec*np.cos(zeta) - vism_zeta)**2
                else: # No Vism_psi, maybe have ra and dec velocities?
                    if 'vism_ra' in pars.keys():
                        veff_ra -= pars['vism_ra']
                    elif vism_ra is not None:
                        veff_ra -= vism_ra
                    if 'vism_dec' in pars.keys():
                        veff_dec -= pars['vism_dec']
                    elif vism_dec is not None:
                        veff_dec -= vism_dec
                    veff2 = (veff_ra*np.sin(zeta) +
                             veff_dec*np.cos(zeta))**2

            # isotropic case
            else:
                if 'vism_ra' in pars.keys():
                    veff_ra -= pars['vism_ra']
                elif vism_ra is not None:
                    veff_ra -= vism_ra
                if 'vism_dec' in pars.keys():
                    veff_dec -= pars['vism_dec']
                elif vism_dec is not None:
                    veff_dec -= vism_dec
                veff2 = veff_ra**2 + veff_dec**2

            veff = np.sqrt(veff2)
            vc_orig = np.cumsum(veff)  # original cumulative sum of velocity

            # even grid in velocity cumulative sum
            vc_new = np.linspace(np.min(vc_orig), np.max(vc_orig),
                                 len(vc_orig))

            for ii in range(0, nf):
                f = interp1d(vc_orig, arin[ii, :], kind='cubic')
                # Make sure the range is valid after rounding
                if max(vc_new) > max(vc_orig):
                    vc_new[np.argmax(vc_new)] = max(vc_orig)
                if min(vc_new) < min(vc_orig):
                    vc_new[np.argmin(vc_new)] = min(vc_orig)
                arout[ii, :] = f(vc_new)
            if hasattr(self, 'lamdyn'):
                for ii in range(0, nf2):
                    f = interp1d(vc_orig, arin2[ii, :], kind='cubic')
                    # Make sure the range is valid after rounding
                    arout2[ii, :] = f(vc_new)
            self.veff_ra = veff_ra
            self.veff_dec = veff_dec
            self.vdyn = arout
            if hasattr(self, 'lamdyn'):
                self.vlamdyn = arout2

        if 'trap' in scale or trap:
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
        Print properties of object
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

    def __init__(self, dyn, name="BasicDyn", header=["BasicDyn"], times=[],
                 freqs=[], nchan=None, nsub=None, bw=None, df=None,
                 freq=None, tobs=None, dt=None, mjd=None):
        """
        Define a basic dynamic spectrum object from an array of fluxes
            and other variables, which can then be passed to the dynspec
            class to access its functions with:
        BasicDyn_Object = BasicDyn(dyn)
        Dynspec_Object = Dynspec(BasicDyn_Object)

        Parameters
        ----------
        dyn : 2D array
            The dynamic spectrum.
        name : str, optional
            Name of the dynamic spectrum. The default is "BasicDyn".
        header : list of str, optional
            Header for the object. The default is ["BasicDyn"].
        times : 1D array, optional
            Time axis. The default is [].
        freqs : 1D array, optional
            Frequency axis. The default is [].
        nchan : int, optional
            Number of frequency channels. The default is None.
        nsub : int, optional
            Number of sub-integrations. The default is None.
        bw : float, optional
            Observation bandwidth. The default is None.
        df : float, optional
            Frequncy channel width. The default is None.
        freq : float, optional
            Observation frequency. The default is None.
        tobs : float, optional
            Observation time duration. The default is None.
        dt : float, optional
            Sub-integration duration. The default is None.
        mjd : float, optional
            MJD of observation. The default is None.

        Raises
        ------
        ValueError
            If the time or frequency axes are left unspecified.

        """

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

    def __init__(self, matfilename):
        """
        Imports simulated dynamic spectra from Matlab code by
            Coles et al. (2010)

        Parameters
        ----------
        matfilename : str
            Path to the mat file.

        Raises
        ------
        NameError
            If variables "spi" or "dlam" are missing from the mat file.

        """

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

    def __init__(self, sim):
        """
        Imports Simulation() object from scint_sim to Dynspec class

        Parameters
        ----------
        sim : Simulation object
            The simulated dynamic spectrum.

        """

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

    def __init__(self, holofile, imholofile=None, df=1, dt=1, fmin=0, mjd=0):
        """
        Imports model dynamic spectrum from holography code of
            Walker et al. (2008).

        Parameters
        ----------
        holofile : str
            Path to fits file of the real component of the dynamic spectrum.
        imholofile : str, optional
            Path to fits file of the imaginary component of the dynamic
            spectrum. The default is None.
        df : float, optional
            Frequncy channel width. The default is 1.
        dt : float, optional
            Sub-integration duration. The default is None.
        fmin : float, optional
            Frequency of the lowest-frequency channel. The default is 0.
        mjd : float, optional
            MJD of observation. The default is None.

        """

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
             min_freq=0, max_freq=5000, verbose=True, max_frac_bw=2):
    """
    Sorts list of dynamic spectra into good and bad files based on some
        user-defined conditions

    Parameters
    ----------
    dynfiles : list of str
        List of paths to the dynamic spectrum files.
    outdir : str, optional
        Directory in which to save the sorted lists. The default is None.
    min_nsub : int, optional
        Minimum number of sub-integrations. The default is 10.
    min_nchan : int, optional
        Minimum number of frequency channels. The default is 50.
    min_tsub : float, optional
        Minimum observation time duration in minutes. The default is 10.
    min_freq : float, optional
        Minimum observation frequency in MHz. The default is 0.
    max_freq : float, optional
        Maximum observation frequency in MHz. The default is 5000.
    verbose : bool, optional
        Print all the things. The default is True.
    max_frac_bw : float, optional
        Maximum ratio of the bandwidth and observation frequency. The default
        is 2.

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
