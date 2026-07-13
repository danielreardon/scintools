#!/usr/bin/env python

"""
scint_sim.py
----------------------------------
Simulation tools for pulsar scintillation.

This module provides three classes:

* `Simulation` - a wave-optics simulator of a thin scattering screen that
  produces a simulated dynamic spectrum (and related products such as the
  phase screen, electric field, and scatter-broadened pulse shape). Based on
  original MATLAB code by Coles et al. (2010), "Scattering of pulsar radio
  emission by the interstellar plasma".
* `ACF` - computes the theoretical autocorrelation function (ACF) and
  secondary spectrum of intensity for strong scintillation, following the
  analytic treatment in Appendix A of Rickett, Coles et al. (2014), ApJ 787,
  161.
* `Brightness` - computes the angular brightness distribution, delay-Doppler
  (secondary) spectrum, and ACF of a scattered wave from the angular
  spectrum, based on Yao et al. (2020) with corrections to the phase
  gradient terms.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from numpy import random
from numpy.random import randn
from numpy.fft import fft2, ifft2
from scipy.special import gamma
from scipy.interpolate import griddata
import scipy.constants as sc
import matplotlib.pyplot as plt
from scintools.scint_utils import is_valid, get_window


class Simulation():
    """
    Wave-optics simulator of a thin scattering screen.

    Generates a random phase screen with a given power-law structure
    function and anisotropy, propagates it through the Fresnel diffraction
    integral to obtain the electric field and intensity as a function of
    observer position and frequency, and packages the result as a simulated
    dynamic spectrum (with physical time/frequency axes) for use elsewhere
    in scintools. Based on original MATLAB code by Coles et al. (2010).
    """

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                 inner=0.001, ns=256, nf=256, dlam=0.25, lamsteps=False,
                 seed=None, nx=None, ny=None, dx=None, dy=None, plot=False,
                 verbose=False, freq=1400, dt=30, mjd=60000, nsub=None,
                 efield=False, noise=None):
        """
        Electromagnetic simulator based on original code by Coles et al.
        (2010).

        On construction, this generates the phase screen, propagates it to
        get the electric field and intensity, computes the dynamic spectrum
        (if ``nf > 1``), and computes the scatter-broadened pulse response.
        The result is stored as ``self.dyn`` with physical axes
        (``self.freqs``, ``self.times``) so that a `Simulation` instance can
        be passed directly to scintools' `Dynspec` class.

        Parameters
        ----------
        mb2 : float, optional
            Max Born parameter, sets the strength of scattering.
        rf : float, optional
            Fresnel scale, used as the length unit for the simulation.
        ds : float, optional
            Spatial step size (in units of `rf`), used for both x and y
            unless overridden by `dx`/`dy`.
        alpha : float, optional
            Structure function exponent (Kolmogorov turbulence = 5/3).
        ar : float, optional
            Axial ratio of the anisotropy of the scattering screen.
        psi : float, optional
            Orientation angle (degrees) of the anisotropy.
        inner : float, optional
            Inner scale of turbulence, in units of `rf`. Should generally be
            smaller than `ds`.
        ns : int, optional
            Number of spatial steps in x and y, used unless overridden by
            `nx`/`ny`.
        nf : int, optional
            Number of frequency steps across the fractional bandwidth
            `dlam`.
        dlam : float, optional
            Fractional bandwidth relative to the centre frequency.
        lamsteps : bool, optional
            If True, take equally-spaced steps in wavelength rather than in
            frequency.
        seed : int or None, optional
            Seed for the random phase screen, for reproducible simulations.
            Use -1 to reshuffle (i.e. use a fresh, unseeded draw). Default
            of None uses NumPy's global random state.
        nx : int or None, optional
            Number of spatial steps in x. Overrides `ns` when set.
        ny : int or None, optional
            Number of spatial steps in y. Overrides `ns` when set.
        dx : float or None, optional
            Spatial step size in x (in units of `rf`). Overrides `ds` when
            set.
        dy : float or None, optional
            Spatial step size in y (in units of `rf`). Overrides `ds` when
            set.
        plot : bool, optional
            If True, plot the screen, intensity, and dynamic spectrum after
            the simulation is computed (calls `plot_all`).
        verbose : bool, optional
            If True, print progress messages while the simulation runs.
        freq : float, optional
            Centre observing frequency, in MHz, used to set the physical
            frequency axis of the simulated dynamic spectrum.
        dt : float, optional
            Subintegration time, in seconds, used to set the physical time
            axis of the simulated dynamic spectrum.
        mjd : float, optional
            MJD of the start of the observation, recorded in the header.
        nsub : int or None, optional
            Number of subintegrations (spatial steps in x) to keep in the
            output dynamic spectrum. If None, all `nx` steps are kept.
        efield : bool, optional
            If True, store the (real part of the) electric field instead of
            the intensity in ``self.dyn``.
        noise : float or None, optional
            Noise level parameter, accepted for interface compatibility.
            Currently unused within this method.
        """

        self.mb2 = mb2
        self.rf = rf
        self.ds = ds
        self.dx = dx if dx is not None else ds
        self.dy = dy if dy is not None else ds
        self.alpha = alpha
        self.ar = ar
        self.psi = psi
        self.inner = inner
        self.nx = nx if nx is not None else ns
        self.ny = ny if ny is not None else ns
        self.nf = nf
        self.dlam = dlam
        self.lamsteps = lamsteps
        self.seed = seed

        # Now run simulation
        self.set_constants()
        if verbose:
            print('Computing screen phase')
        self.get_screen()
        if verbose:
            print('Getting intensity...')
        self.get_intensity(verbose=verbose)
        if nf > 1:
            if verbose:
                print('Computing dynamic spectrum')
            self.get_dynspec()
        if verbose:
            print('Getting impulse response...')
        self.get_pulse()
        if plot:
            self.plot_all()

        # Now prepare simulation for use with scintools, using physical units
        self.name =\
            'sim:mb2={0},ar={1},psi={2},dlam={3}'.format(self.mb2, self.ar,
                                                         self.psi, self.dlam)
        if lamsteps:
            self.name += ',lamsteps'

        self.header = [self.name, 'MJD0: {}'.format(mjd)]
        if efield:
            dyn = np.real(self.spe)
        else:
            dyn = self.spi
        dlam = self.dlam

        self.dt = dt
        self.freq = freq
        self.nsub = int(np.shape(dyn)[0]) if nsub is None else nsub
        self.nchan = int(np.shape(dyn)[1])
        # lams = np.linspace(1-self.dlam/2, 1+self.dlam/2, self.nchan)
        # freqs = np.divide(1, lams)
        # freqs = np.linspace(np.min(freqs), np.max(freqs), self.nchan)
        # self.freqs = freqs*self.freq/np.mean(freqs)
        if not lamsteps:
            self.df = self.freq*self.dlam/(self.nchan - 1)
            self.freqs = self.freq + np.arange(-self.nchan/2,
                                               self.nchan/2, 1)*self.df
        else:
            self.lam = sc.c/(self.freq*10**6)  # centre wavelength in m
            self.dl = self.lam*self.dlam/(self.nchan - 1)
            self.lams = self.lam + np.arange(-self.nchan/2,
                                             self.nchan/2, 1)*self.dl
            self.freqs = sc.c/self.lams/10**6  # in MHz
            self.freq = (np.max(self.freqs) - np.min(self.freqs))/2
        self.bw = max(self.freqs) - min(self.freqs)
        self.times = self.dt*np.arange(0, self.nsub)
        self.df = self.bw/self.nchan
        self.tobs = float(self.times[-1] - self.times[0])
        self.mjd = mjd
        if nsub is not None:
            dyn = dyn[0:nsub, :]
        self.dyn = np.transpose(dyn)

        # # Theoretical arc curvature
        V = self.ds / self.dt
        lambda0 = self.freq  # wavelength, c=1
        k = 2*np.pi/lambda0  # wavenumber
        L = self.rf**2 * k
        # Curvature to use for Dynspec object within scintools
        self.eta = L/(2 * V**2) / 10**6 / np.cos(psi * np.pi/180)**2
        c = sc.c  # m/s (speed of light)
        beta_to_eta = c*1e6/((self.freq*10**6)**2)
        # Curvature for wavelength-rescaled dynamic spectrum
        self.betaeta = self.eta / beta_to_eta

        return

    def set_constants(self):
        """
        Precompute constants used by the simulation.

        Derives the Fresnel-filter constants (`ffconx`, `ffcony`), the
        spatial coherence scale (`s0`), the normalization of the phase
        power spectrum (`consp`), the FFT normalization (`scnorm`), and the
        reference scale (`sref`) from the current values of `nx`, `ny`,
        `dx`, `dy`, `alpha`, `mb2`, and `rf`. Sets these as attributes on
        the instance; does not return a value.
        """
        ns = 1
        lenx = self.nx*self.dx
        leny = self.ny*self.dy
        self.ffconx = (2.0/(ns*lenx*lenx))*(np.pi*self.rf)**2
        self.ffcony = (2.0/(ns*leny*leny))*(np.pi*self.rf)**2
        dqx = 2*np.pi/lenx
        dqy = 2*np.pi/leny
        # dqx2 = dqx*dqx
        # dqy2 = dqy*dqy
        a2 = self.alpha*0.5
        # spow = (1.0+a2)*0.5
        # ap1 = self.alpha+1.0
        # ap2 = self.alpha+2.0
        aa = 1.0+a2
        ab = 1.0-a2
        cdrf = 2.0**(self.alpha)*np.cos(self.alpha*np.pi*0.25)\
            * gamma(aa)/self.mb2
        self.s0 = self.rf*cdrf**(1.0/self.alpha)

        cmb2 = self.alpha*self.mb2 / (4*np.pi *
                                      gamma(ab)*np.cos(self.alpha *
                                                       np.pi*0.25)*ns)
        self.consp = cmb2*dqx*dqy/(self.rf**self.alpha)
        self.scnorm = 1.0/(self.nx*self.ny)

        # ffconlx = ffconx*0.5
        # ffconly = ffcony*0.5
        self.sref = self.rf**2/self.s0
        return

    def get_screen(self):
        """
        Get phase screen in x and y
        """
        random.seed(self.seed)  # Set the seed, if any

        nx2 = int(self.nx/2 + 1)
        ny2 = int(self.ny/2 + 1)

        w = np.zeros([self.nx, self.ny])  # initialize array
        dqx = 2*np.pi/(self.dx*self.nx)
        dqy = 2*np.pi/(self.dy*self.ny)

        # first do ky=0 line
        k = np.arange(2, nx2+1)
        w[k-1, 0] = self.swdsp(kx=(k-1)*dqx, ky=0)
        w[self.nx+1-k, 0] = w[k, 0]
        # then do kx=0 line
        ll = np.arange(2, ny2+1)
        w[0, ll-1] = self.swdsp(kx=0, ky=(ll-1)*dqy)
        w[0, self.ny+1-ll] = w[0, ll-1]
        # now do the rest of the field
        kp = np.arange(2, nx2+1)
        k = np.arange((nx2+1), self.nx+1)
        km = -(self.nx-k+1)
        for il in range(2, ny2+1):
            w[kp-1, il-1] = self.swdsp(kx=(kp-1)*dqx, ky=(il-1)*dqy)
            w[k-1, il-1] = self.swdsp(kx=km*dqx, ky=(il-1)*dqy)
            w[self.nx+1-kp, self.ny+1-il] = w[kp-1, il-1]
            w[self.nx+1-k, self.ny+1-il] = w[k-1, il-1]

        # done the whole screen weights, now generate complex gaussian array
        xyp = np.multiply(w, np.add(randn(self.nx, self.ny),
                                    1j*randn(self.nx, self.ny)))

        xyp = np.real(fft2(xyp))
        self.w = w
        self.xyp = xyp
        return

    def get_intensity(self, verbose=True):
        """
        Propagate the phase screen to the observer plane at each frequency.

        For each of the `nf` frequency channels, applies the appropriate
        Fresnel scaling to the phase screen, propagates it via the Fresnel
        filter (`frfilt3`), and takes a 1D cut through the resulting
        electric field (through the centre row in y) as a function of x.
        Requires `get_screen` to have been run first (i.e. `self.xyp` to
        exist).

        Parameters
        ----------
        verbose : bool, optional
            If True, print progress as a percentage while looping over
            frequency channels.

        Returns
        -------
        None
            Sets ``self.xyi`` (intensity of the last-computed frequency's
            2D field) and ``self.spe`` (complex electric field as a
            function of x and frequency) as attributes.
        """
        spe = np.zeros([self.nx, self.nf],
                       dtype=np.dtype(np.csingle)) + \
            1j*np.zeros([self.nx, self.nf],
                        dtype=np.dtype(np.csingle))
        for ifreq in range(0, self.nf):
            if verbose:
                if ifreq % round(self.nf/100) == 0:
                    print(int(np.floor((ifreq+1)*100/self.nf)), '%')
            if self.lamsteps:
                scale = 1.0 +\
                    self.dlam * (ifreq - 1 - (self.nf / 2)) / (self.nf)
            else:
                frfreq = 1.0 +\
                    self.dlam * (-0.5 + ifreq / self.nf)
                scale = 1 / frfreq
            scaled = scale
            xye = fft2(np.exp(1j * self.xyp * scaled))
            xye = self.frfilt3(xye, scale)
            xye = ifft2(xye)
            gam = 0
            spe[:, ifreq] = xye[:, int(np.floor(self.ny / 2))] / scale**gam

        xyi = np.real(np.multiply(xye, np.conj(xye)))

        self.xyi = xyi
        self.spe = spe
        return

    def get_dynspec(self):
        """
        Compute the dynamic spectrum (intensity) from the electric field.

        Requires `get_intensity` to have been run first (i.e. `self.spe`
        to exist). Also computes the spatial x-axis (`self.x`) and the
        normalized wavelength (`self.lams`) and frequency (`self.freqs`)
        axes across the fractional bandwidth.

        Returns
        -------
        None
            Sets ``self.spi`` (dynamic spectrum, intensity vs x and
            frequency), ``self.x``, ``self.lams``, and ``self.freqs`` as
            attributes.
        """
        if self.nf == 1:
            print('no spectrum because nf=1')

        # dynamic spectrum
        spi = np.real(np.multiply(self.spe, np.conj(self.spe)))
        self.spi = spi

        self.x = np.linspace(0, self.dx*(self.nx), (self.nx))
        ifreq = np.linspace(0, self.nf-1, self.nf)
        lam_norm = 1.0 + self.dlam * (ifreq - 1 - (self.nf / 2)) / self.nf
        self.lams = lam_norm / np.mean(lam_norm)
        frfreq = 1.0 + self.dlam * (-0.5 + ifreq / self.nf)
        self.freqs = frfreq / np.mean(frfreq)
        return

    def get_pulse(self):
        """
        script to get the pulse shape vs distance x from spe

        you usually need a spectral window because the leading edge of the
        pulse response is very steep. it is also attractive to pad the spe file
        with zeros before FT of course this correlates adjacent samples in the
        pulse response
        """
        if not hasattr(self, 'spe'):
            self.get_intensity()

        # get electric field impulse response
        p = np.fft.fft(np.multiply(self.spe, np.blackman(self.nf)), 2*self.nf)
        p = np.real(p*np.conj(p))  # get intensity impulse response
        # shift impulse to middle of window
        self.pulsewin = np.transpose(np.roll(p, self.nf))

        # get phase delay from the phase screen
        # get units of 1/2BW from phase
        self.dm = self.xyp[:, int(self.ny/2)]*self.dlam/np.pi

    def swdsp(self, kx=0, ky=0):
        """
        Amplitude of the phase-screen power spectrum at given wavenumbers.

        Evaluates the square root of the (anisotropic, power-law) phase
        power spectral density, including the isotropic inner-scale
        cutoff, at the wavenumber(s) `(kx, ky)`. Used by `get_screen` to
        build the phase screen from a complex Gaussian random field.

        Parameters
        ----------
        kx : float or numpy.ndarray, optional
            Wavenumber(s) in the x direction.
        ky : float or numpy.ndarray, optional
            Wavenumber(s) in the y direction.

        Returns
        -------
        float or numpy.ndarray
            Amplitude weighting to apply at `(kx, ky)`, same shape as the
            broadcast of `kx` and `ky`.
        """
        cs = np.cos(self.psi*np.pi/180)
        sn = np.sin(self.psi*np.pi/180)
        r = self.ar
        con = np.sqrt(self.consp)
        alf = -(self.alpha+2)/4
        # anisotropy parameters
        a = (cs**2)/r + r*sn**2
        b = r*cs**2 + sn**2/r
        c = 2*cs*sn*(1/r-r)
        q2 = a * np.power(kx, 2) + b * np.power(ky, 2) + c*np.multiply(kx, ky)
        # isotropic inner scale
        out = con*np.multiply(np.power(q2, alf),
                              np.exp(-(np.add(np.power(kx, 2),
                                              np.power(ky, 2))) *
                                     self.inner**2/2))
        return out

    def frfilt3(self, xye, scale):
        """
        Apply the Fresnel-propagation filter to a 2D field, in place.

        Multiplies the four quadrants of the 2D Fourier-domain field `xye`
        by the appropriate Fresnel phase factor (a function of wavenumber
        and `scale`), using the Fresnel-filter constants (`ffconx`,
        `ffcony`) computed by `set_constants`.

        Parameters
        ----------
        xye : numpy.ndarray
            2D complex array (Fourier transform of the field) to filter,
            of shape ``(nx, ny)``. Modified in place.
        scale : float
            Fresnel scaling factor for the current frequency channel
            (relative wavelength/frequency scale).

        Returns
        -------
        numpy.ndarray
            The filtered array (the same object as `xye`, modified in
            place).
        """
        nx2 = int(self.nx / 2) + 1
        ny2 = int(self.ny / 2) + 1
        filt = np.zeros([nx2, ny2], dtype=np.dtype(np.csingle))
        q2x = np.linspace(0, nx2-1, nx2)**2 * scale * self.ffconx
        for ly in range(0, ny2):
            q2 = q2x + (self.ffcony * (ly**2) * scale)
            filt[:, ly] = np.cos(q2) - 1j * np.sin(q2)

        xye[0:nx2, 0:ny2] = np.multiply(xye[0:nx2, 0:ny2], filt[0:nx2, 0:ny2])
        xye[self.nx:nx2-1:-1, 0:ny2] = np.multiply(
            xye[self.nx:nx2-1:-1, 0:ny2], filt[1:(nx2 - 1), 0:ny2])
        xye[0:nx2, self.ny:ny2-1:-1] =\
            np.multiply(xye[0:nx2, self.ny:ny2-1:-1], filt[0:nx2, 1:(ny2-1)])
        xye[self.nx:nx2-1:-1, self.ny:ny2-1:-1] =\
            np.multiply(xye[self.nx:nx2-1:-1, self.ny:ny2-1:-1],
                        filt[1:(nx2-1), 1:(ny2-1)])
        return xye

    def plot_screen(self, subplot=False):
        """
        Plot the simulated phase screen in x and y.

        Computes the screen first via `get_screen` if it has not already
        been computed.

        Parameters
        ----------
        subplot : bool, optional
            If True, draw onto the current axes without calling
            ``plt.show()`` (for use as part of a larger figure, e.g. by
            `plot_all`). If False, show the figure immediately.
        """
        if not hasattr(self, 'xyp'):
            self.get_screen()
        x_steps = np.linspace(0, self.dx*self.nx, self.nx)
        y_steps = np.linspace(0, self.dy*self.ny, self.ny)
        plt.pcolormesh(x_steps, y_steps, np.transpose(self.xyp))
        plt.title("Screen phase")
        plt.ylabel('$y/r_f$')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_intensity(self, subplot=False):
        """
        Plot the observer-side intensity fluctuations in space, at the
        centre frequency.

        Computes the intensity first via `get_intensity` if it has not
        already been computed.

        Parameters
        ----------
        subplot : bool, optional
            If True, draw onto the current axes without calling
            ``plt.show()`` (for use as part of a larger figure, e.g. by
            `plot_all`). If False, show the figure immediately.
        """
        if not hasattr(self, 'xyi'):
            self.get_intensity()
        x_steps = np.linspace(0, self.dx*(self.nx), (self.nx))
        y_steps = np.linspace(0, self.dy*(self.ny), (self.ny))
        plt.pcolormesh(x_steps, y_steps, np.transpose(self.xyi))
        plt.title('Intensity / Mean')
        plt.ylabel('$y/r_f$')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_dynspec(self, subplot=False):
        """
        Plot the simulated dynamic spectrum (intensity vs x and
        frequency/wavelength).

        Computes the dynamic spectrum first via `get_dynspec` if it has
        not already been computed.

        Parameters
        ----------
        subplot : bool, optional
            If True, draw onto the current axes without calling
            ``plt.show()`` (for use as part of a larger figure, e.g. by
            `plot_all`). If False, show the figure immediately.
        """
        if not hasattr(self, 'spi'):
            self.get_dynspec()

        if self.lamsteps:
            plt.pcolormesh(self.x, self.lams, np.transpose(self.spi))
            plt.ylabel(r'Wavelength $\lambda$')
        else:
            plt.pcolormesh(self.x, self.freqs, np.transpose(self.spi))
            plt.ylabel('Frequency f')
        plt.title('Dynamic Spectrum (Intensity/Mean)')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_efield(self, subplot=False):
        """
        Plot the real part of the simulated electric field vs x and
        frequency/wavelength.

        Computes the electric field first via `get_intensity` if it has
        not already been computed.

        Parameters
        ----------
        subplot : bool, optional
            If True, draw onto the current axes without calling
            ``plt.show()``. If False, show the figure immediately.
        """
        if not hasattr(self, 'spe'):
            self.get_intensity()

        if self.lamsteps:
            plt.pcolormesh(self.x, self.lams,
                           np.real(np.transpose(self.spe)))
            plt.ylabel(r'Wavelength $\lambda$')
        else:
            plt.pcolormesh(self.x, self.freqs,
                           np.real(np.transpose(self.spe)))
            plt.ylabel('Frequency f')
        plt.title('Electric field (Intensity/Mean)')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_delay(self, subplot=False):
        """
        Plot the dispersive group delay vs x (at the centre frequency) and
        the pulse-averaged impulse response function vs delay.

        Requires `get_pulse` to have been run first (i.e. `self.dm` and
        `self.pulsewin` to exist).

        Parameters
        ----------
        subplot : bool, optional
            Accepted for interface consistency with the other ``plot_*``
            methods; currently unused (the figure is always shown).
        """
        # get frequency to set the scale, enter in GHz
        Freq = self.freq/1000
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, self.dx*self.nx, self.nx),
                 -self.dm/(2*self.dlam*Freq))
        plt.ylabel('Group delay (ns)')
        plt.xlabel('$x/r_f$')
        plt.subplot(2, 1, 2)
        plt.plot(np.mean(self.pulsewin, axis=1))
        plt.ylabel('Intensity (arb)')
        plt.xlabel('Delay (arb)')
        plt.show()
        return

    def plot_pulse(self, subplot=False):
        """
        Plot the (log) intensity of the scatter-broadened pulse vs x and
        delay, overlaid with the group delay from the phase screen.

        Requires `get_pulse` to have been run first (i.e. `self.dm` and
        `self.pulsewin` to exist).

        Parameters
        ----------
        subplot : bool, optional
            Accepted for interface consistency with the other ``plot_*``
            methods; currently unused (the figure is always shown).
        """
        # get frequency to set the scale, enter in GHz
        Freq = self.freq/1000
        lpw = np.log10(self.pulsewin)
        vmax = np.max(lpw)
        vmin = np.median(lpw) - 3
        plt.pcolormesh(np.linspace(0, self.dx*self.nx, self.nx),
                       (np.arange(0, 3*self.nf/2, 1) - self.nf/2) /
                       (2*self.dlam*Freq),
                       lpw[int(self.nf/2):, :], vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.ylabel('Delay (ns)')
        plt.xlabel('$x/r_f$')
        plt.plot(np.linspace(0, self.dx*self.nx, self.nx),
                 -self.dm/(2*self.dlam*Freq), 'k')  # group delay=-phase delay
        plt.show()

    def plot_all(self):
        """
        Plot the phase screen, intensity, and dynamic spectrum together as
        subplots in a single figure.
        """
        plt.figure(2)
        plt.subplot(2, 2, 1)
        self.plot_screen(subplot=True)
        plt.subplot(2, 2, 2)
        self.plot_intensity(subplot=True)
        plt.subplot(2, 1, 2)
        self.plot_dynspec(subplot=True)
        plt.show()


class ACF():
    """
    Theoretical autocorrelation function (ACF) of scintillation intensity.

    Computes the 2D ACF of intensity vs time and frequency (and the
    corresponding secondary spectrum) directly from the theoretical
    function for strong scintillation given in Appendix A of Rickett,
    Coles et al. (2014), ApJ 787, 161, including the effects of
    anisotropy and a phase gradient (e.g. due to a binary companion).
    The magnitude of the effective velocity is defined to be 1, so time
    and frequency lags are expressed in units of the (isotropic)
    scintillation timescale and decorrelation bandwidth respectively.
    """

    def __init__(self, psi=0, phasegrad=0, theta=0, ar=1, alpha=5/3,
                 taumax=4, dnumax=4, nf=51, nt=51, amp=1, wn=0,
                 spatial_factor=2, resolution_factor=1, core_factor=2,
                 auto_sampling=True, plot=False, display=True):
        """
        Generate an ACF from the theoretical function in Rickett, Coles et
        al. (2014).

        Magnitude of velocity is defined to be 1. On construction, this
        computes the ACF via `calc_acf`.

        Parameters
        ----------
        psi : float, optional
            Angle (degrees) of the velocity vector with respect to the
            major axis of the (brightness) anisotropy.
        phasegrad : float, optional
            Magnitude of the phase gradient (e.g. due to a binary
            companion).
        theta : float, optional
            Angle (degrees) of the phase gradient with respect to the
            velocity vector.
        ar : float, optional
            Axial ratio of the anisotropy.
        alpha : float, optional
            Structure function exponent (Kolmogorov turbulence = 5/3).
        taumax : float, optional
            Number of scintillation timescales to compute the ACF out to
            (equivalently, the ACF in space goes out to ``taumax*V``).
        dnumax : float, optional
            Number of decorrelation bandwidths to compute the ACF out to.
        nf : int, optional
            Number of frequency lag samples in the ACF. If not odd, it is
            incremented by one so the ACF has a well-defined centre.
        nt : int, optional
            Number of time lag samples in the ACF. If not odd, it is
            incremented by one so the ACF has a well-defined centre.
        amp : float, optional
            Amplitude to scale the ACF by.
        wn : float, optional
            Size of the white-noise spike at the origin of the ACF.
        spatial_factor : float, optional
            Multiplier for the spatial extent (`taumax`) used when
            calculating the ACF of the electric field. Only used if
            `auto_sampling` is False.
        resolution_factor : float, optional
            Multiplier applied to the default spatial resolution used when
            calculating the ACF of the electric field. Only used if
            `auto_sampling` is False.
        core_factor : float, optional
            Additional resolution multiplier applied near the origin
            (first frequency-lag sample), where the integrand varies most
            rapidly. Only used if `auto_sampling` is False.
        auto_sampling : bool, optional
            If True, automatically set the spatial sampling factors
            (`sp_fac`, `res_fac`, `core_fac`) based on `ar` and `taumax`,
            to avoid sampling artefacts, overriding `spatial_factor`,
            `resolution_factor`, and `core_factor`. Warning: computation
            time increases sharply for large `ar`.
        plot : bool, optional
            If True, plot the ACF after it is computed.
        display : bool, optional
            If True and `plot` is True, show the plot immediately.
        """

        self.alpha = alpha
        # anisotropy
        self.ar = ar
        self.psi = psi
        # phase gradients
        self.phasegrad = phasegrad
        self.theta = theta
        # amplitude and white noise spikes (for fitting to real data)
        self.amp = amp
        self.wn = wn
        # sampling parameters
        self.taumax = taumax
        spmax = taumax
        self.dnumax = dnumax
        if nf % 2 == 0:
            nf += 1  # make odd so the ACF has a centre
        if nt % 2 == 0:
            nt += 1  # make odd so the ACF has a centre
        self.nf = nf
        self.nt = nt
        if auto_sampling:
            # calculate to 6 spatial scales along major axis
            self.sp_fac = 6 * ar/spmax
            # adjust to 101 pixels, doubles at ar=3
            self.res_fac = 1 + ar/3
            # quadruple near core
            self.core_fac = 4
        else:
            self.sp_fac = spatial_factor
            self.res_fac = resolution_factor
            self.core_fac = core_factor

        # default resolutions with sampling factors = 1
        self.dsp = 4*spmax/(nt-1)

        # calculate the ACF
        self.calc_acf()

        if plot:
            self.plot_acf(display=display)

        return

    def calc_acf(self, plot=False):
        r"""
        Compute the 2D ACF of intensity vs time and frequency.

        Implements the integrals in Appendix A of Rickett, Coles et al.
        (2014) (equations A1 and A2) for the ACF of intensity, given the
        anisotropy and the angular displacement due to any phase gradient.
        ``psi = 0`` defines the brightness-distribution anisotropy as
        aligned with the velocity V; ``theta = 0`` defines the phase
        gradient as aligned with V. (Within the code, the x-axis, see the
        `Vx` and `sigxn` variables, is the major axis of the ACF-efield
        anisotropy, which corresponds to ``psi=90``.)

        Coordinates in the code are with respect to the `ar` major axis,
        so the structure itself does not need to be rotated; instead V and
        the phase gradient are rotated into the structure coordinates.
        The spatial lag ``sn`` is normalized by :math:`s_0` and the
        frequency lag ``dnun`` by :math:`\nu_{0.5}`, the spatial and
        frequency scales respectively; the phase gradient is normalized by
        :math:`1/s_0` (i.e. ``sigxn = gradphix * s0``).

        Parameters
        ----------
        plot : bool, optional
            If True, plot the resulting ACF (via `plot_acf`) after it is
            computed.

        Returns
        -------
        None
            Sets ``self.acf`` (2D ACF of intensity), ``self.tn`` (time-lag
            axis), ``self.fn`` (frequency-lag axis), ``self.sn``
            (equivalent to ``self.tn``), ``self.snp`` (spatial-lag axis of
            the electric-field ACF), and ``self.acf_efield`` (ACF of the
            electric field) as attributes.

        Notes
        -----
        If there is no phase gradient, the ACF is symmetric and only one
        quadrant needs to be calculated; otherwise two quadrants are
        necessary.

        The worst-case sampling is when ``dnun`` is very small: the
        argument of the complex exponential becomes large and aliasing
        will occur. If ``dnun=0.01`` and ``dsp=0.1``, the alias will peak
        at ``snx = 5``; reducing the spatial sampling ``dsp`` to 0.05 will
        push that alias out to ``snx = 8``, though halving ``dsp``
        increases the computation time by a factor of 4. Sampling can be
        tuned via the `spatial_factor`, `resolution_factor`, and
        `core_factor` parameters of `__init__` (or `auto_sampling`).

        The frequency decorrelation is quite linear near the origin and
        looks quasi-exponential; the half-power width is ``dnun = 0.15``,
        so a sampling of 0.05 in frequency is more than adequate, and a
        sampling of 0.1 in ``sn`` is adequate. ``dnun = 0.0`` is divergent
        in this integral, but is obtained trivially from the ACF of the
        electric field directly.

        Note also that Rickett, Coles et al. (2014) equation A2 as printed
        has an error: it would be correct if :math:`\nu` were replaced by
        :math:`\omega`, i.e. with an extra factor of :math:`2\pi`.
        """

        alph2 = self.alpha/2

        spmax = self.taumax
        dnumax = self.dnumax
        dsp = self.dsp
        phasegrad = self.phasegrad
        theta = self.theta
        amp = self.amp
        wn = self.wn
        # arcs are enhanced when velocity is parallel to brightness
        # distribution major axis (psi=0), which is perpendicular to ACF-efield
        xi = 90 - self.psi  # velocity angle w.r.t ACF-efield
        # calculate velocities parallel (Vx) and perpendicular (Vy) to e-field
        Vx = np.cos(xi*np.pi/180)
        Vy = np.sin(xi*np.pi/180)
        # calculate angular offsets parallel (sigxn) and perpendicular (sigyn)
        #    to V
        sigxn = phasegrad * np.cos((xi - theta)*np.pi/180)
        sigyn = phasegrad * np.sin((xi - theta)*np.pi/180)

        ar = self.ar
        sqrtar = np.sqrt(ar)
        # equally spaced dnu array dnu = dnun * nuhalf
        dnun = np.linspace(0, dnumax, int(np.ceil(self.nf/2)))
        ddnun = np.abs(dnun[1] - dnun[0])
        self.ddnun = ddnun
        ndnun = len(dnun)
        sp_fac = self.sp_fac
        res_fac = self.res_fac
        core_fac = self.res_fac * self.core_fac

        # Calculate ACF of e-field
        snp = np.arange(-sp_fac*spmax, sp_fac*spmax + dsp/res_fac, dsp/res_fac)
        SNPX, SNPY = np.meshgrid(snp, snp)
        # ACF of e-field
        gammes = np.exp(-0.5*((SNPX/sqrtar)**2 +
                              (SNPY*sqrtar)**2)**alph2)
        # Increase spatial resolution by factor of core_fac, for first dnu step
        snp2 = np.arange(-sp_fac*spmax, sp_fac*spmax + dsp/core_fac,
                         dsp/core_fac)
        SNPX2, SNPY2 = np.meshgrid(snp2, snp2)
        # ACF of e-field
        gammes2 = np.exp(-0.5*((SNPX2/sqrtar)**2 +
                               (SNPY2*sqrtar)**2)**alph2)

        if phasegrad == 0:
            # calculate only one quadrant tn >= 0
            # equally spaced t array t= tn*S0
            tn = np.linspace(0, (spmax), int(np.ceil(self.nt/2)))
            snx = Vx*tn
            sny = Vy*tn
            gammitv = np.zeros((int(len(snx)), int(ndnun)),
                               dtype=np.complex128)
            # compute dnun=0 first
            gammitv[:, 0] = np.exp(-0.5*((snx/sqrtar)**2 +
                                         (sny*sqrtar)**2)**alph2)
            gammitv[0, 0] += wn/amp
            for isn in range(0, len(snx)):
                ARG = ((SNPX2-snx[isn])**2 + (SNPY2-sny[isn])**2)/(2*dnun[1])
                temp = gammes2 * np.exp(1j*ARG)
                gammitv[isn, 1] = -1j*((dsp/core_fac)**2 *
                                       np.sum(temp)/((2*np.pi)*dnun[1]))
            # Now do remainder of dnu array
            for idn in range(2, ndnun):
                for isn in range(0, len(snx)):
                    ARG = ((SNPX-snx[isn])**2 +
                           (SNPY-sny[isn])**2)/(2*dnun[idn])
                    temp = gammes * np.exp(1j*ARG)
                    gammitv[isn, idn] = -1j*((dsp/res_fac)**2 * np.sum(temp) /
                                             ((2*np.pi)*dnun[idn]))

            # equation A1 convert ACF of E to ACF of I
            gammitv = np.real(gammitv * np.conj(gammitv))

            # Build first half
            nr, nc = np.shape(gammitv)
            gam2 = np.zeros((nr, nc*2-1))
            gam2[:, 0:nc-1] = np.fliplr(gammitv[:, 1:])
            gam2[:, nc-1:] = gammitv
            gam2 = gam2.squeeze()

            # Build full ACF
            gam3 = np.zeros((nr*2-1, nc*2-1))
            gam3[0:nr-1, :] = np.flipud(gam2[1:, :])
            gam3[nr-1:, :] = gam2
            gam3 = np.transpose(gam3)

            t2 = np.concatenate((np.flip(-tn[1:]), tn)).squeeze()
            f2 = np.concatenate((np.flip(-dnun[1:]), dnun)).squeeze()

        else:
            # calculate two quadrants -tmax t < tmax
            # equally spaced t array t= tn*S0
            tn = np.linspace(-(spmax), (spmax), self.nt)
            snx = np.cos(xi*np.pi/180)*tn
            sny = np.sin(xi*np.pi/180)*tn
            # compute dnun=0 first
            gammitv = np.zeros((int(len(snx)), int(ndnun)),
                               dtype=np.complex128)
            gammitv[:, 0] = np.exp(-0.5*((snx/sqrtar)**2 +
                                         (sny*sqrtar)**2)**alph2)
            gammitv[np.argwhere(snx == 0), 0] += wn/amp
            for isn in range(0, len(snx)):
                snxt = snx - 2*sigxn*dnun[1]
                snyt = sny - 2*sigyn*dnun[1]
                ARG = ((SNPX2-snxt[isn])**2 + (SNPY2-snyt[isn])**2)/(2*dnun[1])
                temp = gammes2 * np.exp(1j*ARG)
                gammitv[isn, 1] = -1j*((dsp/core_fac)**2 *
                                       np.sum(temp)/((2*np.pi)*dnun[1]))
            for idn in range(2, ndnun):
                snxt = snx - 2*sigxn*dnun[idn]
                snyt = sny - 2*sigyn*dnun[idn]
                for isn in range(0, len(snx)):
                    ARG = ((SNPX-snxt[isn])**2 +
                           (SNPY-snyt[isn])**2)/(2*dnun[idn])
                    temp = gammes*np.exp(1j*ARG)
                    gammitv[isn, idn] = -1j*((dsp/res_fac)**2 * np.sum(temp) /
                                             ((2*np.pi)*dnun[idn]))

            # equation A1 convert ACF of E to ACF of I
            gammitv = np.real(gammitv * np.conj(gammitv))

            # Build ACF
            nr, nc = np.shape(gammitv)
            gam3 = np.zeros((nr, nc*2-1))
            gam3[:, 0:nc-1] = np.fliplr(np.flipud(gammitv[:, 1:]))
            gam3[:, nc-1:] = gammitv
            gam3 = np.transpose(gam3)

            f2 = np.concatenate((np.flip(-dnun[1:]), dnun)).squeeze()
            t2 = tn

        self.fn = f2
        self.tn = t2
        self.sn = t2
        self.snp = snp
        self.acf = amp * gam3
        self.acf_efield = gammes

        if plot:
            self.plot_acf()

        return

    def plot_acf(self, display=True, contour=True, filled=False):
        """
        Plot the theoretical 2D ACF of intensity vs time and frequency
        lag.

        Requires `calc_acf` to have been run first (i.e. `self.acf`,
        `self.tn`, `self.fn` to exist).

        Parameters
        ----------
        display : bool, optional
            If True, set the plot title and call ``plt.show()``
            immediately.
        contour : bool, optional
            If True and `filled` is False, overlay black contours at
            ``amp * [0.2, 0.4, 0.6, 0.8]``.
        filled : bool, optional
            If True, plot filled contours (``plt.contourf``) instead of a
            pcolormesh.
        """
        # for plotting, we need to expand tn and fn,
        #   since they are pixel edges, not centres
        dtn = np.abs(self.tn[1] - self.tn[0])
        tn_edges = self.tn - dtn/2
        self.tn_edges = np.append(tn_edges, tn_edges[-1] + dtn)

        dfn = np.abs(self.fn[1] - self.fn[0])
        fn_edges = self.fn - dfn/2
        self.fn_edges = np.append(fn_edges, fn_edges[-1] + dfn)

        if not filled:
            plt.pcolormesh(self.tn_edges, self.fn_edges, self.acf)
            if contour:
                plt.contour(self.tn, self.fn, self.acf,
                            self.amp*[0.2, 0.4, 0.6, 0.8], colors='k')

        else:
            plt.contourf(self.tn, self.fn, self.acf,
                         self.amp*[0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.6, 0.7, 0.8, 0.9, 1])

        plt.xlabel(r'Time lag ($\tau/\tau_{d,\rm{iso}}$)')
        plt.ylabel(r'Frequency lag ($\Delta\nu/\Delta\nu_{d,\rm{iso}}$)')
        if display:
            plt.title('ACF of intensity')
            plt.show()

    def plot_acf_efield(self, display=True):
        """
        Plot the ACF of the electric field vs spatial lag.

        Requires `calc_acf` to have been run first (i.e. `self.acf_efield`
        and `self.snp` to exist).

        Parameters
        ----------
        display : bool, optional
            If True, call ``plt.show()`` immediately.
        """
        # for plotting, we need to expand tn and fn,
        #   since they are pixel edges, not centres
        dsnp = np.abs(self.snp[1] - self.snp[0])
        snp_edges = self.snp - dsnp/2
        snp_edges = np.append(snp_edges, snp_edges[-1] + dsnp)

        plt.pcolormesh(snp_edges, snp_edges, self.acf_efield)
        plt.xlabel(r'$S_x$ ($x/s_{d,\rm{iso}}$)')
        plt.ylabel(r'$S_y$ ($y/s_{d,\rm{iso}}$)')
        plt.title('ACF of electric field')
        if display:
            plt.show()

    def calc_sspec(self, window='hanning', window_frac=1):
        """
        Compute the secondary spectrum from the ACF.

        Applies a 2D window to the ACF (`self.acf`) before taking its 2D
        Fourier transform, to reduce edge effects/spectral leakage.

        Parameters
        ----------
        window : str, optional
            Name of the window function to apply along each axis before
            transforming (passed to `scintools.scint_utils.get_window`),
            e.g. 'hanning', 'blackman'.
        window_frac : float, optional
            Fraction of each axis over which the window is applied
            (passed to `scintools.scint_utils.get_window`).

        Returns
        -------
        None
            Sets ``self.sspec`` (secondary spectrum, in dB) as an
            attribute.
        """
        nf, nt = np.shape(self.acf)
        chan_window, subint_window = get_window(nt, nf, window=window,
                                                frac=window_frac)
        arr = np.multiply(chan_window, self.acf)
        arr = np.transpose(np.multiply(subint_window,
                                       np.transpose(arr)))
        arr = np.fft.fftshift(arr)
        arr = np.fft.fft2(arr)
        arr = np.fft.fftshift(arr)
        arr = np.sqrt(np.real(arr * np.conj(arr)))
        self.sspec = 10*np.log10(arr)

    def plot_sspec(self, display=True, vmin=None, vmax=None):
        """
        Plot the secondary spectrum vs time and frequency lag.

        Computes the secondary spectrum first via `calc_sspec` (with its
        default window) if it has not already been computed.

        Parameters
        ----------
        display : bool, optional
            If True, set the plot title and call ``plt.show()``
            immediately.
        vmin : float or None, optional
            Minimum of the colour scale, in dB. If None, defaults to 3 dB
            below the median of the (finite, non-zero) secondary
            spectrum.
        vmax : float or None, optional
            Maximum of the colour scale, in dB. If None, defaults to 3 dB
            below the maximum of the (finite, non-zero) secondary
            spectrum.
        """
        if not hasattr(self, 'sspec'):
            self.calc_sspec()

        sspec = self.sspec
        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval - 3 if vmin is None else vmin
        vmax = maxval - 3 if vmax is None else vmax

        plt.pcolormesh(self.tn, self.fn, sspec, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlabel(r'Delay')
        plt.ylabel(r'Doppler')
        plt.title('Secondary spectrum (dB)')
        if display:
            plt.show()

        return


class Brightness():
    """
    Angular brightness distribution and delay-Doppler (secondary) spectrum
    of a scattered wave interfering with an unscattered wave.

    Based on Yao et al. (2020), modified to correctly account for the
    phase gradient terms and to remove spurious bright points in the
    secondary spectrum (caused by a coordinate singularity) that would
    otherwise create artefacts in the ACF. The angular spectrum is
    defined by the (phase) structure function exponent. The ACF of the
    electric field is calculated first and 2D-FFT'ed to get the
    brightness distribution; the brightness distribution can then be
    offset by a phase gradient, which causes an angular shift as a
    fraction of the half-width of the distribution.
    """

    def __init__(self, ar=1.0, psi=0, alpha=1.67, thetagx=0, thetagy=0,
                 thetarx=0, thetary=0, df=0.02, dt=0.08, dx=0.1,
                 nf=10, nt=80, nx=30, ncuts=5, plot=False, contour=True,
                 figsize=(10, 8), calc_sspec=True, calc_acf=True):
        """
        Simulate the delay-Doppler spectrum from the scattered angular
        spectrum, following Yao et al. (2020).

        Here we assume that the angular spectrum interferes with an
        unscattered wave. First the ACF of the electric field is
        calculated (`calc_brightness`), then it is 2D-FFT'ed to get the
        brightness distribution. The brightness distribution can be
        offset by a phase gradient which causes an angular shift as a
        fraction of the half-width of the brightness distribution.

        The unscattered wave can also be offset by the phase gradient (as
        it would be in weak scattering), or it can be at zero offset (or
        anywhere else). The default is to set the phase gradient angle
        and the reference angle to be equal.

        Parameters
        ----------
        ar : float, optional
            Axial ratio of the anisotropy.
        psi : float, optional
            Orientation angle (degrees) of the anisotropy.
        alpha : float, optional
            Exponent of the phase structure function (Kolmogorov
            turbulence would be 5/3; note the default here, 1.67, is used
            as an approximation of 5/3).
        thetagx : float, optional
            Offset (in x) of the scattered wave due to the phase
            gradient.
        thetagy : float, optional
            Offset (in y) of the scattered wave due to the phase
            gradient.
        thetarx : float, optional
            Reference angle (in x) for the unscattered wave (normally
            equal to `thetagx`).
        thetary : float, optional
            Reference angle (in y) for the unscattered wave (normally
            equal to `thetagy`).
        df : float, optional
            Step size in Doppler (frequency-lag-like) coordinate `fd`
            used when computing the secondary spectrum.
        dt : float, optional
            Step size in delay (time-lag-like) coordinate `td` used when
            computing the secondary spectrum.
        dx : float, optional
            Spatial step size of the electric-field ACF grid, relative to
            the spatial scale.
        nf : float, optional
            Half-extent of the Doppler axis `fd`, which runs from ``-nf``
            to ``+nf`` in steps of `df`.
        nt : float, optional
            Half-extent of the delay axis `td`, which runs from ``-nt``
            to ``+nt`` in steps of `dt`.
        nx : float, optional
            Half-extent of the electric-field ACF spatial grid, which
            runs from ``-nx`` to ``+nx`` in steps of `dx`.
        ncuts : int, optional
            Number of Doppler cuts (at different delays) to plot in
            `plot_cuts`.
        plot : bool, optional
            If True, plot the electric-field ACF and brightness
            distribution after they are computed, and (if `calc_sspec`
            or `calc_acf` are True) the secondary spectrum, cuts, and/or
            ACF.
        contour : bool, optional
            If True and `plot` and `calc_acf` are True, overlay contours
            on the ACF plot.
        figsize : tuple of float, optional
            Figure size (width, height) in inches, used for all plots
            made during construction.
        calc_sspec : bool, optional
            If True, compute the secondary spectrum (`calc_SS`) after the
            brightness distribution.
        calc_acf : bool, optional
            If True, compute the ACF (`calc_acf`) from the secondary
            spectrum after it is computed.
        """

        self.ar = ar
        self.alpha = alpha
        self.thetagx = thetagx
        self.thetagy = thetagy
        self.thetarx = thetarx
        self.thetary = thetary
        self.psi = psi
        self.df = df
        self.dt = dt
        self.dx = dx
        self.nf = nf
        self.nt = nt
        self.nx = nx
        self.ncuts = ncuts

        # Calculate brighness distribution
        self.calc_brightness()
        if plot:
            self.plot_acf_efield(figsize=figsize)
            self.plot_brightness(figsize=figsize)

        # Calculate secondary spectrum
        if calc_sspec:
            self.calc_SS()
            if plot:
                self.plot_sspec(figsize=figsize)
                self.plot_cuts(figsize=figsize)

        # Calculate ACF
        if calc_acf:
            self.calc_acf()
            if plot:
                self.plot_acf(figsize=figsize, contour=contour)

    def calc_brightness(self):
        """
        Compute the angular brightness distribution from the ACF of the
        electric field.

        First builds the (anisotropic, power-law) ACF of the electric
        field, `self.acf_efield`, on a grid spanning ``-nx`` to ``+nx`` in
        steps of `dx` (distances referenced to the spatial scale in the
        X-direction). The brightness distribution is then obtained by
        2D Fourier transforming that ACF.

        Returns
        -------
        None
            Sets ``self.x`` (1D spatial axis), ``self.X``, ``self.Y`` (2D
            coordinate grids), ``self.acf_efield`` (ACF of the electric
            field), and ``self.B`` (angular brightness distribution) as
            attributes.
        """
        # first need to get the brightness distribution from the ACF of the
        # electric field. Reference distances to the spatial scale in the
        # X-direction

        x = np.arange(-self.nx, self.nx, self.dx)
        self.X, self.Y = np.meshgrid(x, x)

        R = (self.ar**2 - 1) / (self.ar**2 + 1)
        cosa = np.cos(2 * (90 - self.psi) * np.pi/180)
        sina = np.sin(2 * (90 - self.psi) * np.pi/180)
        # quadratic coefficients
        a = (1 - R * cosa) / np.sqrt(1 - R**2)
        b = (1 + R * cosa) / np.sqrt(1 - R**2)
        c = -2 * R * sina / np.sqrt(1 - R**2)

        # ACF of electric field
        Rho = np.exp(-0.5*(a * self.X**2 + b * self.Y**2 +
                           c * self.X * self.Y)
                     ** (self.alpha/2))
        # # Original code below with error
        # Rho = np.exp(-(a * self.X**2 + b * self.Y**2 +
        #                c * self.X * self.Y)
        #              ** (self.alpha/2))/2

        self.x = x
        self.acf_efield = Rho

        # get brightness distribution
        B = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(Rho)))
        self.B = np.abs(B)
        return

    def calc_SS(self):
        """
        Compute the delay-Doppler (secondary) spectrum from the brightness
        distribution.

        The secondary spectrum is defined with ``delay = theta**2`` (i.e.
        ``0.5*L/c = 1``) and ``doppler = theta`` (i.e. ``V/lambda = 1``).
        The differential delay `td` and differential Doppler `fd` are
        therefore::

            td = (thetax+thetagx)**2 + (thetay+thetagy)**2
                 - thetagx**2 - thetagy**2
            fd = (thetax + thetagx) - thetagx = thetax
            Jacobian = 1 / (thetay+thetagy)
            thetay + thetagy = sqrt(td - (thetax+thetagx)**2
                                     + thetagx**2 + thetagy**2)

        Requires `calc_brightness` to have been run first (i.e. `self.X`,
        `self.Y`, `self.B` to exist).

        Returns
        -------
        None
            Sets ``self.fd``, ``self.td`` (Doppler and delay axes),
            ``self.thetax``, ``self.thetay`` (angular coordinates
            corresponding to each `(td, fd)` pair), ``self.jacobian``,
            ``self.SS`` (secondary spectrum, linear), and ``self.LSS``
            (secondary spectrum, in dB) as attributes.

        Notes
        -----
        The arc in the secondary spectrum is defined by
        ``(thetay+thetagy) == 0``, where there is a half-order
        singularity. This singularity creates a problem in the code
        because the sampling in `(fd, td)` is not synchronized with the
        arc position, so there can be very bright points if a sample
        happens to lie very close to the singularity. This is not a
        problem for interpreting the secondary spectrum itself, but it
        causes large artefacts when Fourier transforming it to get the
        ACF. So (following Bill Coles' original MATLAB code) the Jacobian
        is limited by not allowing ``(thetay+thetagy)`` to be smaller than
        half the step size in `thetax`/`thetay` (`self.df`).
        """

        fd = np.arange(-self.nf, self.nf, self.df)
        td = np.arange(-self.nt, self.nt, self.dt)
        self.fd = fd
        self.td = td
        # now get the thetax and thetay corresponding to fd and td
        # first initialize arrays all of same size
        amp = np.zeros((len(td), len(fd)))
        thetax = np.zeros((len(td), len(fd)))
        thetay = np.zeros((len(td), len(fd)))
        SS = np.zeros((len(td), len(fd)))
        for ifd in range(0, len(fd)):
            for itd in range(0, len(td)):
                thetax[itd, ifd] = fd[ifd] - self.thetagx + self.thetarx
                thetayplusthetagysq = td[itd] - \
                    (thetax[itd, ifd] + self.thetagx)**2 + self.thetarx**2 + \
                    self.thetary**2
                if thetayplusthetagysq > 0:
                    thymthgy = np.sqrt(thetayplusthetagysq)  # thetay-thetagy
                    thetay[itd, ifd] += thymthgy - self.thetagy
                    if thymthgy < 0.5*self.df:
                        amp[itd, ifd] = 2/self.df  # bound Jacobian
                    else:
                        amp[itd, ifd] = 1/thymthgy  # Jacobian
                else:
                    amp[itd, ifd] = 10**(-6)  # on or outside primary arc

        self.thetax = thetax
        self.thetay = thetay

        # now get secondary spectrum by interpolating in the brightness array
        # and multiplying by the Jacobian of the tranformation from (td,fd) to
        # (thx,thy)
        SS = griddata((np.ravel(self.X), np.ravel(self.Y)), np.ravel(self.B),
                      (np.ravel(thetax), np.ravel(thetay)), method='linear')\
            * np.ravel(amp)
        SS += griddata((np.ravel(self.X), np.ravel(self.Y)), np.ravel(self.B),
                       (np.ravel(thetax), np.ravel(-thetay)), method='linear')\
            * np.ravel(amp)
        SS = np.reshape(SS, (len(td), len(fd)))
        self.jacobian = np.reshape(amp, (len(td), len(fd)))

        # now add the SS with the sign of td and fd changed
        # unfortunately that is not simply reversing the matrix
        # however if you take just SS(1:, 1:) then it can be reversed and
        # added to the original

        SSrev = np.flip(np.flip(SS[1:, 1:], axis=0), axis=1)
        SS[1:, 1:] += SSrev
        self.SS = SS
        self.LSS = 10*np.log10(SS)
        return

    def calc_acf(self):
        """
        Compute the (normalized) ACF from the secondary spectrum.

        Requires `calc_SS` to have been run first (i.e. `self.SS` to
        exist).

        Returns
        -------
        None
            Sets ``self.acf`` (2D ACF vs delay and Doppler lag, normalized
            to a peak of 1) as an attribute.
        """
        acf = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.SS)))
        acf = np.real(acf)
        acf /= np.max(acf)  # normalize acf
        self.acf = acf
        return

    def plot_acf_efield(self, figsize=(6, 6)):
        """
        Plot the ACF of the electric field vs spatial lag.

        Requires `calc_brightness` to have been run first (i.e.
        `self.acf_efield` to exist).

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size (width, height) in inches.
        """
        plt.figure(figsize=figsize)
        plt.pcolormesh(self.x, self.x, self.acf_efield)
        plt.grid(linewidth=0.2)
        plt.colorbar()
        plt.title('ACF of Intensity for ar={0} for ar={0}, psi={1}, alpha={2}'.
                  format(self.ar, self.psi, self.alpha))
        plt.xlabel('X = velocity axis')
        plt.ylabel('Y axis')
        plt.show()

    def plot_brightness(self, figsize=(6, 6)):
        """
        Plot the angular brightness distribution (in dB).

        Requires `calc_brightness` to have been run first (i.e. `self.B`
        to exist).

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size (width, height) in inches.
        """
        plt.figure(figsize=figsize)
        plt.pcolormesh(self.x, self.x, 10*np.log10(self.B))
        plt.grid(linewidth=0.2)
        plt.colorbar()
        plt.title('Brightness (dB) for ar={0}, psi={1}, alpha={2}'.
                  format(self.ar, self.psi, self.alpha))
        plt.xlabel(r'$\theta_x$ = velocity axis')
        plt.ylabel(r'$\theta_y$ axis')
        plt.show()

    def plot_sspec(self, figsize=(6, 6)):
        """
        Plot the delay-Doppler (secondary) spectrum, in dB.

        Requires `calc_SS` to have been run first (i.e. `self.fd`,
        `self.td`, `self.LSS`, `self.SS` to exist). The colour scale is
        set to 3 dB below the median and maximum of the spectrum (over
        pixels where ``self.SS > 1e-6``).

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size (width, height) in inches.
        """
        plt.figure(figsize=figsize)
        plt.pcolormesh(self.fd, self.td, self.LSS)
        plt.colorbar()
        medval = np.median(self.LSS[(self.SS > 1e-6)])
        maxval = np.max(self.LSS[(self.SS > 1e-6)])
        vmin = medval - 3
        vmax = maxval - 3
        plt.clim((vmin, vmax))
        plt.title('Delay-Doppler Spectrum (dB) for ar={0}, psi={1}, alpha={2}'.
                  format(self.ar, self.psi, self.alpha) +
                  '\n Gradient Angle ({0}, {1}) Reference Angle ({2}, {3})'.
                  format(self.thetagx, self.thetagy,
                         self.thetarx, self.thetary))
        plt.ylabel('Delay')
        plt.xlabel('Doppler')
        plt.show()

    def plot_acf(self, figsize=(6, 6), contour=True):
        """
        Plot the ACF vs delay (frequency) and Doppler (time) lag.

        Requires `calc_acf` to have been run first (i.e. `self.acf` to
        exist).

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size (width, height) in inches.
        contour : bool, optional
            If True, overlay black contours at ``[0.2, 0.4, 0.6, 0.8]``
            and a dotted red contour at 0.
        """
        plt.figure(figsize=figsize)
        plt.pcolormesh(self.fd, self.td, self.acf)
        plt.colorbar()
        if contour:
            # put in contours at 0.2, 0.4, 0.6 and 0.8 in black
            plt.contour(self.fd, self.td, self.acf, [0.2, 0.4, 0.6, 0.8],
                        colors='k')
            # add a contour at zero as a dashed white line
            plt.contour(self.fd, self.td, self.acf, [0.0],
                        colors='r', linestyles='dotted')
        plt.title('ACF (Time, Freq) for ar={0}, psi={1}, alpha={2}'.
                  format(self.ar, self.psi, self.alpha) +
                  '\n Gradient Angle ({0}, {1}) Reference Angle ({2}, {3})'.
                  format(self.thetagx, self.thetagy,
                         self.thetarx, self.thetary))
        plt.ylim((-4, 4))
        plt.xlim((-1, 1))
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    def plot_cuts(self, figsize=(6, 6)):
        """
        Plot 1D cuts through the secondary spectrum in Doppler at a
        selection of delays, and a cut in delay at zero Doppler.

        One might want to take some cuts through the ACF. In particular,
        the cut in Doppler at zero delay is invariant with phase gradient
        and is also ``exp(-(time/t0)**alpha)``, so you can confirm that
        the exponent is correct by examining that cut. The cut in
        frequency (the bandwidth) is very sensitive to `ar`, its
        orientation (`psi`), and to phase gradients.

        Requires `calc_SS` to have been run first (i.e. `self.fd`,
        `self.td`, `self.LSS`, `self.SS` to exist).

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size (width, height) in inches, used for each of the
            two figures produced.
        """
        plt.figure(figsize=figsize)
        nt = len(self.td)
        step = int((nt / 2) / (self.ncuts))
        for itdp in range(int(nt/2) + step - 1, nt + step - 1, step):
            plt.plot(self.fd, self.LSS[itdp, :])

        mn = np.min(self.LSS[nt - 1, round(len(self.fd)/2 - 1)])
        yl = plt.ylim()
        plt.ylim((mn - 10, yl[1]))
        plt.title('{0} Cuts in Doppler at '.format(self.ncuts) +
                  'constant Delay for ar={0}, psi={1}, and alpha={2}'.
                  format(self.ar, self.psi, self.alpha) +
                  '\n Gradient Angle ({0}, {1}) Reference Angle ({2}, {3})'.
                  format(self.thetagx, self.thetagy,
                         self.thetarx, self.thetary))
        plt.xlabel('Doppler')
        plt.ylabel('Log Power')
        plt.grid()
        plt.show()

        # zero doppler cut in delay
        plt.figure(figsize=figsize)
        fi = np.argmin(np.abs(self.fd)).squeeze()
        ti = np.argwhere(self.td >= 0).squeeze()
        plt.semilogx(self.td[ti], self.LSS[ti, fi])
        plt.grid()
        plt.title('Cut in Delay at Doppler=0 for ar={}, psi={} and alpha={}'.
                  format(self.ar, self.psi, self.alpha) +
                  '\n Gradient Angle ({0}, {1}) Reference Angle ({2}, {3})'.
                  format(self.thetagx, self.thetagy,
                         self.thetarx, self.thetary))
        plt.xlabel('Delay')
        plt.ylabel('Log Power')
        plt.show()
