#!/usr/bin/env python

"""
scintsim.py
----------------------------------
Simulate scintillation. Based on original MATLAB code by Coles et al. (2010)
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

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                 inner=0.001, ns=256, nf=256, dlam=0.25, lamsteps=False,
                 seed=None, nx=None, ny=None, dx=None, dy=None, plot=False,
                 verbose=False, freq=1400, dt=30, mjd=60000, nsub=None,
                 efield=False, noise=None):
        """
        Electromagnetic simulator based on original code by Coles et al. (2010)

        mb2: Max Born parameter for strength of scattering
        rf: Fresnel scale
        ds (or dx,dy): Spatial step sizes with respect to rf
        alpha: Structure function exponent (Kolmogorov = 5/3)
        ar: Anisotropy axial ratio
        psi: Anisotropy orientation
        inner: Inner scale w.r.t rf - should generally be smaller than ds
        ns (or nx,ny): Number of spatial steps
        nf: Number of frequency steps.
        dlam: Fractional bandwidth relative to centre frequency
        lamsteps: Boolean to choose whether steps in lambda or freq
        seed: Seed number, or use "-1" to shuffle
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
        c = 299792458.0  # m/s
        beta_to_eta = c*1e6/((self.freq*10**6)**2)
        # Curvature for wavelength-rescaled dynamic spectrum
        self.betaeta = self.eta / beta_to_eta

        return

    def set_constants(self):

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
        # routine to plot intensity
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
        # get frequency to set the scale, enter in GHz
        Freq = self.freq/1000
        lpw = np.log10(self.pulsewin)
        vmax = np.max(lpw)
        vmin = np.median(lpw) - 3
        plt.pcolormesh(np.linspace(0, self.dx*self.nx, self.nx),
                       (np.arange(0, 3*self.nf/2, 1) - self.nf/2) /
                       (2*self.dlam*Freq),
                       lpw[int(self.nf/2):, :], vmin=vmin, vmax=vmax)
        plt.colorbar
        plt.ylabel('Delay (ns)')
        plt.xlabel('$x/r_f$')
        plt.plot(np.linspace(0, self.dx*self.nx, self.nx),
                 -self.dm/(2*self.dlam*Freq), 'k')  # group delay=-phase delay
        plt.show()

    def plot_all(self):
        plt.figure(2)
        plt.subplot(2, 2, 1)
        self.plot_screen(subplot=True)
        plt.subplot(2, 2, 2)
        self.plot_intensity(subplot=True)
        plt.subplot(2, 1, 2)
        self.plot_dynspec(subplot=True)
        plt.show()


class ACF():

    def __init__(self, psi=0, phasegrad=0, theta=0, ar=1, alpha=5/3,
                 taumax=4, dnumax=4, nf=51, nt=51, amp=1, wn=0,
                 spatial_factor=2, resolution_factor=1, core_factor=2,
                 auto_sampling=True, plot=False, display=True):
        """
        Generate an ACF from the theoretical function in:
            Rickett et al. (2014)

        Magnitude of velocity defined to be 1

        psi - angle of velocity w.r.t the major axis of (brightness) anisotropy
        phasegrad - magnitude of phase gradient
        theta - angle of phase gradient w.r.t velocity vector
        ar - axial ratio of anisotropy
        alpha - structure function exponent (Kolmogorov = 5/3)
        taumax - number of time scales to calculate ACF to
            (ACF in space goes to taumax*V)
        dnumax - number of frequency scales to calculate ACF to
        nf - size of ACF in frequency. If not odd, returns nf+1
        nt - size of ACF in frequency. If not odd, returns nt+1
        amp - amplitude of ACF
        wn - size of white-noise spike at the origin of ACF
        spatial_factor - multiplier for spmax for calculating ACF e-field
        resolution_factor - multiplier to default resolution for ACF e-field
        core_factor - additional resolution multiplier for the first dnu sample
        auto_sampling - (bool) decide whether to adjust spatial sampling
            automatically to avoid artefacts.
            Warning: computation time blows up for large ar
        plot - (bool) choose whether to plot after computing
        display - (bool) choose whether to display to screen immediately
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
        """
        Computes 2D ACF of intensity vs time and frequency where the
        sampling in the output ACF can be handled automatically.

        requires anisotropy and angular displacement due to phase gradient.
        psi = 0, defines brightness distribution anisotropy aligned with V
        theta = 0, defines phase gradient aligned with V
        (Within the code, x-axis (see Vx, and sigxn parameters) is the major
         axis of ACF-efield anisotropy, which is psi=90)

        implements the integrals in Appendix A of Rickett, Coles et al ApJ 2014
        on the analysis of the double pulsar scintillation equations A1 and A2.
        A2 has an error. It would be correct if nu were replaced by omega,
        i.e. had an extra 2*pi

        coordinates in the code are with respect to ar major axis so we don't
        have to rotate the structure, we put V and phasegrad into the
        structure coordinates.

        The distance sn is normalized by So and the frequency dnun by \nu_{0.5}
        the spatial scale and the frequency scale respectively.
        the phase gradient is normalized by the 1/s0,
        i.e. sigxn = gradphix * s0

        if there is no phase gradient then the acf is symmetric and only one
        quadrant needs to be calculated. Otherwise two quadrants are necessary.

        the worst case sampling is when dnun is very small. Then the argument
        of the complex exponential becomes large and aliasing will occur. If
        dnun=0.01 and dsp=0.1 the alias will peak at snx = 5. Reducing the
        sampling dsp to 0.05 will push that alias out to snx = 8. However
        halving dsp will increase the time by a factor of 4. Sampling
        characteristics can be tuned with spatial_factor, resolution_factor,
        and core_factor parameters

        The frequency decorrelation is quite linear near the origin and looks
        quasi-exponential, the 0.5 width is dnun = 0.15. Sampling of 0.05 is
        more than adequate in frequency. Sampling of 0.1 in sn is adequate

        dnun = 0.0 is divergent with this integral but can be obtained
        trivially from the ACF of the electric field directly
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
            gammitv = np.zeros((int(len(snx)), int(ndnun)), dtype=np.complex_)
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
            gammitv = np.zeros((int(len(snx)), int(ndnun)), dtype=np.complex_)
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
        Plots the simulated ACF
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
        Plots the simulated ACF
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
        Calculate the secondary spectrum
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
        Plots the simulated ACF
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

    def __init__(self, ar=1.0, psi=0, alpha=1.67, thetagx=0, thetagy=0,
                 thetarx=0, thetary=0, df=0.02, dt=0.08, dx=0.1,
                 nf=10, nt=80, nx=30, ncuts=5, plot=False, contour=True,
                 figsize=(10, 8), calc_sspec=True, calc_acf=True):
        """
        Simulate Delay-Doppler Spectrum from Scattered angular spectrum from
        Yao et al. (2020), modified to get the phase gradient terms correctly
        and to clean up the bright points in the secondary spectrum which cause
        artifacts in the ACF

        Here we assume that the angular spectrum interferes with an unscattered
        wave. The angular spectrum is defined by the spectral exponent. First
        the ACF of the field is calculated, then it is 2D-FFT'ed to get the
        brightness distribution. The brightness distribution can be offset by a
        phase gradient which causes an angular shift as a fraction of the half-
        width of the brightness distribution.

        The unscattered wave can also be offset by the phase gradient (as it
        would be in weak scattering), or it can be at zero offset (or anywhere
        else). The default would be to set the phase gradient angle and the
        reference angle to be equal

        params:
            ar: axial ratio
            alpha: exponent of phase structure function
            thetagx: scattered wave offset by phase gradient
            thetagy:
            thetarx: reference angle for unscattered wave (normally thetax)
            thetary:
            dx, nx: spatial resolution and size of e-field ACF, relative to
                spatial scale

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
        now set up the secondary spectrum defined by:

        delay = theta^2, i.e. 0.5 L/c = 1
        doppler = theta, i.e. V/lambda = 1

        therefore differential delay (td), and differential doppler (fd) are:
            td = (thetax+thetagx)^2 +(thetay+thetagy)^2 - thetagx^2-thetagy^2
            fd = (thetax + thetagx) - thetagx = thetax
            Jacobian = 1/(thetay+thetagy)
        thetay + thetagy =
            sqrt(td - (thetax + thetagx)^2 + thetagx^2 + thetagy^2)

        the arc is defined by (thetay+thetagy) == 0 where there is a half order
        singularity.

        The singularity creates a problem in the code because the sampling in
        fd,td is not synchronized with the arc position, so there can be some
        very bright points if the sample happens to lie very close to the
        singularity.

        this is not a problem in interpreting the secondary spectrum, but it
        causes large artifacts when Fourier transforming it to get the ACF.

        So I [Bill Coles, in original Matlab code] have limited the Jacobian by
        not allowing (thetay+thetagy) to be less than half the step size in
        thetax and thetay.
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
        acf = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.SS)))
        acf = np.real(acf)
        acf /= np.max(acf)  # normalize acf
        self.acf = acf
        return

    def plot_acf_efield(self, figsize=(6, 6)):
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
        plot some cuts. One might want to take some cuts through the ACF. In
        particular the ACF cut in doppler at zero delay is invarient with phase
        gradient and is also exp(-(time/t0)^alpha), so you can confirm that
        the exponent is correct by examining that cut.

        the cut in frequency (the bandwidth) is very sensitive to ar and its
        orientation and to phase gradients.
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
