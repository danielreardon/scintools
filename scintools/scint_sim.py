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
from scipy.special import gamma
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.fft import fft2, ifft2, fftshift, ifftshift


class Simulation():

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                 inner=0.001, ns=256, nf=256, dlam=0.25, lamsteps=False,
                 seed=None, nx=None, ny=None, dx=None, dy=None, plot=False,
                 verbose=False, freq=1400, dt=30, mjd=50000, nsub=None,
                 efield=False):
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
        if plot:
            self.plot_all()

        # Now prepare simulation for use with scintools, using physical units
        self.name =\
            'sim:mb2={0},ar={1},psi={2},dlam={3}'.format(self.mb2, self.ar,
                                                         self.psi, self.dlam)
        if lamsteps:
            self.name += ',lamsteps'

        self.header = self.name
        if efield:
            dyn = np.real(self.spe)
        else:
            dyn = self.spi
        dlam = self.dlam

        self.dt = dt
        self.freq = freq
        self.nsub = int(np.shape(dyn)[0]) if nsub is None else nsub
        self.nchan = int(np.shape(dyn)[1])
        lams = np.linspace(1-self.dlam/2, 1+self.dlam/2, self.nchan)
        freqs = np.divide(1, lams)
        freqs = np.linspace(np.min(freqs), np.max(freqs), self.nchan)
        self.freqs = freqs*self.freq/np.mean(freqs)
        self.bw = max(self.freqs) - min(self.freqs)
        self.times = self.dt*np.arange(0, self.nsub)
        self.df = self.bw/self.nchan
        self.tobs = float(self.times[-1] - self.times[0])
        self.mjd = mjd
        if nsub is not None:
            dyn = dyn[0:nsub, :]
        self.dyn = np.transpose(dyn)

        # Theoretical arc curvature
        #V = 1
        #k = 2*pi/lambda0
        #L = rf^2*k

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

        self.x = np.linspace(0, self.dx*(self.nx), (self.nx+1))
        ifreq = np.arange(0, self.nf+1)
        lam_norm = 1.0 + self.dlam * (ifreq - 1 - (self.nf / 2)) / self.nf
        self.lams = lam_norm / np.mean(lam_norm)
        frfreq = 1.0 + self.dlam * (-0.5 + ifreq / self.nf)
        self.freqs = frfreq / np.mean(frfreq)

        return

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

    def plot_all(self):
        plt.figure(2)
        plt.subplot(2, 2, 1)
        self.plot_screen(subplot=True)
        plt.subplot(2, 2, 2)
        self.plot_intensity(subplot=True)
        plt.subplot(2, 1, 2)
        self.plot_dynspec(subplot=True)
        plt.show()


"""
The code below is unfinished, but will eventually allow one to compute the ACF
analytically, including a phase gradient. A dynamic spectrum with phase
gradients (beyond those that arise naturally) can be simulated from this.
"""


#class ACF():
#
#    def __init__(self, s_max=5, dnu_max=5, ns=201, nf=101, ar=2, alpha=5/3,
#                 phasegrad_x=0, phasegrad_y=0, Vx=None, Vy=None, nt=None):
#        """
#        Generate an ACF from the theoretical function in:
#            Rickett et al. (2014)
#
#        s_max - number of coherence spatial scales to calculate over
#        dnu_max - number of decorrelation bandwidths to calculate over
#        ns - number of spatial steps
#        nf - number of decorrelation bandwidth steps
#        ar - axial ratio of diffractive scintillation
#        alpha - exponent for interstellar turbulence
#
#        If ISS spectrum is a Kolmogorov power-law with no inner or outer scale,
#        alpha=5/3
#        """
#        if phasegrad_x == 0 and phasegrad_y == 0 and Vx is None and Vy is None:
#            # No phase gradient, so calculate using the faster method
#            self.calc_acf_fourier(s_max=s_max, dnu_max=dnu_max, ns=ns, nf=nf,
#                                  ar=ar, alpha=alpha)
#        else:
#            self.calc_acf(s_max=s_max, dnu_max=dnu_max, nt=ns, nf=nf, ar=ar,
#                          alpha=alpha, phasegrad_x=phasegrad_x,
#                          phasegrad_y=phasegrad_y)
#        return
#
#    def calc_acf(self, s_max=5, dnu_max=5, nt=201, nf=101, ar=2, alpha=5/3,
#                 phasegrad_x=0, phasegrad_y=0, Vx=10, Vy=10):
#        """
#        Computes 2-D ACF of intensity vs t and v where optimal sampling of t
#        and v is provided with the output ACF.
#
#        Requires velocity and angular displacement due to phase gradient
#        vectors are x, y where x = major axis of spatial structure,
#        i.e. density variations are elongated by "ar" in the x direction.
#        y is 90deg CCW.
#
#        Implements the integrals in Appendix A of Rickett et al. ApJ (2014) on
#        the analysis of the double pulsar scintillation equations A1 and A2.
#        A2 has an error. It would be correct if nu were replaced by omega,
#        i.e. had an extra 2pi
#
#        Coordinates are with respect to ar major axis so we don't have to
#        rotate the structure, we put V and phase gradient vectors in the
#        structure coordinates.
#
#        The distance s_max is normalized by the coherence spatial scale (s0)
#        and the frequency dnu_max by the decorrelation bandwidth. The phase
#        gradient is normalized by the 1/s0, i.e. sigxn = phasegrad_x * s0
#
#        If there is no phase gradient then the acf is symmetric and only one
#        quadrant needs to be calculated. Otherwise two quadrants are necessary.
#
#        New algorithm to compute same integral: Normalized integral is
#        game(sn, dnun) = -j/(2pi)^2 (1/dnun) sum sum (dsn)^2
#        game(snp,0)exp((j/4pi)(1/dnun) | sn - snp|^2
#
#        The worst case sampling is when dnun is very small. Then the argument
#        of the complex exponential becomes large and aliasing will occur. If
#        dnun=0.01 and dsp=0.1 the alias will peak at snx = 5. Reducing the
#        sampling dsp to 0.05 will push that alias out to snx = 8. However
#        halving dsp will increase the time by a factor of 4.
#
#        The frequency decorrelation is quite linear near the origin and looks
#        quasi-exponential, the 0.5 width is dnun = 0.15. Sampling of 0.05 is
#        more than adequate in frequency. Sampling of 0.1 in sn is adequate
#
#        dnun = 0.0 is divergent with this integral but can be obtained
#        trivially from the ACF of the electric field directly
#
#        Use formula S = Vt - 2*sigma_p*dnu/nu
#        equation A6 to get equal t sampling. dt = ds / |V| and tmax = Smax + 2
#        |sigma_p| dnu/nu
#        """
#
#        spmax = s_max
#        dnumax = dnu_max
#        dsp = 2*(spmax)/(nt-1)
#        alph2 = alpha/2
#        ddnun = 2*(dnumax)/(nf-1)
#        Vmag = np.sqrt(Vx**2 + Vy**2)
#        sqrtar = np.sqrt(ar)
#        dnun = np.arange(0, dnumax, ddnun)  # equally spaced dnu array
#        ndnun = len(dnun)
#        sigxn = phasegrad_x
#        sigyn = phasegrad_y
#        # initialize arrays
#        gammiv = np.zeros((spmax/dsp, spmax/dsp, 2*ndnun-1))
#
#        if sigxn == 0 and sigyn == 0:
#            # calculate only one quadrant tn >= 0
#            print('Calculating ACF... w/ one quad')
#            # equally spaced t array t = tn*S0
#            tn = np.arange(0, (spmax/Vmag), (dsp/Vmag))
#            snx = Vx*tn
#            sny = Vy*tn
#            sn = np.arange(-2*spmax, 2*spmax, dsp)
#            SNPX, SNPY = np.meshgrid(sn, sn)
#
#            gammes = np.exp(-0.5*((SNPX/sqrtar)**2 +
#                                  (SNPY*sqrtar)**2)**alph2)  # ACF of e-field
#            # compute dnun=0 first
#            gammitv[:, 0] = np.exp(-0.5*((snx/sqrtar)**2 +
#                                   (sny*sqrtar)**2)**alph2)
#            # now do first dnu step with double spatial resolution
#            sn2 = np.arange(-2*spmax, 2*spmax, dsp/2)
#            SNPX2, SNPY2 = np.meshgrid(sn2, sn2)
#            gammes2 = np.exp(-0.5*((SNPX2/sqrtar)**2 +
#                                   (SNPY2*sqrtar)**2)**alph2)  # ACF of e-field
#            for isn in range(0, len(snx)):
#                ARG = ((SNPX2-snx[isn])**2+(SNPY2-sny[isn])**2)/(2*dnun(2))
#                temp = gammes2*np.exp(1j*ARG)
#                gammitv[isn, 1] = -1j*(dsp/2)**2 *\
#                    np.sum(temp[:])/((2*np.pi)*dnun[1])
#
#            plt.pcolormesh(gammitv)
#            plt.show()
#            # now do remainder of dnu array
#            # for idn in range(2, ndnun):
##                #     snx= snx -2*sigxn*dnun(idn);
##                #     sny = sny - 2*sigyn*dnun(idn);
##            for isn in range(0, len(snx)):
##                ARG = ((SNPX-snx(isn)).^2+(SNPY-sny(isn)).^2)/(2*dnun(idn));
##                temp = gammes.*exp(1i*ARG);
##                gammitv(isn,idn) = -1i*dsp^2*sum(temp(:))/((2*pi)*dnun(idn));
##
##            gammitv=real(gammitv.*conj(gammitv));  %equation A1 convert ACF of E to ACF of I
##            gam2=[fliplr(gammitv(:,2:end)),gammitv];
##            t2=[fliplr(-tn(2:end)),tn];
##            gam3=[flipud(gam2(2:end,:));gam2]';
##            f2=[fliplr(-dnun(2:end)),dnun];
##            s2=t2.*Vmag;
##        else
##            %calculate two quadrants -tmax t < tmax
##            display('Calculating ACF... w/ two quad')
##            tn = -(spmax/Vmag):(dsp/Vmag):(spmax/Vmag);  %equally spaced t array t= tn*S0
##            snx= Vx*tn; sny = Vy*tn;
##            [SNPX,SNPY] = meshgrid(-spmax:dsp:spmax);
##            gammes=exp(-0.5*((SNPX/sqrtar).^2+(SNPY*sqrtar).^2).^alph2); %ACF of e-field
##            %compute dnun=0 first
##            gammitv(:,1)=exp(-0.5*((snx/sqrtar).^2 + (sny*sqrtar).^2).^alph2);
##            for idn=2:ndnun
##                snxt= snx -2*sigxn*dnun(idn);
##                snyt = sny - 2*sigyn*dnun(idn);
##            for isn=1:length(snx);
##                %temp=gammes.*exp(1i*((SNPX-snx(isn)).^2+(SNPY-sny(isn)).^2)/(2*dnun(idn)));
##                %gammitv(isn,idn)= -1i*dsp^2*sum(temp(:))/((2*pi)*dnun(idn));
##                temp=gammes.*exp(1i*((SNPX-snxt(isn)).^2+(SNPY-snyt(isn)).^2)/(2*dnun(idn)));
##                gammitv(isn,idn)= -1i*dsp^2*sum(temp(:))/((2*pi)*dnun(idn));
##            gammitv=real(gammitv.*conj(gammitv));  %equation A1 convert ACF of E to ACF of I
##            gam3=[fliplr(flipud(gammitv(:,2:end))),gammitv]';
##            f2=[fliplr(-dnun(2:end)),dnun];
##            t2=tn;
##            s2=t2.*Vmag;
##        return
#
#    def calc_acf_fourier(self, s_max=5, dnu_max=5, ns=201, nf=101, ar=2,
#                         alpha=5/3):
#        """
#        #Compute ACF in the Fourier domain when there is no phase gradient
#        """
#        snmax = s_max
#        dnunmax = dnu_max
#        alph2 = alpha/2
#        hns = int(np.floor(ns/2))
#        sn = np.arange(-hns, hns+1)
#        sn = sn*2*snmax/(ns-1)
#        kn = np.arange(-hns, hns+1)
#        kn = kn/(2*snmax)
#        dnun = np.arange(nf)
#        dnun = dnun*dnunmax/(nf-1)
#        SNX, SNY = np.meshgrid(sn, sn)
#        SNX = ifftshift(SNX, 1)
#        SNY = ifftshift(SNY, 0)
#        KNX, KNY = np.meshgrid(kn, kn)
#        KNX = ifftshift(KNX, 1)
#        KNY = ifftshift(KNY, 0)
#        KN2 = np.add(np.power(KNX, 2), np.power(KNY, 2))  # KN2 is fftshifted
#        # Get medium parameters assuming x is the major axis
#        sqrtar = np.sqrt(ar)
#        # compute 2-D spatial ACF for dnun=0 and the corresponding 2-D spectrum
#        # ACF of e-field
#        gammes = np.exp(-0.5*(np.power(np.add(np.power(SNX/sqrtar, 2),
#                                              np.power(SNY*sqrtar, 2)),
#                                       alph2)))
#        pk0 = fft2(gammes)
#        # initialize arrays
#        gammi = np.zeros((ns, ns, 2*nf-1))
#        # first dnun=0
#        gammi[:, :, nf-1] = fftshift(np.multiply(gammes, np.conj(gammes)))
#        # then rest of dnun
#        for idnun in range(1, nf):
#            temp = np.multiply(pk0,
#                               np.exp(np.multiply(-1j*2*np.pi**2*dnun[idnun],
#                                                  KN2)))
#            temp = ifft2(temp)
#            # Make sure python realises this is real, to avoid warnings
#            temp = np.real(fftshift(np.multiply(temp, np.conj(temp))))
#            gammi[:, :, nf+(idnun-1)] = temp
#            gammi[:, :, nf-(idnun-1)] = temp
#
#        self.gammix = np.transpose(gammi[hns, :, :])
#        self.gammiy = np.transpose(gammi[:, hns, :])
#        return
#
#    def plot_acf(self):
#        """
#        Plots the simulated ACF
#        """
#        plt.figure(1)
#        plt.pcolormesh(self.gammix)
#        plt.show()
#        plt.figure(2)
#        plt.pcolormesh(self.gammiy)
#        plt.show()
#        return
#

