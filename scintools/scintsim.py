#!/usr/bin/env python

"""
scintsim.py
----------------------------------
Simulate scintillation. Based on original MATLAB code by Coles et al. (2010)
"""

import numpy as np
from numpy import random
from scipy.special import gamma
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from numpy.fft import ifftshift


class Simulation():

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                 inner=0.001, ns=256, nf=256, dlam=0.5, lamsteps=False,
                 seed=None, verbose=True, nx=None, ny=None, dx=None, dy=None):
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
        self.get_screen()

        return

    def set_constants(self):

        ns = 1
        lenx = self.nx*self.dx
        leny = self.ny*self.dy
        # ffconx = (2.0/(ns*lenx*lenx))*(np.pi*self.rf)**2
        # ffcony = (2.0/(ns*leny*leny))*(np.pi*self.rf)**2
        dqx = 2*np.pi/lenx
        dqy = 2*np.pi/leny
        # dqx2 = dqx*dqx
        # dqy2 = dqy*dqy
        a2 = self.alpha*0.5
        # spow = (1.0+a2)*0.5
        # ap1 = self.alpha+1.0
        # ap2 = self.alpha+2.0
        # aa = 1.0+a2
        ab = 1.0-a2
        # cdrf = 2.0**(self.alpha)*np.cos(self.alpha*np.pi*0.25)
        #                                 *gamma(aa)/self.mb2
        # s0 = self.rf*cdrf**(1.0/self.alpha)

        cmb2 = self.alpha*self.mb2 / (4*np.pi *
                                      gamma(ab)*np.cos(self.alpha *
                                                       np.pi*0.25)*ns)
        self.consp = cmb2*dqx*dqy/(self.rf**self.alpha)
        # scnorm = 1.0/(self.nx*self.ny)

        # ffconlx = ffconx*0.5
        # ffconly = ffcony*0.5
        # sref = self.rf**2/s0
        return

    def plot_screen(self):
        if not hasattr(self, 'xyp'):
            self.getscreen()
        x_steps = np.linspace(0, self.dx*self.nx, self.nx)
        y_steps = np.linspace(0, self.dy*self.ny, self.ny)
        plt.pcolormesh(y_steps, x_steps, self.xyp)
        plt.ylabel('$y/r_f$')
        plt.xlabel('$x/r_f$')
        plt.show()
        return

    def get_screen(self):
        """
        Get phase screen in x and y
        """
        random.seed(self.seed)  # Set the seed, if any

        nx2 = int(self.nx/2)
        ny2 = int(self.ny/2)

        w = np.zeros([self.nx, ny2])  # initialize array
        dqx = 2*np.pi/(self.dx*self.nx)
        dqy = 2*np.pi/(self.dy*self.ny)
        # first do ky=0 line
        k = np.arange(1, nx2, 1)
        w[k, 0] = self.swdsp(kx=(k)*dqx, ky=1)
        w[self.nx-k, 0] = w[k, 0]
        # then do kx=0 line
        ll = np.arange(1, ny2, 1)
        w[0, ll] = self.swdsp(kx=1, ky=(ll)*dqy)
        # now do the rest of the field
        kp = np.arange(1, nx2, 1)
        k = np.arange(nx2, self.nx, 1)
        km = -(self.nx-k)
        for il in range(0, ny2):
            w[kp, il] = self.swdsp(kx=(kp)*dqx, ky=(il)*dqy)
            w[k, il] = self.swdsp(kx=km*dqx, ky=(il)*dqy)
        # done the whole screen weights, now generate complex gaussian array
        xyp = np.zeros([self.nx, self.ny], dtype=np.dtype(np.csingle))
        xyp[0:self.nx, 0:ny2] = np.multiply(w, np.add(randn(self.nx, ny2),
                                                      1j*randn(self.nx, ny2)))
        # now fill in the other half because xyp is hermitean
        xyp[nx2:self.nx,
            ny2:self.ny] = np.transpose(
                            np.flip(
                             np.flip(
                              np.transpose(
                               np.conj(xyp[0:(nx2),
                                           0:(ny2)])), axis=0), axis=1))
        xyp[0:nx2, ny2:self.ny] = np.flip(np.conj(xyp[0:nx2, 0:ny2]), axis=1)
        xyp = np.real(fft2(xyp))
        self.xyp = xyp
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
        q2 = a*np.power(kx, 2) + b*np.power(ky, 2) + c*np.multiply(kx, ky)
        # isotropic inner scale
        out = con*np.multiply(np.power(q2, alf),
                              np.exp(-(np.add(np.power(kx, 2),
                                              np.power(ky, 2))) *
                              self.inner**2/2))
        return out


class ACF():

    def __init__(self, s_max=5, dnu_max=5, ns=201, nf=101, ar=2, alpha=5/3,
                 phasegrad_x=0, phasegrad_y=0, Vx=None, Vy=None, nt=None):
        """
        Generate an ACF from the theoretical function in:
            Rickett et al. (2014)

        s_max - number of coherence spatial scales to calculate over
        dnu_max - number of decorrelation bandwidths to calculate over
        ns - number of spatial steps
        nf - number of decorrelation bandwidth steps
        ar - axial ratio of diffractive scintillation
        alpha - exponent for interstellar turbulence

        If ISS spectrum is a Kolmogorov power-law with no inner or outer scale,
        alpha=5/3
        """
        if phasegrad_x == 0 and phasegrad_y == 0 and Vx is None and Vy is None:
            # No phase gradient, so calculate using the faster method
            self.calc_acf_fourier(s_max=s_max, dnu_max=dnu_max, ns=ns, nf=nf,
                                  ar=ar, alpha=alpha)
        else:
            self.calc_acf(s_max=s_max, dnu_max=dnu_max, nt=ns, nf=nf, ar=ar,
                          alpha=alpha, phasegrad_x=phasegrad_x,
                          phasegrad_y=phasegrad_y)
        return

    def calc_acf(self, s_max=5, dnu_max=5, nt=201, nf=101, ar=2, alpha=5/3,
                 phasegrad_x=0, phasegrad_y=0, Vx=10, Vy=10):
        """
        Computes 2-D ACF of intensity vs t and v where optimal sampling of t
        and v is provided with the output ACF.

        Requires velocity and angular displacement due to phase gradient
        vectors are x, y where x = major axis of spatial structure,
        i.e. density variations are elongated by "ar" in the x direction.
        y is 90deg CCW.

        Implements the integrals in Appendix A of Rickett et al. ApJ (2014) on
        the analysis of the double pulsar scintillation equations A1 and A2.
        A2 has an error. It would be correct if nu were replaced by omega,
        i.e. had an extra 2pi

        Coordinates are with respect to ar major axis so we don't have to
        rotate the structure, we put V and phase gradient vectors in the
        structure coordinates.

        The distance s_max is normalized by the coherence spatial scale (s0)
        and the frequency dnu_max by the decorrelation bandwidth. The phase
        gradient is normalized by the 1/s0, i.e. sigxn = phasegrad_x * s0

        If there is no phase gradient then the acf is symmetric and only one
        quadrant needs to be calculated. Otherwise two quadrants are necessary.

        New algorithm to compute same integral: Normalized integral is
        game(sn, dnun) = -j/(2pi)^2 (1/dnun) sum sum (dsn)^2
        game(snp,0)exp((j/4pi)(1/dnun) | sn - snp|^2

        The worst case sampling is when dnun is very small. Then the argument
        of the complex exponential becomes large and aliasing will occur. If
        dnun=0.01 and dsp=0.1 the alias will peak at snx = 5. Reducing the
        sampling dsp to 0.05 will push that alias out to snx = 8. However
        halving dsp will increase the time by a factor of 4.

        The frequency decorrelation is quite linear near the origin and looks
        quasi-exponential, the 0.5 width is dnun = 0.15. Sampling of 0.05 is
        more than adequate in frequency. Sampling of 0.1 in sn is adequate

        dnun = 0.0 is divergent with this integral but can be obtained
        trivially from the ACF of the electric field directly

        Use formula S = Vt - 2*sigma_p*dnu/nu
        equation A6 to get equal t sampling. dt = ds / |V| and tmax = Smax + 2
        |sigma_p| dnu/nu
        """

        spmax = s_max
        dnumax = dnu_max
        dsp = 2*(spmax)/(nt-1)
        ddnun = 2*(dnumax)/(nf-1)
        Vmag = np.sqrt(Vx**2 + Vy**2)
        sqrtar = np.sqrt(ar)
        dnun = np.arange(0, dnumax, ddnun)  # equally spaced dnu array
        ndnun = len(dnun)
        sigxn = phasegrad_x
        sigyn = phasegrad_y

        if sigxn == 0 and sigyn == 0:
            # calculate only one quadrant tn >= 0
            print('Calculating ACF... w/ one quad')
            tn = 0:(dsp./Vmag):(spmax./Vmag)  # equally spaced t array t= tn*S0
            snx= Vx.*tn
            sny = Vy.*tn
            [SNPX,SNPY] = meshgrid(-2*spmax:dsp:2*spmax);
            gammes=exp(-0.5*((SNPX/sqrtar).^2+(SNPY*sqrtar).^2).^alph2); %ACF of e-field
            %compute dnun=0 first
            gammitv(:,1)=exp(-0.5*((snx/sqrtar).^2 + (sny*sqrtar).^2).^alph2);
            %now do first dnu step with double spatial resolution
            [SNPX2,SNPY2] = meshgrid(-2*spmax:(dsp/2):2*spmax);
            gammes2=exp(-0.5*((SNPX2/sqrtar).^2+(SNPY2*sqrtar).^2).^alph2); %ACF of e-field
            for isn=1:length(snx);
                ARG=((SNPX2-snx(isn)).^2+(SNPY2-sny(isn)).^2)/(2*dnun(2));
                temp=gammes2.*exp(1i*ARG);
                gammitv(isn,2)= -1i*(dsp/2)^2*sum(temp(:))/((2*pi)*dnun(2));
            %now do remainder of dnu array
            for idn=3:ndnun
            %     snx= snx -2*sigxn*dnun(idn);
            %     sny = sny - 2*sigyn*dnun(idn);
            for isn=1:length(snx);
                ARG=((SNPX-snx(isn)).^2+(SNPY-sny(isn)).^2)/(2*dnun(idn));
                temp=gammes.*exp(1i*ARG);
                gammitv(isn,idn)= -1i*dsp^2*sum(temp(:))/((2*pi)*dnun(idn));

            gammitv=real(gammitv.*conj(gammitv));  %equation A1 convert ACF of E to ACF of I
            gam2=[fliplr(gammitv(:,2:end)),gammitv];
            t2=[fliplr(-tn(2:end)),tn];
            gam3=[flipud(gam2(2:end,:));gam2]';
            f2=[fliplr(-dnun(2:end)),dnun];
            s2=t2.*Vmag;
        else
            %calculate two quadrants -tmax t < tmax
            display('Calculating ACF... w/ two quad')
            tn = -(spmax/Vmag):(dsp/Vmag):(spmax/Vmag);  %equally spaced t array t= tn*S0
            snx= Vx*tn; sny = Vy*tn;
            [SNPX,SNPY] = meshgrid(-spmax:dsp:spmax);
            gammes=exp(-0.5*((SNPX/sqrtar).^2+(SNPY*sqrtar).^2).^alph2); %ACF of e-field
            %compute dnun=0 first
            gammitv(:,1)=exp(-0.5*((snx/sqrtar).^2 + (sny*sqrtar).^2).^alph2);
            for idn=2:ndnun
                snxt= snx -2*sigxn*dnun(idn);
                snyt = sny - 2*sigyn*dnun(idn);
            for isn=1:length(snx);
                %temp=gammes.*exp(1i*((SNPX-snx(isn)).^2+(SNPY-sny(isn)).^2)/(2*dnun(idn)));
                %gammitv(isn,idn)= -1i*dsp^2*sum(temp(:))/((2*pi)*dnun(idn));
                temp=gammes.*exp(1i*((SNPX-snxt(isn)).^2+(SNPY-snyt(isn)).^2)/(2*dnun(idn)));
                gammitv(isn,idn)= -1i*dsp^2*sum(temp(:))/((2*pi)*dnun(idn));
            gammitv=real(gammitv.*conj(gammitv));  %equation A1 convert ACF of E to ACF of I
            gam3=[fliplr(flipud(gammitv(:,2:end))),gammitv]';
            f2=[fliplr(-dnun(2:end)),dnun];
            t2=tn;
            s2=t2.*Vmag;
        return

    def calc_acf_fourier(self, s_max=5, dnu_max=5, ns=201, nf=101, ar=2,
                         alpha=5/3):
        """
        Compute ACF in the Fourier domain when there is no phase gradient
        """
        snmax = s_max
        dnunmax = dnu_max
        alph2 = alpha/2
        hns = int(np.floor(ns/2))
        sn = np.arange(-hns, hns+1)
        sn = sn*2*snmax/(ns-1)
        kn = np.arange(-hns, hns+1)
        kn = kn/(2*snmax)
        dnun = np.arange(nf)
        dnun = dnun*dnunmax/(nf-1)
        SNX, SNY = np.meshgrid(sn, sn)
        SNX = ifftshift(SNX, 1)
        SNY = ifftshift(SNY, 0)
        KNX, KNY = np.meshgrid(kn, kn)
        KNX = ifftshift(KNX, 1)
        KNY = ifftshift(KNY, 0)
        KN2 = np.add(np.power(KNX, 2), np.power(KNY, 2))  # KN2 is fftshifted
        # Get medium parameters assuming x is the major axis
        sqrtar = np.sqrt(ar)
        # compute 2-D spatial ACF for dnun=0 and the corresponding 2-D spectrum
        # ACF of e-field
        gammes = np.exp(-0.5*(np.power(np.add(np.power(SNX/sqrtar, 2),
                                              np.power(SNY*sqrtar, 2)),
                                       alph2)))
        pk0 = fft2(gammes)
        # initialize arrays
        gammi = np.zeros((ns, ns, 2*nf-1))
        # first dnun=0
        gammi[:, :, nf-1] = fftshift(np.multiply(gammes, np.conj(gammes)))
        # then rest of dnun
        for idnun in range(1, nf):
            temp = np.multiply(pk0,
                               np.exp(np.multiply(-1j*2*np.pi**2*dnun[idnun],
                                                  KN2)))
            temp = ifft2(temp)
            # Make sure python realises this is real, to avoid warnings
            temp = np.real(fftshift(np.multiply(temp, np.conj(temp))))
            gammi[:, :, nf+(idnun-1)] = temp
            gammi[:, :, nf-(idnun-1)] = temp

        self.gammix = np.transpose(gammi[hns, :, :])
        self.gammiy = np.transpose(gammi[:, hns, :])
        return

    def plot_acf(self):
        """
        Plots the simulated ACF
        """
        plt.pcolormesh(self.gammix)
        plt.show()
        plt.pcolormesh(self.gammiy)
        plt.show()
        return
