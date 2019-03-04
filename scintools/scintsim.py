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

class Simulation():
    def __init__(self,mb2=2,rf=1,ds=0.01,alpha=5/3,ar=1,psi=0,inner=0.001,
                 ns=256,nf=256,dlam=0.5,lamsteps=False,seed=None,verbose=True,
                 nx=None,ny=None,dx=None,dy=None):
        """
        mb2: Max Born parameter for strength of scattering
        rf: Fresnel scale
        ds (or dx,dy): Spatial step sizes with respect to rf
        alpha: Structure function exponent (Kolmogorov = 5/3)
        ar: Anisotropy axial ratio
        psi: Anisotropy orientation
        inner: Inner scale w.r.t rf - should generally be smaller than step size
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
        
        self.set_constants()
        self.get_screen()
        
        return
    
    def set_constants(self):
        
        ns = 1
        lenx = self.nx*self.dx 
        leny = self.ny*self.dy
        #ffconx = (2.0/(ns*lenx*lenx))*(np.pi*self.rf)**2
        #ffcony = (2.0/(ns*leny*leny))*(np.pi*self.rf)**2
        dqx = 2*np.pi/lenx 
        dqy=2*np.pi/leny
        #dqx2 = dqx*dqx 
        #dqy2 = dqy*dqy
        a2 = self.alpha*0.5
        #spow = (1.0+a2)*0.5
        #ap1 = self.alpha+1.0
        #ap2 = self.alpha+2.0
        #aa = 1.0+a2
        ab = 1.0-a2
        #cdrf = 2.0**(self.alpha)*np.cos(self.alpha*np.pi*0.25)*gamma(aa)/self.mb2
        #s0 = self.rf*cdrf**(1.0/self.alpha)
        
        cmb2 = self.alpha*self.mb2/(4*np.pi*gamma(ab)*np.cos(self.alpha*np.pi*0.25)*ns)
        self.consp = cmb2*dqx*dqy/(self.rf**self.alpha)
        #scnorm = 1.0/(self.nx*self.ny)
        
        #ffconlx = ffconx*0.5
        #ffconly = ffcony*0.5
        #sref = self.rf**2/s0
        return
    
    def plot_screen(self):
        if not hasattr(self, 'xyp'): self.getscreen()
        plt.pcolormesh(self.xyp)
        plt.show()
        return
    
    
    def get_screen(self):
        """
        Get phase screen in x and y
        """
        random.seed(self.seed) #Set the seed, if any
        
        nx2 = int(self.nx/2)
        ny2 = int(self.ny/2)
        
        w = np.zeros([self.nx, ny2]) #initialize array
        dqx = 2*np.pi/(self.dx*self.nx) 
        dqy = 2*np.pi/(self.dy*self.ny)
        #first do ky=0 line
        k = np.arange(1,nx2,1)
        w[k,0] = self.swdsp(kx=(k)*dqx,ky=1)
        w[self.nx-k,0] = w[k,0]
        #then do kx=0 line
        l = np.arange(1,ny2,1);
        w[0,l] = self.swdsp(kx=1,ky=(l)*dqy)   #(kx,ky,axirat,angle,lout,lin,consp,alpha)
        #w(1,ny+2-l)=w(1,l);
        #now do the rest of the field
        kp = np.arange(1,nx2,1) 
        k = np.arange(nx2,self.nx,1) 
        km = -(self.nx-k)
        for il in range(0,ny2):
            w[kp,il] = self.swdsp(kx=(kp)*dqx,ky=(il)*dqy)
            w[k,il] = self.swdsp(kx=km*dqx, ky=(il)*dqy)
        #done the whole screen weights, now generate complex gaussian array
        xyp = np.zeros([self.nx,self.ny],dtype=np.dtype(np.csingle))
        xyp[0:self.nx,0:ny2] = np.multiply(w,np.add(randn(self.nx,ny2), 1j*randn(self.nx,ny2)))   #half screen
        #now fill in the other half because xyp is hermitean
        xyp[nx2:self.nx,ny2:self.ny] = np.flip(np.flip(np.transpose(np.conj(xyp[0:(nx2),0:(ny2)])),axis=0),axis=1)
        xyp[0:nx2,ny2:self.ny] = np.flip(np.conj(xyp[0:nx2,0:ny2]),axis=1)
        xyp = np.real(fft2(xyp))    
        self.xyp = xyp
        return
    

    def swdsp(self,kx=0,ky=0):
        cs = np.cos(self.psi*np.pi/180)
        sn = np.sin(self.psi*np.pi/180)
        r = self.ar
        con = np.sqrt(self.consp) 
        alf = -(self.alpha+2)/4
        #anisotropy parameters
        a = (cs**2)/r + r*sn**2 
        b = r*cs**2 + sn**2/r 
        c = 2*cs*sn*(1/r-r)
        q2 = a*np.power(kx,2) + b*np.power(ky,2) + c*np.multiply(kx,ky)
        out = con*np.multiply(np.power(q2,alf), 
                              np.exp(-(np.add(np.power(kx,2), 
                                           np.power(ky,2)))*self.inner**2/2))  #isotropic inner scale
        return out
        
        
        
        
        
    
        
        