#!/usr/bin/env python

"""
ththmod.py
----------------------------------
Code for handling theta-theta transformation by Daniel Baker
"""

import numpy as np
import astropy.units as u
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def svd_model(arr, nmodes=1):
    u, s, w = np.linalg.svd(arr)
    s[nmodes:] = 0
    S = np.zeros(([len(u), len(w)]), np.complex128)
    S[:len(s), :len(s)] = np.diag(s)
    model = np.dot(np.dot(u, S), w)
    return (model)


def chi_par(x, A, x0, C):
    """
    Parabola for fitting to chisq curve.
    """
    return A*(x - x0)**2 + C


def thth_map(SS, tau, fd, eta, edges):
    """Map from Secondary Spectrum to theta-theta space

    Arguments:
    SS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins(symmetric about 0)
    """

    # Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    # Calculate theta1 and th2 arrays
    th1 = np.ones((th_cents.shape[0], th_cents.shape[0])) * th_cents
    th2 = th1.T

    # tau and fd step sizes
    dtau = np.diff(tau).mean()
    dfd = np.diff(fd).mean()

    # Find bin in SS space that each point maps back to
    tau_inv = (((eta * (th1**2 - th2**2))*u.mHz**2
                - tau[0] + dtau/2)//dtau).astype(int)
    fd_inv = (((th1 - th2)*u.mHz - fd[0] + dfd/2)//dfd).astype(int)

    # Define thth
    thth = np.zeros(tau_inv.shape, dtype=complex)

    # Only fill thth points that are within the SS
    pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fd_inv < fd.shape[0])
    thth[pnts] = SS[tau_inv[pnts], fd_inv[pnts]]

    # Preserve flux
    thth *= np.abs(2*eta*(th2-th1)).value

    # Force Hermetian
    thth -= np.tril(thth)
    thth += np.conjugate(np.triu(thth).T)
    thth -= np.diag(np.diag(thth))
    thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
    thth = np.nan_to_num(thth)

    return thth


def thth_redmap(SS, tau, fd, eta, edges):
    """
    Map from Secondary Spectrum to theta-theta space for the largest
    possible filled in sqaure within edges

    Arguments:
    SS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins(symmetric about 0)
    """

    # Find full thth
    thth = thth_map(SS, tau, fd, eta, edges)

    # Find region that is fully within SS
    th_cents = (edges[1:]+edges[:-1])/2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    th_pnts = ((th_cents**2)*eta.value < np.abs(tau.max().value)) * \
        (np.abs(th_cents) < np.abs(fd.max()).value/2)
    thth_red = thth[th_pnts, :][:, th_pnts]
    edges_red = th_cents[th_pnts]
    edges_red = (edges_red[:-1]+edges_red[1:])/2
    edges_red = np.concatenate((np.array([edges_red[0] -
                                          np.diff(edges_red).mean()]),
                                edges_red, np.array([edges_red[-1] +
                                                     np.diff(edges_red).
                                                     mean()])))
    return thth_red, edges_red


def rev_map(thth, tau, fd, eta, edges):
    """
    Map back from theta-theta space to SS space

    Arguments:
    thth -- 2d theta-theta spectrum
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins(symmetric about 0)
    """

    # Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]

    fd_map=(th_cents[np.newaxis,:]-th_cents[:,np.newaxis])
    tau_map=eta.value*(th_cents[np.newaxis,:]**2-th_cents[:,np.newaxis]**2)
    fd_edges=(np.linspace(0,fd.shape[0],fd.shape[0]+1)-.5)*(fd[1]-fd[0]).value+fd[0].value
    tau_edges=(np.linspace(0,tau.shape[0],tau.shape[0]+1)-.5)*(tau[1]-tau[0]).value+tau[0].value
    
    recov=np.histogram2d(np.ravel(fd_map),
                         np.ravel(tau_map),
                         bins=(fd_edges,tau_edges),
                         weights=np.ravel(thth/np.abs(2*eta*fd_map.T)).real)[0] +\
            np.histogram2d(np.ravel(fd_map),
                         np.ravel(tau_map),
                         bins=(fd_edges,tau_edges),
                         weights=np.ravel(thth/np.abs(2*eta*fd_map.T)).imag)[0]*1j
    recov/=np.histogram2d(np.ravel(fd_map),
                         np.ravel(tau_map),
                         bins=(fd_edges,tau_edges))[0]
    recov=np.nan_to_num(recov)
    return(recov.T)

def modeler(SS, tau, fd, eta, edges,fd2=None,tau2=None):
    if fd2==None:
        fd2=fd
    if tau2==None:
        tau2=tau
    thth_red,edges_red=thth_redmap(SS, tau, fd, eta, edges)
    ##Find first eigenvector and value
    w,V=eigsh(thth_red,1)
    w=w[0]
    V=V[:,0]
    ##Use larges eigenvector/value as model
    thth2_red=np.outer(V,np.conjugate(V))
    thth2_red*=np.abs(w)
    ##Map back to SS for high
#    thth2_red[thth_red==0]=0
    recov=rev_map(thth2_red,tau2,fd2,eta,edges_red)
    model=2*np.fft.ifft2(np.fft.ifftshift(recov)).real
    return(thth_red,thth2_red,recov,model,edges_red)

def chisq_calc(dspec,SS, tau, fd, eta, edges,mask,N,fd2=None,tau2=None):
    model=modeler(SS, tau, fd, eta, edges,fd2,tau2)[3][:dspec.shape[0],:dspec.shape[1]]
    chisq=np.sum((model-dspec)[mask]**2)/N
    return(chisq)

def G_revmap(w,V,eta,edges,tau,fd):
    th_cents=(edges[1:]+edges[:-1])/2
    th_cents-=th_cents[np.abs(th_cents)==np.abs(th_cents).min()]
    screen=np.conjugate(V[:,np.abs(w)==np.abs(w).max()][:,0]*np.sqrt(w[np.abs(w)==np.abs(w).max()]))
#     screen/=np.abs(2*eta*th_cents).value
    dtau=np.diff(tau).mean()
    dfd=np.diff(fd).mean()
    fd_map=(((th_cents*fd.unit)-fd[0] +dfd/2)//dfd).astype(int)
    tau_map=(((eta*(th_cents*fd.unit)**2)-tau[0]+dtau/2)//dtau).astype(int)
    pnts=(fd_map>0)*(tau_map>0)*(fd_map<fd.shape[0])*(tau_map<tau.shape[0])
    SS_G=np.zeros((tau.shape[0],fd.shape[0]),dtype=complex)
    SS_G[tau_map[pnts],fd_map[pnts]]=screen[pnts]
    G=np.fft.ifft2(np.fft.ifftshift(SS_G))
    return(G)

def len_arc(x,eta):
    a=2*eta
    return((a*x*np.sqrt((a*x)**2 + 1) +np.arcsinh(a*x))/(2.*a))

def arc_edges(eta,dfd,dtau,fd_max,n):
    x_max=fd_max/dfd
    eta_ul=dfd**2*eta/dtau
    l_max=len_arc(x_max.value,eta_ul.value)
    dl=l_max/(n//2 - .5)
    x=np.zeros(int(n//2))
    x[0]=dl/2
    for i in range(x.shape[0]-1):
        x[i+1]=x[i]+dl/(np.sqrt(1+(2*eta_ul*x[i])**2))
    edges=np.concatenate((-x[::-1],x))*dfd.value
    return(edges) 

def ext_find(x, y):
    dx = np.diff(x).mean()
    dy = np.diff(y).mean()
    ext = [(x[0] - dx / 2).value, (x[-1] + dx / 2).value,
           (y[0] - dy / 2).value, (y[-1] + dy / 2).value]
    return (ext)

def fft_axis(x, unit, pad=1):
    fx = np.fft.fftshift(
        np.fft.fftfreq(pad * x.shape[0], x[1] - x[0]).to_value(unit)) * unit
    return (fx)

def sample_plot(dspec,
                SS,
                thth,
                thth2,
                SS2,
                dspec2,
                etas,
                chisq,
                time,
                freq,
                fd,
                tau,
		edges,
                fdm,
                taum,
                eta_fit=0,
                fit_str=None,
                err_str=None):
    if eta_fit == 0:
        eta_fit = etas[chisq == chisq.min()].mean()
    eta_low = etas.min()
    eta_high = etas.max()
    ##Plotting variables
    SS_ext = ext_find(fd, tau)
    dspec_ext = ext_find(time.to(u.min), freq)
    SS_min = np.median(np.abs(SS)**2)
    SS_max = np.max(np.abs(2 * thth2)**2) * np.exp(-3)

    ##Compare model to data in plots
    grid = plt.GridSpec(4, 2, wspace=0.4, hspace=0.3)
    plt.figure(figsize=(8, 16))
    plt.subplot(grid[0, 0])
    plt.imshow(dspec,
               aspect='auto',
               origin='lower',
               extent=dspec_ext,
               vmin=0,
               vmax=1)
    plt.title('Dynamic Spectrum')
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.subplot(grid[0, 1])
    plt.imshow(dspec2,
               aspect='auto',
               origin='lower',
               extent=dspec_ext,
               vmin=0,
               vmax=1)
    plt.title('Dynamic Spectrum Model')
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.subplot(grid[1, 0])
    plt.imshow(np.abs(SS)**2,
               norm=LogNorm(),
               aspect='auto',
               origin='lower',
               extent=SS_ext,
               vmin=SS_min,
               vmax=SS_max)
    plt.xlim((-fdm, fdm))
    plt.ylim((0, taum))
    plt.plot(fd, eta_low * (fd**2), 'k')
    plt.plot(fd, eta_high * (fd**2), 'k')
    plt.plot(fd, eta_fit * (fd**2), 'r')
    plt.title('Secondary Spectrum')
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.subplot(grid[1, 1])
    plt.imshow(np.abs(2 * SS2)**2,
               norm=LogNorm(),
               aspect='auto',
               origin='lower',
               extent=SS_ext,
               vmin=SS_min,
               vmax=SS_max)
    plt.xlim((-fdm, fdm))
    plt.ylim((0, taum))
    plt.plot(fd, etas.min() * (fd**2), 'k')
    plt.plot(fd, etas.max() * (fd**2), 'k')
    plt.plot(fd, eta_fit * (fd**2), 'r')
    plt.title('Secondary Spectrum Model')
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')

    thth_min = np.median(np.abs(thth)) / 10
    thth_max = np.max(np.abs(thth))

    plt.subplot(grid[2, 0])
    plt.imshow(np.abs(thth),
               norm=LogNorm(),
               aspect='auto',
               origin='lower',
               extent=[edges[0], edges[-1], edges[0], edges[-1]],
               vmin=thth_min,
               vmax=thth_max)
    plt.title(r'$\theta-\theta$')
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.subplot(grid[2, 1])
    plt.imshow(np.abs(thth2),
               norm=LogNorm(),
               aspect='auto',
               origin='lower',
               extent=[edges[0], edges[-1], edges[0], edges[-1]],
               vmin=thth_min,
               vmax=thth_max)
    plt.title(r'$\theta-\theta$')
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.subplot(grid[3, :])
    plt.plot(etas, chisq)
    if not err_str == None:
        plt.axvline(eta_fit.value,label=r'%s $\pm$ %s $s^3$' % (fit_str, err_str))
    elif eta_fit>0:
        plt.axvline(eta_fit.value)
    plt.xlabel(r'$\eta$ ($s^3$)')
    plt.ylabel(r'$\chi^2$')
    plt.title('Curvature Search')
    plt.legend()

