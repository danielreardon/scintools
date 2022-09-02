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
from matplotlib.colors import LogNorm,SymLogNorm
from scipy.optimize import curve_fit

def svd_model(arr, nmodes=1):
    """
    Model a matrix using the first nmodes modes of the singular value decomposition

    Arguments:
    arr -- 2d numpy array ti be modeled
    nmodes -- Number os SVD modes to use in reconstruction
    """
    u, s, w = np.linalg.svd(arr)
    s[nmodes:] = 0
    S = np.zeros(([len(u), len(w)]), np.complex128)
    S[:len(s), :len(s)] = np.diag(s)
    model = np.dot(np.dot(u, S), w)
    return (model)


def chi_par(x, A, x0, C):
    """
    Parabola for fitting to chisq curve.

    Arguments:
    x -- numpy array of x coordinates of fit
    A -- 
    x0 -- x coordinate of parabola extremum
    C -- y coordinate of extremum
    """
    return A*(x - x0)**2 + C


def thth_map(CS, tau, fd, eta, edges,hermetian=True):
    """Map from Secondary Spectrum to theta-theta space

    Arguments:
    CS -- Conjugate Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins(symmetric about 0)
    """
    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)
    # Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    # Calculate theta1 and th2 arrays
    th1 = np.ones((th_cents.shape[0], th_cents.shape[0])) * th_cents
    th2 = th1.T

    # tau and fd step sizes
    dtau = np.diff(tau).mean()
    dfd = np.diff(fd).mean()

    # Find bin in CS space that each point maps back to
    tau_inv = (((eta * (th1**2 - th2**2))
                - tau[0] + dtau/2)//dtau).astype(int)
    fd_inv = (((th1 - th2) - fd[0] + dfd/2)//dfd).astype(int)

    # Define thth
    thth = np.zeros(tau_inv.shape, dtype=complex)

    # Only fill thth points that are within the CS
    pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fd_inv < fd.shape[0])
    thth[pnts] = CS[tau_inv[pnts], fd_inv[pnts]]

    # Preserve flux (int
    thth *= np.sqrt(np.abs(2*eta*(th2-th1)).value)
    if hermetian:
        # Force Hermetian
        thth -= np.tril(thth)
        thth += np.conjugate(np.triu(thth).T)
        thth -= np.diag(np.diag(thth))
        thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
        thth = np.nan_to_num(thth)

    return thth


def thth_redmap(CS, tau, fd, eta, edges,hermetian=True):
    """
    Map from Secondary Spectrum to theta-theta space for the largest
    possible filled in sqaure within edges

    Arguments:
    CS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins(symmetric about 0)
    """

    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    # Find full thth
    thth = thth_map(CS, tau, fd, eta, edges,hermetian)

    # Find region that is fully within CS
    th_cents = (edges[1:]+edges[:-1])/2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    th_pnts = ((th_cents**2)*eta < np.abs(tau.max())) * \
        (np.abs(th_cents) < np.abs(fd.max())/2)
    thth_red = thth[th_pnts, :][:, th_pnts]
    edges_red = th_cents[th_pnts]
    edges_red = (edges_red[:-1]+edges_red[1:])/2
    edges_red = np.concatenate((np.array([edges_red[0].value -
                                          np.diff(edges_red.value).mean()]),
                                edges_red.value, np.array([edges_red[-1].value +
                                                     np.diff(edges_red.value).
                                                     mean()])))*edges_red.unit
    return thth_red, edges_red


def rev_map(thth, tau, fd, eta, edges,isdspec=True):
    """
    Map back from theta-theta space to CS space

    Arguments:
    thth -- 2d theta-theta spectrum
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins(symmetric about 0)
    """

    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    # Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]

    fd_map=(th_cents[np.newaxis,:]-th_cents[:,np.newaxis])
    tau_map=eta.value*(th_cents[np.newaxis,:]**2-th_cents[:,np.newaxis]**2)
    fd_edges=(np.linspace(0,fd.shape[0],fd.shape[0]+1)-.5)*(fd[1]-fd[0]).value+fd[0].value
    tau_edges=(np.linspace(0,tau.shape[0],tau.shape[0]+1)-.5)*(tau[1]-tau[0]).value+tau[0].value
    
    ## Bind TH-TH points back into Conjugate Spectrum
    recov=np.histogram2d(np.ravel(fd_map.value),
                         np.ravel(tau_map.value),
                         bins=(fd_edges,tau_edges),
                         weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).real)[0] +\
            np.histogram2d(np.ravel(fd_map.value),
                         np.ravel(tau_map.value),
                         bins=(fd_edges,tau_edges),
                         weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).imag)[0]*1j
    norm=np.histogram2d(np.ravel(fd_map.value),
                         np.ravel(tau_map.value),
                         bins=(fd_edges,tau_edges))[0]
    if isdspec:
        recov += np.histogram2d(np.ravel(-fd_map.value),
                            np.ravel(-tau_map.value),
                            bins=(fd_edges,tau_edges),
                            weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).real)[0] -\
                np.histogram2d(np.ravel(-fd_map.value),
                            np.ravel(-tau_map.value),
                            bins=(fd_edges,tau_edges),
                            weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).imag)[0]*1j
        norm+=np.histogram2d(np.ravel(-fd_map.value),
                            np.ravel(-tau_map.value),
                            bins=(fd_edges,tau_edges))[0] 
    recov/=norm
    recov=np.nan_to_num(recov)
    return(recov.T)

def modeler(CS, tau, fd, eta, edges,fd2=None,tau2=None):
    """
    Create theta-theta array as well as model theta-theta, Conjugate Spectrum and Dynamic Spectrum
    from data conjugate spectrum and curvature

    Arguments:
    CS -- 2d complex numpy array of the Conjugate Spectrum
    tau -- 1d array of tau values for conjugate spectrum in ascending order with units
    fd -- 1d array of fd values for conjugate spectrum in ascending order with units
    eta -- curvature of main arc in units of (tau/fd^2)
    edges -- 1d array of coordinate of bin edges in theta-theta array
    fd2 --  fd values for reverse theta-theta map (defaults to fd)
    tau2 -- tau values for reverse theta-theta map (defaults to tau)
    """
    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    if fd2==None:
        fd2=fd
    else:
        fd2 = unit_checks(fd2,'fd2',u.mHz)
    if tau2==None:
        tau2=tau
    else:
        tau2 = unit_checks(tau2,'tau2',u.us)
    thth_red,edges_red=thth_redmap(CS, tau, fd, eta, edges)
    ##Find first eigenvector and value
    w,V=eigsh(thth_red,1,which='LA')
    w=w[0]
    V=V[:,0]
    ##Use larges eigenvector/value as model
    thth2_red=np.outer(V,np.conjugate(V))
    thth2_red*=np.abs(w)
    ##Map back to CS for high
#    thth2_red[thth_red==0]=0
    recov=rev_map(thth2_red,tau2,fd2,eta,edges_red)
    model=np.fft.ifft2(np.fft.ifftshift(recov)).real
    return(thth_red,thth2_red,recov,model,edges_red,w,V)

def chisq_calc(dspec,CS, tau, fd, eta, edges,mask,N,fd2=None,tau2=None):
    """
    Calculate chisq value for modeled dynamic spectrum for a given curvature

    Arguments:
    dspec -- 2d array of observed dynamic spectrum
    CS -- 2d array of conjugate spectrum
    tau -- 1d array of tau values for conjugate spectrum in ascending order with units
    fd -- 1d array of fd values for conjugate spectrum in ascending order with units
    eta -- curvature of main arc in units of (tau/fd^2)
    edges -- 1d array of coordinate of bin edges in theta-theta array
    mask -- 2d boolean array of points in dynamic spectrum for fitting
    N -- Variance of dynamic spectrum noise
    fd2 --  fd values for reverse theta-theta map (defaults to fd)
    tau2 -- tau values for reverse theta-theta map (defaults to tau)

    """

    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    model=modeler(CS, tau, fd, eta, edges,fd2,tau2)[3][:dspec.shape[0],:dspec.shape[1]]
    chisq=np.sum((model-dspec)[mask]**2)/N
    return(chisq)

def Eval_calc(CS, tau, fd, eta, edges):
    """
    Calculates the dominant eigenvalue for the theta-theta matrix from a given conjugate spectrum
    and curvature.


    Arguments:
    CS -- 2d complex numpy array of the Conjugate Spectrum
    tau -- 1d array of tau values for conjugate spectrum in ascending order with units
    fd -- 1d array of fd values for conjugate spectrum in ascending order with units
    eta -- curvature of main arc in units of (tau/fd^2)
    edges -- 1d array of coordinate of bin edges in theta-theta array
    """

    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    thth_red,edges_red=thth_redmap(CS, tau, fd, eta, edges)
    ##Find first eigenvector and value
    v0=np.copy(thth_red[thth_red.shape[0]//2,:])
    v0/=np.sqrt((np.abs(v0)**2).sum())
    w,V=eigsh(thth_red,1,v0=v0,which='LA')
    return(np.abs(w[0]))

def G_revmap(w,V,eta,edges,tau,fd):

    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    th_cents=(edges[1:]+edges[:-1])/2
    th_cents-=th_cents[np.abs(th_cents)==np.abs(th_cents).min()]
    screen=np.conjugate(V[:,np.abs(w)==np.abs(w).max()][:,0]*np.sqrt(w[np.abs(w)==np.abs(w).max()]))
#     screen/=np.abs(2*eta*th_cents).value
    dtau=np.diff(tau).mean()
    dfd=np.diff(fd).mean()
    fd_map=(((th_cents*fd.unit)-fd[0] +dfd/2)//dfd).astype(int)
    tau_map=(((eta*(th_cents*fd.unit)**2)-tau[0]+dtau/2)//dtau).astype(int)
    pnts=(fd_map>0)*(tau_map>0)*(fd_map<fd.shape[0])*(tau_map<tau.shape[0])
    CS_G=np.zeros((tau.shape[0],fd.shape[0]),dtype=complex)
    CS_G[tau_map[pnts],fd_map[pnts]]=screen[pnts]
    G=np.fft.ifft2(np.fft.ifftshift(CS_G))
    return(G)

def len_arc(x,eta):
    """
    Calculate distance along arc with curvature eta to points (x,eta x**2) (DEVELOPMENT ONLY)
    """
    a=2*eta
    return((a*x*np.sqrt((a*x)**2 + 1) +np.arcsinh(a*x))/(2.*a))

def arc_edges(eta,dfd,dtau,fd_max,n):
    '''
    Calculate evenly spaced in arc length edges array (DEVELOPMENT ONLY)

    Arguments:
    eta -- Curvature 
    dfd -- fD resolution of conjugate spectrum
    dtau -- tau resolution of conjugate spectrum
    fd_max largest fD value in edges array
    b -- Integer number of points in array (assumed to be even)
    '''
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
    '''
    Determine extent for imshow to center bins at given coordinates

    x -- x coordinates of data
    y -- y coordiantes of data

    '''
    dx = np.diff(x).mean()
    dy = np.diff(y).mean()
    ext = [(x[0] - dx / 2).value, (x[-1] + dx / 2).value,
           (y[0] - dy / 2).value, (y[-1] + dy / 2).value]
    return (ext)

def fft_axis(x, unit, pad=0):
    '''
    Calculates fourier space coordinates from data space coordinates.

    Arguments
    x -- Astropy 
    unit -- desired unit for fourier coordinates
    pad -- integer giving how many additional copies of the data are padded in this direction
    '''
    fx = np.fft.fftshift(
        np.fft.fftfreq((pad+1) * x.shape[0], x[1] - x[0]).to_value(unit)) * unit
    return (fx)


def single_search(params):
    """
    Curvature Search for a single chunk of a dynamic spectrum. Designed for use with MPI4py
    

    Arguments:
    params -- A tuple containing
        dspec2 -- The chunk of the dynamic spectrum
        freq2 -- The frequency channels of that chunk (with units)
        time -- The time bins of that chunk (with units)
        eta_low -- The lower limit of curvatures to search (with units)
        eta_high -- the upper limit of curvatures to search (with units)
        edges -- The bin edges for Theta-Theta
        name -- A string filename used if plotting
        plot -- A bool controlling if the result should be plotted
        neta -- Number of curvatures to test
        coher -- A bool for whether to use coherent (True) or incoherent (False) theta-theta
    """
    
    ## Reap Parameters
    dspec2,freq,time,eta_low,eta_high,edges,name,plot,fw,npad,neta,coher=params

    ## Verify units
    time = unit_checks(time,'time2',u.s)
    freq = unit_checks(freq,'freq2',u.MHz)
    eta_low = unit_checks(eta_low,'eta_low',u.s**3)
    eta_high = unit_checks(eta_high,'eta_high',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    ## Curvature Range to Search Over
    etas = np.linspace(eta_low, eta_high, neta)

    ## Calculate fD and tau arrays
    fd = fft_axis(time, u.mHz, npad)
    tau = fft_axis(freq, u.us, npad)

    ## Pad dynamic Spectrum
    dspec_pad = np.pad(dspec2,
                   ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
                   mode='constant',
                   constant_values=dspec2.mean())

    ## Calculate Conjugate Spectrum
    CS = np.fft.fft2(dspec_pad)
    CS = np.fft.fftshift(CS)
    eigs = np.zeros(etas.shape)
    if coher:
        ## Loop over all curvatures
        for i in range(eigs.shape[0]):
            try:
                ## Find largest Eigenvalue for curvature
                eigs[i] = Eval_calc(CS, tau, fd, etas[i], edges)
            except:
                ## Set eigenvalue to NaN in event of failure
                eigs[i]=np.nan
    else:
        SS = np.abs(CS)**2
        ## Loop over all curvatures
        for i in range(eigs.shape[0]):
            try:
                ## Find largest Eigenvalue for curvature
                eigs[i] = Eval_calc(SS, tau, fd, etas[i], edges)
            except:
                ## Set eigenvalue to NaN in event of failure
                eigs[i]=np.nan   
    
    ## Fit eigenvalue peak
    try:
        ## Remove failed curvatures
        etas=etas[np.isfinite(eigs)]
        eigs=eigs[np.isfinite(eigs)]

        ## Reduced range around peak to be withing fw times curvature of maximum eigenvalue
        etas_fit = etas[np.abs(etas - etas[eigs == eigs.max()]) < fw * etas[eigs == eigs.max()]]
        eigs_fit = eigs[np.abs(etas - etas[eigs == eigs.max()]) < fw * etas[eigs == eigs.max()]]

        ## Initial Guesses
        C = eigs_fit.max()
        x0 = etas_fit[eigs_fit == C][0].value
        if x0 == etas_fit[0].value:
            A = (eigs_fit[-1] - C) / ((etas_fit[-1].value - x0)**2)
        else:
            A = (eigs_fit[0] - C) / ((etas_fit[0].value - x0)**2)

        ## Fit parabola around peak
        popt, pcov = curve_fit(chi_par,
                                etas_fit.value,
                                eigs_fit,
                                p0=np.array([A, x0, C]))

        ## Record curvauture fit and error
        eta_fit = popt[1]*u.us/u.mHz**2
        eta_sig = np.sqrt((eigs_fit - chi_par(etas_fit.value, *popt)).std() / np.abs(popt[0]))*u.us/u.mHz**2
    except:
        ## Return NaN for curvautre and error if fitting fails
        popt=None
        eta_fit=np.nan
        eta_sig=np.nan

    ## Plotting
    try:
        if plot:
            ## Create diagnostic plots where requested
            PlotFunc(dspec2,time,freq,CS,fd,tau,edges,eta_fit,eta_sig,etas,eigs,etas_fit,popt)
            plt.savefig(name)
            plt.close()
    except:
        print('Plotting Error',flush=True)

    ## Progress Report
    print('Chunk completed (eta = %s +- %s at %s)' %(eta_fit,eta_sig,freq.mean()),flush=True)
    return(eta_fit,eta_sig,freq.mean(),time.mean(),eigs)

def PlotFunc(dspec,time,freq,CS,fd,tau,
            edges,eta_fit,eta_sig,etas,measure,etas_fit,fit_res,
            tau_lim=None,method='eigenvalue'):
    '''
    Plotting script to look at invidivual chunks

    Arguments
    dspec -- 2D numpy array containing the dynamic spectrum
    time -- 1D numpy array of the dynamic spectrum time bins (with units)
    freq -- 1D numpy array of the dynamic spectrum frequency channels (with units)
    CS -- 2D numpy array of the conjugate spectrum
    fd -- 1D numpy array of the SS fd bins (with units)
    tau -- 1D numpy array of the SS tau bins (with units)
    edges -- 1D numpy array with the bin edges for theta-theta
    eta_fit -- Best fit curvature
    eta_sig -- Error on best fir curvature
    etas -- 1D numpy array of curvatures searched over
    measure -- 1D numpy array with largest eigenvalue (method = 'eigenvalue') or chisq value (method = 'chisq') for etas
    etas_fit -- Subarray of etas used for fitting
    fit_res -- Fit parameters for parabola at extremum
    tau_lim -- Largest tau value for SS plots
    method -- Either 'eigenvalue' or 'chisq' depending on how curvature was found
    '''

    ## Verify units
    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    edges = unit_checks(edges,'edges',u.mHz)
    eta_fit = unit_checks(eta_fit,'eta_fit',u.s**3)
    eta_sig = unit_checks(eta_sig,'eta_sig',u.s**3)
    etas = unit_checks(etas,'etas',u.s**3)
    etas_fit = unit_checks(etas_fit,'etas_fit',u.s**3)

    if not tau_lim==None:
        tau_lim = unit_checks(tau_lim,'tau_lim',u.us)
    else:
        tau_lim=tau.max()

    ## Determine fd limits 
    fd_lim=min(2*edges.max(),fd.max()).value

    ## Determine TH-TH and model
    if np.isnan(eta_fit):
        eta=etas.mean()
        thth_red, thth2_red, recov, model, edges_red,w,V = modeler(CS, tau, fd, etas.mean(), edges)
    else:
        eta=eta_fit
        thth_red, thth2_red, recov, model, edges_red,w,V = modeler(CS, tau, fd, eta_fit, edges)

    ## Create model Wavefield and Conjugate Wavefield
    ththE_red=thth_red*0
    ththE_red[ththE_red.shape[0]//2,:]=np.conjugate(V)*np.sqrt(w)
    ##Map back to time/frequency space
    recov_E=rev_map(ththE_red,tau,fd,eta,edges_red,isdspec = False)
    model_E=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec.shape[0],:dspec.shape[1]]
    model_E*=(dspec.shape[0]*dspec.shape[1]/4)
    model_E[dspec>0]=np.sqrt(dspec[dspec>0])*np.exp(1j*np.angle(model_E[dspec>0]))
    model_E=np.pad(model_E,
                    (   (0,CS.shape[0]-model_E.shape[0]),
                        (0,CS.shape[1]-model_E.shape[1])),
                    mode='constant',
                    constant_values=0)
    recov_E=np.abs(np.fft.fftshift(np.fft.fft2(model_E)))**2
    model_E=model_E[:dspec.shape[0],:dspec.shape[1]]
    N_E=recov_E[:recov_E.shape[0]//4,:].mean()

    ## Create derotated thth
    thth_derot=thth_red*np.conjugate(thth2_red)

    ##Plots
    grid=plt.GridSpec(6,2)
    plt.figure(figsize=(8,24))
    
    ## Data Dynamic Spectrum
    plt.subplot(grid[0,0])
    plt.imshow(dspec,
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower',
            vmin=0,vmax=dspec.max())
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Data Dynamic Spectrum')

    ## Model Dynamic Spectrum
    plt.subplot(grid[0,1])
    plt.imshow(model[:dspec.shape[0],:dspec.shape[1]],
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower',
            vmin=0,vmax=dspec.max())
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Model Dynamic Spectrum')

    ## Data Secondary Spectrum
    plt.subplot(grid[1,0])
    plt.imshow(np.abs(CS)**2,
            norm=LogNorm(vmin=np.median(np.abs(CS)**2),vmax=np.abs(CS).max()**2),
            origin='lower',
            aspect='auto',
            extent=ext_find(fd,tau))
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim.value))
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.title('Data Secondary Spectrum')
    plt.plot(fd,eta*(fd**2),'r',alpha=.7)
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')

    ## Model Secondary Spectrum
    plt.subplot(grid[1,1])
    plt.imshow(np.abs(recov)**2,
            norm=LogNorm(vmin=np.median(np.abs(CS)**2),vmax=np.abs(CS).max()**2),
            origin='lower',
            aspect='auto',
            extent=ext_find(fd,tau))
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim.value))
    plt.title('Model Secondary Spectrum')

    ## Data TH-TH
    plt.subplot(grid[2,0])
    plt.imshow(np.abs(thth_red)**2,
            norm=LogNorm(vmin=np.median(np.abs(thth_red)**2),vmax=np.abs(thth_red).max()**2),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0].value,edges_red[-1].value,edges_red[0].value,edges_red[-1].value])
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Data $\theta-\theta$')

    ## Model TH-TH
    plt.subplot(grid[2,1])
    plt.imshow(np.abs(thth2_red)**2,
            norm=LogNorm(vmin=np.median(np.abs(thth_red)**2),vmax=np.abs(thth_red).max()**2),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0].value,edges_red[-1].value,edges_red[0].value,edges_red[-1].value])
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Model $\theta-\theta$')

    ## Derotated TH-TH Real Part
    plt.subplot(grid[3,0])
    plt.imshow(thth_derot.real,
            norm=SymLogNorm(np.median(np.abs(thth_red)**2),vmin=-np.abs(thth_derot).max(),vmax=np.abs(thth_derot).max()),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0].value,edges_red[-1].value,edges_red[0].value,edges_red[-1].value])
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Derotated $\theta-\theta$ (real)')

    ## Derotated TH-TH Imaginary Part
    plt.subplot(grid[3,1])
    plt.imshow(thth_derot.imag,
            norm=SymLogNorm(np.median(np.abs(thth_red)**2),vmin=-np.abs(thth_derot).max(),vmax=np.abs(thth_derot).max()),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0].value,edges_red[-1].value,edges_red[0].value,edges_red[-1].value])
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Derotated $\theta-\theta$ (imag)')

    ##Plot Eigenvalue vs Curvature
    plt.subplot(grid[4,:])
    plt.plot(etas,measure)
    if not np.isnan(eta_fit):
        exp_fit = int(('%.0e' % eta_fit.value)[2:])
        exp_err = int(('%.0e' % eta_sig.value)[2:])
        fmt = "{:.%se}" % (exp_fit - exp_err)
        fit_string = fmt.format(eta_fit.value)[:2 + exp_fit - exp_err]
        err_string = '0%s' % fmt.format(10**(exp_fit) + eta_sig.value)[1:]
        
        plt.plot(etas_fit,
            chi_par(etas_fit.value, *fit_res),
            label=r'$\eta$ = %s $\pm$ %s $s^3$' % (fit_string, err_string))
        plt.legend()
    if method == 'eigenvalue':
        plt.title('Eigenvalue Search')
        plt.ylabel(r'Largest Eigenvalue')
    else:
        plt.title('Chisquare Search')
        plt.ylabel(r'$\chi^2$')
    plt.xlabel(r'$\eta$ ($s^3$)')

    ## Phase of Wavefield
    plt.subplot(grid[5,0])
    plt.imshow(np.angle(model_E),
            cmap='twilight',
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower',
            vmin=-np.pi,vmax=np.pi)
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Recovered Phases')

    ## Secondary Wavefield
    plt.subplot(grid[5,1])
    plt.imshow(recov_E,
            norm=LogNorm(vmin=N_E),
            origin='lower',
            aspect='auto',
            extent=ext_find(fd,tau))
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim.value))
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.title('Recovered Secondary Wavefield')
    plt.colorbar()
    plt.tight_layout()

def VLBI_chunk_retrieval(params):
    '''
    Performs phase retrieval on a single time/frequency chunk using multiple dynamic spectra and visibilities.
    Designed for use in parallel phase retreival code

    Arguments
    params -- tuple of relevant parameters
    '''

    ## Read parameters
    dspec2_list,edges,time2,freq2,eta,idx_t,idx_f,npad,n_dish = params

    ## Verify unit compatability
    time2 = unit_checks(time2,'time2',u.s)
    freq2 = unit_checks(freq2,'freq2',u.MHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    ## Progress reporting
    print("Starting Chunk %s-%s" %(idx_f,idx_t),flush=True)

    ## Determine fd and tau coordinates of Conjugate Spectrum
    fd = fft_axis(time2, u.mHz, npad)
    tau = fft_axis(freq2, u.us, npad)

    ## Determine which sprectra in dspec2_list are dynamic spectra
    dspec_args=(n_dish*(n_dish+1))/2-np.cumsum(np.linspace(1,n_dish,n_dish))
    thth_red=list()
    for i in range(len(dspec2_list)):
        ## Pad dynamic spectrum to help with peiodicity problem
        dspec_pad = np.pad(dspec2_list[i],
                    ((0, npad * dspec2_list[i].shape[0]), (0, npad * dspec2_list[i].shape[1])),
                    mode='constant',
                    constant_values=dspec2_list[i].mean())

        ## Calculate Conjugate Spectrum (or Conjugate Visibility)
        CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
        if np.isin(i,dspec_args):
            ## Calculate TH-TH for dynamic spectra
            thth_single,edges_red=thth_redmap(CS,tau,fd,eta,edges)
        else:
            ## Calculate THTH for Visiblities
            thth_single,edges_red=thth_redmap(CS,tau,fd,eta,edges,hermetian=False)
        ## Append TH-THT of spectrum to list
        thth_red.append(thth_single)

    ##Determine size of individual spectra
    thth_size=thth_red[0].shape[0]

    ##Create array for composite TH-TH
    thth_comp=np.zeros((thth_size*n_dish,thth_size*n_dish),dtype=complex)

    ## Loop over all pairs of dishes
    for d1 in range(n_dish):
        for d2 in range(n_dish-d1):
            ##Determine position of pair in thth_red list
            idx=int(((n_dish*(n_dish+1))//2)-(((n_dish-d1)*(n_dish-d1+1))//2)+d2)

            ## Insert THTH in apropriate location in composite THTH
            thth_comp[d1*thth_size:(d1+1)*thth_size,(d1+d2)*thth_size:(d1+d2+1)*thth_size]=np.conjugate(thth_red[idx].T)
            ## Make composite TH-TH hermitian by including complex conjugate transpose in mirrored location (Dynamic spectra are along the diagonal and are already Hermitian)
            thth_comp[(d1+d2)*thth_size:(d1+d2+1)*thth_size,d1*thth_size:(d1+1)*thth_size]=thth_red[idx]

    ## Find largest eigvalue and its eigenvector
    w,V=eigsh(thth_comp,1,which='LA')
    w=w[0]
    V=V[:,0]
    thth_temp=np.zeros((thth_size,thth_size),dtype=complex)
    model_E=list()
    ## Loop over all dishes
    for d in range(n_dish):
        ## Build Model TH-TH for dish
        thth_temp*=0
        thth_temp[thth_size//2,:]=np.conjugate(V[d*thth_size:(d+1)*thth_size])*np.sqrt(w)
        recov_E=rev_map(thth_temp,tau,fd,eta,edges_red,isdspec = False)
        ## Map back to frequency/time space
        model_E_temp=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec2_list[0].shape[0],:dspec2_list[0].shape[1]]
        model_E_temp*=(dspec2_list[0].shape[0]*dspec2_list[0].shape[1]/4)
        model_E.append(model_E_temp)
    ##Progress Report
    print("Chunk %s-%s success" %(idx_f,idx_t),flush=True)
    return(model_E,idx_f,idx_t)

def single_chunk_retrieval(params):
    '''
    Performs phase retrieval on a single time/frequency chunk.
    Designed for use in parallel phase retreival code

    Arguments
    params -- tuple of relevant parameters
    '''
    
    ## Read parameters
    dspec2,edges,time2,freq2,eta,idx_t,idx_f,npad = params

    ## Verify unit compatability
    time2 = unit_checks(time2,'time2',u.s)
    freq2 = unit_checks(freq2,'freq2',u.MHz)
    eta = unit_checks(eta,'eta',u.s**3)
    edges = unit_checks(edges,'edges',u.mHz)

    ## Progress Reporting
    print("Starting Chunk %s-%s" %(idx_f,idx_t),flush=True)

    ## Determine fd and tau coordinates of Conjugate Spectrum
    fd = fft_axis(time2, u.mHz, npad)
    tau = fft_axis(freq2, u.us, npad)

    ## Pad dynamic spectrum to help with peiodicity problem
    dspec_pad = np.pad(dspec2,
                   ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
                   mode='constant',
                   constant_values=dspec2.mean())

    ## Compute Conjugate Spectrum
    CS = np.fft.fft2(dspec_pad)
    CS = np.fft.fftshift(CS)

    ## Try phase retrieval on chunk
    try:
        ## Calculate Reduced TH-TH and largest eigenvalue/vector pair
        thth_red, thth2_red, recov, model, edges_red,w,V = modeler(CS, tau, fd, eta, edges)

        ## Build model TH-TH for wavefield
        ththE_red=thth_red*0
        ththE_red[ththE_red.shape[0]//2,:]=np.conjugate(V)*np.sqrt(w)
        ##Map back to time/frequency space
        recov_E=rev_map(ththE_red,tau,fd,eta,edges_red,isdspec = False)
        model_E=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec2.shape[0],:dspec2.shape[1]]
        model_E*=(dspec2.shape[0]*dspec2.shape[1]/4)
        ## Progress Reporting
        print("Chunk %s-%s success" %(idx_f,idx_t),flush=True)
    except Exception as e:
        ## If chunk cannot be recovered print Error and return zero array (Prevents failure of a single chunk from ending phase retrieval)
        print(e,flush=True)
        model_E=np.zeros(dspec2.shape,dtype=complex)
    return(model_E,idx_f,idx_t)

def mask_func(w):
    """ Mask function used to weight chunks for mosaic function

    Arguments:
    w -- Integer length of array to be masked
    """
    x=np.linspace(0,w-1,w)
    return(np.sin((np.pi/2)*x/w)**2)

def mosaic(chunks):
    """ Combine recovered wavefield chunks into a single composite wavefield by correcting for random phase rotation and stacking

    Arguments:
    chunks -- Numpy Array of recovered chunks in the form [Chunk # in Freq, Chunk # in Time, Freq within chunk, Time within chunk]
    """
    ## Determine chunks sizes and number in time and freq
    nct=chunks.shape[1]
    ncf=chunks.shape[0]
    cwf=chunks.shape[2]
    cwt=chunks.shape[3]

    ## Prepare E_recov array (output wavefield)
    E_recov=np.zeros(((ncf-1)*(cwf//2)+cwf,(nct-1)*(cwt//2)+cwt),dtype=complex)

    ##Loop over all chunks
    for cf in range(ncf):
        for ct in range(nct):
            ## Select new chunk 
            chunk_new=chunks[cf,ct,:,:]

            ## Find overlap with current wavefield
            chunk_old=E_recov[cf*cwf//2:cf*cwf//2+cwf,ct*cwt//2:ct*cwt//2+cwt]
            mask=np.ones(chunk_new.shape)

            ##Determine Mask for new chunk (chunks will have higher weights towards their centre)
            if cf>0:
                ## All chunks but the first in frequency overlap for the first half in frequency
                mask[:cwf//2,:]*=mask_func(cwf//2)[:,np.newaxis]
            if cf<ncf-1:
                ## All chunks but the last in frequency overlap for the second half in frequency
                mask[cwf//2:,:]*=1-mask_func(cwf//2)[:,np.newaxis]
            if ct>0:
                ## All chunks but the first in time overlap for the first half in time
                mask[:,:cwt//2]*=mask_func(cwt//2)
            if ct<nct-1:
                ## All chunks but the last in time overlap for the second half in time
                mask[:,cwt//2:]*=1-mask_func(cwt//2)
            ##Average phase difference between new chunk and existing wavefield
            rot=np.angle((chunk_old*np.conjugate(chunk_new)*mask).mean())
            ## Add masked and roated new chunk to wavefield
            E_recov[cf*cwf//2:cf*cwf//2+cwf,ct*cwt//2:ct*cwt//2+cwt]+=chunk_new*mask*np.exp(1j*rot)
    return(E_recov)

def two_curve_map(CS, tau, fd, eta1, edges1,eta2,edges2):
    """Map from Secondary Spectrum to theta-theta space allowing for arclets with different curvature

    Arguments:
    CS -- Conjugate Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta1-- curvature of the main arc with the units of tau and fd
    edges1 -- 1d numpy array with the edges of the theta bins along the main arc
    eta2-- curvature of the arclets with the units of tau and fd
    edges2 -- 1d numpy array with the edges of the theta bins along the arclets
    """
    tau = unit_checks(tau,'tau',u.us)
    fd = unit_checks(fd,'fd',u.mHz)
    eta1 = unit_checks(eta1,'eta1',u.s**3)
    edges1 = unit_checks(edges1,'edges1',u.mHz)
    eta2 = unit_checks(eta2,'eta2',u.s**3)
    edges2 = unit_checks(edges2,'edges2',u.mHz)   

    # Find bin centers
    th_cents1 = (edges1[1:] + edges1[:-1]) / 2
    th_cents2 = (edges2[1:] + edges2[:-1]) / 2

    # Calculate theta1 and th2 arrays
    th1 = np.ones((th_cents2.shape[0], th_cents1.shape[0])) * th_cents1
    th2 = np.ones((th_cents2.shape[0], th_cents1.shape[0])) * th_cents2[:,np.newaxis]

    # tau and fd step sizes
    dtau = np.diff(tau).mean()
    dfd = np.diff(fd).mean()

    # Find bin in CS space that each point maps back to
    tau_inv = (((eta1 * th1**2 - eta2*th2**2)
                - tau[1] + dtau/2)//dtau).astype(int)
    fd_inv = (((th1 - th2) - fd[1] + dfd/2)//dfd).astype(int)

    # Define thth
    thth = np.zeros(tau_inv.shape, dtype=complex)

    # Only fill thth points that are within the CS
    pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]-1) * (fd_inv < fd.shape[0]-1)
    thth[pnts] = CS[tau_inv[pnts], fd_inv[pnts]]

    # Preserve flux (int
    thth *= np.sqrt(np.abs(2*eta1*th1-2*eta2*th2)).value

    th2_max=np.sqrt(tau.max()/eta2)
    th1_max=np.sqrt(tau.max()/eta1)
    th_cents1 = (edges1[1:] + edges1[:-1]) / 2
    th_cents2 = (edges2[1:] + edges2[:-1]) / 2
    pnts_1=np.abs(th_cents1)<th1_max
    pnts_2=np.abs(th_cents2)<th2_max
    edges_red1=np.zeros(pnts_1[pnts_1].shape[0]+1)*edges1.unit
    edges_red1[:-1]=edges1[:-1][pnts_1]
    edges_red1[-1]=edges1[1:][pnts_1].max()
    edges_red2=np.zeros(pnts_2[pnts_2].shape[0]+1)*edges2.unit
    edges_red2[:-1]=edges2[:-1][pnts_2]
    edges_red2[-1]=edges2[1:][pnts_2].max()
    thth_red=thth[pnts_2,:][:,pnts_1]
    th_cents1 = (edges_red1[1:] + edges_red1[:-1]) / 2
    th_cents2 = (edges_red2[1:] + edges_red2[:-1]) / 2
    return(thth_red,edges_red1,edges_red2)


def unit_checks(var,name,desired):
    """Checks for unit compatibility (Used internally to help provide useful errors)
    
    Arguments:
    var -- Variable whose units are to be checked
    name -- String name of variable for displaying errors
    desired -- Astropy Unit thatvar should have
    
    """
    var*=u.dimensionless_unscaled
    ## Check if var has units
    if u.dimensionless_unscaled.is_equivalent(var.unit):
        ## If there are no units assign desired units and tell user
        var*=desired
        print(f'{name} missing units. Assuming {desired}.')
    elif desired.is_equivalent(var.unit):
        ## If var has correct units (or equivalent)
        var=var.to(desired)
    else:
        ## If var has incompatible units return error
        raise u.UnitConversionError(f'{name} units ({var.unit}) not equivalent to {desired}')
    return(var)

def min_edges(fd_lim,fd,tau,eta,factor=2):
    """Calculates minimum size of edges array such that the Conjugate Spectrum is oversampled by at least some margin at all points
    Arguments:
    fd_lim -- Largest value of fd to search out to along main arc
    fd -- Array of fD values in Conjugate Spcetrum
    tau --  Array of tau values in Conjugate Spectrum
    eta -- The curvature to calculate edges for (When doing a curvature search this should be the largest curvature to search over)
    factor -- Over sample factor 
    """

    fd_lim = unit_checks(fd_lim,'fD Limit',fd.unit)
    ## Calculate largest fd step such that adjacent points at the top of the arc are within 1/factor tau steps
    dtau_lim = (tau[1]-tau[0])/factor
    dtau_lim /= 2*eta*fd_lim
    ## Calculate largest fd step such that points at the bottom of the arc are within 1/factor fd steps
    dfd_lim = (fd[1]-fd[0])/factor
    ## Cacululate number of points needed to cover -fd_lim to fd_lim with smaller step size limit
    npoints = (2*fd_lim)//(min(dfd_lim,dtau_lim.to(u.mHz)))
    ## Ensure even number of edges points
    npoints += np.mod(npoints,2)
    return(np.linspace(-fd_lim,fd_lim,int(npoints)))
