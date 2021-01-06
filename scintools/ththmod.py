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
from scipy.optimize import curve_fit

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


def thth_map(SS, tau, fd, eta, edges,hermetian=True):
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


def thth_redmap(SS, tau, fd, eta, edges,hermetian=True):
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
    thth = thth_map(SS, tau, fd, eta, edges,hermetian)

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


def rev_map(thth, tau, fd, eta, edges,isdspec=True):
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
                         weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).real)[0] +\
            np.histogram2d(np.ravel(fd_map),
                         np.ravel(tau_map),
                         bins=(fd_edges,tau_edges),
                         weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).imag)[0]*1j
    norm=np.histogram2d(np.ravel(fd_map),
                         np.ravel(tau_map),
                         bins=(fd_edges,tau_edges))[0]
    if isdspec:
        recov += np.histogram2d(np.ravel(-fd_map),
                            np.ravel(-tau_map),
                            bins=(fd_edges,tau_edges),
                            weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).real)[0] -\
                np.histogram2d(np.ravel(-fd_map),
                            np.ravel(-tau_map),
                            bins=(fd_edges,tau_edges),
                            weights=np.ravel(thth/np.sqrt(np.abs(2*eta*fd_map.T).value)).imag)[0]*1j
        norm+=np.histogram2d(np.ravel(-fd_map),
                            np.ravel(-tau_map),
                            bins=(fd_edges,tau_edges))[0] 
    recov/=norm
    recov=np.nan_to_num(recov)
    return(recov.T)

def modeler(SS, tau, fd, eta, edges,fd2=None,tau2=None):
    if fd2==None:
        fd2=fd
    if tau2==None:
        tau2=tau
    thth_red,edges_red=thth_redmap(SS, tau, fd, eta, edges)
    ##Find first eigenvector and value
    w,V=eigsh(thth_red,1,which='LA')
    w=w[0]
    V=V[:,0]
    ##Use larges eigenvector/value as model
    thth2_red=np.outer(V,np.conjugate(V))
    thth2_red*=np.abs(w)
    ##Map back to SS for high
#    thth2_red[thth_red==0]=0
    recov=rev_map(thth2_red,tau2,fd2,eta,edges_red)
    model=np.fft.ifft2(np.fft.ifftshift(recov)).real
    return(thth_red,thth2_red,recov,model,edges_red,w,V)

def chisq_calc(dspec,SS, tau, fd, eta, edges,mask,N,fd2=None,tau2=None):
    model=modeler(SS, tau, fd, eta, edges,fd2,tau2)[3][:dspec.shape[0],:dspec.shape[1]]
    chisq=np.sum((model-dspec)[mask]**2)/N
    return(chisq)

def Eval_calc(SS, tau, fd, eta, edges):
    thth_red,edges_red=thth_redmap(SS, tau, fd, eta, edges)
    ##Find first eigenvector and value
    v0=thth_red[thth_red.shape[0]//2,:]
    v0/=np.sqrt((np.abs(v0)**2).sum())
    w,V=eigsh(thth_red,1,v0=v0,which='LA')
    return(np.abs(w[0]))

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
    '''
    
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
    Curvature Search for a single chunk of a dynamic spectrum.
    

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
    """
    dspec2,freq2,time2,eta_low,eta_high,edges,name,plot,fw,npad=params

    etas = np.linspace(eta_low, eta_high, 100) * u.us / u.mHz**2


    fd = fft_axis(time2, u.mHz, npad)
    tau = fft_axis(freq2, u.us, npad)

    dspec_pad = np.pad(dspec2,
                   ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
                   mode='constant',
                   constant_values=dspec2.mean())

    SS = np.fft.fft2(dspec_pad)
    SS = np.fft.fftshift(SS)
    eigs = np.zeros(etas.shape)
    for i in range(eigs.shape[0]):
        try:
            eigs[i] = Eval_calc(SS, tau, fd, etas[i], edges)
        except:
            eigs[i]=np.nan
    try:
        etas=etas[np.isfinite(eigs)]
        eigs=eigs[np.isfinite(eigs)]

        etas_fit = etas[np.abs(etas - etas[eigs == eigs.max()]) < fw * etas[eigs == eigs.max()]]
        eigs_fit = eigs[np.abs(etas - etas[eigs == eigs.max()]) < fw * etas[eigs == eigs.max()]]

        C = eigs_fit.max()
        x0 = etas_fit[eigs_fit == C][0].value
        if x0 == etas_fit[0].value:
            A = (eigs_fit[-1] - C) / ((etas_fit[-1].value - x0)**2)
        else:
            A = (eigs_fit[0] - C) / ((etas_fit[0].value - x0)**2)
        popt, pcov = curve_fit(chi_par,
                                etas_fit.value,
                                eigs_fit,
                                p0=np.array([A, x0, C]))
        eta_fit = popt[1]*u.us/u.mHz**2
        eta_sig = np.sqrt((eigs_fit - chi_par(etas_fit.value, *popt)).std() / np.abs(popt[0]))*u.us/u.mHz**2
    except:
        popt=None
        eta_fit=np.nan
        eta_sig=np.nan
    try:
        if plot:
            PlotFunc(dspec2,time2,freq2,SS,fd,tau,edges,eta_fit,eta_sig,etas,eigs,etas_fit,popt)
            plt.savefig(name)
            plt.close()
    except:
        print('Plotting Error',flush=True)
    print('Chunk completed (eta = %s +- %s at %s)' %(eta_fit,eta_sig,freq2.mean()),flush=True)
    return(eta_fit,eta_sig,freq2.mean(),time2.mean(),eigs)

def PlotFunc(dspec,time,freq,SS,fd,tau,
            edges,eta_fit,eta_sig,etas,measure,etas_fit,fit_res,
            tau_lim=None,method='eigenvalue'):
    '''
    Plotting script to look at invidivual chunks

    Arguments
    dspec -- 2D numpy array containing the dynamic spectrum
    time -- 1D numpy array of the dynamic spectrum time bins (with units)
    freq -- 1D numpy array of the dynamic spectrum frequency channels (with units)
    SS -- 2D numpy array of the conjugate spectrum
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
    fd_lim=min(2*edges.max(),fd.max().value)
    if np.isnan(eta_fit):
        eta=etas.mean()
        thth_red, thth2_red, recov, model, edges_red,w,V = modeler(SS, tau, fd, etas.mean(), edges)
    else:
        eta=eta_fit
        thth_red, thth2_red, recov, model, edges_red,w,V = modeler(SS, tau, fd, eta_fit, edges)
    ththE_red=thth_red*0
    ththE_red[ththE_red.shape[0]//2,:]=np.conjugate(V)*np.sqrt(w)
    ##Map back to time/frequency space
    recov_E=rev_map(ththE_red,tau,fd,eta,edges_red,isdspec = False)
    model_E=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec.shape[0],:dspec.shape[1]]
    model_E*=(dspec.shape[0]*dspec.shape[1]/4)
    model_E[dspec>0]=np.sqrt(dspec[dspec>0])*np.exp(1j*np.angle(model_E[dspec>0]))
    model_E=np.pad(model_E,
                    (   (0,SS.shape[0]-model_E.shape[0]),
                        (0,SS.shape[1]-model_E.shape[1])),
                    mode='constant',
                    constant_values=0)
    recov_E=np.abs(np.fft.fftshift(np.fft.fft2(model_E)))**2
    model_E=model_E[:dspec.shape[0],:dspec.shape[1]]
    N_E=recov_E[:recov_E.shape[0]//4,:].mean()

    grid=plt.GridSpec(5,2)
    plt.figure(figsize=(8,20))
    plt.subplot(grid[0,0])
    plt.imshow(dspec,
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower',
            vmin=0,vmax=dspec.max())
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Data Dynamic Spectrum')
    plt.subplot(grid[0,1])
    plt.imshow(model[:dspec.shape[0],:dspec.shape[1]],
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower',
            vmin=0,vmax=dspec.max())
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Model Dynamic Spectrum')
    plt.subplot(grid[1,0])
    plt.imshow(np.abs(SS)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=ext_find(fd,tau),
            vmin=np.median(np.abs(SS)**2),vmax=np.abs(SS).max()**2)
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim))
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.title('Data Secondary Spectrum')
    plt.plot(fd,eta*(fd**2),'r',alpha=.7)
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.subplot(grid[1,1])
    plt.imshow(np.abs(recov)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=ext_find(fd,tau),
            vmin=np.median(np.abs(SS)**2),vmax=np.abs(SS).max()**2)
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim))
    plt.title('Model Secondary Spectrum')
    plt.subplot(grid[2,0])
    plt.imshow(np.abs(thth_red)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0],edges_red[-1],edges_red[0],edges_red[-1]],
            vmin=np.median(np.abs(thth_red)**2),vmax=np.abs(thth_red).max()**2)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Data $\theta-\theta$')
    plt.subplot(grid[2,1])
    plt.imshow(np.abs(thth2_red)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0],edges_red[-1],edges_red[0],edges_red[-1]],
            vmin=np.median(np.abs(thth_red)**2),vmax=np.abs(thth_red).max()**2)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Data $\theta-\theta$')
    plt.subplot(grid[3,:])
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
    plt.subplot(grid[4,0])
    plt.imshow(np.angle(model_E),
            cmap='twilight',
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower',
            vmin=-np.pi,vmax=np.pi)
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Recovered Phases')
    plt.subplot(grid[4,1])
    plt.imshow(recov_E,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=ext_find(fd,tau),
            vmin=N_E)
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim))
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.title('Recovered Wavefield')
    plt.colorbar()
    plt.tight_layout()

def VLBI_chunk_retrieval(params):
    '''
    Performs phase retrieval on a single time/frequency chunk using multiple dynamic spectra and visibilities.
    Designed for use in parallel phase retreival code

    Arguments
    params -- tuple of relevant parameters
    '''
    dspec2_list,edges,time2,freq2,eta,idx_t,idx_f,npad,n_dish = params
    print("Starting Chunk %s-%s" %(idx_f,idx_t),flush=True)
    fd = fft_axis(time2, u.mHz, npad)
    tau = fft_axis(freq2, u.us, npad)

    thth_red=list()
    for i in range(len(dspec2_list)):
        dspec_pad = np.pad(dspec2_list[i],
                    ((0, npad * dspec2_list[i].shape[0]), (0, npad * dspec2_list[i].shape[1])),
                    mode='constant',
                    constant_values=dspec2_list[i].mean())

        SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
        thth_single,edges_red=thth_redmap(SS,tau,fd,eta,edges)
        thth_red.append(thth_single)
    thth_size=thth_red[0].shape[0]
    thth_comp=np.zeros((thth_size*n_dish,thth_size*n_dish),dtype=complex)
    for d1 in range(n_dish):
        for d2 in range(n_dish-d1):
            idx=int(((n_dish*(n_dish+1))//2)-(((n_dish-d1)*(n_dish-d1+1))//2)+d2)
            thth_comp[d1*thth_size:(d1+1)*thth_size,(d1+d2)*thth_size:(d1+d2+1)*thth_size]=thth_red[idx]
            thth_comp[(d1+d2)*thth_size:(d1+d2+1)*thth_size,d1*thth_size:(d1+1)*thth_size]=np.conjugate(thth_red[idx].T)
    w,V=eigsh(thth_comp,1,which='LA')
    w=w[0]
    V=V[:,0]
    thth_temp=np.zeros((thth_size,thth_size),dtype=complex)
    model_E=list()
    for d in range(n_dish):
        thth_temp*=0
        thth_temp[thth_size//2,:]=np.conjugate(V[d*thth_size:(d+1)*thth_size])*np.sqrt(w)
        recov_E=rev_map(thth_temp,tau,fd,eta,edges_red,isdspec = False)
        model_E_temp=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec2_list[0].shape[0],:dspec2_list[0].shape[1]]
        model_E_temp*=(dspec2_list[0].shape[0]*dspec2_list[0].shape[1]/4)
        model_E.append(model_E_temp)
    print("Chunk %s-%s success" %(idx_f,idx_t),flush=True)
    return(model_E,idx_f,idx_t)

def single_chunk_retrieval(params):
    '''
    Performs phase retrieval on a single time/frequency chunk.
    Designed for use in parallel phase retreival code

    Arguments
    params -- tuple of relevant parameters
    '''
    dspec2,edges,time2,freq2,eta,idx_t,idx_f,npad = params
    print("Starting Chunk %s-%s" %(idx_f,idx_t),flush=True)
    fd = fft_axis(time2, u.mHz, npad)
    tau = fft_axis(freq2, u.us, npad)

    dspec_pad = np.pad(dspec2,
                   ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
                   mode='constant',
                   constant_values=dspec2.mean())

    SS = np.fft.fft2(dspec_pad)
    SS = np.fft.fftshift(SS)

    try:
        thth_red, thth2_red, recov, model, edges_red,w,V = modeler(SS, tau, fd, eta, edges)

        ththE_red=thth_red*0
        ththE_red[ththE_red.shape[0]//2,:]=np.conjugate(V)*np.sqrt(w)
        ##Map back to time/frequency space
        recov_E=rev_map(ththE_red,tau,fd,eta,edges_red,isdspec = False)
        model_E=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec2.shape[0],:dspec2.shape[1]]
        model_E*=(dspec2.shape[0]*dspec2.shape[1]/4)
        print("Chunk %s-%s success" %(idx_f,idx_t),flush=True)
    except Exception as e:
        print(e,flush=True)
        model_E=np.zeros(dspec2.shape,dtype=complex)
    return(model_E,idx_f,idx_t)

def mask_func(w):
    x=np.linspace(0,w-1,w)
    return(np.sin((np.pi/2)*x/w)**2)

def mosaic(chunks):
    nct=chunks.shape[1]
    ncf=chunks.shape[0]
    cwf=chunks.shape[2]
    cwt=chunks.shape[3]
    E_recov=np.zeros(((ncf-1)*(cwf//2)+cwf,(nct-1)*(cwt//2)+cwt),dtype=complex)

    for cf in range(ncf):
        for ct in range(nct):
            chunk_new=chunks[cf,ct,:,:]
            chunk_old=E_recov[cf*cwf//2:cf*cwf//2+cwf,ct*cwt//2:ct*cwt//2+cwt]
            mask=np.ones(chunk_new.shape)
            if cf>0:
                mask[:cwf//2,:]*=mask_func(cwf//2)[:,np.newaxis]
            if cf<ncf-1:
                mask[cwf//2:,:]*=1-mask_func(cwf//2)[:,np.newaxis]
            if ct>0:
                mask[:,:cwt//2]*=mask_func(cwt//2)
            if ct<nct-1:
                mask[:,cwt//2:]*=1-mask_func(cwt//2)
            rot=np.angle((chunk_old*np.conjugate(chunk_new)*mask).mean())
            E_recov[cf*cwf//2:cf*cwf//2+cwf,ct*cwt//2:ct*cwt//2+cwt]+=chunk_new*mask*np.exp(1j*rot)
    return(E_recov)

