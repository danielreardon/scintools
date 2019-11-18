import numpy as np
import astropy.units as u
from scipy.sparse.linalg import eigsh

def chi_par(x,A,x0,C):
    """Parabola for fitting to chisq curve."""
    return(A*(x-x0)**2+C)

def thth_map(SS, tau, fd, eta, edges):
    """Map from Secondary Spectrum to theta-theta space

    Arguments:
    SS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins (symmetric about 0)
    """

    ##Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    ##Calculate theta1 and th2 arrays
    th1 = np.ones((th_cents.shape[0], th_cents.shape[0])) * th_cents
    th2 = th1.T
    
    ##tau and fd step sizes
    dtau=np.diff(tau).mean()
    dfd=np.diff(fd).mean()
    
    ##Find bin in SS space that each point maps back to
    tau_inv = (((eta * (th1**2 - th2**2))*u.mHz**2 -tau[0] +dtau/2)//dtau).astype(int)        
    fd_inv = (((th1 - th2)*u.mHz - fd[0] + dfd/2)//dfd).astype(int)
    
    ##Define thth
    thth = np.zeros(tau_inv.shape,dtype=complex)

    ##Only fill thth points that are within the SS
    pnts = (tau_inv>0) * (tau_inv < tau.shape[0]) * (fd_inv < fd.shape[0])
    thth[pnts] = SS[tau_inv[pnts], fd_inv[pnts]]

    ##Preserve flux
    thth*=np.abs(2*eta*(th2-th1)).value

    ##Force Hermetian
    thth-=np.tril(thth)
    thth+=np.conjugate(np.triu(thth).T)
    thth-=np.diag(np.diag(thth))
    thth-=np.diag(np.diag(thth[::-1,:]))[::-1,:]
    thth=np.nan_to_num(thth)
    return (thth)

def thth_redmap(SS, tau, fd, eta, edges):
    """Map from Secondary Spectrum to theta-theta space for the largest possible filled in sqaure within edges

    Arguments:
    SS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins (symmetric about 0)
    """

    ##Find full thth
    thth=thth_map(SS, tau, fd, eta, edges)

    ##Find region that is fully within SS
    th_cents=(edges[1:]+edges[:-1])/2
    th_cents-=th_cents[np.abs(th_cents)==np.abs(th_cents).min()]
    th_pnts=((th_cents**2)*eta.value<np.abs(tau.max().value)) * (np.abs(th_cents)<np.abs(fd.max()).value/2)
    thth_red=thth[th_pnts,:][:,th_pnts]
    edges_red=th_cents[th_pnts]
    edges_red=(edges_red[:-1]+edges_red[1:])/2
    edges_red=np.concatenate((np.array([edges_red[0]-np.diff(edges_red).mean()]),
                                edges_red,
                                np.array([edges_red[-1]+np.diff(edges_red).mean()])))
    return(thth_red,edges_red)

def rev_map(thth,tau,fd,eta,edges):
    """Map back from theta-theta space to SS space

    Arguments:
    thth -- 2d theta-theta spectrum
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins (symmetric about 0)
    """

    ##Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]

    ##Place each SS point within a bin in thth space
    th1 = ((tau[:,np.newaxis]/eta + fd[np.newaxis,:]**2)/(2*fd[np.newaxis,:])).value
    th2 = th1 - fd[np.newaxis,:].value
    th1 -= edges[0] - np.diff(edges).mean()/2
    th1/=np.diff(edges).mean()
    th2 -= edges[0] - np.diff(edges).mean()/2
    th2 /= np.diff(edges).mean()
    th1 = np.floor(th1).astype(int)
    th2 = np.floor(th2).astype(int)
    ##Only fill points that map into given thth range
    pnts = (th1 >= 0) * (th2>=0) * (th1 < edges.shape[0]-1)  * (th2 < edges.shape[0]-1)
    ##Define SS array
    recov=np.zeros((tau.shape[0],fd.shape[0]),dtype='complex')
    ##Fill and normalize SS
    th_dif=th_cents[:,np.newaxis]-th_cents[np.newaxis,:]
    recov[pnts] = thth[th2[pnts],th1[pnts]]/np.abs(2*eta*th_dif[th2[pnts],th1[pnts]]).value
    recov = np.nan_to_num(recov)
    return(recov)

def chisq_calc(SS, tau, fd, eta, edges,mask,N):
    """Calculate chisq of model SS using a 1d screen of given curvature

    Arguments:
    SS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins (symmetric about 0)
    mask -- boolian array showing with points in SS to fit
    N -- Variance of each point in SS
    """

    ## Find reduced thth
    thth_red,edges_red=thth_redmap(SS, tau, fd, eta, edges)
    w,V=eigsh(thth_red,1)
    w=w[0]
    V=V[:,0]
    ##Use larges eigenvector/value as model
    thth2_red=np.outer(V,np.conjugate(V))
    thth2_red*=np.abs(w)
    thth2_red[thth_red==0]=0
    ##Invert model thth
    SS_rev=rev_map(thth2_red,tau,fd,eta,edges_red)
    ##Compare to data
    chisq=np.sum(((np.abs(SS_rev-SS)**2)/N)[mask])
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

def curve_dspec(dspec,time,freq,eta_low,eta_high,edges=None,var=None,mask=None,verbose=False):
    """Calculate chisq of model SS using a 1d screen of given curvature

    Arguments:
    SS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins (symmetric about 0)
    mask -- boolian array showing with points in SS to fit
    N -- Variance of each point in SS
    """

    SS=np.fft.fftshift(np.fft.fft2(dspec)/np.sqrt(dspec.shape[0]*dspec.shape[1]))
    fd=np.fft.fftshift(np.fft.fftfreq(time.shape[0],time[1]-time[0]).to(u.mHz))*u.mHz
    tau=np.fft.fftshift(np.fft.fftfreq(freq.shape[0],freq[1]-freq[0]).to(u.us))*u.us
    var = (np.abs(SS[tau>5*tau.max()/6,:][:,np.abs(fd)>5*fd.max()/6])**2).mean() if var is None else var
    edges=np.linspace(-(fd.value.max()/2),fd.value.max()/2,2*max(SS.shape)) if edges is None else edges
    etas=np.linspace(eta_low.value,eta_high.value,1000)*u.us/u.mHz**2
    chisq=np.zeros(etas.shape[0])
    for i in range(etas.shape[0]):
        eta=etas[i]
        ## Find reduced thth
        thth_red,edges_red=thth_redmap(SS, tau, fd, eta, edges)
        w,V=eigsh(thth_red,1)
        w=w[0]
        V=V[:,0]
        ##Use larges eigenvector/value as model
        thth2_red=np.outer(V,np.conjugate(V))
        thth2_red*=np.abs(w)
        thth2_red[thth_red==0]=0
        ##Invert model thth
        SS_rev=rev_map(thth2_red,tau,fd,eta,edges_red)
        ##Compare to data
        dspec_mod=np.fft.ifft2(np.fft.ifft2(SS_rev)*np.sqrt(dspec.shape[0]*dspec.shape[1])).real
        if not mask is None:
            chisq[i]=np.sum(((np.abs(dspec-2*dspec_mod)**2)/var)[mask])
        else:
            chisq[i]=np.sum(((np.abs(dspec-2*dspec_mod)**2)/var))
        if verbose:
            if np.mod(i+1,etas.shape[0]//10)==0:
                print('%s Complete' %(i+1))
    return(etas,chisq)

    def curve_sspec(dspec,time,freq,eta_low,eta_high,coher=False,edges=None,var=None,mask=None,verbose=False):
    """Calculate chisq of model SS using a 1d screen of given curvature

    Arguments:
    SS -- Secondary Spectrum in [tau,fd] order with (0,0) in center
    tau -- Time lags in ascending order
    fd -- doppler frequency in ascending order
    eta -- curvature with the units of tau and fd
    edges -- 1d numpy array with the edges of the theta bins (symmetric about 0)
    mask -- boolian array showing with points in SS to fit
    N -- Variance of each point in SS
    """

    SS=np.fft.fftshift(np.fft.fft2(dspec)/np.sqrt(dspec.shape[0]*dspec.shape[1]))
    if not coher:
        SS=np.abs(SS)**2
    fd=np.fft.fftshift(np.fft.fftfreq(time.shape[0],time[1]-time[0]).to(u.mHz))*u.mHz
    tau=np.fft.fftshift(np.fft.fftfreq(freq.shape[0],freq[1]-freq[0]).to(u.us))*u.us
    if not coher:
        var = (SS[tau>5*tau.max()/6,:][:,np.abs(fd)>5*fd.max()/6]).mean() if var is None else var
    else:
        var = (np.abs(SS[tau>5*tau.max()/6,:][:,np.abs(fd)>5*fd.max()/6])**2).mean() if var is None else var
    edges=np.linspace(-(fd.value.max()/2),fd.value.max()/2,2*max(SS.shape)) if edges is None else edges
    etas=np.linspace(eta_low.value,eta_high.value,1000)*u.us/u.mHz**2
    chisq=np.zeros(etas.shape[0])
    mask = (tau>0)[:,np.newaxis]*(np.abs(fd)>0) if mask is None else mask
    for i in range(etas.shape[0]):
        chisq[i]=chisq_calc(SS, tau, fd, eta, edges,mask,var)
    return(etas,chisq)