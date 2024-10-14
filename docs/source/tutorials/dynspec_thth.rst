.. _dynspec_thth:
***********************************
Theta-Theta with the Dynspec Class
***********************************

This tutorial describes how to use the Dynspec class to perform theta-theta analysis on simulated data
to recover a curvature and perform phase retrieval.

The code used in this example can be downloaded from:

:Python script:
    :jupyter-download-script:`dynspec_thth.py <dynspec_thth>`
:Jupyter notebook:
    :jupyter-download-notebook:`dynspec_thth.ipynb <dynspec_thth>`


Imports
=======

.. jupyter-execute::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    import scintools.ththmod as thth
    from matplotlib.colors import LogNorm,SymLogNorm
    from scipy.optimize import curve_fit, minimize
    from scipy.sparse.linalg import eigsh
    from scintools.dynspec import BasicDyn, Dynspec

Load Data
=========

For this example we use the the simulated data found in
scintools/examples/data/ththsims. This includes the wavefield for a
single one dimensional screen with no noise that we then convert into a
dynamic spectrum and add some simple noise to.

.. jupyter-execute::

    ##Load Sample Data
    arch = np.load("../scintools/examples/data/ththsims/Sample_Data.npz")
    time = arch["t_s"] * u.s
    freq = arch["f_MHz"] * u.MHz
    wf = arch["Espec"]
    ##Create noisy dynamic spectrum
    dspec = np.abs(wf) ** 2 + np.random.normal(0, 20, wf.shape)
    dspec /= dspec.mean()

A quick visualization of the data.

.. jupyter-execute::

    fd=thth.fft_axis(time,u.mHz)
    tau=thth.fft_axis(freq,u.us)
    ##Sample spectra from portion of data
    plt.figure(figsize=(8, 10))
    plt.subplot(211)
    plt.imshow(
        dspec[:, :],
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(time.to(u.min), freq[:]),
    )
    SS = np.fft.fftshift(np.abs(np.fft.fft2(dspec[:, :])) ** 2)
    plt.xlabel("Time (min)")
    plt.ylabel("Freq (MHz)")
    plt.title("Dynamic Spectrum")
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(
        SS,
        norm=LogNorm(vmax=1e8, vmin=1e4),
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(fd, tau),
    )
    plt.ylim((0, 3))
    plt.colorbar()
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")
    plt.title("Secondary Spectrum")

Running :math:`\theta-\theta` with Dynspec
==========================================

We begin by loading the data into the Dynspec class using Basicdyn

.. jupyter-execute::

    ## build a BasicDyn object using the arrays imported earlier
    bDyne = BasicDyn(
        name='Sample Data',
        header=["Sample Data"],
        times=time.value,
        freqs=freq.value,
        dyn=dspec,
        nsub=time.shape[0],
        nchan=freq.shape[0],
        dt=(time[1] - time[0]).value,
        df=(freq[1] - freq[0]).value,
    )
    
    ## Convert into a Dynspec object
    dyn = Dynspec(dyn=bDyne, process=False)

Determine :math:`\theta-\theta` parameters
------------------------------------------

We can run :math:`\theta-\theta` with a minimum number of arguments to
get a sense of the data. In particular we note that the apexes of the
most distant arclets above are only at :math:`f_D\approx0.3~\rm{mHz}`.

.. jupyter-execute::

    dyn.prep_thetatheta(verbose=False,edges_lim=.3)

This run, which uses the whole observation in a single chunk, uses the
Hough transform method to get a reasonable range of curvatures to search
over (:math:`\eta\approx 66~\rm{s}^3` to
:math:`\eta\approx108~\rm{s}^3`), but wants an extremely high resolution
in the :math:`\theta-\theta` matrix with 10402 points in edges. Running
this full sized matrix is likely to be memory intensive and slow.
However, :math:`\theta-\theta` typically performs better on smaller
chunks anyway, so we consider shrinking the chunks to use only 128
frequency channels each.

.. jupyter-execute::

    dyn.prep_thetatheta(verbose=True,cwf=128,edges_lim=.3)
    dyn.thetatheta_single()

We can see that the smaller chunk size has greatly reduced the size of
the :math:`\theta-\theta` matrix and allows for us to begin looking at
the results of a single chunk. However, we can see that in this case the
asymmetry of the arc has resulted in a bright inner feature on the left
arm of the parabola that has skewed the curvature high. We will have to
shift our search range down. Since the current curvature (chosen to be
the center of the search range since no peak was found) puts points on
the arc approximately twice as high as the data appears to need, we
lower the bottom of the search region to :math:`30~\rm{s}^3` and top to
:math:`60~\rm{s}^3`

.. jupyter-execute::

    dyn.prep_thetatheta(verbose=True,cwf=128,edges_lim=.3,eta_min=30*u.s**3,eta_max=60*u.s**3)
    dyn.thetatheta_single()

This new seach region contains a nice peak whose curvature produces a
reasonable model for a portion of the dynamic spectrum. The lower
maximum curvature also reduces the number of points needed in the
:math:`\theta-\theta` map to ensure oversampling which will improve our
run times. We can further shrink our search region to focus on the peak
while also reducing the channels per chunk since only about half the
current chunk appears to be fully modelled by :math:`\theta-\theta`.

.. jupyter-execute::

    dyn.prep_thetatheta(verbose=True,cwf=64,edges_lim=.3,eta_min=30*u.s**3,eta_max=50*u.s**3)
    dyn.thetatheta_single()

Curvature Fitting
-----------------

Now that we have found reasonable values for our :math:`\theta-\theta`
parameters, we can attempt a curvature seach over every chunk.

.. jupyter-execute::

    dyn.fit_thetatheta(verbose=False,plot=True)

Parallel Curvature Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the curvature search on each chunk is independant, scintools
allows for parallelization using worker pools. This example uses Pool
from the multiprocessing package, but MPIPool from the mpipool package
also works.

.. jupyter-execute::

    from multiprocessing import Pool

.. jupyter-execute::

    with Pool(4) as pool:
        dyn.fit_thetatheta(verbose=False,plot=True,pool=pool)

Phase Retrieval
---------------

Using the best fit global curvature, we can now perform phase retrieval
on a set of overlapping chunks and stacking them using the moasaic
method as described in [Baker2022]\_.

.. jupyter-execute::

    dyn.calc_wavefield()

We can visualize the resulting conjugate wavefield to get a sense of how
well the retrieval has functioned. Note that when calculating
:math:`f_D` and :math:`\tau` we cut down the size of the respective axes
using the size of the recovered wavefield. While not important in this
example, if the chunks don’t line up perfectly with the end of the
dynamic spectrum along an axis, the uncovered region is not modeled in
the resulting wavefield.

.. jupyter-execute::

    fd=thth.fft_axis(dyn.times[:dyn.wavefield.shape[1]]*u.s,u.mHz)
    tau=thth.fft_axis(dyn.freqs[:dyn.wavefield.shape[0]]*u.MHz,u.us)
    CWF=np.fft.fftshift(np.fft.fft2(dyn.wavefield))
    SWF=np.abs(CWF)**2
    plt.figure()
    plt.imshow(SWF,origin='lower',aspect='auto',extent=thth.ext_find(fd,tau),norm=LogNorm(vmin=np.percentile(SWF,75)))
    plt.ylim(-1,tau.max().value)
    plt.xlabel(r'$f_D~\left(\rm{mHz}\right)$')
    plt.ylabel(r'$\tau~\left(\mu\rm{s}\right)$')

The resuling conjugate wavefield has a nice sharp arc with a collection
of clear idividual images. We can improve the result further by
employing the Gerchberg-Saxton algorithm to force the constraints that
the wavefield squared is the dynamic spectrum and that there should bu
no images with :math:`\tau<0~\mu\rm{s}`

.. jupyter-execute::

    dyn.calc_wavefield(gs=True)
    CWF=np.fft.fftshift(np.fft.fft2(dyn.wavefield))
    SWF=np.abs(CWF)**2
    plt.figure()
    plt.imshow(SWF,origin='lower',aspect='auto',extent=thth.ext_find(fd,tau),norm=LogNorm(vmin=np.percentile(SWF,75)))
    plt.ylim(-1,tau.max().value)
    plt.xlabel(r'$f_D~\left(\rm{mHz}\right)$')
    plt.ylabel(r'$\tau~\left(\mu\rm{s}\right)$')

This new version helps to sharpen the features further. Since we have
access to the underlying wavefield, we can compare it to our result. In
particular we can define a cross wavefield was :math:`C = \bar{W}W^{*}`
where :math:`\bar{W}` is the true wavefield and :math:`W` is our
recovered wavefield. For a perfect recovery, this should reproduce the
dynamic spectrum in the real part and have 0 imaginary part

.. jupyter-execute::

    C = wf[:dyn.wavefield.shape[0],:dyn.wavefield.shape[1]]*np.conjugate(dyn.wavefield)
    grid=plt.GridSpec(nrows=1,ncols=17)
    plt.figure()
    plt.subplot(grid[0,:8])
    plt.imshow(np.real(C),origin='lower',aspect='auto',
               extent=thth.ext_find(time[:dyn.wavefield.shape[0]].to(u.min),freq[:dyn.wavefield.shape[1]]),
               cmap='bwr',vmin=-30,vmax=30)
    plt.ylabel(r'$\nu~\left(\rm{MHz}\right($')
    plt.xlabel(r'$t~\left(\rm{min}\right($')
    plt.title('Real')
    ax=plt.subplot(grid[0,8:16])
    im=plt.imshow(np.imag(C),origin='lower',aspect='auto',
               extent=thth.ext_find(time[:dyn.wavefield.shape[0]].to(u.min),freq[:dyn.wavefield.shape[1]]),
               cmap='bwr',vmin=-30,vmax=30)
    plt.yticks([])
    plt.xlabel(r'$t~\left(\rm{min}\right($')
    plt.title('Imaginary')
    plt.colorbar(cax = plt.subplot(grid[0,-1]))
    plt.tight_layout()

In this case we can see that there is some phase difference between the
true wavefield and the recovered wavefield. This is expected since
:math:`\theta-\theta` does not yield a unique solution since any global
phase rotation of the wavefield is undectable in the dynamic spectrum.
This means we are generically off by some unknown offset
:math:`\phi_{\rm{offset}}`

.. jupyter-execute::

    phi = np.angle(C.mean())
    C2 = C*np.exp(-1j*phi)
    plt.figure()
    plt.subplot(grid[0,:8])
    plt.imshow(np.real(C2),origin='lower',aspect='auto',
               extent=thth.ext_find(time[:dyn.wavefield.shape[0]].to(u.min),freq[:dyn.wavefield.shape[1]]),
               cmap='bwr',vmin=-30,vmax=30)
    plt.ylabel(r'$\nu~\left(\rm{MHz}\right($')
    plt.xlabel(r'$t~\left(\rm{min}\right($')
    plt.title('Real')
    ax=plt.subplot(grid[0,8:16])
    im=plt.imshow(np.imag(C2),origin='lower',aspect='auto',
               extent=thth.ext_find(time[:dyn.wavefield.shape[0]].to(u.min),freq[:dyn.wavefield.shape[1]]),
               cmap='bwr',vmin=-30,vmax=30)
    plt.yticks([])
    plt.xlabel(r'$t~\left(\rm{min}\right($')
    plt.title('Imaginary')
    plt.colorbar(cax = plt.subplot(grid[0,-1]))
    plt.tight_layout()

In gereral, this phase offset is unimportant, but if you want to compare
two wavefields (for example from different stations or polarizations)
they will have different value of :math:`\phi_{\rm{offset}}` that will
need to be corrected for.

