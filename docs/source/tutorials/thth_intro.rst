.. _thth_intro:
***********************************
Basic Theta-Theta
***********************************

This tutorial describes how to use the Dynspec class to perform theta-theta analysis on simulated data
to recover a curvature and perform phase retrieval.

The code used in this example can be downloaded from:

:Python script:
    :jupyter-download-script:`thth_intro.py <thth_intro>`
:Jupyter notebook:
    :jupyter-download-notebook:`thth_intro.ipynb <thth_intro>`

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
    # dspec /= dspec.mean()

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

Intro to :math:`\theta-\theta`
==============================

The :math:`\theta-\theta` map works by remapping the arcs and inverted
arclets in the conjugate spectrum into a space where they form straight
lines. For a given curvature :math:`\eta`, this is achieved by
converting to the coordinates :math:`\theta_1` and :math:`theta_2` given
by:

:math:`\theta_1 = \left(\frac{\tau}{\eta} + f_D\right)/2`

and

:math:`\theta_2 = \left(\frac{\tau}{\eta} - f_D\right)/2`

Curvature Dependance
--------------------

To see the effect of curvature on the :math:`\theta-\theta` map, we
consider 3 values around the true value of
:math:`\eta\approx44~\mu\rm{s}~\rm{mHz}^{-2}` (or :math:`\rm{s}^3`) for
a small portion of the data. We first select the small portion of the
data and calculate the mean subtracted conjugate spectrum. We also pad
the spectrum to improve performance.

.. jupyter-execute::

    cwf = 64
    cwt = dspec.shape[1]
    
    dspec2 = np.copy(dspec[:cwf, :cwt])
    freq2 = freq[:cwf]
    time2 = time[:cwt]
    
    dspec2 -= dspec2.mean()
    
    npad = 3
    dspec_pad = np.pad(
        dspec2,
        ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
        mode="constant",
        constant_values=0,
    )
    
    CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
    SS = np.abs(CS) ** 2
    
    fd = thth.fft_axis(time2, u.mHz, npad)
    tau = thth.fft_axis(freq2, u.us, npad)

We now select sample curvatures around our true value and form the
corresponding :math:`\theta-\theta` matrices. We must also select a
resolution for our :math:`\theta-\theta` map using the edges array. This
array defines the edges of pixels in the :math:`\theta-\theta` matrix
and should be symmetric about 0 with an even number if points. For this
example we extend it out to :math:`.03~\rm{mHz}` to include the apexes
of the most distance arclets.

.. jupyter-execute::

    
    ## Choose sample curvatures
    etas = np.array([12.5, 44, 100]) * u.us / u.mHz**2
    
    edges = thth.min_edges(0.3 * u.mHz, fd, tau, etas.max(), 1)
    
    
    
    
    ## Create thth matrices for example curvatures
    thths = list()
    
    for i in range(etas.shape[0]):
        """
        The redmap function returns both the thth matrix and a truncated edges array to ensure the
        matrix never samples points from outside the conjugate spectrum.
        """
        thth_red, edges_red = thth.thth_redmap(CS, tau, fd, etas[i], edges)
        thths.append((thth_red, edges_red))

Plotting the :math:`\theta-\theta` matrices shows the straight
horrizontal and vertical line when the correct curvature is used. When
the curvature is too small, the resulting matrix is stretched along the
:math:`\theta_1=\theta_2` diagonal, while too large a curvature causes
stretching along the :math:`\theta_1=-\theta_2` diagonal.

.. jupyter-execute::

    
    vmax = np.array(
        [np.percentile(np.abs(thths[i][0]) ** 2, 99) for i in range(len(thths))]
    ).max()
    vmin = np.array(
        [np.percentile(np.abs(thths[i][0]) ** 2, 50) for i in range(len(thths))]
    ).min()
    
    plt.figure(figsize=(4, 12))
    for i in range(etas.shape[0]):
        plt.subplot(3, 1, i + 1)
        plt.imshow(
            np.abs(thths[i][0]) ** 2,
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax,
            ),
            extent=[
                thths[i][1][0].value,
                thths[i][1][-1].value,
                thths[i][1][0].value,
                thths[i][1][-1].value,
            ],
            origin="lower",
        )
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.title(r"$\eta$ =%s $s^3$" % etas[i].to_value(u.s**3))
        plt.colorbar()
    plt.tight_layout()
    
    ##Show secondary spectrum and sample curvatures
    plt.figure(figsize=(8, 8))
    plt.imshow(
        SS,
        norm=LogNorm(
            vmax=np.percentile(SS, 99),
            vmin=np.percentile(SS, 50),
        ),
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(fd, tau),
    )
    plt.ylim((0, 3))
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")
    plt.title("Secondary Spectrum")
    for i in range(3):
        plt.plot(
            fd, etas[i] * fd**2, label=r"$\eta$ = %s $s^3$" % etas[i].to_value(u.s**3)
        )
    plt.xlim((-0.5, 0.5))
    plt.legend()

Single Chunk Curvature Fitting
------------------------------

To begin curvature fitting, we first set a few parameters for the search
and test them on a single chunk of the dynamic spectrum:

cwf: The chunk width in frequency (number of channels)

cwt: The chunk width in time (number of sub integrations)

eta_low: The lowest curvature to search

eta_high: The highest curvature to search

edges_max: The most distant arclet apex to include in the model

nedge: The number of points in the edges array (must be even)

fw: The fractional width around the peak of the eigenvalue curve to fit
a parabola to

.. jupyter-execute::

    ##Sample Curvature Search and Plots for Small Chunk of Data
    ##Frequency width (channels)
    cwf = 64
    ##Time width (integrations)
    cwt = dspec.shape[1]
    
    ##Define range of curvatures to search (find a point clearly above/below the main arc)
    eta_low = 0.5 * u.us / (0.04 * u.mHz**2)
    eta_high = 4 * u.us / (0.04 * u.mHz**2)
    
    ## Detmine size/resolution of theta-theta
    edges_max = 0.4 * u.mHz
    nedge = 512
    
    ## Fitting width
    fw = 0.1

As before, we mean subtract and zero pad the dynamic spectrum before
forming the conjugate spectrum.

.. jupyter-execute::

    dspec2 = np.copy(dspec[:cwf, :cwt])
    freq2 = freq[:cwf]
    time2 = time[:cwt]
    
    mn = dspec2.mean()
    
    
    npad = 3
    dspec_pad = np.pad(
        dspec2-mn,
        ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
        mode="constant",
        constant_values=0,
    )
    
    ##Form SS and coordinate arrays
    CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
    fd = thth.fft_axis(time2, u.mHz, npad)
    tau = thth.fft_axis(freq2, u.us, npad)

We then set a grid of curvatures to seach for an calculate the maximum
eigenvalues of the corresponsing :math:`\theta-\theta` matrices.

.. jupyter-execute::

    
    ##Setup for chisq search
    etas = np.linspace(eta_low.value, eta_high.value, 100) * eta_low.unit
    eigs = np.zeros(etas.shape[0])
    edges = np.linspace(-edges_max, edges_max, nedge)
    
    ##Determine chisq for each curvature
    for i in range(etas.shape[0]):
        eta = etas[i]
        """
        Instead of doing the inverse mapping, and looking for a minimal chisquare, we look for the largest eigenvalue.
        This line is the only user facing difference between the two methods
        """
        eigs[i] = thth.Eval_calc(CS, tau, fd, eta, edges)
    
    plt.figure()
    plt.plot(etas,eigs)
    plt.xlabel(r'$\eta\left(\rm{s}^3\right)$')


Since there is a clear peak, we can fit a parabola to it and plot the
model from the resulting best fir curvature.

.. jupyter-execute::

    
    ##Fit for a parabola around the minimum
    e_min = etas[eigs == eigs.max()][0]
    etas_fit = etas[np.abs(etas - e_min) < fw * e_min]
    eigs_fit = eigs[np.abs(etas - e_min) < fw * e_min]
    C = eigs_fit.max()
    x0 = etas_fit[eigs_fit == C][0].value
    A = (eigs_fit[0] - C) / ((etas_fit[0].value - x0) ** 2)
    popt, pcov = curve_fit(thth.chi_par, etas_fit.value, eigs_fit, p0=np.array([A, x0, C]))
    eta_fit = popt[1] * etas.unit
    eta_sig = (
        np.sqrt(-(eigs_fit - thth.chi_par(etas_fit.value, *popt)).std() / popt[0])
        * etas.unit
    )
    
    thth.plot_func(
        dspec2,
        time2,
        freq2,
        CS,
        fd,
        tau,
        edges,
        eta_fit,
        eta_sig,
        etas,
        eigs,
        etas_fit,
        popt,
    )

Frequency Dependence of Curvature
---------------------------------

Now that we have parameters that work for a single chunk, we loop over
all chunks to get the frequency dependence of the curvature.

.. jupyter-execute::

    ##Number of frequency chunks across the observation
    ncf = dspec.shape[0] // cwf
    ##Number of time chunks across the observation
    nct = dspec.shape[1] // cwt
    
    
    ##Arrays for curvatures in each frequency chunk
    f0 = np.zeros((ncf, nct)) * u.MHz
    eta_evo = np.zeros((ncf, nct)) * u.us / u.mHz**2
    eta_evo_err = np.zeros((ncf, nct)) * u.us / u.mHz**2
    
    ##Loop over chunks
    for fc in range(ncf):
        for tc in range(nct):
            ##Define dspec and freq array for chunk
            dspec2 = np.copy(dspec[fc * cwf : (fc + 1) * cwf, tc * cwt : (tc + 1) * cwt])
            dspec2-=np.nanmean(dspec2)
            freq2 = freq[fc * cwf : (fc + 1) * cwf]
            time2 = time[tc * cwt : (tc + 1) * cwt]
            f0[fc, tc] = freq2.mean()
    
            ##Pad before forming dynamic spectrum
            npad = 3
            dspec_pad = np.pad(
                dspec2,
                ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
                mode="constant",
                constant_values=0,
            )
    
            ##Form SS and coordinate arrays
            CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
            fd = thth.fft_axis(time2, u.mHz, npad)
            tau = thth.fft_axis(freq2, u.us, npad)
    
            ##Setup for chisq search
            etas = np.linspace(eta_low.value, eta_high.value, 100) * eta_low.unit
            evals = np.zeros(etas.shape[0])
            edges = np.linspace(-edges_max, edges_max, nedge)
            ##Determine largest eigenvalue for each curvature
            for i in range(etas.shape[0]):
                eta = etas[i]
                evals[i] = thth.Eval_calc(CS, tau, fd, eta, edges)
    
            e_min = etas[evals == evals.max()][0]
            etas_fit = etas[np.abs(etas - e_min) < fw * e_min]
            evals_fit = evals[np.abs(etas - e_min) < fw * e_min]
            C = evals_fit.max()
            x0 = etas_fit[evals_fit == C][0].value
            A = (evals_fit[0] - C) / ((etas_fit[0].value - x0) ** 2)
            popt, pcov = curve_fit(
                thth.chi_par, etas_fit.value, evals_fit, p0=np.array([A, x0, C])
            )
            eta_fit = popt[1] * etas.unit
            eta_sig = (
                np.sqrt(-(evals_fit - thth.chi_par(etas_fit.value, *popt)).std() / popt[0])
                * etas.unit
            )
        
            eta_evo[fc] = eta_fit
            eta_evo_err[fc] = eta_sig

We then use a weighted sum to fit :math:`\eta=\frac{A}{\nu^2}`

.. jupyter-execute::

    def eta_func(f0, A):
        return A / (f0**2)
    
    fref = 1400 * u.MHz
    
    A = (
        np.nansum(eta_evo / (f0 * eta_evo_err) ** 2)
        / np.nansum(1 / ((f0**2) * eta_evo_err) ** 2)
    ).to(u.s**3 * u.MHz**2)
    A_err = np.sqrt(1 / np.nansum(2 / ((f0**2) * eta_evo_err) ** 2)).to(
        u.s**3 * u.MHz**2
    )
    
    etaLS_ref = A / fref**2
    errLS_ref = A_err / fref**2
    
    exp_fit = int(("%.0e" % etaLS_ref.value)[2:])
    exp_err = int(("%.0e" % errLS_ref.value)[2:])
    fmt = "{:.%se}" % (exp_fit - exp_err)
    fit_string = fmt.format(etaLS_ref.value)[: 2 + exp_fit - exp_err]
    err_string = "0%s" % fmt.format(10 ** (exp_fit) + errLS_ref.value)[1:]
    if err_string[exp_fit-exp_err+1]=='1':
        exp_err-=1
        fmt = "{:.%se}" % (exp_fit - exp_err)
        fit_string = fmt.format(etaLS_ref.value)[: 2 + exp_fit - exp_err]
        err_string = "0%s" % fmt.format(10 ** (exp_fit) + errLS_ref.value)[1:]
    
    plt.figure()
    plt.errorbar(
        np.ravel(f0.value),
        np.ravel(eta_evo.value),
        yerr=np.ravel(eta_evo_err.value),
        fmt=".",
    )
    plt.plot(
        f0.value,
        eta_func(f0.value, A),
        label=r"$\eta_{%s}$ = %s $\pm$ %s  $s^3$" % (fref.value, fit_string, err_string),
    )
    plt.title(r"Curvature Evolution")
    plt.xlabel("Freq (MHz)")
    plt.ylabel(r"$\eta$ ($s^3$)")
    plt.legend()

Phase Retrieval
---------------

Using the best fit global curvature, we can now perform phase retrieval
on a set of overlapping chunks. This would typically be done using the
same edges as for the curvature fitting. However, for illustrative
purposes we have shrunk the edges so that the most distant arclets are
not covered by the :math:`\theta-\theta` matrix

.. jupyter-execute::

    hwt = cwt // 2
    hwf = cwf // 2
    
    ncf = (dspec.shape[0] - hwf) // hwf
    nct = (dspec.shape[1] - hwt) // hwt
    
    ##Array for storing chunks
    chunks = np.zeros((ncf, nct, cwf, cwt), dtype=complex)
    edges = np.linspace(-.2,.2,512)*u.mHz
    for cf in range(ncf):
        for ct in range(nct):
            ##Select Chunk and determine curvature
            dspec2 = np.copy(dspec[cf * hwf : cf * hwf + cwf, ct * hwt : ct * hwt + cwt])
            dspec2 -= dspec2.mean()
            freq2 = freq[cf * hwf : cf * hwf + cwf]
            time2 = time[ct * hwt : ct * hwt + cwt]
            eta = A /(freq2.mean()**2)
    
            ##Pad
            dspec_pad = np.pad(
                dspec2,
                ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
                mode="constant",
                constant_values=dspec2.mean(),
            )
    
            CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
            fd = thth.fft_axis(time2, u.mHz, npad)
            tau = thth.fft_axis(freq2, u.us, npad)
    
            ##Create and decompose theta-theta
            thth_red, edges_red = thth.thth_redmap(CS, tau, fd, eta, edges)
            w, V = eigsh(thth_red, 1)
            w = w[0]
            V = V[:, 0]
    
            ##Construct 1D theta-theta
            thth2_red = thth_red * 0
            thth2_red[thth2_red.shape[0] // 2, :] = np.conjugate(V) * np.sqrt(w)
            ##Map back to time/frequency space
            recov = thth.rev_map(thth2_red, tau, fd, eta, edges_red, hermetian=False)
            model_E = np.fft.ifft2(np.fft.ifftshift(recov))[
                : dspec2.shape[0], : dspec2.shape[1]
            ]
            model_E *= dspec2.shape[0] * dspec2.shape[1] / 2
            model_E = model_E[: dspec2.shape[0], : dspec2.shape[1]]
            chunks[cf, ct, :, :] = model_E

Single chunks produce good results for the dynamic spectra in that chunk

.. jupyter-execute::

    ds_model = np.abs(chunks[0, 0, :, :]) ** 2
    plt.figure()
    plt.subplot(211)
    plt.imshow(
        ds_model / ds_model.mean(),
        vmax=10,
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(time[:cwt].to(u.min), freq[:cwf]),
    )
    plt.xticks([])
    plt.ylabel("Freq (MHz)")
    plt.title("Single Chunk Model")
    plt.subplot(212)
    plt.imshow(
        dspec[:cwf, :cwt] / (dspec[:cwf, :cwt].mean()),
        vmax=10,
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(time[:cwt].to(u.min), freq[:cwf]),
    )
    plt.xlabel("Time (Min)")
    plt.ylabel("Freq (MHz)")
    plt.title("Single Chunk Data")
    plt.tight_layout()

To combine the chunks, we stack them using the moasaic method as
described in [Baker2022]\_. We first compare the phases of overlapping
chunks

.. jupyter-execute::

    plt.figure()
    plt.subplot(311)
    plt.imshow(
        np.angle(chunks[0, 0, 32:, :]),
        cmap="twilight",
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(time[:cwt].to(u.min), freq[cwf // 2 : cwf]),
    )
    plt.xticks([])
    plt.ylabel("Freq (MHz)")
    plt.title("Phase of top half of chunk 0-0")
    plt.subplot(312)
    plt.imshow(
        np.angle(chunks[1, 0, :32, :]),
        cmap="twilight",
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(time[:cwt].to(u.min), freq[cwf // 2 : cwf]),
    )
    plt.xticks([])
    plt.ylabel("Freq (MHz)")
    plt.title("Phase of bottom half of chunk 1-0")
    plt.subplot(313)
    plt.imshow(
        np.angle(chunks[0, 0, 32:, :] * np.conjugate(chunks[1, 0, :32, :])),
        vmin=-np.pi,
        vmax=np.pi,
        cmap="twilight",
        origin="lower",
        aspect="auto",
        extent=thth.ext_find(time[:cwt].to(u.min), freq[cwf // 2 : cwf]),
    )
    plt.ylabel("Freq (MHz)")
    plt.xlabel("Time (Min)")
    plt.title("Phase difference")
    plt.tight_layout()

Starting at the lowest frequency and earliest time chunk, we position
the chunk within the larger wavefield. For all subsequent chunks (first
in time then frequency) we find the average phase difference between the
chunk and any overlapping areas in the larger wavefield that already
have chunks added. We then remove this phse difference from the chunk
and add it to the wavefield. Since recovery performance tends to be best
in the centers of chunks, the chunks are weighted using a sinusoidal
weight to ease out the edges.

.. jupyter-execute::

    phases = thth.rotInit(chunks)
    
    E_recov = thth.rotMos(chunks, phases)
    E_recov*=np.sqrt(dspec.mean()/np.abs(E_recov**2).mean())
    
    
    dspec_ext = thth.ext_find(time.to(u.hour), freq)
    plt.figure(figsize=(8, 16))
    plt.subplot(222)
    plt.imshow(np.abs(E_recov) ** 2,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=200,
        extent=dspec_ext,
    )
    plt.xlabel("Time (hrs)")
    plt.ylabel("Freq (MHz)")
    plt.title("Dynamic Spectrum Model")
    plt.subplot(221)
    plt.imshow(dspec,
               aspect="auto",
               origin="lower",
               vmin=0,
               vmax=200,
               extent=dspec_ext,)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Freq (MHz)")
    plt.title("Dynamic Spectrum")
    plt.subplot(224)
    plt.imshow(np.angle(E_recov),
        cmap="twilight",
        aspect="auto",
        origin="lower",
        extent=dspec_ext,
    )
    plt.xlabel("Time (hrs)")
    plt.ylabel("Freq (MHz)")
    plt.title("Wave Field Phase Model")
    plt.subplot(223)
    plt.imshow(np.angle(wf),
        cmap="twilight",
        aspect="auto",
        origin="lower",
        extent=dspec_ext,
    )
    plt.xlabel("Time (hrs)")
    plt.ylabel("Freq (MHz)")
    plt.title("Wave Field Phase")

To search for any regions of the conjugate wavefield that may have been
missed in the original theta-theta recovery, we apply the
Gerchberg-Saxton algorthim. This effectively deconvolves the full
conjugate spectrum using our recovered wavefield as an initial guess.

.. jupyter-execute::

    niter=100
    posdspec = np.isfinite(dspec) * (dspec > 0)
    fd = thth.fft_axis(time,u.mHz)
    tau = thth.fft_axis(freq,u.us)
    wfForced = np.copy(E_recov)
    wfForced *= np.sqrt(dspec[posdspec].mean() / np.abs(wfForced[posdspec]**2).mean())
    wfForced[posdspec] = np.sqrt(dspec[posdspec]) * np.exp(1j*np.angle(wfForced[posdspec]))
    for i in range(niter):
        CWF = np.fft.fftshift(np.fft.fft2(wfForced))
        CWF[tau < 0] = 0
        wfForced = np.fft.ifft2(np.fft.ifftshift(CWF))
        wfForced[posdspec] = np.sqrt(dspec[posdspec]) * np.exp(1j*np.angle(wfForced[posdspec]))

Comparing the initial mosaic wavefield recovery to the forced amplitude
version, we can see that the high :math:`\tau` images are still
recovered even though they were not part of the original
:math:`\theta-\theta` map.

.. jupyter-execute::

    fd = thth.fft_axis(time,u.mHz)
    tau = thth.fft_axis(freq,u.us)
    SS_ext_full = thth.ext_find(fd, tau)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.imshow(
        np.abs(np.fft.fftshift(np.fft.fft2(E_recov))) ** 2,
        norm=LogNorm(
            vmax=1e11,
            vmin=1e6,
        ),
        origin="lower",
        aspect="auto",
        extent=SS_ext_full,
    )
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")
    plt.title("Wavefield Model")
    plt.xlim((-1, 1))
    plt.ylim((-1, 3))
    plt.subplot(132)
    plt.imshow(
        np.abs(np.fft.fftshift(np.fft.fft2(wf))) ** 2,
        norm=LogNorm(
            vmax=1e11,
            vmin=1e6,
        ),
        origin="lower",
        aspect="auto",
        extent=SS_ext_full,
    )
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")
    plt.title("Wavefield")
    plt.xlim((-1, 1))
    plt.ylim((-1, 3))
    
    plt.subplot(133)
    plt.imshow(
        np.abs(np.fft.fftshift(np.fft.fft2(wfForced))) ** 2,
        norm=LogNorm(
            vmax=1e11,
            vmin=1e6,
        ),
        origin="lower",
        aspect="auto",
        extent=SS_ext_full,
    )
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")
    plt.title("Wavefield Model\n(Forced Amplitudes)")
    plt.xlim((-1, 1))
    plt.ylim((-1, 3))
