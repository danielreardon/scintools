Dynspec class
=============

The ``Dynspec`` class, found in the ``dynspec`` module, is the main entry point
for processing and analysing observed pulsar dynamic spectra. It supports
loading dynamic spectra from disk or from a simulation, cleaning and
normalising them, computing autocorrelation functions (ACFs) and secondary
spectra, measuring scintillation arcs, and estimating scintillation parameters.

A full, runnable walkthrough is provided in the ``arc_modelling`` example
notebook in ``scintools/examples``. The sections below summarise the most
commonly used features.

Importing dynamic spectra
-------------------------

A ``Dynspec`` object is most commonly loaded from a `psrflux`-format dynamic
spectrum file::

    from scintools.dynspec import Dynspec

    dyn = Dynspec(filename="J0437-4715.dynspec", process=False)

Setting ``process=False`` loads the data without running the default
processing chain, so that each processing step can be applied manually. Pass
``verbose=False`` to suppress progress output.

A ``Dynspec`` object can also be built from a :doc:`Simulation <simulation>`
object, or from an in-memory flux array via the ``BasicDyn`` helper::

    dyn = Dynspec(dyn=my_sim)           # from a scint_sim.Simulation object
    dyn = Dynspec(dyn=BasicDyn(...))    # from arrays of flux, times, freqs

Multiple observations can be concatenated in time by adding ``Dynspec``
objects together::

    dyn_tot = dyn1 + dyn2

Basic plotting
--------------

To view the dynamic spectrum::

    dyn.plot_dyn()

Pass ``lamsteps=True`` to plot against wavelength rather than frequency (after
scaling, see `Processing`_). The ACF and secondary spectrum can be plotted
directly with::

    dyn.plot_acf()
    dyn.plot_sspec()

Processing
----------

The dynamic spectrum can be cleaned, cropped, normalised and resampled in
several ways. A typical processing chain is::

    dyn.trim_edges()     # remove empty channels/subints from the edges
    dyn.refill()         # interpolate over gaps (e.g. zapped RFI)
    dyn.correct_dyn()    # normalise out bandpass and time-variable gain
    dyn.scale_dyn()      # rescale the frequency axis to wavelength (lambda)

Other useful processing methods include:

    * ``crop_dyn(fmin, fmax, tmin, tmax)`` - crop the dynamic spectrum in
      frequency and/or time.
    * ``refill(method='linear')`` / ``refill(method='mean')`` - choose the
      interpolation method used to fill gaps.
    * ``zap()`` - flag outlying pixels.

For example, to combine several observations into a single long dynamic
spectrum, each observation is trimmed, refilled and corrected before being
added::

    dyn_tot = Dynspec(filename=dyn_files[0], verbose=False)
    dyn_tot.trim_edges()
    dyn_tot.refill()
    dyn_tot.correct_dyn()
    for dyn_file in dyn_files[1:]:
        dyn_new = Dynspec(filename=dyn_file, verbose=False)
        dyn_new.trim_edges()
        dyn_new.refill(method='linear')
        dyn_new.correct_dyn()
        dyn_tot += dyn_new

Secondary spectra
-----------------

The secondary spectrum is the two-dimensional power spectrum of the dynamic
spectrum. It is computed with::

    dyn.calc_sspec()
    dyn.plot_sspec(lamsteps=True)

Pre-whitening and post-darkening are applied by default. Passing
``plot=True`` to ``calc_sspec`` computes and plots in a single step.

Measuring scintillation arcs
----------------------------

Scintillation arcs appear as parabolic features in the secondary spectrum, and
their curvature encodes the geometry and velocity of the scattering screen.
The arc curvature is measured with ``fit_arc``::

    dyn.fit_arc(
        lamsteps=True,
        plot=True,
        etamin=2,
        etamax=400,
        delmax=1,
        numsteps=1e3,
        nsmooth=7,
        startbin=2,
    )

The fitted curvature and its uncertainty are stored on the object::

    print("Curvature:", dyn.betaeta)
    print("Uncertainty:", dyn.betaetaerr)

The fitted arc can be overlaid on the secondary spectrum::

    dyn.plot_sspec(lamsteps=True, plotarc=True)

The secondary spectrum can also be normalised about the fitted arc curvature
with ``norm_sspec``::

    dyn.norm_sspec(lamsteps=True, plot=True, delmax=0.3, cutmid=3, startbin=5)

Scintillation parameters
------------------------

Scintillation timescale, decorrelation bandwidth and related parameters are
estimated from the ACF with ``get_scint_params``::

    dyn.get_scint_params(method="acf2d_approx", plot=True)

The measured parameters (e.g. the scintillation timescale ``dyn.tau`` and
decorrelation bandwidth ``dyn.dnu``) are stored as attributes on the object
for downstream modelling.
