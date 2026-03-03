Dynspec class
=============

The :class:`~scintools.dynspec.Dynspec` class is the primary interface for
loading, processing, and analysing pulsar dynamic spectra in scintools.

For questions or support please email Daniel Reardon: dreardon@swin.edu.au

Overview
--------

A dynamic spectrum is a two-dimensional array of flux density as a function of
time and radio frequency.  The :class:`~scintools.dynspec.Dynspec` class
provides methods to:

- Load dynamic spectra from ``psrflux``-format files or from
  :class:`~scintools.scint_sim.Simulation` objects.
- Clean, crop, normalise, and resample data.
- Compute secondary spectra and fit parabolic arc curvatures.
- Compute autocorrelation functions (ACFs) and measure scintillation
  timescales and bandwidths.
- Visualise all intermediate and final data products.

See :doc:`examples` for complete, runnable worked examples.

Importing dynamic spectra
-------------------------

A :class:`~scintools.dynspec.Dynspec` object can be loaded from a
``psrflux``-format dynamic spectrum file::

    from scintools.dynspec import Dynspec
    dyn = Dynspec(filename='myobs.dynspec', verbose=True)

It can also be initialised directly from a
:class:`~scintools.scint_sim.Simulation` object::

    from scintools.scint_sim import Simulation
    from scintools.dynspec import Dynspec
    sim = Simulation()
    dyn = Dynspec(dyn=sim, verbose=True)

Basic plotting
--------------

To view the dynamic spectrum use::

    dyn.plot_dyn()

Additional keyword arguments control the colour scale, axis labels, and
whether to display or save the figure.

Processing
----------

The dynamic spectrum can be cleaned, cropped, normalised, and resampled in
several ways::

    dyn.trim_edges()          # remove zero-padded edge channels/subints
    dyn.zap()                 # interactive RFI flagging
    dyn.refill()              # interpolate over flagged data
    dyn.crop_dyn(tmin, tmax, fmin, fmax)  # crop in time and/or frequency
    dyn.scale_dyn()           # normalise each sub-integration

Secondary spectra
-----------------

The secondary spectrum is computed via a 2-D Fourier transform of the dynamic
spectrum.  Pre-whitening and post-darkening are applied by default::

    dyn.calc_sspec()
    dyn.plot_sspec()

Arc curvature
-------------

Parabolic arc curvatures are measured using
:meth:`~scintools.dynspec.Dynspec.fit_arc`, which searches for the arc
curvature :math:`\eta` that maximises power along the arc::

    dyn.calc_sspec()
    dyn.fit_arc(plot=True)
    print(dyn.eta, dyn.etaerr)

See :doc:`examples` for a complete arc-fitting workflow and guidance on
selecting input parameters.

ACF analysis
------------

The autocorrelation function is computed with
:meth:`~scintools.dynspec.Dynspec.calc_acf` and fitted with
:meth:`~scintools.dynspec.Dynspec.fit_acf`::

    dyn.calc_acf()
    dyn.fit_acf()
    print(dyn.tau, dyn.dnu)   # scintillation timescale and bandwidth

API reference
-------------

.. autoclass:: scintools.dynspec.Dynspec
   :members:
   :undoc-members:
   :show-inheritance:

