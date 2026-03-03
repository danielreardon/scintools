Examples
========

This page contains practical, runnable examples demonstrating common
scintools workflows.  All examples assume that scintools and its dependencies
are installed (see the project `README <https://github.com/danielreardon/scintools>`_).

.. contents:: Contents
   :local:
   :depth: 2


Loading and visualising a dynamic spectrum
------------------------------------------

**Loading from file**

Dynamic spectra stored in ``psrflux`` format can be loaded directly::

    from scintools.dynspec import Dynspec

    dyn = Dynspec(filename='J0437-4715.dynspec', verbose=True)

The ``verbose=True`` flag (default) prints a summary of the observation
metadata (pulsar name, MJD, frequency, bandwidth, etc.) on load.

**Basic plotting**

Display the dynamic spectrum with the default colour scale::

    dyn.plot_dyn()

Save the figure to a file instead of displaying it::

    dyn.plot_dyn(filename='dynspec.png', display=False)

Customise the colour limits::

    dyn.plot_dyn(vmin=0, vmax=5)

**Loading from a simulation**

A :class:`~scintools.scint_sim.Simulation` object can be passed directly::

    from scintools.scint_sim import Simulation
    from scintools.dynspec import Dynspec

    sim = Simulation(mb2=2, rf=1, ds=0.01, alpha=5/3)
    dyn = Dynspec(dyn=sim, verbose=True)
    dyn.plot_dyn()


Processing dynamic spectra
--------------------------

**Cleaning and normalisation**

Remove short or bad sub-integrations and normalise each sub-integration to
remove bandpass structure::

    dyn.trim_edges()    # strip zero-padded boundary channels/sub-integrations
    dyn.refill()        # interpolate over flagged (zero) channels
    dyn.scale_dyn()     # normalise each sub-integration

**Cropping**

Crop to a sub-band and time range (units: MHz and minutes)::

    dyn.crop_dyn(tmin=0, tmax=30, fmin=1300, fmax=1500)

**Resampling**

Reduce frequency or time resolution by averaging::

    dyn.zap_minmax()    # flag outlier channels/sub-integrations
    dyn.refill()

**Common pitfalls**

- Always call ``refill()`` after flagging to ensure that masked pixels do
  not propagate as zeros into the FFT.
- Normalisation (``scale_dyn``) should be applied before computing the
  secondary spectrum or ACF.


Computing secondary spectra
---------------------------

**Basic calculation**

Compute the secondary spectrum using the default pre-whitening and
post-darkening::

    dyn.calc_sspec()
    dyn.plot_sspec()

The secondary spectrum is stored in ``dyn.sspec`` with conjugate variables
``dyn.tdel`` (differential delay, μs) and ``dyn.fdop`` (differential Doppler
frequency, mHz).

**Wavelength-scaled secondary spectrum**

For a display that is independent of the observing frequency, use equal
steps in wavelength (``lamsteps=True``)::

    dyn.calc_sspec(lamsteps=True)
    dyn.plot_sspec(lamsteps=True)

**Interpreting results**

Parabolic arcs in the secondary spectrum arise from a thin scattering screen.
The curvature :math:`\eta` (in units of s³) depends on the effective velocity
and the geometry of the scattering screen.  See
:meth:`~scintools.dynspec.Dynspec.fit_arc` for how to measure :math:`\eta`.


Arc fitting workflow
--------------------

**Complete example**

::

    from scintools.dynspec import Dynspec

    dyn = Dynspec(filename='J0437-4715.dynspec')
    dyn.trim_edges()
    dyn.refill()
    dyn.scale_dyn()
    dyn.calc_sspec()
    dyn.fit_arc(plot=True)

    print(f"Arc curvature eta = {dyn.eta:.4f} +/- {dyn.etaerr:.4f} s^3")

**Parameter selection guide**

+-------------------+------------------------------------------------------------+
| Parameter         | Guidance                                                   |
+===================+============================================================+
| ``delmax``        | Maximum differential delay (μs) to include in the fit.    |
|                   | Set to ~half the total delay range of your spectrum.       |
+-------------------+------------------------------------------------------------+
| ``numsteps``      | Number of trial curvatures; 10 000 is sufficient for most  |
|                   | observations.                                              |
+-------------------+------------------------------------------------------------+
| ``startbin``      | Rows near zero-delay to exclude (avoids DC artefacts).     |
+-------------------+------------------------------------------------------------+
| ``cutmid``        | Columns near zero-Doppler to exclude (avoids DC artefacts).|
+-------------------+------------------------------------------------------------+
| ``constraint``    | ``[eta_min, eta_max]`` to limit the search range if the    |
|                   | approximate curvature is known a priori.                   |
+-------------------+------------------------------------------------------------+
| ``nsmooth``       | Window length for the Savitzky–Golay smoothing filter      |
|                   | applied to the summed power profile. Must be an odd        |
|                   | integer ≥ 3.                                               |
+-------------------+------------------------------------------------------------+

**Fitting each side of the arc separately**

When the arc is clearly asymmetric (e.g. due to anisotropic scattering),
use ``asymm=True`` to fit the left and right wings independently::

    dyn.fit_arc(asymm=True, plot=True)
    print(dyn.eta_left, dyn.eta_right)

**Troubleshooting**

- *No clear arc visible*: try increasing ``startbin`` or adjusting
  ``low_power_diff``/``high_power_diff`` to widen the fitting region.
- *Fit returns forward parabola*: the peak power sits at zero curvature.
  Try increasing ``constraint[0]`` or reducing ``delmax``.
- *Error is very large*: the arc is broad or noisy.  Try reducing
  ``nsmooth`` or using ``log_parabola=True``.


ACF analysis
------------

**Computing the ACF**

::

    dyn.calc_acf()
    dyn.plot_acf()

The 2-D ACF is stored in ``dyn.acf`` with lag axes ``dyn.tau_samples``
(time lags in minutes) and ``dyn.dnu_samples`` (frequency lags in MHz).

**Fitting the ACF**

Fit a 1-D model along the time and frequency axes simultaneously::

    dyn.fit_acf()
    print(f"Scintillation timescale tau = {dyn.tau:.1f} +/- {dyn.tauerr:.1f} s")
    print(f"Decorrelation bandwidth dnu = {dyn.dnu:.3f} +/- {dyn.dnuerr:.3f} MHz")

For a full 2-D fit that also measures the phase gradient (drift) in the
ACF::

    dyn.fit_acf(method='2d')
    print(dyn.acf_tilt, dyn.phasegrad)

**Extracting scintillation parameters**

Once ``tau`` and ``dnu`` are measured, derive the scintillation velocity::

    from scintools.scint_utils import scint_velocity
    viss = scint_velocity(params=None, dnu=dyn.dnu, tau=dyn.tau,
                          freq=dyn.freq)
    print(f"Scintillation velocity Viss = {viss:.1f} km/s")


Simulations
-----------

**Creating a simulated dynamic spectrum**

Simulate a dynamic spectrum using the electromagnetic wave propagation code
of Coles et al. (2010)::

    from scintools.scint_sim import Simulation
    from scintools.dynspec import Dynspec

    sim = Simulation(mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                     ns=256, nf=256, dlam=0.25, freq=1400, dt=30)
    dyn = Dynspec(dyn=sim, verbose=True)
    dyn.plot_dyn()

**Inspecting the theoretical arc curvature**

The simulation computes the theoretical arc curvature automatically::

    print(f"Theoretical eta = {sim.eta:.4f} s^3")

    dyn.calc_sspec()
    dyn.fit_arc(plot=True)
    print(f"Measured eta    = {dyn.eta:.4f} +/- {dyn.etaerr:.4f} s^3")

**Parameter studies**

Run multiple simulations to study how arc curvature varies with screen
geometry::

    import numpy as np
    from scintools.scint_sim import Simulation
    from scintools.dynspec import Dynspec

    screen_distances = np.linspace(0.1, 0.9, 9)
    eta_values = []
    for s in screen_distances:
        sim = Simulation(mb2=2, rf=1, ds=0.01, seed=42)
        dyn = Dynspec(dyn=sim, verbose=False)
        dyn.calc_sspec()
        dyn.fit_arc(plot=False, display=False)
        eta_values.append(dyn.eta)

    print("Screen distance vs measured curvature:")
    for s, eta in zip(screen_distances, eta_values):
        print(f"  s={s:.1f}  eta={eta:.4f}")

**Common pitfalls**

- The ``seed`` parameter controls the random screen realisation.  Set it to
  a fixed value for reproducible results.
- For lamsteps simulations, pass ``lamsteps=True`` to both
  :class:`~scintools.scint_sim.Simulation` and
  :class:`~scintools.dynspec.Dynspec`.
