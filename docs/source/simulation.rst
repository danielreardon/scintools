Simulation class
================

The ``Simulation`` class, found in the ``scint_sim`` module, is a wave-optics
simulator of a thin scattering screen. It generates a random phase screen with
a given power-law structure function and anisotropy, propagates it through the
Fresnel diffraction integral to obtain the electric field and intensity as a
function of observer position and frequency, and packages the result as a
simulated dynamic spectrum with physical time/frequency axes. A ``Simulation``
instance can be passed directly to the :doc:`Dynspec class <dynspec>` (via the
``SimDyn`` wrapper) for further analysis.

This is based on the original MATLAB code by Coles et al. (2010),
`"Scattering of pulsar radio emission by the interstellar plasma"
<https://ui.adsabs.harvard.edu/abs/2010ApJ...717.1206C>`_.

Simulating a dynamic spectrum from turbulence
---------------------------------------------

The simulation runs on construction of the object, so a dynamic spectrum can be
generated with the default parameters in a single line::

    from scintools.scint_sim import Simulation

    sim = Simulation()

The resulting object exposes the phase screen, intensity, electric field and
dynamic spectrum, each of which can be inspected with a dedicated plotting
method (see `Plotting`_ below).

Input parameters
----------------

The most commonly adjusted parameters of ``Simulation`` are:

    * **mb2** (`float`) - the maximum Born parameter, which controls the
      strength of scattering. Default is 2.
    * **rf** (`float`) - the Fresnel scale, used as the length unit for the
      simulation. Default is 1.
    * **ds** (`float`) - the spatial step size in units of ``rf``, used for both
      the x and y axes unless overridden by ``dx``/``dy``. Default is 0.01.
    * **alpha** (`float`) - the structure-function exponent. 5/3 (default) is a
      Kolmogorov spectrum.
    * **ar** (`float`) - the axial ratio of the anisotropy of the scattering
      screen. Default is 1 (isotropic).
    * **psi** (`float`) - the orientation angle of the anisotropy, in degrees,
      relative to the x axis. Default is 0.
    * **inner** (`float`) - the inner scale of turbulence, in units of ``rf``.
      Should generally be smaller than ``ds``. Default is 0.001.
    * **ns** (`int`) - the size of the simulation in spatial steps, used for
      both axes unless overridden by ``nx``/``ny``. The total length in
      refractive scales is ``ns * ds``. Default is 256.
    * **nf** (`int`) - the number of frequency channels across the fractional
      bandwidth ``dlam``. Default is 256.
    * **dlam** (`float`) - the fractional bandwidth relative to the centre
      frequency. Default is 0.25.
    * **lamsteps** (`bool`) - if True, take equally-spaced steps in
      wavelength rather than in frequency. Default is False.
    * **seed** (`int or None`) - the seed used to generate the random phase
      screen, for reproducible experiments. Use -1 to reshuffle (draw a
      fresh, unseeded screen). Default is None (NumPy's global random
      state).

The number and size of spatial steps can be set independently in x and y,
overriding ``ns``/``ds``:

    * **nx** (`int or None`) - number of spatial steps in x. Overrides
      ``ns`` when set. Default is None.
    * **ny** (`int or None`) - number of spatial steps in y. Overrides
      ``ns`` when set. Default is None.
    * **dx** (`float or None`) - spatial step size in x, in units of
      ``rf``. Overrides ``ds`` when set. Default is None.
    * **dy** (`float or None`) - spatial step size in y, in units of
      ``rf``. Overrides ``ds`` when set. Default is None.

The following parameters control what happens at construction time:

    * **plot** (`bool`) - if True, plot the screen, intensity, and dynamic
      spectrum after the simulation is computed (via ``plot_all``).
      Default is False.
    * **verbose** (`bool`) - if True, print progress messages while the
      simulation runs. Default is False.

Optional parameters for physical units
---------------------------------------

By default the simulation is dimensionless. The following parameters attach
physical time and frequency axes to the output dynamic spectrum so that it can
be handed to the ``Dynspec`` class:

    * **freq** (`float`) - the centre observing frequency, in MHz. Default is
      1400.
    * **dt** (`float`) - the subintegration time, in seconds. Default is 30.
    * **mjd** (`float`) - the MJD of the start of the observation. Default is
      60000.
    * **nsub** (`int or None`) - the number of subintegrations to keep in the
      output dynamic spectrum. If None (default), all ``nx`` steps are kept.
    * **efield** (`bool`) - if True, store the real part of the electric
      field instead of the intensity in the output dynamic spectrum.
      Default is False.
    * **noise** (`float or None`) - accepted for interface compatibility
      with other scintools classes; currently unused. Default is None.

A fully specified example
-------------------------

To simulate a strongly scattered, anisotropic screen with a reproducible seed::

    mb2 = 20    # Born variance, controls the strength of scattering
    ar = 2      # axial ratio of anisotropy
    psi = 30    # angle of anisotropy relative to the x axis, in degrees
    dlam = 0.33 # fractional bandwidth
    ns = 1024   # size of the simulation in spatial steps
    nf = 256    # number of frequency channels across dlam
    seed = 1    # seed for the random phase screen, for reproducibility

    my_sim = Simulation(mb2=mb2, ar=ar, psi=psi, dlam=dlam,
                        ns=ns, nf=nf, seed=seed)

Plotting
--------

Each component of the simulation can be visualised directly::

    my_sim.plot_screen()     # the simulated phase screen in x and y
    my_sim.plot_intensity()  # observer-side intensity fluctuations in space
    my_sim.plot_dynspec()    # the simulated dynamic spectrum
    my_sim.plot_efield()     # the real part of the electric field
    my_sim.plot_delay()      # dispersive group delay and impulse response
    my_sim.plot_pulse()      # the scatter-broadened pulse in space
    my_sim.plot_all()        # the screen, intensity and dynamic spectrum

Using a simulation with the Dynspec class
------------------------------------------

Because the simulated dynamic spectrum carries physical axes, the ``Simulation``
object can be loaded into a ``Dynspec`` object for the same analysis applied to
observed data::

    from scintools.dynspec import Dynspec

    dyn = Dynspec(dyn=my_sim)
    dyn.plot_dyn()
