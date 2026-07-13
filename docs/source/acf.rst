ACF class
================

The scintools ``ACF`` class, found in the ``scint_sim`` module, features tools for simulating and plotting autocorrelation functions. Based on the theoretical treatment found in Appendix A of `Rickett, Coles et al. (2014) <https://iopscience.iop.org/article/10.1088/0004-637X/787/2/161/>`_. The magnitude of the effective velocity is defined to be 1, so time and frequency lags are expressed in units of the (isotropic) scintillation timescale and decorrelation bandwidth respectively.

.. raw:: html

	<em class="property">class </em><code class="descclassname">scintools.</code><code class="descname">ACF</code><span class="sig-paren">(</span><em>psi=0, phasegrad=0, theta=0, ar=1, alpha=5/3, taumax=4, dnumax=4, nf=51, nt=51, amp=1, wn=0, spatial_factor=2, resolution_factor=1, core_factor=2, auto_sampling=True, plot=False, display=True</em><span class="sig-paren">)</span>

\
		ACF class. On construction, computes the ACF from the theoretical function.

		**Parameters:**
				*   **psi** (`float, optional`) - angle (degrees) of the velocity vector with respect to the major axis of the (brightness) anisotropy.
				*   **phasegrad** (`float, optional`) - magnitude of the phase gradient (e.g. due to a binary companion).
				*   **theta** (`float, optional`) - angle (degrees) of the phase gradient with respect to the velocity vector.
				*   **ar** (`float, optional`) - axial ratio of anisotropy.
				*   **alpha** (`float, optional`) - structure function exponent. 5/3 is a Kolmogorov profile (default) while 2 is Gaussian.
				*   **taumax** (`float, optional`) - number of scintillation timescales to compute the ACF out to (equivalently, the ACF in space goes out to ``taumax*V``).
				*   **dnumax** (`float, optional`) - number of decorrelation bandwidths to compute the ACF out to.
				*   **nf** (`int, optional`) - number of frequency-lag samples in the ACF. If not odd, it is incremented by one so the ACF has a well-defined centre.
				*   **nt** (`int, optional`) - number of time-lag samples in the ACF. If not odd, it is incremented by one so the ACF has a well-defined centre.
				*   **amp** (`float, optional`) - amplitude to scale the ACF by. By default the ACF peaks at 1.
				*   **wn** (`float, optional`) - size of the white-noise spike at the origin of the ACF.
				*   **spatial_factor** (`float, optional`) - multiplier for the spatial extent (``taumax``) used when calculating the ACF of the electric field. Only used if ``auto_sampling`` is False.
				*   **resolution_factor** (`float, optional`) - multiplier applied to the default spatial resolution used when calculating the ACF of the electric field. Only used if ``auto_sampling`` is False.
				*   **core_factor** (`float, optional`) - additional resolution multiplier applied near the origin (first frequency-lag sample). Only used if ``auto_sampling`` is False.
				*   **auto_sampling** (`bool, optional`) - if True, automatically set the spatial sampling factors based on ``ar`` and ``taumax`` to avoid sampling artefacts, overriding ``spatial_factor``, ``resolution_factor``, and ``core_factor``. Warning: computation time increases sharply for large ``ar``.
				*   **plot** (`bool, optional`) - if True, plot the ACF after it is computed.
				*   **display** (`bool, optional`) - if True and ``plot`` is True, show the plot immediately.

Example
-------

.. code-block:: python

	from scintools.scint_sim import ACF

	# Quick look with default parameters
	acf_obj = ACF()
	acf_obj.plot_acf()          # the ACF
	acf_obj.plot_acf_efield()   # the ACF of the electric field
	acf_obj.plot_sspec()        # the secondary spectrum (Hanning window applied by default)
	acf_obj.calc_sspec(window='blackman', window_frac=1.0)  # change the window and its size
	acf_obj.plot_sspec()

	# An ACF with anisotropy and a phase gradient (e.g. a binary companion)
	ar = 2            # axial ratio of anisotropy
	psi = 30          # angle of velocity relative to major axis (defined as x)
	phasegrad = 0.2   # magnitude of phase gradient
	theta = 0         # angle of phase gradient relative to velocity
	taumax = 4        # number of scintillation timescales to compute to
	dnumax = 4        # number of decorrelation bandwidths to compute to
	nt = 51           # number of time samples in the ACF
	nf = 51           # number of frequency samples in the ACF
	auto_sampling = True  # dynamically set spatial properties to avoid ACF artefacts

	my_acf = ACF(psi=psi, phasegrad=phasegrad, theta=theta, ar=ar,
	             taumax=taumax, dnumax=dnumax, nt=nt, nf=nf)
	my_acf.plot_acf()
	my_acf.plot_sspec()

	# Raw arrays are available for custom plotting
	acf = my_acf.acf
	t = my_acf.tn
	f = my_acf.fn

Methods
-------

.. raw:: html

	<code class="descname">calc_acf</code><span class="sig-paren">(</span><em>plot=False</em><span class="sig-paren">)</span>

\

		Computes the 2D ACF of intensity vs time and frequency lag, implementing the integrals in Appendix A of Rickett, Coles et al. (2014). This is called automatically on construction of an ``ACF`` object, and populates ``self.acf``, ``self.tn`` (time-lag axis), ``self.fn`` (frequency-lag axis), ``self.snp`` (spatial-lag axis of the electric-field ACF), and ``self.acf_efield`` (ACF of the electric field).

		**Parameters:**
				*   **plot** (`bool, optional`) - plot the resulting ACF after it is computed.

.. raw:: html

	<code class="descname">plot_acf</code><span class="sig-paren">(</span><em>display=True, contour=True, filled=False</em><span class="sig-paren">)</span>

\

		Plot the theoretical 2D ACF of intensity vs time and frequency lag. Requires ``calc_acf`` to have been run first.

		**Parameters:**
				*   **display** (`bool, optional`) - if True, set the plot title and show the plot immediately.
				*   **contour** (`bool, optional`) - if True and ``filled`` is False, overlay black contours at ``amp * [0.2, 0.4, 0.6, 0.8]``.
				*   **filled** (`bool, optional`) - if True, plot filled contours instead of a pcolormesh.

.. raw:: html

	<code class="descname">plot_acf_efield</code><span class="sig-paren">(</span><em>display=True</em><span class="sig-paren">)</span>

\

		Plot the ACF of the electric field vs spatial lag. Requires ``calc_acf`` to have been run first.

		**Parameters:**
				*   **display** (`bool, optional`) - if True, show the plot immediately.

.. raw:: html

	<code class="descname">calc_sspec</code><span class="sig-paren">(</span><em>window='hanning', window_frac=1</em><span class="sig-paren">)</span>

\

		Compute the secondary spectrum from the ACF, applying a 2D window before taking the 2D Fourier transform to reduce edge effects. Populates ``self.sspec`` (in dB).

		**Parameters:**
				*   **window** (`str, optional`) - name of the window function to apply along each axis before transforming, e.g. ``'hanning'`` (default), ``'hamming'``, ``'blackman'``, or ``'bartlett'``.
				*   **window_frac** (`float, optional`) - fraction of each axis over which the window is applied.

.. raw:: html

	<code class="descname">plot_sspec</code><span class="sig-paren">(</span><em>display=True, vmin=None, vmax=None</em><span class="sig-paren">)</span>

\

		Plot the secondary spectrum vs time and frequency lag. Computes the secondary spectrum first via ``calc_sspec`` (with its default window) if it has not already been computed.

		**Parameters:**
				*   **display** (`bool, optional`) - if True, set the plot title and show the plot immediately.
				*   **vmin** (`float or None, optional`) - minimum of the colour scale, in dB. If None, defaults to 3 dB below the median of the (finite, non-zero) secondary spectrum.
				*   **vmax** (`float or None, optional`) - maximum of the colour scale, in dB. If None, defaults to 3 dB below the maximum of the (finite, non-zero) secondary spectrum.
