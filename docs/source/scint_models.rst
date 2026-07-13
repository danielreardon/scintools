scint_models module
===================

The ``scint_models`` module contains various functions for modelling and fitting scintillation data.

Fitting
-------

.. raw:: html

	<code class="descname">fitter</code><span class="sig-paren">(</span><em>model, params, args, mcmc=False, pos=None, nwalkers=100, steps=1000, burn=0.2, progress=True, workers=1, nan_policy='raise', max_nfev=None, thin=10, is_weighted=True</em><span class="sig-paren">)</span>

\

		Common entry point for fitting one of the residual-returning models in this module using ``lmfit``. Wraps `model` and `params` in an ``lmfit`` ``Minimizer`` and either performs a least-squares minimisation, or (if ``mcmc=True``) runs the ``emcee`` MCMC sampler.

		**Parameters:**
				*   **model** (`callable`) - a model function from this module (e.g. ``tau_acf_model``, ``scint_acf_model``, ``scint_acf_model_2d_approx``, ``powerspectrum_model``) that takes ``params`` followed by the contents of `args` and returns the (weighted) residuals between data and model.
				*   **params** (`lmfit Parameters() object`) - initial/starting parameters for the fit.
				*   **args** (`tuple`) - extra positional arguments passed to `model` after `params` (typically some combination of ``xdata``, ``ydata``, and ``weights``).
				*   **mcmc** (`bool`, optional) - if True, sample the posterior with ``emcee`` instead of doing a least-squares fit. The default is False.
				*   **pos** (`array-like`, optional) - initial walker positions for ``emcee`` (only used if `mcmc` is True). The default is None.
				*   **nwalkers** (`int`, optional) - number of ``emcee`` walkers (only used if `mcmc` is True). The default is 100.
				*   **steps** (`int`, optional) - number of ``emcee`` steps to run (only used if `mcmc` is True). The default is 1000.
				*   **burn** (`float`, optional) - fraction of `steps` to discard as burn-in (only used if `mcmc` is True). The default is 0.2.
				*   **progress** (`bool`, optional) - whether ``emcee`` prints a progress bar (only used if `mcmc` is True). The default is True.
				*   **workers** (`int`, optional) - number of parallel workers for ``emcee`` (only used if `mcmc` is True). The default is 1.
				*   **nan_policy** (`str`, optional) - how ``lmfit`` should handle NaNs in the residuals for the least-squares fit (only used if `mcmc` is False). The default is 'raise'.
				*   **max_nfev** (`int`, optional) - maximum number of function evaluations for the least-squares fit (only used if `mcmc` is False). The default is None.
				*   **thin** (`int`, optional) - only accept every `thin`-th ``emcee`` sample (only used if `mcmc` is True). The default is 10.
				*   **is_weighted** (`bool`, optional) - whether the residuals returned by `model` are already weighted (only used if `mcmc` is True). The default is True.
		**Returns:**
				*   The ``lmfit`` fit result object (``MinimizerResult``) returned by ``Minimizer.minimize()`` or ``Minimizer.emcee()``.

ACF fitting
-----------

.. raw:: html

	<code class="descname">tau_acf_model</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Models a 1D cut through the center of the ACF along the time axis and returns the residuals.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), timescale at :math:`1/e` width ('tau'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), and white-noise spike ('wn').
				*   **xdata** (`numpy 1D array`) - time of sub-integrations from the center of the ACF to the maximum time.
				*   **ydata** (`numpy 1D array`) - ACF pixel values corresponding to xdata and running through the center of the ACF.
				*   **weights** (`numpy 1D array`) - weights of the data
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">dnu_acf_model</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Models a 1D cut through the center of the ACF along the frequency axis and returns the residuals.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), decorrelation bandwidth at half power ('dnu'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), and white-noise spike ('wn').
				*   **xdata** (`numpy 1D array`) - frequency of the channels from the center of the ACF to the maximum frequency.
				*   **ydata** (`numpy 1D array`) - ACF pixel values corresponding to xdata and running through the center of the ACF.
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">scint_acf_model</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Applies ``tau_acf_model`` and ``dnu_acf_model`` simultaneously and returns the concatenated residuals.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), timescale at :math:`1/e` width ('tau'), decorrelation bandwidth at half power ('dnu'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), and white-noise spike ('wn'), as well as half the number of sub-integrations in the ACF ('nt').
				*   **xdata** (`numpy 1D array`) - time of sub-integrations from the center of the ACF to the maximum time concatenated with the frequency of the channels from the center of the ACF to the maximum frequency.
				*   **ydata** (`numpy 1D array`) - ACF pixel values corresponding to xdata and running through the center of the ACF.
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">scint_acf_model_2d_approx</code><span class="sig-paren">(</span><em>params, tdata, fdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Models an approximate 2D ACF that incorporates a phase gradient.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), timescale at :math:`1/e` width ('tau'), decorrelation bandwidth at half power ('dnu'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), white-noise spike ('wn'), and phase gradient ('phasegrad'), as well as the observation frequency ('freq'), total observation time ('tobs'), and half the number of sub-integrations in the ACF ('nt').
				*   **tdata** (`numpy 1D array`) - times of sub-integrations along the desired range, centered on the sub-integration left of that of the white-noise spike.
				*   **fdata** (`numpy 1D array`) - frequencies of channels along the desired range, centered on the channel below that of the white-noise spike.
				*   **ydata** (`numpy 1D array`) - ACF cropped to the range of times and frequencies matching ``tdata`` and ``fdata``.
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">scint_acf_model_2d</code><span class="sig-paren">(</span><em>params, ydata, weights</em><span class="sig-paren">)</span>

\

		Models an analytical 2D ACF using the ``scint_sim`` ``ACF`` class. This method is significantly slower than ``scint_acf_model_2d_approx()``.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit timescale at :math:`1/e` width ('tau'), decorrelation bandwidth at half power ('dnu'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), axial ratio of anisotropy ('ar'), orientation of anisotropy ('psi'), phase gradient ('phasegrad'), rotation of the phase gradient ('theta'), and amplitude ('amp'), as well as the total observation time ('tobs'), total bandwidth of the observation ('bw'), and the number of sub-integrations and channels used to build the model ACF ('nt', 'nf').
				*   **ydata** (`numpy 1D array`) - ACF cropped symmetrically around its center to a desired range.
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">tau_sspec_model</code><span class="sig-paren">(</span><em>params, xdata, ydata</em><span class="sig-paren">)</span>

\

		Models a 1D cut through the center of the ACF along the time axis and applies a Fourier transform.

		**Parameters:**
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), timescale at :math:`1/e` width ('tau'), and index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov).
				*   **xdata** (`numpy 1D array`) - time of sub-integrations from the center of the ACF to the maximum time.
				*   **ydata** (`numpy 1D array`) - profile from secondary spectrum corresponding to the ACF to model summed along all columns (all :math:`f_t`).
		**Returns:**
				*   Residual of model and data, weighted by the model itself

.. raw:: html

	<code class="descname">dnu_sspec_model</code><span class="sig-paren">(</span><em>params, xdata, ydata</em><span class="sig-paren">)</span>

\

		Models a 1D cut through the center of the ACF along the frequency axis and applies a Fourier tranform.

		**Parameters:**
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp') and decorrelation bandwidth at half power ('dnu').
				*   **xdata** (`numpy 1D array`) - frequency of the channels from the center of the ACF to the maximum frequency.
				*   **ydata** (`numpy 1D array`) - profile from secondary spectrum corresponding to the ACF to model summed along all rows (all :math:`f_\tau` or :math:`f_\lambda`).
		**Returns:**
				*   Residual of model and data, weighted by the model itself

.. raw:: html

	<code class="descname">scint_sspec_model</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Applies ``tau_sspec_model`` and ``dnu_sspec_model`` simultaneously and returns the concatenated residuals.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), timescale at :math:`1/e` width ('tau'), decorrelation bandwidth at half power ('dnu'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), and white-noise spike ('wn'), as well as half the number of sub-integrations in the ACF ('nt').
				*   **xdata** (`numpy 1D array`) - time of sub-integrations from the center of the ACF to the maximum time concatenated with the frequency of the channels from the center of the ACF to the maximum frequency.
				*   **ydata** (`numpy 1D array`) - profile from secondary spectrum corresponding to the ACF to model summed along all columns (all :math:`f_t`) concatenated with it summed along all rows (all :math:`f_\tau` or :math:`f_\lambda`).
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

Secondary spectrum fitting
--------------------------

.. raw:: html

	<code class="descname">powerspectrum_model</code><span class="sig-paren">(</span><em>params, xdata, ydata</em><span class="sig-paren">)</span>

\

		Models a red-noise power spectrum (e.g. of the frequency-averaged secondary spectrum vs :math:`\sqrt{\eta}`) as a power law plus a white-noise floor, and returns the residuals.

		**Parameters:**
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the white-noise level ('wn'), power-law amplitude ('amp'), and power-law index ('alpha').
				*   **xdata** (`numpy 1D array`) - independent variable (e.g. square root of delay).
				*   **ydata** (`numpy 1D array`) - power spectrum values corresponding to `xdata`.
		**Returns:**
				*   Residual of model and data (not multiplied by weights)

.. raw:: html

	<code class="descname">arc_power_curve</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Returns a template for the power curve in secondary spectrum against :math:`\sqrt{\eta}` or normalised :math:`f_t`.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object.
				*   **xdata** (`numpy 1D array`) - square root arc curvatures.
				*   **ydata** (`numpy 1D array`) - secondary spectrum power profile. 
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">fit_parabola</code><span class="sig-paren">(</span><em>x, y</em><span class="sig-paren">)</span>

\

		Fit a parabola and return the value and error for the peak.

		**Parameters:** 
				*   **x** (`numpy 1D array`) - x values of the peak profile
				*   **y** (`numpy 1D array`) - y values of the peak profile
		**Returns:** 
				*   y values of the fit
				*   Fit peak value
				*   Fit peak error

.. raw:: html

	<code class="descname">fit_log_parabola</code><span class="sig-paren">(</span><em>x, y</em><span class="sig-paren">)</span>

\

		Fit a log-parabola and return the value and error for the peak.

		**Parameters:** 
				*   **x** (`numpy 1D array`) - x values of the peak profile
				*   **y** (`numpy 1D array`) - y values of the peak profile
		**Returns:** 
				*   y values of the fit
				*   Fit peak value
				*   Fit peak error

.. raw:: html

	<code class="descname">arc_curvature</code><span class="sig-paren">(</span><em>params, ydata, weights, true_anomaly, vearth_ra, vearth_dec, mjd=None, model_only=False, return_veff=False</em><span class="sig-paren">)</span>

\

		Models the arc curvature and returns the residuals (or, if ``model_only=True``, the model curvature itself).

		**Parameters:**
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing current best-fit values for the distance to the pulsar in kpc ('d'), the fractional screen distance ('s'), plus the parameters required by ``effective_velocity_annual``, and optionally an anisotropy model flag ('nmodel'), anisotropy angle ('zeta'), and the ISM velocity in RA and dec ('vism_ra', 'vism_dec') or, for the anisotropic case, along ``zeta`` ('vism_zeta'). The deprecated parameter names 'psi' and 'vism_psi' are no longer supported.
				*   **ydata** (`numpy 1D array`) - arc curvature data.
				*   **weights** (`numpy 1D array`) - weights of the data.
				*   **true_anomaly** (`numpy 1D array`) - true anolmalies corresponding to the data.
				*   **vearth_ra** (`numpy 1D array`) - Earth velocity in RA corresponding to the data.
				*   **vearth_dec** (`numpy 1D array`) - Earth velocity in dec corresponding to the data.
				*   **mjd** (`numpy 1D array`, optional) - MJDs corresponding to the data, used for the 'OMDOT' correction. The default is None.
				*   **model_only** (`bool`, optional) - if True, return the model curvature instead of the residuals. The default is False.
				*   **return_veff** (`bool`, optional) - if True (and ``model_only=True``), also return the ISM-subtracted effective velocity in RA and dec. The default is False.
		**Returns:**
				*   Weighted residual of model and data (or the model curvature, and optionally the effective velocity components, if ``model_only=True``)

.. raw:: html

	<code class="descname">arc_weak</code><span class="sig-paren">(</span><em>ftn, ar=1, psi=0, alpha=11/3</em><span class="sig-paren">)</span>

\

		Models the 1D weak-scattering scintillation arc Doppler profile (power vs normalised Doppler frequency), for a possibly anisotropic scattering screen.

		**Parameters:**
				*   **ftn** (`numpy 1D array`) - the normalised Doppler frequency (x-axis), where ``ftn = 1`` is the arc.
				*   **ar** (`float`, optional) - anisotropy axial ratio. The default is 1 (isotropic).
				*   **psi** (`float`, optional) - orientation angle of the anisotropy, in degrees. The default is 0.
				*   **alpha** (`float`, optional) - index of the turbulence spectrum. The default is 11/3 (Kolmogorov).
		**Returns:**
				*   The model Doppler profile

.. raw:: html

	<code class="descname">arc_weak_2d</code><span class="sig-paren">(</span><em>fdop, tdel, eta=1, ar=1, psi=0, alpha=11/3</em><span class="sig-paren">)</span>

\

		Models the 2D weak-scattering secondary spectrum along a scintillation arc, for a possibly anisotropic scattering screen.

		**Parameters:**
				*   **fdop** (`numpy 1D array`) - the Doppler frequency (x-axis) coordinates of the model secondary spectrum.
				*   **tdel** (`numpy 1D array`) - the delay (y-axis) coordinates of the model secondary spectrum.
				*   **eta** (`float`, optional) - arc curvature. The default is 1.
				*   **ar** (`float`, optional) - anisotropy axial ratio. The default is 1 (isotropic).
				*   **psi** (`float`, optional) - orientation angle of the anisotropy, in degrees. The default is 0.
				*   **alpha** (`float`, optional) - index of the turbulence spectrum. The default is 11/3 (Kolmogorov). Currently unused by the model, which always uses a fixed exponent of -11/6.
		**Returns:**
				*   The model secondary spectrum (2D array)

Velocity models
---------------

.. raw:: html

	<code class="descname">effective_velocity_annual</code><span class="sig-paren">(</span><em>params, true_anomaly, vearth_ra, vearth_dec, mjd=None</em><span class="sig-paren">)</span>

\

		Computes the effective velocity including annual (Earth orbital and proper motion) and pulsar orbital terms. Does NOT include the IISM velocity, but the returned effective velocity is expressed in the IISM frame.

		**Parameters:**
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing, for a binary pulsar, the projected semi-major axis in lt-s ('A1'), orbital period in days ('PB'), orbital eccentricity ('ECC'), longitude of periastron in degrees ('OM'), inclination via one of 'KIN' (degrees), 'COSI', or 'SINI', and optionally an inclination-sense flag ('sense') and periastron-advance rate in deg/yr ('OMDOT', requires 'T0' and ``mjd``); also, optionally, the proper motion in RA and dec ('PMRA', 'PMDEC'), and always the longitude of ascending node in degrees ('KOM'), the distance to the pulsar in kpc ('d'), and the fractional screen distance ('s').
				*   **true_anomaly** (`numpy 1D array`) - true anolmalies to compute over.
				*   **vearth_ra** (`numpy 1D array`) - Earth velocity in RA to compute over.
				*   **vearth_dec** (`numpy 1D array`) - Earth velocity in dec to compute over.
				*   **mjd** (`numpy 1D array`, optional) - MJDs corresponding to ``true_anomaly``, used for the 'OMDOT' correction. The default is None.
		**Returns:**
				*   Effective velocity in RA
				*   Effective velocity in dec
				*   Pulsar velocity in RA
				*   Pulsar velocity in dec

.. raw:: html

	<code class="descname">veff_thin_screen</code><span class="sig-paren">(</span><em>params, ydata, weights, true_anomaly, vearth_ra, vearth_dec, mjd=None</em><span class="sig-paren">)</span>

\

		Models the effective velocity implied by a thin-screen scattering geometry and returns the residuals. Uses Eq. 4 from Rickett et al. (2014) for the anisotropy coefficients.

		**Parameters:**
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit fractional screen distance ('s') and pulsar distance in kpc ('d'), plus the parameters required by ``effective_velocity_annual``, and optionally a scaling factor ('kappa', default 1), an anisotropy model flag ('nmodel'), and, for the anisotropic case, an axial ratio parameter ('R') and anisotropy angle in degrees ('psi'). ISM velocity in RA and dec ('vism_ra', 'vism_dec') may also be given and is subtracted from the effective velocity.
				*   **ydata** (`numpy 1D array`) - measured effective velocity data.
				*   **weights** (`numpy 1D array`) - weights of the data.
				*   **true_anomaly** (`numpy 1D array`) - true anolmalies corresponding to the data.
				*   **vearth_ra** (`numpy 1D array`) - Earth velocity in RA corresponding to the data.
				*   **vearth_dec** (`numpy 1D array`) - Earth velocity in dec corresponding to the data.
				*   **mjd** (`numpy 1D array`, optional) - MJDs corresponding to the data, used for the 'OMDOT' correction. The default is None.
		**Returns:**
				*   Weighted residual of model and data

