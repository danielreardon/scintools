scint_models module
===================

The ``scint_models`` module contains various functions for modelling and fitting scintillation data.

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
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), timescale at :math:`1/e` width ('tau'), decorrelation bandwidth at half power ('dnu'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), white-noise spike ('wn'), phase gradient in x and y ('phasegrad_x', 'phasegrad_y'), axial ratio of anisotropy ('ar'), orientation of anisotropy relative to y ('psi'), and effective velocity in x and y ('v_ra', 'v_dec'), as well as the total bandwidth of the observation ('bw'), total observation time ('tobs'), and half the number of sub-integrations and channels in the ACF ('nt', 'nf').
				*   **ydata** (`numpy 1D array`) - ACF cropped symmetrically around its center to a desired range.
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">tau_sspec_model</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Models a 1D cut through the center of the ACF along the time axis and applies a Fourier transform.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), timescale at :math:`1/e` width ('tau'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), and white-noise spike ('wn').
				*   **xdata** (`numpy 1D array`) - time of sub-integrations from the center of the ACF to the maximum time.
				*   **ydata** (`numpy 1D array`) - profile from secondary spectrum corresponding to the ACF to model summed along all columns (all :math:`f_t`).
				*   **weights** (`numpy 1D array`) - weights of the data
		**Returns:** 
				*   Weighted residual of model and data

.. raw:: html

	<code class="descname">dnu_sspec_model</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Models a 1D cut through the center of the ACF along the frequency axis and applies a Fourier tranform.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the current best-fit amplitude ('amp'), decorrelation bandwidth at half power ('tau'), index of exponential function ('alpha', 2 is Gaussian, 5/3 is Kolmogorov), and white-noise spike ('wn').
				*   **xdata** (`numpy 1D array`) - frequency of the channels from the center of the ACF to the maximum frequency.
				*   **ydata** (`numpy 1D array`) - profile from secondary spectrum corresponding to the ACF to model summed along all rows (all :math:`f_\tau` or :math:`f_\lambda`).
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

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

	<code class="descname">arc_curvature</code><span class="sig-paren">(</span><em>params, ydata, weights, true_anomaly, vearth_ra, vearth_dec</em><span class="sig-paren">)</span>

\

		Models the arc curvature and returns the residuals.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing current best-fit values for the distance to the pulsar ('d'), the fractional screen distance ('s'), and optionally the ISM velocity in RA and dec ('vism_ra', 'vism_dec').
				*   **ydata** (`numpy 1D array`) - arc curvature data.
				*   **weights** (`numpy 1D array`) - weights of the data.
				*   **true_anomaly** (`numpy 1D array`) - true anolmalies corresponding to the data.
				*   **vearth_ra** (`numpy 1D array`) - Earth velocity in RA corresponding to the data.
				*   **vearth_dec** (`numpy 1D array`) - Earth velocity in dec corresponding to the data.
		**Returns:** 
				*   Weighted residual of model and data

Velocity models
---------------

.. raw:: html

	<code class="descname">effective_velocity_annual</code><span class="sig-paren">(</span><em>params, true_anomaly, vearth_ra, vearth_dec</em><span class="sig-paren">)</span>

\

		Models the arc curvature and returns the residuals.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object containing the pulsar's projected semi-major axis in lt-s ('A1'), orbital period in days ('PB'), orbital eccentricity ('ECC'), longitude of periastron in degrees ('OM'), inclination in degrees ('KIN'), longitude of ascending node in degrees ('KOM'), and optionally the proper motion in RA and dec ('PMRA', 'PMDEC'), as well as the distance to the pulsar ('d') and the fractional screen distance ('s').
				*   **true_anomaly** (`numpy 1D array`) - true anolmalies to compute over.
				*   **vearth_ra** (`numpy 1D array`) - Earth velocity in RA to compute over.
				*   **vearth_dec** (`numpy 1D array`) - Earth velocity in dec to compute over.
		**Returns:** 
				*   Effective velocity in RA
				*   Effective velocity in dec
				*   Pulsar velocity in RA
				*   Pulsar velocity in dec

.. raw:: html

	<code class="descname">thin_screen</code><span class="sig-paren">(</span><em>params, xdata, ydata, weights</em><span class="sig-paren">)</span>

\

		Models the thin screen effective velocity and returns the residuals.

		**Parameters:** 
				*   **params** (`lmfit Parameters() object`) - ``lmfit`` ``Parameters()`` object.
				*   **xdata** (`numpy 1D array`) - 
				*   **ydata** (`numpy 1D array`) - screen effective velocity data
				*   **weights** (`numpy 1D array`) - weights of the data.
		**Returns:** 
				*   Weighted residual of model and data

