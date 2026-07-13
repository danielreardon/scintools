scint_utils module
==================
Astrophysical modelling
-----------------------

.. raw:: html

	<code class="descname">get_ssb_delay</code><span class="sig-paren">(</span><em>mjds, raj, decj, message=True</em><span class="sig-paren">)</span>

\

		Get Romer delay to Solar System Barycentre (SSB) for correction of site arrival times to barycentric.

		**Parameters:**
				*   **mjds** (`numpy 1D array`) - list of modified Julian dates to calculate over
				*   **raj** (`str`) - right ascension (J2000) of the pulsar, e.g. ``'hh:mm:ss'``
				*   **decj** (`str`) - declination (J2000) of the pulsar, e.g. ``'dd:mm:ss'``
				*   **message** (`bool, optional`) - if True, print a reminder that the returned delays should be added to site arrival times
		**Returns:**
				*   List of Romer delays, in seconds
.. raw:: html

	<code class="descname">get_earth_velocity</code><span class="sig-paren">(</span><em>mjds, raj, decj, radial=False</em><span class="sig-paren">)</span>

\

		Calculates the component of Earth's velocity transverse to the line of sight, in RA and DEC. Optionally also returns the radial velocity.

		**Parameters:**
				*   **mjds** (`numpy 1D array`) - list of modified Julian dates to calculate over
				*   **raj** (`str`) - right ascension (J2000) of the pulsar, e.g. ``'hh:mm:ss'``
				*   **decj** (`str`) - declination (J2000) of the pulsar, e.g. ``'dd:mm:ss'``
				*   **radial** (`bool, optional`) - if True, also return Earth's radial velocity component
		**Returns:**
				*   List of Earth velocities in RA, in km/s
				*   List of Earth velocities in dec, in km/s
				*   List of Earth radial velocities, in km/s (only if ``radial=True``)

.. raw:: html

	<code class="descname">make_lsr</code><span class="sig-paren">(</span><em>d, raj, decj, pmra, pmdec, vr=0</em><span class="sig-paren">)</span>

\

		Converts a pulsar's barycentric proper motion and radial velocity to proper motion in the Local Standard of Rest (LSR) frame, using the IAU standard solar motion with respect to the LSR.

		**Parameters:**
				*   **d** (`float`) - distance to the pulsar, in kpc
				*   **raj** (`str`) - right ascension (J2000) of the pulsar, e.g. ``'hh:mm:ss'``
				*   **decj** (`str`) - declination (J2000) of the pulsar, e.g. ``'dd:mm:ss'``
				*   **pmra** (`float`) - proper motion in right ascension (``pm_ra_cosdec``), in mas/yr
				*   **pmdec** (`float`) - proper motion in declination, in mas/yr
				*   **vr** (`float, optional`) - radial velocity, in km/s. The default is 0.
		**Returns:**
				*   Proper motion components (RA, Dec) of the pulsar in the LSR frame, in mas/yr

.. raw:: html

	<code class="descname">get_true_anomaly</code><span class="sig-paren">(</span><em>mjds, pars</em><span class="sig-paren">)</span>

\

		Calculates true anomalies for an array of barycentric MJDs and a parameter dictionary

		**Parameters:** 
				*   **mjds** (`numpy 1D array`) - list of modified Julian dates to calculate over
				*   **pars** (`dict`) - parameter file containing the orbital eccentricity ('ECC'), binary period ('PB'), MJD of binary period measurement ('T0'), and binary period derivative, :math:`\dot P` ('PBDOT').
		**Returns:**
				*   List of true anomalies

.. raw:: html

	<code class="descname">get_binphase</code><span class="sig-paren">(</span><em>mjds, pars</em><span class="sig-paren">)</span>

\

		Calculates binary phase (true anomaly plus argument of periastron) for an array of barycentric MJDs and a parameter dictionary.

		**Parameters:**
				*   **mjds** (`numpy 1D array`) - list of modified Julian dates to calculate over
				*   **pars** (`dict`) - parameter dictionary as used by ``get_true_anomaly()``, plus (for non-ELL1 models) the argument of periastron ``OM`` (degrees) and optionally its derivative ``OMDOT`` (deg/yr)
		**Returns:**
				*   List of binary phases, in radians

.. raw:: html

	<code class="descname">differential_velocity</code><span class="sig-paren">(</span><em>params, sun_velocity=220, screen_velocity=220, radius=8</em><span class="sig-paren">)</span>

\

		Approximates the differential velocity between the scattering screen and the Sun assuming zero-inclination circular galactic orbits. Useful for determining the intrinsic ISM velocity.

		**Parameters:**
				*   **params** (`dict`) - parameters containing the pulsar RAJ and DECJ, pulsar distance ``d``, screen fractional distance ``s``, and anisotropy angle ``psi``
				*   **sun_velocity** (`float, optional`) - orbital speed of the Sun in km/s. The default is 220.
				*   **screen_velocity** (`float, optional`) - orbital speed of the scattering screen in km/s, assuming a flat galactic rotation curve. The default is 220.
				*   **radius** (`float, optional`) - radius of the Sun's orbit about the galactic center, in kpc. The default is 8.
		**Returns:**
				*   RA component of the differential velocity, in km/s
				*   Dec component of the differential velocity, in km/s

.. raw:: html

	<code class="descname">scint_velocity</code><span class="sig-paren">(</span><em>params, dnu, tau, freq, dnuerr=None, tauerr=None, a=2.53e4</em><span class="sig-paren">)</span>

\

		Calculate the scintillation velocity from the ACF scintillation bandwidth and timescale, using the thin-screen scintillation velocity equation.

		**Parameters:**
				*   **params** (`dict or lmfit Parameters, optional`) - parameters containing the pulsar distance ``d`` and screen fractional distance ``s`` (and their uncertainties). If None, ``a`` is used directly as the thin-screen coefficient.
				*   **dnu** (`float or numpy array`) - scintillation bandwidth, in MHz
				*   **tau** (`float or numpy array`) - scintillation timescale, in seconds
				*   **freq** (`float or numpy array`) - observing frequency, in MHz
				*   **dnuerr** (`float or numpy array, optional`) - uncertainty on ``dnu``
				*   **tauerr** (`float or numpy array, optional`) - uncertainty on ``tau``
				*   **a** (`float, optional`) - fixed thin-screen coefficient used when ``params`` is None. The default is 2.53e4.
		**Returns:**
				*   Scintillation velocity, in km/s
				*   Uncertainty on the scintillation velocity, in km/s (only if ``dnuerr`` and ``tauerr`` are both given)

.. raw:: html

	<code class="descname">acor</code><span class="sig-paren">(</span><em>arr</em><span class="sig-paren">)</span>

\

		Calculates the characteristic (50%) autocorrelation length of an array, e.g. a time series.

		**Parameters:**
				*   **arr** (`array`) - array of numbers, e.g. a time series
		**Returns:**
				*   Characteristic (50%) autocorrelation length (int)

Reading and writing data
------------------------

.. raw:: html

	<code class="descname">read_dynlist</code><span class="sig-paren">(</span><em>file_path</em><span class="sig-paren">)</span>

\

		Reads list of dynamic spectra filenames from path

		**Parameters:** 
				*   **file_path** (`str`) - file path containing the dynamic spectra files
		**Returns:** 
				*   List of dynamic spectrum file paths

.. raw:: html

	<code class="descname">write_results</code><span class="sig-paren">(</span><em>filename, dyn=None</em><span class="sig-paren">)</span>

\

		Appends dynamic spectrum information and parameters of interest to file

		**Parameters:** 
				*   **filename** (`str`) - path of the file to write to
				*   **dyn** (`Dynspec object`) - Dynspec object

.. raw:: html

	<code class="descname">read_results</code><span class="sig-paren">(</span><em>filename</em><span class="sig-paren">)</span>

\

		Reads a CSV results file written by `write_results()`

		**Parameters:**
				*   **filename** (`str`) - path of the file to read from
		**Returns:**
				*   Dictionary of parameters from file

.. raw:: html

	<code class="descname">search_and_replace</code><span class="sig-paren">(</span><em>filename, search, replace</em><span class="sig-paren">)</span>

\

		Reads a text file, replaces all occurrences of one substring with another, and overwrites the file with the result.

		**Parameters:**
				*   **filename** (`str`) - path to the text file to edit in place
				*   **search** (`str`) - substring to search for
				*   **replace** (`str`) - substring to substitute in place of ``search``

.. raw:: html

	<code class="descname">save_fits</code><span class="sig-paren">(</span><em>filename, dyn</em><span class="sig-paren">)</span>

\

		Saves a dynamic spectrum to a FITS file as the primary HDU.

		**Parameters:**
				*   **filename** (`str`) - path of the FITS file to write (must not already exist)
				*   **dyn** (`Dynspec object`) - Dynspec object whose ``dyn`` array is written to the FITS file

.. raw:: html

	<code class="descname">float_array_from_dict</code><span class="sig-paren">(</span><em>dictionary, key</em><span class="sig-paren">)</span>

\

		Convert an array stored in dictionary to a numpy array

		**Parameters:** 
				*   **dictionary** (`dict`) - dictionary containing the array
				*   **key** (`str`) - key of the array
		**Returns:** 
				*   numpy array from stored array

.. raw:: html

	<code class="descname">read_par</code><span class="sig-paren">(</span><em>parfile</em><span class="sig-paren">)</span>

\

		Reads a parameter file and return a dictionary of parameter names and values

		**Parameters:** 
				*   **parfile** (`str`) - path to parameter file for conversion
		**Returns:** 
				*   resulting dictionary

.. raw:: html

	<code class="descname">pars_to_params</code><span class="sig-paren">(</span><em>pars, params=None</em><span class="sig-paren">)</span>

\

		Converts a dictionary of parameter file parameters from ``read_par()`` to an ``lmfit`` ``Parameters()`` object to use in models. By default, parameters are not varied.

		**Parameters:** 
				*   **pars** (`dict`) - dictionary of parameters
				*   **params** (`lmfit Parameters() object, optional`) - ``lmfit`` ``Parameters()`` object to append parameters to. If None, initializes new object.
		**Returns:** 
				*   appended ``lmfit`` ``Parameters()`` object

Other utilities
---------------

.. raw:: html

	<code class="descname">clean_archive</code><span class="sig-paren">(</span><em>archive, template=None, bandwagon=0.99, channel_threshold=5, subint_threshold=5, output_directory=None</em><span class="sig-paren">)</span>

\

		Cleans a psrchive archive using the ``coast_guard`` ``surgical`` and ``bandwagon`` cleaners, then unloads the cleaned archive to disk with a ``.clean`` extension.

		**Parameters:**
				*   **archive** (`str or psrchive archive object`) - path to, or already-loaded psrchive archive object, to be cleaned
				*   **template** (`str, optional`) - path to a template profile. Currently unused by this function, reserved for future use with the cleaners.
				*   **bandwagon** (`float, optional`) - ``badchantol`` value passed to the ``bandwagon`` cleaner. The default is 0.99.
				*   **channel_threshold** (`float, optional`) - ``chanthresh`` (sigma threshold for flagging channels) passed to the ``surgical`` cleaner. The default is 5.
				*   **subint_threshold** (`float, optional`) - ``subintthresh`` (sigma threshold for flagging subintegrations) passed to the ``surgical`` cleaner. The default is 5.
				*   **output_directory** (`str, optional`) - directory to output the cleaned archive. If None, writes alongside the input archive.

.. raw:: html

	<code class="descname">is_valid</code><span class="sig-paren">(</span><em>array</em><span class="sig-paren">)</span>

\

		Returns boolean array of values that are finite an not nan.

		**Parameters:** 
				*   **array** (`numpy ndarray`) - input array
		**Returns:** 
				*   Boolean ndarray

.. raw:: html

	<code class="descname">slow_FT</code><span class="sig-paren">(</span><em>dynspec, freqs</em><span class="sig-paren">)</span>

\

		Slow FT of dynamic spectrum along points of t*(f / fref), account for phase scaling of f_D. Given a uniform t axis, this reduces to a regular FT.

		Uses Olaf's c-implemation if possible, otherwise reverts to a slow, pure Python/numpy method.

    		Reference freq is currently hardcoded to the middle of the band.

		**Parameters:** 
				*   **dynspec** (`numpy 2D array`) - input dynamic spectrum
				*   **freqs** (`numpy 1D array`) - frequency axis of the dynamic spectrum in MHz
		**Returns:** 
				*   Fourier-transformed dynamic spectrum

.. raw:: html

	<code class="descname">svd_model</code><span class="sig-paren">(</span><em>arr, nmodes=1</em><span class="sig-paren">)</span>

\

		Take SVD of a dynamic spectrum, divide by the largest N modes

		**Parameters:**
				*   **arr** (`numpy 2D array`) - time/freq visibility matrix (input dynamic spectrum)
				*   **nmodes** (`int, optional`) - number of leading singular values (modes) to keep when constructing the model; higher-order modes are zeroed. The default is 1.
		**Returns:**
				*   Array divided by absolute value of SVD model
				*   SVD model

.. raw:: html

	<code class="descname">make_dynspec</code><span class="sig-paren">(</span><em>archive, template=None, phasebin=1</em><span class="sig-paren">)</span>

\

		Creates a `psrflux`-format dynamic spectrum from an archive ``$ psrflux -s [template] -e dynspec [archive]``. **Note:** this function is a placeholder for future functionality and is not yet implemented.

		**Parameters:**
				*   **archive** (`str or psrchive archive object`) - path to, or already-loaded psrchive archive object, to create the dynamic spectrum from
				*   **template** (`str, optional`) - path to a template profile, passed to ``psrflux -s``
				*   **phasebin** (`int, optional`) - number of pulse phase bins to integrate over when forming the dynamic spectrum. The default is 1.

.. raw:: html

	<code class="descname">make_pickle</code><span class="sig-paren">(</span><em>obj, filepath</em><span class="sig-paren">)</span>

\

		Pickles ``obj`` to ``filepath``, writing in chunks so that this also works for objects whose pickled representation is larger than 2GB.

		**Parameters:**
				*   **obj** (`object`) - Python object to pickle (e.g. a Dynspec object)
				*   **filepath** (`str`) - path of the file to write the pickle to

.. raw:: html

	<code class="descname">load_pickle</code><span class="sig-paren">(</span><em>filepath</em><span class="sig-paren">)</span>

\

		Loads a pickled object from ``filepath``, reading in chunks so that this also works for pickle files larger than 2GB. Counterpart to ``make_pickle()``.

		**Parameters:**
				*   **filepath** (`str`) - path of the pickle file to read
		**Returns:**
				*   The unpickled Python object

.. raw:: html

	<code class="descname">autocorr</code><span class="sig-paren">(</span><em>arr</em><span class="sig-paren">)</span>

\

		Does a slow (direct, nested-loop) calculation of the 2D autocorrelation of an input masked array.

		**Parameters:**
				*   **arr** (`numpy masked array`) - 2D masked array to autocorrelate
		**Returns:**
				*   2D autocorrelation of ``arr``, normalised so the maximum value is 1

.. raw:: html

	<code class="descname">cov_to_corr</code><span class="sig-paren">(</span><em>cov</em><span class="sig-paren">)</span>

\

		Calculates the correlation matrix from a covariance matrix.

		**Parameters:**
				*   **cov** (`numpy 2D array`) - covariance matrix
		**Returns:**
				*   Correlation matrix corresponding to ``cov``

.. raw:: html

	<code class="descname">difference</code><span class="sig-paren">(</span><em>x</em><span class="sig-paren">)</span>

\

		Unlike ``numpy.diff``, computes differences between the centres of neighbouring elements in ``x``, returning an array the same size as ``x``.

		**Parameters:**
				*   **x** (`array_like`) - input 1D array
		**Returns:**
				*   Array of centred differences, the same length as ``x``

.. raw:: html

	<code class="descname">mjd_to_year</code><span class="sig-paren">(</span><em>mjd</em><span class="sig-paren">)</span>

\

		Converts Modified Julian Date(s) to a decimal (Besselian) year.

		**Parameters:**
				*   **mjd** (`float or array_like`) - Modified Julian Date(s) to convert
		**Returns:**
				*   Decimal year(s) corresponding to ``mjd``

.. raw:: html

	<code class="descname">find_nearest</code><span class="sig-paren">(</span><em>arr, val</em><span class="sig-paren">)</span>

\

		Returns the index of an array (``arr``) that is nearest to value (``val``).

		**Parameters:**
				*   **arr** (`array_like`) - array to search
				*   **val** (`float`) - value to find the nearest element to
		**Returns:**
				*   Index of the element of ``arr`` closest to ``val``

.. raw:: html

	<code class="descname">longest_run_of_zeros</code><span class="sig-paren">(</span><em>arr</em><span class="sig-paren">)</span>

\

		Finds the length of the longest run of consecutive zeros in an array.

		**Parameters:**
				*   **arr** (`array_like`) - input 1D array (or iterable) of numbers
		**Returns:**
				*   Length of the longest consecutive run of zero-valued elements in ``arr``

.. raw:: html

	<code class="descname">interp_nan_2d</code><span class="sig-paren">(</span><em>array, method='linear'</em><span class="sig-paren">)</span>

\

		Fills in NaN (and other invalid) values of a 2D array by interpolating from the surrounding valid pixels.

		**Parameters:**
				*   **array** (`array_like`) - 2D input array possibly containing NaN or otherwise invalid values
				*   **method** (`str, optional`) - interpolation method (``'linear'``, ``'nearest'``, or ``'cubic'``). The default is ``'linear'``.
		**Returns:**
				*   2D array with invalid values filled in by interpolation

.. raw:: html

	<code class="descname">centres_to_edges</code><span class="sig-paren">(</span><em>arr</em><span class="sig-paren">)</span>

\

		Takes an array of pixel-centres, and returns an array of pixel-edges. Assumes the pixel-centres are evenly spaced.

		**Parameters:**
				*   **arr** (`array_like`) - 1D array of evenly-spaced pixel-centre values
		**Returns:**
				*   Array of pixel-edge values, one longer than ``arr``

.. raw:: html

	<code class="descname">get_window</code><span class="sig-paren">(</span><em>nt, nf, window='hanning', frac=0.1</em><span class="sig-paren">)</span>

\

		Returns tapering windows to apply along the time and frequency axes before computing an FFT.

		**Parameters:**
				*   **nt** (`int`) - length of the time axis
				*   **nf** (`int`) - length of the frequency axis
				*   **window** (`str, optional`) - window type: ``'hanning'``, ``'hamming'``, ``'blackman'``, or ``'bartlett'``. The default is ``'hanning'``.
				*   **frac** (`float, optional`) - fraction of each axis length to taper at the edges. The default is 0.1.
		**Returns:**
				*   Tapering window of length ``nt``, for the time axis
				*   Tapering window of length ``nf``, for the frequency axis

.. raw:: html

	<code class="descname">calculate_curvature_peak_probability</code><span class="sig-paren">(</span><em>power_data, noise_level, smooth=True, curvatures=None, log=False</em><span class="sig-paren">)</span>

\

		Calculates the (unnormalised) Gaussian probability distribution of the arc curvature "power vs curvature" profile, modelling the profile as a Gaussian peak at its maximum with standard deviation given by ``noise_level``.

		**Parameters:**
				*   **power_data** (`array_like`) - power (Doppler profile) as a function of curvature, for one or more observations
				*   **noise_level** (`float or array_like`) - noise standard deviation of ``power_data``, used as the Gaussian width (and smoothing sigma if ``smooth=True``)
				*   **smooth** (`bool, optional`) - if True, smooth ``power_data`` with a Gaussian filter of sigma ``noise_level`` before computing the probability. The default is True.
				*   **curvatures** (`array_like, optional`) - curvature values corresponding to ``power_data``. Currently unused.
				*   **log** (`bool, optional`) - if True, return the log-probability instead of the probability. The default is False.
		**Returns:**
				*   (Log-)probability distribution, same shape as ``power_data``

.. raw:: html

	<code class="descname">save_curvature_data</code><span class="sig-paren">(</span><em>dyn, filename=None</em><span class="sig-paren">)</span>

\

		Saves the "power vs curvature" arc-fitting data and noise level to a ``.npz`` file, for later use with ``calculate_curvature_peak_probability()`` or ``curvature_log_likelihood()``.

		**Parameters:**
				*   **dyn** (`Dynspec object`) - Dynspec object with arc-curvature fitting results (e.g. ``eta_array``, ``norm_sspec_avg``, and ``noise``)
				*   **filename** (`str, optional`) - output file path. If None, defaults to ``dyn.name + 'curvature_data'``.

.. raw:: html

	<code class="descname">curvature_log_likelihood</code><span class="sig-paren">(</span><em>power, nfdop, noise, model_nfdop</em><span class="sig-paren">)</span>

\

		Calculates the log likelihood of a model prediction for ``nfdop`` by taking the likelihood function for each observation to be a probability density calculated from the doppler profile.

		**Parameters:**
				*   **power** (`array_like`) - doppler profile(s)
				*   **nfdop** (`array_like`) - nfdop values for doppler profile(s)
				*   **noise** (`float or array_like`) - noise value for each profile
				*   **model_nfdop** (`float or array_like`) - model prediction for nfdop for each profile
		**Returns:**
				*   Log likelihood of the input data (float)

