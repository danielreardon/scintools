scint_utils module
==================
Astrophysical modelling
-----------------------

.. raw:: html

	<code class="descname">get_ssb_delay</code><span class="sig-paren">(</span><em>mjds, raj, decj</em><span class="sig-paren">)</span>

\

		Get Romer delay to Solar System Barycentre (SSB) for correction of site arrival times to barycentric.

		**Parameters:** 
				*   **mjds** (`numpy 1D array`) - list of modified Julian dates to calculate over
				*   **raj** (`float`) - right ascension (J2000) of the pulsar in degrees
				*   **decj** (`float`) - declination (J2000) of the pulsar in degrees
		**Returns:** 
				*   List of Romer delays
.. raw:: html

	<code class="descname">get_earth_velocity</code><span class="sig-paren">(</span><em>mjds, raj, decj</em><span class="sig-paren">)</span>

\

		Calculates the component of Earth's velocity transverse to the line of sight, in RA and DEC

		**Parameters:** 
				*   **mjds** (`numpy 1D array`) - list of modified Julian dates to calculate over
				*   **raj** (`float`) - right ascension (J2000) of the pulsar in degrees
				*   **decj** (`float`) - declination (J2000) of the pulsar in degrees
		**Returns:** 
				*   List of Earth velocities in RA
				*   List of Earth velocities in dec

.. raw:: html

	<code class="descname">get_true_anomaly</code><span class="sig-paren">(</span><em>mjds, pars</em><span class="sig-paren">)</span>

\

		Calculates true anomalies for an array of barycentric MJDs and a parameter dictionary

		**Parameters:** 
				*   **mjds** (`numpy 1D array`) - list of modified Julian dates to calculate over
				*   **pars** (`dict`) - parameter file containing the orbital eccentricity ('ECC'), binary period ('PB'), MJD of binary period measurement ('T0'), and binary period derivative, :math:`\dot P` ('PBDOT').
		**Returns:** 
				*   List of true anomalies

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

	<code class="descname">clean_archive</code><span class="sig-paren">(</span><em>archive, template=None, bandwagon=0.99, channel_threshold=7, subint_threshold=5, output_directory=None</em><span class="sig-paren">)</span>

\

		Cleans a psrchive archive object using ``coast_guard``.

		**Parameters:** 
				*   **archive** (`psarchive archive object`) - psarchive archive
				*   **template** (`str, optional`) - 
				*   **bandwagon** (`float, optional`) - bandwagon value
				*   **channel_threshold** (`float, optional`) - channel threshold
				*   **subint_threshold** (`float, optional`) - sub-integration threshold
				*   **output_directory** (`str`) - directory to output the cleaned archive

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
				*   **arr** (`numpy 2D array`) - input dynamic spectrum
				*   **nmodes** (`int, optional`) - number of singular values to compute up to.
		**Returns:** 
				*   Array divided by absolute value of SVD model
				*   SVD model

.. raw:: html

	<code class="descname">make_dynspec</code><span class="sig-paren">(</span><em>archive, template=None, phasebin=1</em><span class="sig-paren">)</span>

\

		Creates a `psrflux`-format dynamic spectrum from an archive ``$ psrflux -s [template] -e dynspec [archive]``

		**Parameters:** 
				*   **archive** (`psarchive archive object`) - psarchive archive
				*   **template** (`str, optional`) - 
				*   **phasebin** (`int, optional`) - 

.. raw:: html

	<code class="descname">remove_duplicates</code><span class="sig-paren">(</span><em>dyn_files</em><span class="sig-paren">)</span>

\

		Filters out dynamic spectra from simultaneous observations.

		**Parameters:** 
				*   **dyn_files** (`list or numpy 1D array or str`) - list of dynamic spectrum file paths

		**Returns:** 
				*   Filtered list of dynamic spectrum file paths

.. raw:: html

	<code class="descname">make_pickle</code><span class="sig-paren">(</span><em>dyn, process=True, sspec=True, acf=True, lamsteps=True</em><span class="sig-paren">)</span>

\

		Pickles a dynamic spectrum object.

		**Parameters:** 
				*   **dyn** (`Dynspec object`) - dynamic spectrum Dynspec object.
				*   **process** (`bool, optional`) - perform default processing. Involves trimming the edges, refilling, correction, and calculation of the ACF and secondary spectrum.
				*   **sspec** (`numpy 2D array, optional`) - input secondary spectrum.
				*   **acf** (`numpy 2D array, optional`) - input autocorrelation function.
				*   **lamsteps** (`bool, optional`) - option to use wavelength steps instead of default frequency steps.

