ACF class
================

The scintools ``ACF`` class, found in the ``scint_sim`` module, features tools for simulating and plotting autocorrelation functions. Based on the theoretical treatment found in Appendix A of `Rickett, Coles et al. (2014) <https://iopscience.iop.org/article/10.1088/0004-637X/787/2/161/>`_.

.. raw:: html

	<em class="property">class </em><code class="descclassname">scintools.</code><code class="descname">ACF</code><span class="sig-paren">(</span><em>s_max=5, dnu_max=5, ns=256, nf=256, ar=1, alpha=5/3, phasegrad_x=0, phasegrad_y=0, V_x=1, V_y=0, psi=0, amp=1</em><span class="sig-paren">)</span>

\
		Simulation class.

		**Parameters:** 
				*   **s_max** (`float, optional`) - number of coherence spatial scales to calculate over.
				*   **dnu_max** (`float, optional`) - number of decorrelation bandwidths to calculate over.
				*   **ns** (`int, optional`) - number of spatial steps
				*   **nf** (`int, optional`) - number of frequency steps.
				*   **ar** (`float, optional`) - axial ratio of anisotropy
				*   **alpha** (`float, optional`) - structure function exponent. 5/3 is a Kolmogorov profile (default) while 2 is Gaussian.
				*   **phasegrad_x** (`float, optional`) - phase gradient in x
				*   **phasegrad_y** (`float, optional`) - phase gradient in y
				*   **V_x** (`float, optional`) - effective velocity in the x direction
				*   **V_y** (`float, optional`) - effective velocity in the y direction
				*   **psi** (`float, optional`) - orientation of anisotropy in degrees
				*   **amp** (`float, optional`) - amplitude to scale the ACF by. By default ACF is normalized to the range [0, 1].

Methods
-------

.. raw:: html

	<code class="descname">calc_acf</code><span class="sig-paren">(</span><em>plot=False</em><span class="sig-paren">)</span>

\

        	Computes 2D ACF of intensity vs t and v where optimal sampling of t and v is provided with the output ACF.

		**Parameters:** 
				*   **plot** (`bool, optional`) - plot the simulated ACF

.. raw:: html

	<code class="descname">plot_acf</code><span class="sig-paren">(</span><em></em><span class="sig-paren">)</span>

\

        	Plot the simulated ACF

