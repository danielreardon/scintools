===============================
Scintools
===============================

SCINTOOLS (SCINtillation TOOLS)
is a package for the analysis and simulation of pulsar scintillation data. This code can be used for: processing observed dynamic spectra, computing secondary spectra and ACFs, measuring scintillation arcs, simulating dynamic spectra, and modelling pulsar transverse velocities through scintillation arcs or diffractive timescales. 

* This is currently considered a pre-release only
* Comes with absolutely no warranty
* Free software under MIT license
* `Limited documentation available here <https://scintools.readthedocs.io/en/latest/index.html>`_
* Usage examples located in scintools/examples
* Please email Daniel Reardon for further questions relating to usage: dreardon@swin.edu.au

===============================
Referencing
===============================

If your work makes use of Scintools, please cite `Reardon et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020arXiv200912757R>`_ or the `Astrophysics Source Code Library record <https://ui.adsabs.harvard.edu/abs/2020ascl.soft11019R>`_ and provide a url link to this github page. If utilising scintools for scintillation timescale, bandwidth, or phase gradient measurements via the autocorrelation function, also cite `Reardon et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023arXiv230316338R>`_ . If you use the *ththmod.py* module, cite `Sprenger et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.1114S>`_ and `Baker et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.4573B>`_ . If using the electromagnetic simulation software (*Simulation* class in *scint_sim.py*), also cite `Coles et al. (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...717.1206C>`_

Below is a list of some works that have used Scintools.

* Reardon et al. (2020): `"Precision orbital dynamics from interstellar scintillation arcs for PSR J0437-4715", <https://ui.adsabs.harvard.edu/abs/2020ApJ...904..104R>`_
* Rickett et al. (2021): `"Scintillation Arcs in Pulsar B0450-18", <https://ui.adsabs.harvard.edu/abs/2021ApJ...907...49R>`_
* Wang, Y. et al. (2021): `"ASKAP observations of multiple rapid scintillators reveal a degrees-long plasma filament", <https://ui.adsabs.harvard.edu/abs/2021MNRAS.tmp..186W>`_
* Johnston et al. (2021): `"A supernova remnant association for the fast-moving pulsar PSR J0908-4913", <https://ui.adsabs.harvard.edu/abs/2021MNRAS.507L..41J>`_
* Hamid (2021): `"A Study of Birefringent Scintillation Towards the Millisecond Pulsar J0437-4715", <http://hdl.handle.net/10292/14786>`_
* McKee et al. (2021): `"Probing the local interstellar medium with scintillometry of the bright pulsar B1133+16", <https://ui.adsabs.harvard.edu/abs/2022ApJ...927...99M>`_
* Baker et al. (2022): `"Interstellar interferometry: precise curvature measurement from pulsar secondary spectra", <https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.4573B>`_
* Mall et al. (2022): `"Modelling annual scintillation arc variations in PSR J1643-1224 using the Large European Array for Pulsars", <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.1104M>`_
* Chen et al. (2022): `"Interstellar Scintillation of PSR J2048-1616", <https://ui.adsabs.harvard.edu/abs/2022ApJ...927...14C>`_
* Walker et al. (2022): `"Orbital dynamics and extreme scattering event properties from long-term scintillation observations of PSR J1603-7202", <https://ui.adsabs.harvard.edu/abs/2022ApJ...933...16W>`_
* Zhu et al. (2022): `"Pulsar Double-lensing Sheds Light on the Origin of Extreme Scattering Events", <https://ui.adsabs.harvard.edu/abs/2023ApJ...950..109Z>`_
* Ding et al. (2023): `"The MSPSRπ catalogue: VLBA astrometry of 18 millisecond pulsars", <https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.4982D>`_
* Askew et al. (2022): `"Analysis of the ionized interstellar medium and orbital dynamics of PSR J1909-3744 using scintillation arcs", <https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.5086A>`_
* Reardon et al. (2023) `"Determining electron column density fluctuations in a dominant scattering region using pulsar scintillation", <https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.6392R>`_
* Main et al. (2023): `"Modelling annual scintillation velocity variations of FRB 20201124A", <https://ui.adsabs.harvard.edu/abs/2023MNRAS.522L..36M>`_
* Zhu et al. (2023): `"Pulsar Double Lensing Sheds Light on the Origin of Extreme Scattering Events", <https://ui.adsabs.harvard.edu/abs/2023ApJ...950..109Z>`_
* Baker et al. (2023): `"High-resolution VLBI astrometry of pulsar scintillation screens with the θ - θ transform", <https://ui.adsabs.harvard.edu/abs/2023MNRAS.525..211B>`_
* Yonghua et al. (2023): `"Interstellar scintillation observations for PSR J0835-4510 at 6656 MHz", <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1246X>`_
* Ocker et al. (2024): `"Pulsar Scintillation through Thick and Thin: Bow Shocks, Bubbles, and the Broader Interstellar Medium", <https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.7568O>`_
* Wu et al. (2024): `"Scintillation Arc from FRB 20220912A", <https://ui.adsabs.harvard.edu/abs/2024SCPMA..6719512W>`_
* Turner et al. (2024): `"A Simultaneous Dual-Frequency Scintillation Arc Survey of Six Bright Canonical Pulsars Using the Upgraded Giant Metrewave Radio Telescope", <https://ui.adsabs.harvard.edu/abs/2024ApJ...961..101T>`_
* Wang, Z. et al. (2024): `"Probing the Interstellar Medium from Scintillation of the Swooshing Pulsar B0919+06", <https://ui.adsabs.harvard.edu/abs/2024ApJ...968..109W>`_
* Jang et al. (2024): `"Timing and scintillation studies of PSR J1439−5501", <https://ui.adsabs.harvard.edu/abs/2024A%26A...689A.296J>`_
* Baoda et al. (2024): `"Timing and Scintillation Studies of Pulsars in Globular Cluster M3 (NGC 5272) with FAST", <https://ui.adsabs.harvard.edu/abs/2024ApJ...972...43L>`_
* Turner et al. (2024): `"The Pulsar Science Collaboratory: Multiepoch Scintillation Studies of Pulsars", <https://ui.adsabs.harvard.edu/abs/2024ApJ...977..205T>`_
* Wang, R. et al. (2025): `"Is the Gum Nebula an important interstellar scattering disk of background pulsars?", <https://ui.adsabs.harvard.edu/abs/2025SCPMA..6839512W>`_
* Wang, Y. et al. (2025): `"The Discovery of a 41 s Radio Pulsar PSR J0311+1402 with ASKAP", <https://ui.adsabs.harvard.edu/abs/2025ApJ...982L..53W>`_
* Wang, Z. et al. (2025): `"Frequency-dependent Emission of the Millisecond Pulsar B1937+21 with the Parkes Ultrawideband Receiver", <https://ui.adsabs.harvard.edu/abs/2025ApJ...987...43W>`_
* Reardon et al. (2025): `"Bow shock and Local Bubble plasma unveiled by the scintillating millisecond pulsar J0437‒4715", <https://ui.adsabs.harvard.edu/abs/2025NatAs...9.1053R>`_


Full list is here:https://ui.adsabs.harvard.edu/search/filter_property_fq_property=AND&filter_property_fq_property=property%3A%22refereed%22&fq=%7B!type%3Daqp%20v%3D%24fq_property%7D&fq_property=(property%3A%22refereed%22)&q=%20full%3A%22scintools%22&sort=date%20desc%2C%20bibcode%20desc&p_=0






