#!/usr/bin/env python

"""
ththmod.py
----------------------------------
Code for handling theta-theta transformation by Daniel Baker
"""

import numpy as np
import astropy.units as u
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.optimize import curve_fit
import warnings


def svd_model(arr, nmodes=1):
    """
    Model a matrix using the first nmodes modes of the singular value
    decomposition

    Parameters
    ----------
    arr : 2D Array
        The array to be modeled with the SVD
    nmodes: int, optional
        Number of modes used in the SVD model. Defaults to 1
    """
    u, s, w = np.linalg.svd(arr)
    s[nmodes:] = 0
    S = np.zeros(([len(u), len(w)]), np.complex128)
    S[: len(s), : len(s)] = np.diag(s)
    model = np.dot(np.dot(u, S), w)
    return model


def chi_par(x, A, x0, C):
    """
    Parabola for fitting to chisq curve.

    Parameters
    ----------
    x : 1D Array
        X coordinates of the
    A : float
        Coefficient for the quadratic term
    x0 : float
        X coordinate of apex
    X : float
        Y Coordinate of apex
    """
    return A * (x - x0) ** 2 + C


def thth_map(CS, tau, fd, eta, edges, hermetian=True):
    """
    Maping from Conjugate Spectrum space to theta-theta space

    Parameters
    ----------
    CS : 2D Array
        Conjugate Spectrum
    tau : 1D Array
        Time delay coordinates for the CS (us).
    fd : 1D Array
        Doppler shift coordinates for the CS (mHz)
    eta : float
        Arc curvature (s**3)
    edges: 1D Array
        Bin edges for theta-theta mapping (mHz). Should have an even number of
        points and be symmetric about 0
    hermetian: bool, optional
        Force theta-theta to be hermetian symmetric
        (as expected from dynamic spectra). Defaults to True
    """

    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)
    # Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    # Calculate theta1 and th2 arrays
    th1 = np.ones((th_cents.shape[0], th_cents.shape[0])) * th_cents
    th2 = th1.T

    # tau and fd step sizes
    dtau = np.diff(tau).mean()
    dfd = np.diff(fd).mean()

    # Find bin in CS space that each point maps back to
    tau_inv = (
        ((eta * (th1**2 - th2**2)) - tau[0] + dtau / 2) // dtau
    ).astype(int)
    fd_inv = (((th1 - th2) - fd[0] + dfd / 2) // dfd).astype(int)

    # Define thth
    thth = np.zeros(tau_inv.shape, dtype=complex)

    # Only fill thth points that are within the CS
    pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fd_inv < fd.shape[0])
    thth[pnts] = CS[tau_inv[pnts], fd_inv[pnts]]

    # Preserve flux (int
    thth *= np.sqrt(np.abs(2 * eta * (th2 - th1)).value)
    if hermetian:
        # Force Hermetian
        thth -= np.tril(thth)
        thth += np.conjugate(np.triu(thth).T)
        thth -= np.diag(np.diag(thth))
        thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
        thth = np.nan_to_num(thth)

    return thth


def thth_redmap(CS, tau, fd, eta, edges, hermetian=True):
    """
    Map from Conjugate Spectrum to theta-theta space for the largest
    possible filled in sqaure within edges

    Parameters
    ----------
    CS : 2D Array
        Conjugate Spectrum
    tau : 1D Array
        Time delay coordinates for the CS (us).
    fd : 1D Array
        Doppler shift coordinates for the CS (mHz)
    eta : float
        Arc curvature (s**3)
    edges: 1D Array
        Bin edges for theta-theta mapping (mHz). Should have an even number of
        points and be symmetric about 0
    hermetian: bool, optional
        Force theta-theta to be hermetian symmetric
        (as expected from dynamic spectra). Defaults to True
    """

    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    # Find full thth
    thth = thth_map(CS, tau, fd, eta, edges, hermetian)

    # Find region that is fully within CS
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    th_pnts = ((th_cents**2) * eta < np.abs(tau.max())) * (
        np.abs(th_cents) < np.abs(fd.max()) / 2
    )
    thth_red = thth[th_pnts, :][:, th_pnts]
    edges_red = th_cents[th_pnts]
    edges_red = (edges_red[:-1] + edges_red[1:]) / 2
    edges_red = (
        np.concatenate(
            (
                np.array(
                    [edges_red[0].value - np.diff(edges_red.value).mean()]
                ),
                edges_red.value,
                np.array(
                    [edges_red[-1].value + np.diff(edges_red.value).mean()]
                ),
            )
        )
        * edges_red.unit
    )
    return thth_red, edges_red


def rev_map(thth, tau, fd, eta, edges, hermetian=True):
    """
    Inverse map from theta-theta to Conjugate Spectrum space.

    Parameters
    ----------
    thth : 2D Array
        Theta-theta matrix to be mapped
    tau : 1D Array
        Time delay coordinates for the CS (us).
    fd : 1D Array
        Doppler shift coordinates for the CS (mHz)
    eta : float
        Arc curvature (s**3)
    edges: 1D Array
        Bin edges for theta-theta mapping (mHz). Should have an even number of
        points and be symmetric about 0
    hermetian: bool, optional
        Force CS to be hermetian symmetric
        (as expected from dynamic spectra). Defaults to True
    """

    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    # Find bin centers
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]

    fd_map = th_cents[np.newaxis, :] - th_cents[:, np.newaxis]
    tau_map = eta.value * (
        th_cents[np.newaxis, :] ** 2 - th_cents[:, np.newaxis] ** 2
    )
    fd_edges = (np.linspace(0, fd.shape[0], fd.shape[0] + 1) - 0.5) * (
        fd[1] - fd[0]
    ).value + fd[0].value
    tau_edges = (np.linspace(0, tau.shape[0], tau.shape[0] + 1) - 0.5) * (
        tau[1] - tau[0]
    ).value + tau[0].value

    # Bind TH-TH points back into Conjugate Spectrum
    with np.errstate(divide="ignore", invalid="ignore"):
        recov = (
            np.histogram2d(
                np.ravel(fd_map.value),
                np.ravel(tau_map.value),
                bins=(fd_edges, tau_edges),
                weights=np.ravel(
                    thth / np.sqrt(np.abs(2 * eta * fd_map.T).value)
                ).real,
            )[0]
            + np.histogram2d(
                np.ravel(fd_map.value),
                np.ravel(tau_map.value),
                bins=(fd_edges, tau_edges),
                weights=np.ravel(
                    thth / np.sqrt(np.abs(2 * eta * fd_map.T).value)
                ).imag,
            )[0]
            * 1j
        )
        norm = np.histogram2d(
            np.ravel(fd_map.value),
            np.ravel(tau_map.value),
            bins=(fd_edges, tau_edges),
        )[0]
        if hermetian:
            recov += (
                np.histogram2d(
                    np.ravel(-fd_map.value),
                    np.ravel(-tau_map.value),
                    bins=(fd_edges, tau_edges),
                    weights=np.ravel(
                        thth / np.sqrt(np.abs(2 * eta * fd_map.T).value)
                    ).real,
                )[0]
                - np.histogram2d(
                    np.ravel(-fd_map.value),
                    np.ravel(-tau_map.value),
                    bins=(fd_edges, tau_edges),
                    weights=np.ravel(
                        thth / np.sqrt(np.abs(2 * eta * fd_map.T).value)
                    ).imag,
                )[0]
                * 1j
            )
            norm += np.histogram2d(
                np.ravel(-fd_map.value),
                np.ravel(-tau_map.value),
                bins=(fd_edges, tau_edges),
            )[0]
        recov /= norm
        recov = np.nan_to_num(recov)
    return recov.T


def modeler(CS, tau, fd, eta, edges, hermetian=True):
    """
    Create theta-theta array as well as model theta-theta, Conjugate Spectrum
    and Dynamic Spectrum from data conjugate spectrum and curvature

    Parameters
    ----------
    CS : 2D Array
        Theta-theta matrix to be mapped
    tau : 1D Array
        Time delay coordinates for the CS (us).
    fd : 1D Array
        Doppler shift coordinates for the CS (mHz)
    eta : float
        Arc curvature (s**3)
    edges: 1D Array
        Bin edges for theta-theta mapping (mHz). Should have an even number of
        points and be symmetric about 0
    hermetian: bool, optional
        Force CS to be hermetian symmetric
        (as expected from dynamic spectra). Defaults to True
    """

    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    thth_red, edges_red = thth_redmap(
        CS, tau, fd, eta, edges, hermetian=hermetian
    )
    if hermetian:
        # Use eigenvalue decomposition if hermetian
        # Find first eigenvector and value
        w, V = eigsh(thth_red, 1, which="LA")
        w = w[0]
        V = V[:, 0]
        # Use larges eigenvector/value as model
        thth2_red = np.outer(V, np.conjugate(V))
        thth2_red *= np.abs(w)
    else:
        # Use Singular value decomposition if not hermetian
        U, S, W = np.linalg.svd(thth_red)
        U = U[:, 0]
        W = W[0, :]
        S = S[0]
        thth2_red = np.outer(U[:, 0], W[0, :]) * S[0]
    recov = rev_map(thth2_red, tau, fd, eta, edges_red, hermetian=hermetian)
    model = np.fft.ifft2(np.fft.ifftshift(recov))
    if hermetian:
        model = model.real
        return (thth_red, thth2_red, recov, model, edges_red, w, V)
    else:
        return (thth_red, thth2_red, recov, model, edges_red, U, S, W)


def chisq_calc(dspec, CS, tau, fd, eta, edges, N, mask=None):
    """
    Calculates the chisquared value for the model dynamic spectrum generated
    with the theta-theta model

    Parameters
    ----------
    dspec : 2D Array
        Dynamic spectrum to be fit to
    CS : 2D Array
        Theta-theta matrix to be mapped
    tau : 1D Array
        Time delay coordinates for the CS (us).
    fd : 1D Array
        Doppler shift coordinates for the CS (mHz)
    eta : float
        Arc curvature (s**3)
    edges: 1D Array
        Bin edges for theta-theta mapping (mHz). Should have an even number of
        points and be symmetric about 0
    N : 2D Array
        Standard deviation of the noise at each point in dspec
    mask 2D Array bool, optional
        Sets which points in dspec to calculate chisquared for.
        Defaults to all finite points
    """

    if mask is None:
        mask = np.isfinite(dspec)
    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    model = modeler(CS, tau, fd, eta, edges)[3][
        : dspec.shape[0], : dspec.shape[1]
    ]
    chisq = np.sum((model - dspec)[mask] ** 2) / N
    return chisq


def Eval_calc(CS, tau, fd, eta, edges):
    """
    Calculates to dominate eigenvalue for the theta-matrix generated from CS
    with curvature eta and bin edges edges

    Parameters
    ----------
    CS : 2D Array
        Theta-theta matrix to be mapped
    tau : 1D Array
        Time delay coordinates for the CS (us).
    fd : 1D Array
        Doppler shift coordinates for the CS (mHz)
    eta : float
        Arc curvature (s**3)
    edges: 1D Astropy Quantity Array
        Bin edges for theta-theta mapping (mHz). Should have an even number of
        points and be symmetric about 0 (mHz)
    """

    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    thth_red, edges_red = thth_redmap(CS, tau, fd, eta, edges)
    # Find first eigenvector and value
    v0 = np.copy(thth_red[thth_red.shape[0] // 2, :])
    v0 /= np.sqrt((np.abs(v0) ** 2).sum())
    w, V = eigsh(thth_red, 1, v0=v0, which="LA")
    return np.abs(w[0])


def len_arc(x, eta):
    """
    Calculate distance along arc with curvature eta to points (x,eta x**2)
    (DEVELOPMENT ONLY)

    Parameters
    ----------
    x : float
        X offset from apex of parabola
    eta : float
        Arc curvature of parabola
    """
    a = 2 * eta
    return (a * x * np.sqrt((a * x) ** 2 + 1) + np.arcsinh(a * x)) / (2.0 * a)


def arc_edges(eta, dfd, dtau, fd_max, n):
    """
    Calculate evenly spaced in arc length edges array (DEVELOPMENT ONLY)

    Parameters:
    ----------
    eta : float
        The curvature of the parabola
    dfd : float
        Doppler shift resolution of conjugate spectrum (mHz)
    dtau : float
        Time delay resolution of conjugate spectrum (us)
    fd_max : float
        Extent of arc in Doppler shift
    n  : int
        Integer number of points in array (must be even)
    """

    x_max = fd_max / dfd
    eta_ul = dfd**2 * eta / dtau
    l_max = len_arc(x_max.value, eta_ul.value)
    dl = l_max / (n // 2 - 0.5)
    x = np.zeros(int(n // 2))
    x[0] = dl / 2
    for i in range(x.shape[0] - 1):
        x[i + 1] = x[i] + dl / (np.sqrt(1 + (2 * eta_ul * x[i]) ** 2))
    edges = np.concatenate((-x[::-1], x)) * dfd.value
    return edges


def ext_find(x, y):
    """
    Determine extent for imshow to center bins at given coordinates

    Parameters
    ----------
    x : 1D Array Quantity
        X coordinates of data
    y : 1D Array Quantity
        Y coordiantes of data

    """
    dx = np.diff(x).mean()
    dy = np.diff(y).mean()
    ext = [
        (x[0] - dx / 2).value,
        (x[-1] + dx / 2).value,
        (y[0] - dy / 2).value,
        (y[-1] + dy / 2).value,
    ]
    return ext


def fft_axis(x, unit, pad=0):
    """
    Calculates fourier space coordinates from data space coordinates.

    Parameters
    ----------
    x : 1D Array Quantity
        Coordinate array to find Fourier conjugate of
    unit : Atropy Unit
        Desired unit for fourier coordinates
    pad : int, optional
        Integer giving how many additional copies of the data are padded in
        this direction
    """
    fx = (
        np.fft.fftshift(
            np.fft.fftfreq((pad + 1) * x.shape[0], x[1] - x[0]).to_value(unit)
        )
        * unit
    )
    return fx


def singularvalue_calc(
    CS, tau, fd, eta, edges, etaArclet, edgesArclet, centerCut
):
    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)
    etaArclet = unit_checks(etaArclet, "etaArclet", u.s**3)
    edgesArclet = unit_checks(edgesArclet, "edgesArclet", u.mHz)
    centerCut = unit_checks(centerCut, "Center Cut", u.mHz)

    thth_red, edges_red1, edges_red2 = two_curve_map(
        CS, tau, fd, eta, edges, etaArclet, edgesArclet
    )
    cents1 = (edges_red1[1:] + edges_red1[:-1]) / 2
    thth_red[:, np.abs(cents1) < centerCut] = 0
    U, S, W = np.linalg.svd(thth_red)
    return S[0]


def single_search_thin(params):
    """
    Curvature Search for a single chunk of a dynamic spectrum. Designed for use
    with MPI4py

    Parameters
    ----------
    params : List
        Contains the following
        dspec2 -- The chunk of the dynamic spectrum
        freq -- The frequency channels of that chunk (with units)
        time -- The time bins of that chunk (with units)
        eta_l -- The lower limit of curvatures to search (with units)
        eta_h -- the upper limit of curvatures to search (with units)
        edges -- The bin edges for Theta-Theta
        name -- A string filename used if plotting
        plot -- A bool controlling if the result should be plotted
        neta -- Number of curvatures to test
        coher -- A bool for whether to use coherent (True) or incoherent
                (False) theta-theta
        verbose -- A bool for how many updates the search prints
    """

    # Read Parameters
    (
        dspec2,
        freq,
        time,
        etas,
        edges,
        name,
        plot,
        fw,
        npad,
        coher,
        verbose,
        edgesArclet,
        centerCut,
    ) = params

    # Verify units
    time = unit_checks(time, "time", u.s)
    freq = unit_checks(freq, "freq", u.MHz)
    etas = unit_checks(etas, "etas", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    # Calculate fD and tau arrays
    fd = fft_axis(time, u.mHz, npad)
    tau = fft_axis(freq, u.us, npad)

    # Pad dynamic Spectrum
    dspec_pad = np.pad(
        dspec2,
        ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
        mode="constant",
        constant_values=dspec2.mean(),
    )

    # Calculate Conjugate Spectrum
    CS = np.fft.fft2(dspec_pad)
    CS = np.fft.fftshift(CS)
    eigs = np.zeros(etas.shape)
    if coher:
        # Loop over all curvatures
        for i in range(eigs.shape[0]):
            try:
                # Find largest Eigenvalue for curvature
                eigs[i] = singularvalue_calc(
                    CS,
                    tau,
                    fd,
                    etas[i],
                    edges,
                    etas[i],
                    edgesArclet,
                    centerCut,
                )
            except Exception as e:
                if verbose:
                    print(e)
                # Set eigenvalue to NaN in event of failure
                eigs[i] = np.nan
    else:
        SS = np.abs(CS) ** 2
        # Loop over all curvatures
        for i in range(eigs.shape[0]):
            try:
                # Find largest Eigenvalue for curvature
                eigs[i] = singularvalue_calc(
                    SS,
                    tau,
                    fd,
                    etas[i],
                    edges,
                    etas[i],
                    edgesArclet,
                    centerCut,
                )
            except Exception as e:
                if verbose:
                    print(e)
                # Set eigenvalue to NaN in event of failure
                eigs[i] = np.nan

    # Fit eigenvalue peak
    try:
        # Remove failed curvatures
        etas = etas[np.isfinite(eigs)]
        eigs = eigs[np.isfinite(eigs)]

        # Reduced range around peak to be withing fw times curvature of
        #     maximum eigenvalue
        etas_fit = etas[
            np.abs(etas - etas[eigs == eigs.max()])
            < fw * etas[eigs == eigs.max()]
        ]
        eigs_fit = eigs[
            np.abs(etas - etas[eigs == eigs.max()])
            < fw * etas[eigs == eigs.max()]
        ]

        # Initial Guesses
        C = eigs_fit.max()
        x0 = etas_fit[eigs_fit == C][0].value
        if x0 == etas_fit[0].value:
            A = (eigs_fit[-1] - C) / ((etas_fit[-1].value - x0) ** 2)
        else:
            A = (eigs_fit[0] - C) / ((etas_fit[0].value - x0) ** 2)

        # Fit parabola around peak
        popt, pcov = curve_fit(
            chi_par, etas_fit.value, eigs_fit, p0=np.array([A, x0, C])
        )

        # Record curvauture fit and error
        eta_fit = popt[1] * u.us / u.mHz**2
        eta_sig = (
            np.sqrt(
                (eigs_fit - chi_par(etas_fit.value, *popt)).std()
                / np.abs(popt[0])
            )
            * u.us
            / u.mHz**2
        )
    except Exception as e:
        if verbose:
            print(e)
        # Return NaN for curvautre and error if fitting fails
        popt = None
        eta_fit = np.nan
        eta_sig = np.nan

    # Plotting
    try:
        if plot:
            # Create diagnostic plots where requested
            PlotFunc(
                dspec2,
                time,
                freq,
                CS,
                fd,
                tau,
                edges,
                eta_fit,
                eta_sig,
                etas,
                eigs,
                etas_fit,
                popt,
            )
            plt.savefig(name)
            plt.close()
    except Exception as e:
        if verbose:
            print(e)
        # always print
        print("Plotting Error", flush=True)

    if verbose:
        # Progress Report
        print(
            "Chunk completed (eta = %s +- %s at %s)"
            % (eta_fit, eta_sig, freq.mean()),
            flush=True,
        )
    return (eta_fit, eta_sig, freq.mean(), time.mean(), eigs)


def single_search(params):
    """
    Curvature Search for a single chunk of a dynamic spectrum. Designed for use
    with MPI4py

    Parameters
    ----------
    params : List
        Contains the following
        dspec2 -- The chunk of the dynamic spectrum
        freq -- The frequency channels of that chunk (with units)
        time -- The time bins of that chunk (with units)
        eta_l -- The lower limit of curvatures to search (with units)
        eta_h -- the upper limit of curvatures to search (with units)
        edges -- The bin edges for Theta-Theta
        name -- A string filename used if plotting
        plot -- A bool controlling if the result should be plotted
        neta -- Number of curvatures to test
        coher -- A bool for whether to use coherent (True) or incoherent
                (False) theta-theta
        verbose -- A bool for how many updates the search prints
    """

    # Read Parameters
    (
        dspec2,
        freq,
        time,
        etas,
        edges,
        name,
        plot,
        fw,
        npad,
        coher,
        verbose,
    ) = params

    # Verify units
    time = unit_checks(time, "time", u.s)
    freq = unit_checks(freq, "freq", u.MHz)
    etas = unit_checks(etas, "etas", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    # Calculate fD and tau arrays
    fd = fft_axis(time, u.mHz, npad)
    tau = fft_axis(freq, u.us, npad)

    # Pad dynamic Spectrum
    dspec_pad = np.pad(
        dspec2,
        ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
        mode="constant",
        constant_values=dspec2.mean(),
    )

    # Calculate Conjugate Spectrum
    CS = np.fft.fft2(dspec_pad)
    CS = np.fft.fftshift(CS)
    eigs = np.zeros(etas.shape)
    if coher:
        # Loop over all curvatures
        for i in range(eigs.shape[0]):
            try:
                # Find largest Eigenvalue for curvature
                eigs[i] = Eval_calc(CS, tau, fd, etas[i], edges)
            except Exception as e:
                if verbose:
                    print(e)
                # Set eigenvalue to NaN in event of failure
                eigs[i] = np.nan
    else:
        SS = np.abs(CS)
        # Loop over all curvatures
        for i in range(eigs.shape[0]):
            try:
                # Find largest Eigenvalue for curvature
                eigs[i] = Eval_calc(SS, tau, fd, etas[i], edges)
            except Exception as e:
                if verbose:
                    print(e)
                # Set eigenvalue to NaN in event of failure
                eigs[i] = np.nan

    # Fit eigenvalue peak
    try:
        # Remove failed curvatures
        etas = etas[np.isfinite(eigs)]
        eigs = eigs[np.isfinite(eigs)]

        # Reduced range around peak to be withing fw times curvature of
        #     maximum eigenvalue
        etas_fit = etas[
            np.abs(etas - etas[eigs == eigs.max()])
            < fw * etas[eigs == eigs.max()]
        ]
        eigs_fit = eigs[
            np.abs(etas - etas[eigs == eigs.max()])
            < fw * etas[eigs == eigs.max()]
        ]

        # Initial Guesses
        C = eigs_fit.max()
        x0 = etas_fit[eigs_fit == C][0].value
        if x0 == etas_fit[0].value:
            A = (eigs_fit[-1] - C) / ((etas_fit[-1].value - x0) ** 2)
        else:
            A = (eigs_fit[0] - C) / ((etas_fit[0].value - x0) ** 2)

        # Fit parabola around peak
        popt, pcov = curve_fit(
            chi_par, etas_fit.value, eigs_fit, p0=np.array([A, x0, C])
        )

        # Record curvauture fit and error
        eta_fit = popt[1] * u.us / u.mHz**2
        eta_sig = (
            np.sqrt(
                (eigs_fit - chi_par(etas_fit.value, *popt)).std()
                / np.abs(popt[0])
            )
            * u.us
            / u.mHz**2
        )
    except Exception as e:
        if verbose:
            print(e)
        # Return NaN for curvautre and error if fitting fails
        popt = None
        eta_fit = np.nan
        eta_sig = np.nan

    # Plotting
    try:
        if plot:
            # Create diagnostic plots where requested
            PlotFunc(
                dspec2,
                time,
                freq,
                CS,
                fd,
                tau,
                edges,
                eta_fit,
                eta_sig,
                etas,
                eigs,
                etas_fit,
                popt,
            )
            plt.savefig(name)
            plt.close()
    except Exception as e:
        if verbose:
            print(e)
        # always print
        print("Plotting Error", flush=True)

    if verbose:
        # Progress Report
        print(
            "Chunk completed (eta = %s +- %s at %s)"
            % (eta_fit, eta_sig, freq.mean()),
            flush=True,
        )
    return (eta_fit, eta_sig, freq.mean(), time.mean(), eigs)


def PlotFunc(
    dspec,
    time,
    freq,
    CS,
    fd,
    tau,
    edges,
    eta_fit,
    eta_sig,
    etas,
    measure,
    etas_fit,
    fit_res,
    tau_lim=None,
    method="eigenvalue",
):
    """
    Plotting script to look at invidivual chunks

    Parameters
    ----------
    dspec : 2D Array
        The Dynamic Spectrum
    time : 1D Astropy Quantity Array
        Time bins of the Dynamic Spectrum (s)
    freq : 1D Astropy Quantity Array
        Frequency channels of the Dynamic Spectrum (MHz)
    CS : 2D Array
        Conjugate Spectrum
    fd : 1D Astropy Quantity Array
        Doppler Shift coordinates of the Conjugate Spectrum (mHz)
    tau : 1D Astropy Quantity Array
        Time Delay coordinates of the Conjugate Spectrum (us)
    edges : 1D Astropy Quantity Array
        Theta-Theta bin edges (mHz)
    eta_fit : Astropy Quantity
        Best fit Arc Curvature (s**3)
    eta_sig : Astropy Quantity
        Error on best fit Arc Curvature (s**3)
    etas : 1D Astropy Quantity Array
        Curvatures search over (s**3)
    measure : 1D Array
        Largest eigenvalue (method = 'eigenvalue') or chisq value
        (method = 'chisq') for each eta
    etas_fit : 1D Astropy Quantity Array
        Subarray of etas used for fitting
    fit_res : tuple
        Fit parameters for parabola at extremum
    tau_lim : Float Quantity, optional
        Largest tau value for SS plots
    method : String, optional
        Either 'eigenvalue' or 'chisq' depending on how curvature was found.
        Defaults to "eigenvalue"
    """

    # Verify units
    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    edges = unit_checks(edges, "edges", u.mHz)
    eta_fit = unit_checks(eta_fit, "eta_fit", u.s**3)
    eta_sig = unit_checks(eta_sig, "eta_sig", u.s**3)
    etas = unit_checks(etas, "etas", u.s**3)
    etas_fit = unit_checks(etas_fit, "etas_fit", u.s**3)

    if tau_lim is not None:
        tau_lim = unit_checks(tau_lim, "tau_lim", u.us)
    else:
        tau_lim = tau.max()

    # Determine fd limits
    fd_lim = min(2 * edges.max(), fd.max()).value

    # Determine TH-TH and model
    if np.isnan(eta_fit):
        eta = etas.mean()
        thth_red, thth2_red, recov, model, edges_red, w, V = modeler(
            CS, tau, fd, etas.mean(), edges
        )
    else:
        eta = eta_fit
        thth_red, thth2_red, recov, model, edges_red, w, V = modeler(
            CS, tau, fd, eta_fit, edges
        )

    # Create model Wavefield and Conjugate Wavefield
    ththE_red = thth_red * 0
    ththE_red[ththE_red.shape[0] // 2, :] = np.conjugate(V) * np.sqrt(w)
    # Map back to time/frequency space
    recov_E = rev_map(ththE_red, tau, fd, eta, edges_red, hermetian=False)
    model_E = np.fft.ifft2(np.fft.ifftshift(recov_E))[
        : dspec.shape[0], : dspec.shape[1]
    ]
    model_E *= dspec.shape[0] * dspec.shape[1] / 4
    model_E[dspec > 0] = np.sqrt(dspec[dspec > 0]) * np.exp(
        1j * np.angle(model_E[dspec > 0])
    )
    model_E = np.pad(
        model_E,
        (
            (0, CS.shape[0] - model_E.shape[0]),
            (0, CS.shape[1] - model_E.shape[1]),
        ),
        mode="constant",
        constant_values=0,
    )
    recov_E = np.abs(np.fft.fftshift(np.fft.fft2(model_E))) ** 2
    model_E = model_E[: dspec.shape[0], : dspec.shape[1]]
    N_E = recov_E[: recov_E.shape[0] // 4, :].mean()

    model = model[: dspec.shape[0], : dspec.shape[1]]
    model -= model.mean()
    model *= np.nanstd(dspec) / np.std(model)
    model += np.nanmean(dspec)

    # Create derotated thth
    thth_derot = thth_red * np.conjugate(thth2_red)

    # Plots
    grid = plt.GridSpec(6, 2)
    plt.figure(figsize=(8, 24))

    # Data Dynamic Spectrum
    plt.subplot(grid[0, 0])
    plt.imshow(
        dspec,
        aspect="auto",
        extent=ext_find(time.to(u.min), freq),
        origin="lower",
        vmin=np.nanmean(dspec) - 5 * np.nanstd(dspec),
        vmax=np.nanmean(dspec) + 5 * np.nanstd(dspec),
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Freq (MHz)")
    plt.title("Data Dynamic Spectrum")

    # Model Dynamic Spectrum
    plt.subplot(grid[0, 1])
    plt.imshow(
        model[: dspec.shape[0], : dspec.shape[1]],
        aspect="auto",
        extent=ext_find(time.to(u.min), freq),
        origin="lower",
        vmin=np.nanmean(dspec) - 5 * np.nanstd(dspec),
        vmax=np.nanmean(dspec) + 5 * np.nanstd(dspec),
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Freq (MHz)")
    plt.title("Model Dynamic Spectrum")

    # Data Secondary Spectrum
    plt.subplot(grid[1, 0])
    plt.imshow(
        np.abs(CS) ** 2,
        norm=LogNorm(
            vmin=np.median(np.abs(CS) ** 2), vmax=np.abs(CS).max() ** 2
        ),
        origin="lower",
        aspect="auto",
        extent=ext_find(fd, tau),
    )
    plt.xlim((-fd_lim, fd_lim))
    plt.ylim((0, tau_lim.value))
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")
    plt.title("Data Secondary Spectrum")
    plt.plot(fd, eta * (fd**2), "r", alpha=0.7)
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")

    # Model Secondary Spectrum
    plt.subplot(grid[1, 1])
    plt.imshow(
        np.abs(recov) ** 2,
        norm=LogNorm(
            vmin=np.median(np.abs(CS) ** 2), vmax=np.abs(CS).max() ** 2
        ),
        origin="lower",
        aspect="auto",
        extent=ext_find(fd, tau),
    )
    plt.xlim((-fd_lim, fd_lim))
    plt.ylim((0, tau_lim.value))
    plt.title("Model Secondary Spectrum")

    # Data TH-TH
    plt.subplot(grid[2, 0])
    plt.imshow(
        np.abs(thth_red) ** 2,
        norm=LogNorm(
            vmin=np.median(np.abs(thth_red) ** 2),
            vmax=np.abs(thth_red).max() ** 2,
        ),
        origin="lower",
        aspect="auto",
        extent=[
            edges_red[0].value,
            edges_red[-1].value,
            edges_red[0].value,
            edges_red[-1].value,
        ],
    )
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(r"Data $\theta-\theta$")

    # Model TH-TH
    plt.subplot(grid[2, 1])
    plt.imshow(
        np.abs(thth2_red) ** 2,
        norm=LogNorm(
            vmin=np.median(np.abs(thth_red) ** 2),
            vmax=np.abs(thth_red).max() ** 2,
        ),
        origin="lower",
        aspect="auto",
        extent=[
            edges_red[0].value,
            edges_red[-1].value,
            edges_red[0].value,
            edges_red[-1].value,
        ],
    )
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(r"Model $\theta-\theta$")

    # Derotated TH-TH Real Part
    plt.subplot(grid[3, 0])
    plt.imshow(
        thth_derot.real,
        norm=SymLogNorm(
            np.median(np.abs(thth_red) ** 2),
            vmin=-np.abs(thth_derot).max(),
            vmax=np.abs(thth_derot).max(),
        ),
        origin="lower",
        aspect="auto",
        extent=[
            edges_red[0].value,
            edges_red[-1].value,
            edges_red[0].value,
            edges_red[-1].value,
        ],
    )
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(r"Derotated $\theta-\theta$ (real)")

    # Derotated TH-TH Imaginary Part
    plt.subplot(grid[3, 1])
    plt.imshow(
        thth_derot.imag,
        norm=SymLogNorm(
            np.median(np.abs(thth_red) ** 2),
            vmin=-np.abs(thth_derot).max(),
            vmax=np.abs(thth_derot).max(),
        ),
        origin="lower",
        aspect="auto",
        extent=[
            edges_red[0].value,
            edges_red[-1].value,
            edges_red[0].value,
            edges_red[-1].value,
        ],
    )
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(r"Derotated $\theta-\theta$ (imag)")

    # Plot Eigenvalue vs Curvature
    plt.subplot(grid[4, :])
    plt.plot(etas, measure)
    if not np.isnan(eta_fit):
        fit_string, err_string = errString(eta_fit, eta_sig)
        plt.plot(
            etas_fit,
            chi_par(etas_fit.value, *fit_res),
            label=r"$\eta$ = %s $\pm$ %s $s^3$" % (fit_string, err_string),
        )
        plt.legend()
    if method == "eigenvalue":
        plt.title("Eigenvalue Search")
        plt.ylabel(r"Largest Eigenvalue")
    else:
        plt.title("Chisquare Search")
        plt.ylabel(r"$\chi^2$")
    plt.xlabel(r"$\eta$ ($s^3$)")

    # Phase of Wavefield
    plt.subplot(grid[5, 0])
    plt.imshow(
        np.angle(model_E),
        cmap="twilight",
        aspect="auto",
        extent=ext_find(time.to(u.min), freq),
        origin="lower",
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Freq (MHz)")
    plt.title("Recovered Phases")

    # Secondary Wavefield
    plt.subplot(grid[5, 1])
    plt.imshow(
        recov_E,
        norm=LogNorm(vmin=N_E),
        origin="lower",
        aspect="auto",
        extent=ext_find(fd, tau),
    )
    plt.xlim((-fd_lim, fd_lim))
    plt.ylim((0, tau_lim.value))
    plt.xlabel(r"$f_D$ (mHz)")
    plt.ylabel(r"$\tau$ (us)")
    plt.title("Recovered Secondary Wavefield")
    plt.colorbar()
    plt.tight_layout()


def VLBI_chunk_retrieval(params):
    """
    Performs phase retrieval on a single time/frequency chunk using multiple
    dynamic spectra and visibilities. Designed for use in parallel phase
    retreival code

    Parameters
    ----------
    params : List
        Contains
        dspec2_list : List
            A list of the overlapping dynamic spectra for all single dishes
            and visibilities ordered as [I1 , V12, ..., V1N, I2, V23,...,IN]
        edges : 1D Astropy Quantity Array
            Bin edges for theta-theta mapping (mHz). Should have an even number
            of points and be symmetric about 0
        time : 1D Astropy Quanity Array
            Time bins for the section of the spectra being examined (s)
        freq : 1D Astropy Quantity Array
            Frequency channels for the section of the spectra being examined
        eta : Float Quantity
            Arc curvature for the section of the spectra being examined
        idx_t : int
            Time index of chunk being examined
        idx_f : int
            Frequency index of chunk being examined
        npad : int
            Number of zeros paddings to add to end of dspec
        n_dish : int
            Number of stations used for VLBI
        verbose : bool
            Control the number of print statements

    """

    # Read parameters
    (
        dspec2_list,
        edges,
        time,
        freq,
        eta,
        idx_t,
        idx_f,
        npad,
        n_dish,
        verbose,
    ) = params

    # Verify unit compatability
    time = unit_checks(time, "time2", u.s)
    freq = unit_checks(freq, "freq2", u.MHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    if verbose:
        # Progress reporting
        print("Starting Chunk %s-%s" % (idx_f, idx_t), flush=True)

    # Determine fd and tau coordinates of Conjugate Spectrum
    fd = fft_axis(time, u.mHz, npad)
    tau = fft_axis(freq, u.us, npad)

    # Determine which sprectra in dspec2_list are dynamic spectra
    dspec_args = (n_dish * (n_dish + 1)) / 2 - np.cumsum(
        np.linspace(1, n_dish, n_dish)
    )
    thth_red = list()
    for i in range(len(dspec2_list)):
        if np.isin(i, dspec_args):
            # Pad dynamic spectrum to help with peiodicity problem
            dspec_pad = np.pad(
                dspec2_list[i],
                (
                    (0, npad * dspec2_list[i].shape[0]),
                    (0, npad * dspec2_list[i].shape[1]),
                ),
                mode="constant",
                constant_values=dspec2_list[i].mean(),
            )

            # Calculate Conjugate Spectrum (or Conjugate Visibility)
            CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
            # Calculate TH-TH for dynamic spectra
            thth_single, edges_red = thth_redmap(CS, tau, fd, eta, edges)
        else:
            # Pad dynamic spectrum to help with peiodicity problem
            dspec_pad = np.pad(
                dspec2_list[i],
                (
                    (0, npad * dspec2_list[i].shape[0]),
                    (0, npad * dspec2_list[i].shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )

            # Calculate Conjugate Spectrum (or Conjugate Visibility)
            CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
            # Calculate THTH for Visiblities
            thth_single, edges_red = thth_redmap(
                CS, tau, fd, eta, edges, hermetian=False
            )
        # Append TH-THT of spectrum to list
        thth_red.append(thth_single)

    # Determine size of individual spectra
    thth_size = thth_red[0].shape[0]

    # Create array for composite TH-TH
    thth_comp = np.zeros(
        (thth_size * n_dish, thth_size * n_dish), dtype=complex
    )

    # Loop over all pairs of dishes
    for d1 in range(n_dish):
        for d2 in range(n_dish - d1):
            # Determine position of pair in thth_red list
            idx = int(
                ((n_dish * (n_dish + 1)) // 2)
                - (((n_dish - d1) * (n_dish - d1 + 1)) // 2)
                + d2
            )

            # Insert THTH in apropriate location in composite THTH
            thth_comp[
                d1 * thth_size: (d1 + 1) * thth_size,
                (d1 + d2) * thth_size: (d1 + d2 + 1) * thth_size,
            ] = np.conjugate(thth_red[idx].T)
            # Make composite TH-TH hermitian by including complex conjugate
            #     transpose in mirrored location (Dynamic spectra are along the
            #     diagonal and are already Hermitian)
            thth_comp[
                (d1 + d2) * thth_size: (d1 + d2 + 1) * thth_size,
                d1 * thth_size: (d1 + 1) * thth_size,
            ] = thth_red[idx]

    # Find largest eigvalue and its eigenvector
    w, V = eigsh(thth_comp, 1, which="LA")
    w = w[0]
    V = V[:, 0]
    thth_temp = np.zeros((thth_size, thth_size), dtype=complex)
    model_E = list()
    # Loop over all dishes
    for d in range(n_dish):
        # Build Model TH-TH for dish
        thth_temp *= 0
        thth_temp[thth_size // 2, :] = np.conjugate(
            V[d * thth_size: (d + 1) * thth_size]
        ) * np.sqrt(w)
        recov_E = rev_map(thth_temp, tau, fd, eta, edges_red, hermetian=False)
        # Map back to frequency/time space
        model_E_temp = np.fft.ifft2(np.fft.ifftshift(recov_E))[
            : dspec2_list[0].shape[0], : dspec2_list[0].shape[1]
        ]
        model_E_temp *= dspec2_list[0].shape[0] * dspec2_list[0].shape[1] / 4
        model_E.append(model_E_temp)
    if verbose:
        # Progress Report
        print("Chunk %s-%s success" % (idx_f, idx_t), flush=True)
    return (model_E, idx_f, idx_t)


def single_chunk_retrieval(params):
    """
    Performs phase retrieval on a single time/frequency chunk.
    Designed for use in parallel phase retreival code

    Parameters
    ----------
    params : List
        Contains
        dspec2 : 2D Array
            Section of the Dynamic Spectrum to be analyzed
        edges : 1D Astropy Quantity Array
            Bin edges for theta-theta mapping (mHz). Should have an even number
            of points and be symmetric about 0
        time : 1D Astropy Quanity Array
            Time bins for the section of the spectrum being examined (s)
        freq : 1D Astropy Quantity Array
            Frequency channels for the section of the spectrum being examined
        eta : Float Quantity
            Arc curvature for the section of the spectrum being examined
        idx_t : int
            Time index of chunk being examined
        idx_f : int
            Frequency index of chunk being examined
        npad : int
            Number of zeros paddings to add to end of dspec
        verbose : bool
            Control the number of print statements

    """

    # Read parameters
    dspec2, edges, time, freq, eta, idx_t, idx_f, npad, verbose = params

    # Verify unit compatability
    time2 = unit_checks(time, "time2", u.s)
    freq2 = unit_checks(freq, "freq2", u.MHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    if verbose:
        # Progress Reporting
        print("Starting Chunk %s-%s" % (idx_f, idx_t), flush=True)

    # Determine fd and tau coordinates of Conjugate Spectrum
    fd = fft_axis(time2, u.mHz, npad)
    tau = fft_axis(freq2, u.us, npad)

    # Pad dynamic spectrum to help with peiodicity problem
    dspec_pad = np.pad(
        dspec2,
        ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
        mode="constant",
        constant_values=dspec2.mean(),
    )

    # Compute Conjugate Spectrum
    CS = np.fft.fft2(dspec_pad)
    CS = np.fft.fftshift(CS)

    # Try phase retrieval on chunk
    try:
        # Calculate Reduced TH-TH and largest eigenvalue/vector pair
        thth_red, thth2_red, recov, model, edges_red, w, V = modeler(
            CS, tau, fd, eta, edges
        )

        # Build model TH-TH for wavefield
        ththE_red = thth_red * 0
        ththE_red[ththE_red.shape[0] // 2, :] = np.conjugate(V) * np.sqrt(w)
        # Map back to time/frequency space
        recov_E = rev_map(ththE_red, tau, fd, eta, edges_red, hermetian=False)
        model_E = np.fft.ifft2(np.fft.ifftshift(recov_E))[
            : dspec2.shape[0], : dspec2.shape[1]
        ]
        model_E *= dspec2.shape[0] * dspec2.shape[1] / 4
        if verbose:
            # Progress Reporting
            print("Chunk %s-%s success" % (idx_f, idx_t), flush=True)
    except Exception as e:
        # If chunk cannot be recovered print Error and return zero array
        #     (Prevents failure of a single chunk from ending phase retrieval)
        print(e, flush=True)
        model_E = np.zeros(dspec2.shape, dtype=complex)
    return (model_E, idx_f, idx_t)


def mask_func(w):
    """
    Mask function used to weight chunks for mosaic function

    Parameters
    ----------
    w : int
        Length of data to be masked
    """
    x = np.linspace(0, w - 1, w)
    return np.sin((np.pi / 2) * x / w) ** 2


def mosaic(chunks):
    """Combine recovered wavefield chunks into a single composite wavefield by
    correcting for random phase rotation and stacking

    Parameters
    ----------
    chunks 4D Astropy Quantity Array
        Recovered wavfields for all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    """
    # Determine chunks sizes and number in time and freq
    nct = chunks.shape[1]
    ncf = chunks.shape[0]
    cwf = chunks.shape[2]
    cwt = chunks.shape[3]

    # Prepare E_recov array (output wavefield)
    E_recov = np.zeros(
        ((ncf - 1) * (cwf // 2) + cwf, (nct - 1) * (cwt // 2) + cwt),
        dtype=complex,
    )

    # Loop over all chunks
    for cf in range(ncf):
        for ct in range(nct):
            # Select new chunk
            chunk_new = chunks[cf, ct, :, :]

            # Find overlap with current wavefield
            chunk_old = E_recov[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ]
            mask = np.ones(chunk_new.shape)

            # Determine Mask for new chunk (chunks will have higher weights
            #     towards their centre)
            if cf > 0:
                # All chunks but the first in frequency overlap for the first
                #     half in frequency
                mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
            if cf < ncf - 1:
                # All chunks but the last in frequency overlap for the second
                #     half in frequency
                mask[cwf // 2:, :] *= 1 - mask_func(cwf // 2)[:, np.newaxis]
            if ct > 0:
                # All chunks but the first in time overlap for the first half
                #     in time
                mask[:, : cwt // 2] *= mask_func(cwt // 2)
            if ct < nct - 1:
                # All chunks but the last in time overlap for the second half
                #     in time
                mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
            # Average phase difference between new chunk and existing wavefield
            rot = np.angle((chunk_old * np.conjugate(chunk_new) * mask).mean())
            # Add masked and roated new chunk to wavefield
            E_recov[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ] += (
                chunk_new * mask * np.exp(1j * rot)
            )
    return E_recov


def two_curve_map(CS, tau, fd, eta1, edges1, eta2, edges2):
    """
    Map from Secondary Spectrum to theta-theta space allowing for arclets with
    different curvature

    Parameters
    ----------
    CS : 2D Array
        Conjugate Spectrum in [tau,fd] order with (0,0) in center
    tau : 1D Astropy Quantity Array
        Time Delay coordinates of Conjugate Spectrum (us)
    fd : 1D Astropy Quanity Array
        Doppler Shift coordinates of Conjugate Spectrum (mHz)
    eta1 : Float Quantity
        Arc Curvature of the main arc (s**3)
    edges1 : 1D Astropy Quantity Array
        Bin edges in theta1 for theta-theta mapping (mHz).
    eta2 : Float Quanity
        Arc Curvature of the Inverted Arclets
    edges2 : 1D Astropy Quantity Array
            Bin edges for in theta2 theta-theta mapping (mHz).
    """
    tau = unit_checks(tau, "tau", u.us)
    fd = unit_checks(fd, "fd", u.mHz)
    eta1 = unit_checks(eta1, "eta1", u.s**3)
    edges1 = unit_checks(edges1, "edges1", u.mHz)
    eta2 = unit_checks(eta2, "eta2", u.s**3)
    edges2 = unit_checks(edges2, "edges2", u.mHz)

    # Find bin centers
    th_cents1 = (edges1[1:] + edges1[:-1]) / 2
    th_cents2 = (edges2[1:] + edges2[:-1]) / 2

    # Calculate theta1 and th2 arrays
    th1 = np.ones((th_cents2.shape[0], th_cents1.shape[0])) * th_cents1
    th2 = (
        np.ones((th_cents2.shape[0], th_cents1.shape[0]))
        * th_cents2[:, np.newaxis]
    )

    # tau and fd step sizes
    dtau = np.diff(tau).mean()
    dfd = np.diff(fd).mean()

    # Find bin in CS space that each point maps back to
    tau_inv = (
        ((eta1 * th1**2 - eta2 * th2**2) - tau[1] + dtau / 2) // dtau
    ).astype(int)
    fd_inv = (((th1 - th2) - fd[1] + dfd / 2) // dfd).astype(int)

    # Define thth
    thth = np.zeros(tau_inv.shape, dtype=complex)

    # Only fill thth points that are within the CS
    pnts = (
        (tau_inv > 0)
        * (tau_inv < tau.shape[0] - 1)
        * (fd_inv < fd.shape[0] - 1)
    )
    thth[pnts] = CS[tau_inv[pnts], fd_inv[pnts]]

    # Preserve flux (int
    thth *= np.sqrt(np.abs(2 * eta1 * th1 - 2 * eta2 * th2)).value

    th2_max = np.sqrt(tau.max() / eta2)
    th1_max = np.sqrt(tau.max() / eta1)
    th_cents1 = (edges1[1:] + edges1[:-1]) / 2
    th_cents2 = (edges2[1:] + edges2[:-1]) / 2
    pnts_1 = np.abs(th_cents1) < th1_max
    pnts_2 = np.abs(th_cents2) < th2_max
    edges_red1 = np.zeros(pnts_1[pnts_1].shape[0] + 1) * edges1.unit
    edges_red1[:-1] = edges1[:-1][pnts_1]
    edges_red1[-1] = edges1[1:][pnts_1].max()
    edges_red2 = np.zeros(pnts_2[pnts_2].shape[0] + 1) * edges2.unit
    edges_red2[:-1] = edges2[:-1][pnts_2]
    edges_red2[-1] = edges2[1:][pnts_2].max()
    thth_red = thth[pnts_2, :][:, pnts_1]
    th_cents1 = (edges_red1[1:] + edges_red1[:-1]) / 2
    th_cents2 = (edges_red2[1:] + edges_red2[:-1]) / 2
    return (thth_red, edges_red1, edges_red2)


def unit_checks(var, name, desired):
    """
    Checks for unit compatibility
    (Used internally to help provide useful errors)

    Parameters
    ----------
    var : Astropy Quanity
        Variable whose units are to be checked
    name : String
        Name of variable for displaying errors
    desired : Astropy Unit
        Desired unit

    """
    var *= u.dimensionless_unscaled
    # Check if var has units
    if u.dimensionless_unscaled.is_equivalent(var.unit):
        # If there are no units assign desired units and tell user
        var *= desired
        warnings.warn(f"{name} missing units. Assuming {desired}.")
    elif desired.is_equivalent(var.unit):
        # If var has correct units (or equivalent)
        var = var.to(desired)
    else:
        # If var has incompatible units return error
        raise u.UnitConversionError(
            f"{name} units ({var.unit}) not equivalent to {desired}"
        )
    return var


def min_edges(fd_lim, fd, tau, eta, factor=2):
    """
    Calculates minimum size of edges array such that the Conjugate Spectrum is
    oversampled by at least some margin at all points

    Parameters
    ----------
    fd_lim : Float Quantity
        Largest value of fd to search out to along main arc
    fd : 1D Astropy Quanity Array
        Doppler Shift coordinates of Conjugate Spectrum (mHz)
    tau : 1D Astropy Quantity Array
        Time Delay coordinates of Conjugate Spectrum (us)
    eta : Float Quanity
        The curvature to calculate edges for
        (When doing a curvature search this should be the largest curvature
         to search over)
    factor : Float
        Over sample factor
    """

    fd_lim = unit_checks(fd_lim, "fD Limit", fd.unit)
    # Calculate largest fd step such that adjacent points at the top of the arc
    #     are within 1/factor tau steps
    dtau_lim = (tau[1] - tau[0]) / factor
    dtau_lim /= 2 * eta * fd_lim
    # Calculate largest fd step such that points at the bottom of the arc are
    #     within 1/factor fd steps
    dfd_lim = (fd[1] - fd[0]) / factor
    # Cacululate number of points needed to cover -fd_lim to fd_lim with
    #     smaller step size limit
    npoints = (2 * fd_lim) // (min(dfd_lim, dtau_lim.to(u.mHz)))
    # Ensure even number of edges points
    npoints += np.mod(npoints, 2)
    return np.linspace(-fd_lim, fd_lim, int(npoints))


def rotMos(chunks, x):
    """
    Combine recovered wavefield chunks into a single composite wavefield by
    correcting for random phase rotation and stacking

    Parameters
    ----------
    chunks :4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    x : 1D Array
        phase rotation of each chunk after the first.
    """
    # Determine chunks sizes and number in time and freq
    nct = chunks.shape[1]
    ncf = chunks.shape[0]
    cwf = chunks.shape[2]
    cwt = chunks.shape[3]

    # Prepare E_recov array (output wavefield)
    E_recov = np.zeros(
        ((ncf - 1) * (cwf // 2) + cwf, (nct - 1) * (cwt // 2) + cwt),
        dtype=complex,
    )

    # Loop over all chunks
    for cf in range(ncf):
        for ct in range(nct):
            # Select new chunk
            chunk_new = np.copy(chunks[cf, ct, :, :])

            # Find overlap with current wavefield
            mask = np.ones(chunk_new.shape)

            # Determine Mask for new chunk (chunks will have higher weights
            #     towards their centre)
            if cf > 0:
                # All chunks but the first in frequency overlap for the first
                #     half in frequency
                mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
            if cf < ncf - 1:
                # All chunks but the last in frequency overlap for the second
                #     half in frequency
                mask[cwf // 2:, :] *= 1 - mask_func(cwf // 2)[:, np.newaxis]
            if ct > 0:
                # All chunks but the first in time overlap for the first half
                #     in time
                mask[:, : cwt // 2] *= mask_func(cwt // 2)
            if ct < nct - 1:
                # All chunks but the last in time overlap for the second half
                #     in time
                mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
            rot = 0
            if cf > 0 or ct > 0:
                rot = x[nct * cf + ct - 1]
            # Add masked and roated new chunk to wavefield
            E_recov[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ] += (
                chunk_new * mask * np.exp(1j * rot)
            )
    return E_recov


def rotFit(x, chunks):
    """
    Calculates the sum of the dynamic spectrum for the wavefield produced by
    rotMos. Should be maximized when all chunks add coherently

    Parameters
    ----------
    x : 1D Array
        phase rotation of each chunk after the first.
    chunks :4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    """
    E = rotMos(chunks, x)
    res = -np.sum(np.abs(E) ** 2)
    return res


def rotInit(chunks):
    """
    Provides an initial estimate for the phase rotations for rotMos or rotfit

    Parameters
    ----------
    chunks :4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    """
    # Determine chunks sizes and number in time and freq
    nct = chunks.shape[1]
    ncf = chunks.shape[0]
    cwf = chunks.shape[2]
    cwt = chunks.shape[3]

    # Prepare E_recov array (output wavefield)
    E_recov = np.zeros(
        ((ncf - 1) * (cwf // 2) + cwf, (nct - 1) * (cwt // 2) + cwt),
        dtype=complex,
    )

    x = np.zeros((ncf * nct - 1))
    # Loop over all chunks
    for cf in range(ncf):
        for ct in range(nct):
            # Select new chunk
            chunk_new = np.copy(chunks[cf, ct, :, :])

            # Find overlap with current wavefield
            chunk_old = E_recov[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ]
            mask = np.ones(chunk_new.shape)

            # Determine Mask for new chunk (chunks will have higher weights
            #     towards their centre)
            if cf > 0:
                # All chunks but the first in frequency overlap for the first
                #     half in frequency
                mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
            if cf < ncf - 1:
                # All chunks but the last in frequency overlap for the second
                #     half in frequency
                mask[cwf // 2:, :] *= 1 - mask_func(cwf // 2)[:, np.newaxis]
            if ct > 0:
                # All chunks but the first in time overlap for the first half
                #     in time
                mask[:, : cwt // 2] *= mask_func(cwt // 2)
            if ct < nct - 1:
                # All chunks but the last in time overlap for the second half
                #     in time
                mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
            # Average phase difference between new chunk and existing wavefield
            rot = np.angle((chunk_old * np.conjugate(chunk_new) * mask).mean())
            # Add masked and roated new chunk to wavefield
            E_recov[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ] += (
                chunk_new * mask * np.exp(1j * rot)
            )
            if cf > 0 or ct > 0:
                x[cf * nct + ct - 1] = rot
    return x


def rotDer(x, chunks):
    """
    Analytic calculation of the gradient of rotFit for a given set of phase
    rotations and chunks

    Parameters
    ----------
    x : 1D Array
        phase rotation of each chunk after the first.
    chunks : 4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    """
    nct = chunks.shape[1]
    ncf = chunks.shape[0]
    cwf = chunks.shape[2]
    cwt = chunks.shape[3]
    E0 = rotMos(chunks, x)
    derivative = np.zeros(x.shape)
    for cf in range(ncf):
        for ct in range(nct):
            if cf > 0 or ct > 0:
                # Select new chunk
                y = np.copy(chunks[cf, ct, :, :])

                # Find overlap with current wavefield
                xx = np.copy(
                    E0[
                        cf * cwf // 2: cf * cwf // 2 + cwf,
                        ct * cwt // 2: ct * cwt // 2 + cwt,
                    ]
                )
                mask = np.ones(y.shape)

                # Determine Mask for new chunk (chunks will have higher weights
                #     towards their centre)
                if cf > 0:
                    # All chunks but the first in frequency overlap for the
                    #     first half in frequency
                    mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
                if cf < ncf - 1:
                    # All chunks but the last in frequency overlap for the
                    #     second half in frequency
                    mask[cwf // 2:, :] *= (
                        1 - mask_func(cwf // 2)[:, np.newaxis]
                    )
                if ct > 0:
                    # All chunks but the first in time overlap for the first
                    #     half in time
                    mask[:, : cwt // 2] *= mask_func(cwt // 2)
                if ct < nct - 1:
                    # All chunks but the last in time overlap for the second
                    #     half in time
                    mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
                y *= mask
                rot = x[nct * cf + ct - 1]
                xx -= y * np.exp(1j * rot)
                derivative[nct * cf + ct - 1] = np.sum(
                    2 * np.imag(np.conjugate(xx) * y * np.exp(1j * rot))
                )
    return derivative


def fullMos(chunks, p):
    """Combine recovered wavefield chunks into a single composite wavefield by
    correcting for random phase rotation, rescaling, and stacking

    Parameters
    ----------
    chunks : 4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    p : 1D Array
        Phase rotation of each chunk after the first and then amplitude scaling
        for all chunks
    """
    # Determine chunks sizes and number in time and freq
    nct = chunks.shape[1]
    ncf = chunks.shape[0]
    cwf = chunks.shape[2]
    cwt = chunks.shape[3]

    # Prepare E_recov array (output wavefield)
    E_recov = np.zeros(
        ((ncf - 1) * (cwf // 2) + cwf, (nct - 1) * (cwt // 2) + cwt),
        dtype=complex,
    )

    # Loop over all chunks
    for cf in range(ncf):
        for ct in range(nct):
            idx = cf * nct + ct
            # Select new chunk
            chunk_new = np.copy(chunks[cf, ct, :, :])

            # Find overlap with current wavefield
            mask = np.ones(chunk_new.shape)

            # Determine Mask for new chunk (chunks will have higher weights
            #     towards their centre)
            if cf > 0:
                # All chunks but the first in frequency overlap for the first
                #     half in frequency
                mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
            if cf < ncf - 1:
                # All chunks but the last in frequency overlap for the second
                #     half in frequency
                mask[cwf // 2:, :] *= 1 - mask_func(cwf // 2)[:, np.newaxis]
            if ct > 0:
                # All chunks but the first in time overlap for the first half
                #     in time
                mask[:, : cwt // 2] *= mask_func(cwt // 2)
            if ct < nct - 1:
                # All chunks but the last in time overlap for the second half
                #     in time
                mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
            if idx > 0:
                phi = p[idx - 1]
            else:
                phi = 0
            A = p[idx + ncf * nct - 1]
            # Add masked and roated new chunk to wavefield
            E_recov[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ] += (
                A * chunk_new * mask * np.exp(1j * phi)
            )
    return E_recov


def fullMosFit(p, chunks, dspec, N):
    """Combine recovered wavefield chunks into a single composite wavefield by
    correcting for random phase rotation, rescaling, and stacking

    Parameters
    ----------
    p : 1D Array
        Phase rotation of each chunk after the first and then amplitude scaling
        for all chunks c
    chunks : 4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    dspec : 2D Array
        Dynamic Spectrum to be fit to
    N : 2D Array
        Point by point noise standard deviation for Dynamic Spectrum
    """
    W = fullMos(chunks, p)
    M = np.abs(W) ** 2
    res = np.nansum(
        np.power(
            (M - dspec[: M.shape[0], : M.shape[1]])
            / N[: M.shape[0], : M.shape[1]],
            2,
        )
    )
    return res


def fullMosGrad(p, chunks, dspec, N):
    """Analysic gradient of fullMosFit

    Parameters
    ----------
    p : 1D Array
        Phase rotation of each chunk after the first and then amplitude scaling
        for all chunks c
    chunks : 4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    dspec : 2D Array
        Dynamic Spectrum to be fit to
    N : 2D Array
        Point by point noise standard deviation for Dynamic Spectrum

    """
    nct = chunks.shape[1]
    ncf = chunks.shape[0]
    cwf = chunks.shape[2]
    cwt = chunks.shape[3]
    W = fullMos(chunks, p)
    M = np.abs(W) ** 2
    weight = 4 * (M - dspec)
    grad = np.zeros(p.shape[0])
    for cf in range(ncf):
        for ct in range(nct):
            idx = cf * nct + ct
            y = np.copy(chunks[cf, ct, :, :])

            # Find overlap with current wavefield
            xx = weight[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ]
            Nse = N[
                cf * cwf // 2: cf * cwf // 2 + cwf,
                ct * cwt // 2: ct * cwt // 2 + cwt,
            ]
            mask = np.ones(y.shape)

            # Determine Mask for new chunk (chunks will have higher weights
            #     towards their centre)
            if cf > 0:
                # All chunks but the first in frequency overlap for the first
                #     half in frequency
                mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
            if cf < ncf - 1:
                # All chunks but the last in frequency overlap for the second
                #     half in frequency
                mask[cwf // 2:, :] *= 1 - mask_func(cwf // 2)[:, np.newaxis]
            if ct > 0:
                # All chunks but the first in time overlap for the first half
                #     in time
                mask[:, : cwt // 2] *= mask_func(cwt // 2)
            if ct < nct - 1:
                # All chunks but the last in time overlap for the second half
                #     in time
                mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
            y *= mask
            if idx > 0:
                phi = p[idx - 1]
            else:
                phi = 0
            A = p[idx + ncf * nct - 1]

            temp = np.conjugate(
                np.nansum(
                    xx
                    * y
                    * np.exp(1j * phi)
                    * np.conjugate(
                        W[
                            cf * cwf // 2: cf * cwf // 2 + cwf,
                            ct * cwt // 2: ct * cwt // 2 + cwt,
                        ]
                    )
                    / Nse**2
                )
            )
            if idx > 0:
                grad[idx - 1] = A * temp.imag
            grad[idx + ncf * nct - 1] = temp.real
    return grad


def fullMosHess(p, chunks, dspec, N):
    """Analysic Hessian of fullMosFit

    Parameters
    ----------
    p : 1D Array
        Phase rotation of each chunk after the first and then amplitude scaling
        for all chunks c
    chunks : 4D Astropy Quanity Array
        Recovered wavefields of all chunks in the form
        [Chunk# in Freq, Chunk# in Time, Freq within chunk, Time within chunk]
    dspec : 2D Array
        Dynamic Spectrum to be fit to
    N : 2D Array
        Point by point noise standard deviation for Dynamic Spectrum

    """

    nct = chunks.shape[1]
    ncf = chunks.shape[0]
    cwf = chunks.shape[2]
    cwt = chunks.shape[3]
    W = fullMos(chunks, p)
    Ws = np.conjugate(W)
    M = np.abs(W) ** 2
    weight = M - dspec
    H = np.zeros((p.shape[0], p.shape[0]))
    for cfN in range(ncf):
        for ctN in range(nct):
            idxN = cfN * nct + ctN
            yN = np.copy(chunks[cfN, ctN, :, :])

            # Find overlap with current wavefield
            wtN = weight[
                cfN * cwf // 2: cfN * cwf // 2 + cwf,
                ctN * cwt // 2: ctN * cwt // 2 + cwt,
            ]
            NseN = N[
                cfN * cwf // 2: cfN * cwf // 2 + cwf,
                ctN * cwt // 2: ctN * cwt // 2 + cwt,
            ]
            mask = np.ones(yN.shape)

            # Determine Mask for new chunk (chunks will have higher weights
            #     towards their centre)
            if cfN > 0:
                # All chunks but the first in frequency overlap for the first
                #     half in frequency
                mask[: cwf // 2, :] *= mask_func(cwf // 2)[:, np.newaxis]
            if cfN < ncf - 1:
                # All chunks but the last in frequency overlap for the second
                #     half in frequency
                mask[cwf // 2:, :] *= 1 - mask_func(cwf // 2)[:, np.newaxis]
            if ctN > 0:
                # All chunks but the first in time overlap for the first half
                #     in time
                mask[:, : cwt // 2] *= mask_func(cwt // 2)
            if ctN < nct - 1:
                # All chunks but the last in time overlap for the second half
                #     in time
                mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
            yN *= mask
            idpN = idxN - 1
            if idpN > -1:
                phiN = p[idpN]
            else:
                phiN = 0
            yN *= np.exp(1j * phiN)
            idAN = idxN + ncf * nct - 1
            AN = p[idAN]
            tempN = (
                yN
                * Ws[
                    cfN * cwf // 2: cfN * cwf // 2 + cwf,
                    ctN * cwt // 2: ctN * cwt // 2 + cwt,
                ]
            )

            for dt in np.array([-1, 0, 1]):
                if dt == -1:
                    olNt = np.linspace(0, cwt // 2 - 1, cwt // 2).astype(int)
                    olMt = np.linspace(cwt // 2, cwt - 1, cwt // 2).astype(int)
                elif dt == 0:
                    olNt = np.linspace(0, cwt - 1, cwt).astype(int)
                    olMt = np.linspace(0, cwt - 1, cwt).astype(int)
                else:
                    olMt = np.linspace(0, cwt // 2 - 1, cwt // 2).astype(int)
                    olNt = np.linspace(cwt // 2, cwt - 1, cwt // 2).astype(int)
                for df in np.array([-1, 0, 1]):
                    if df == -1:
                        olNf = np.linspace(0, cwf // 2 - 1, cwf // 2).astype(
                            int
                        )
                        olMf = np.linspace(cwf // 2, cwf - 1, cwf // 2).astype(
                            int
                        )
                    elif df == 0:
                        olNf = np.linspace(0, cwf - 1, cwf).astype(int)
                        olMf = np.linspace(0, cwf - 1, cwf).astype(int)
                    else:
                        olMf = np.linspace(0, cwf // 2 - 1, cwf // 2).astype(
                            int
                        )
                        olNf = np.linspace(cwf // 2, cwf - 1, cwf // 2).astype(
                            int
                        )
                    cfM = cfN + df
                    ctM = ctN + dt
                    if -1 < cfM < ncf and -1 < ctM < nct:
                        idxM = cfM * nct + ctM
                        yM = np.copy(chunks[cfM, ctM, :, :])
                        mask = np.ones(yN.shape)

                        # Determine Mask for new chunk (chunks will have higher
                        #     weights towards their centre)
                        if cfM > 0:
                            # All chunks but the first in frequency overlap for
                            #     the first half in frequency
                            mask[: cwf // 2, :] *= mask_func(cwf // 2)[
                                :, np.newaxis
                            ]
                        if cfM < ncf - 1:
                            # All chunks but the last in frequency overlap for
                            #     the second half in frequency
                            mask[cwf // 2:, :] *= (
                                1 - mask_func(cwf // 2)[:, np.newaxis]
                            )
                        if ctM > 0:
                            # All chunks but the first in time overlap for the
                            #     first half in time
                            mask[:, : cwt // 2] *= mask_func(cwt // 2)
                        if ctM < nct - 1:
                            # All chunks but the last in time overlap for the
                            #     second half in time
                            mask[:, cwt // 2:] *= 1 - mask_func(cwt // 2)
                        yM *= mask
                        idpM = idxM - 1
                        if idpM > -1:
                            phiM = p[idpM]
                        else:
                            phiM = 0
                        yM *= np.exp(1j * phiM)
                        idAM = idxM + ncf * nct - 1
                        AM = p[idAM]
                        tempM = (
                            yM
                            * Ws[
                                cfM * cwf // 2: cfM * cwf // 2 + cwf,
                                ctM * cwt // 2: ctM * cwt // 2 + cwt,
                            ]
                        )

                        dAndAm = 8 * np.real(tempM[olMf][:, olMt]) * np.real(
                            tempN[olNf][:, olNt]
                        ) + 4 * wtN[olNf][:, olNt] * np.real(
                            np.conjugate(yM[olMf][:, olMt]) * yN[olNf][:, olNt]
                        )
                        dAndAm = np.sum(dAndAm / (NseN[olNf][:, olNt] ** 2))
                        H[idAN, idAM] = dAndAm
                        H[idAM, idAN] = dAndAm
                        if idxM > 0:
                            dAndpm = -8 * AM * np.imag(
                                tempM[olMf][:, olMt]
                            ) * np.real(tempN[olNf][:, olNt]) + 4 * wtN[olNf][
                                :, olNt
                            ] * AM * np.imag(
                                yN[olNf][:, olNt]
                                * np.conjugate(yM[olMf][:, olMt])
                            )
                            if idxM == idxN:
                                dAndpm -= (
                                    4
                                    * wtN[olNf][:, olNt]
                                    * np.imag(tempN[olNf][:, olNt])
                                )
                            dAndpm = np.sum(
                                dAndpm / (NseN[olNf][:, olNt] ** 2)
                            )
                            H[idAN, idpM] = dAndpm
                            H[idpM, idAN] = dAndpm
                            if idxN > 0:
                                dpndpm = 8 * AN * AM * np.imag(
                                    tempM[olMf][:, olMt]
                                ) * np.imag(
                                    tempN[olNf][:, olNt]
                                ) + 4 * AN * AM * wtN[
                                    olNf
                                ][
                                    :, olNt
                                ] * np.real(
                                    np.conjugate(yM[olMf][:, olMt])
                                    * yN[olNf][:, olNt]
                                )
                                if idxM == idxN:
                                    dpndpm -= (
                                        4
                                        * AN
                                        * wtN[olNf][:, olNt]
                                        * np.real(tempN[olNf][:, olNt])
                                    )
                                dpndpm = np.sum(
                                    dpndpm / (NseN[olNf][:, olNt] ** 2)
                                )
                                H[idpM, idpN] = dpndpm
                                H[idpN, idpM] = dpndpm
    return H


def errString(fit, sig):
    """
    Convert a measurement and a standard deviation into strings in scientific
    notion for plot labels

     Parameters
    ----------
    fit : Float Quantity
        Fitted Value
    sig : Float Quanity
        Standard Deviation of fit

    """
    exp_fit = int(("%.0e" % fit.value)[2:])
    exp_err = int(("%.0e" % sig.value)[2:])

    if exp_err == exp_fit:
        fmt = "{:.%se}" % (exp_fit - exp_err)
        fit_string = fmt.format(fit.value)
        fit_string = fit_string[: fit_string.index("e")]
        err_string = fmt.format(sig.value)
        if err_string[0] == "1":
            exp_err -= 1
            fmt = "{:.%se}" % (exp_fit - exp_err)
            fit_string = fmt.format(fit.value)
            fit_string = fit_string[: fit_string.index("e")]
            err_string = fmt.format(sig.value)
    elif exp_fit > exp_err:
        fmt = "{:.%se}" % (exp_fit - exp_err)
        fit_string = fmt.format(fit.value)
        fit_string = fit_string[: fit_string.index("e")]
        err_string = "0%s" % fmt.format(10**exp_fit + sig.value)
        err_string = err_string[err_string.index("."):]
        if err_string[1] == "1":
            exp_err -= 1
            fmt = "{:.%se}" % (exp_fit - exp_err)
            fit_string = fmt.format(fit.value)
            fit_string = fit_string[: fit_string.index("e")]
            err_string = "0%s" % fmt.format(10**exp_fit + sig.value)
            err_string = err_string[err_string.index("."):]
        err_string = "0" + err_string
    else:
        fmt = "{:.%se}" % (exp_err - exp_fit)
        err_string = fmt.format(sig.value)
        err_string = err_string
        fit_string = "0%s" % fmt.format(10**exp_err + fit.value)
        fit_string = (
            "0" + fit_string[fit_string.index("."): fit_string.index("e")]
        )
    if err_string[err_string.index("e"):] == "e+00":
        err_string = err_string[: err_string.index("e")]

    return (fit_string, err_string)


def errCalc(etas, eigs, fitPars):
    M = chi_par(etas.value, *fitPars)
    sigEstimate = np.std(eigs - M)
    x0Err = (
        np.sqrt(
            2
            / np.sum(
                4
                * fitPars[0]
                * (2 * fitPars[0] * (fitPars[1] - etas.value) ** 2 + M - eigs)
            )
        )
        * sigEstimate
    )
    return x0Err


def calc_asymmetry(params):
    """
    Calculates the arc asymmetry from the theta-theta transform

    Parameters
    ----------
    params : List
        Contains
        dspec2 : 2D Array
            Section of the Dynamic Spectrum to be analyzed
        edges : 1D Astropy Quantity Array
            Bin edges for theta-theta mapping (mHz). Should have an even number
            of points and be symmetric about 0
        time : 1D Astropy Quanity Array
            Time bins for the section of the spectrum being examined (s)
        freq : 1D Astropy Quantity Array
            Frequency channels for the section of the spectrum being examined
        eta : Float Quantity
            Arc curvature for the section of the spectrum being examined
        idx_t : int
            Time index of chunk being examined
        idx_f : int
            Frequency index of chunk being examined
        npad : int
            Number of zeros paddings to add to end of dspec
        verbose : bool
            Control the number of print statements

    """

    # Read parameters
    dspec2, edges, time, freq, eta, idx_t, idx_f, npad, verbose = params

    # Verify unit compatability
    time2 = unit_checks(time, "time2", u.s)
    freq2 = unit_checks(freq, "freq2", u.MHz)
    eta = unit_checks(eta, "eta", u.s**3)
    edges = unit_checks(edges, "edges", u.mHz)

    if verbose:
        # Progress Reporting
        print("Starting Chunk %s-%s" % (idx_f, idx_t), flush=True)

    # Determine fd and tau coordinates of Conjugate Spectrum
    fd = fft_axis(time2, u.mHz, npad)
    tau = fft_axis(freq2, u.us, npad)

    # Pad dynamic spectrum to help with peiodicity problem
    dspec_pad = np.pad(
        dspec2,
        ((0, npad * dspec2.shape[0]), (0, npad * dspec2.shape[1])),
        mode="constant",
        constant_values=dspec2.mean(),
    )

    # Compute Conjugate Spectrum
    CS = np.fft.fft2(dspec_pad)
    CS = np.fft.fftshift(CS)

    # Try phase retrieval on chunk
    try:
        # Calculate Reduced TH-TH and largest eigenvalue/vector pair
        thth_red, thth2_red, recov, model, edges_red, w, V = modeler(
            CS, tau, fd, eta, edges
        )
        cents = (edges_red[1:] + edges_red[:-1]) / 2
        leftV = V[: (cents.shape[0] - 1) // 2]
        rightV = V[1 + (cents.shape[0] - 1) // 2:]
        asymm = (np.sum(np.abs(leftV) ** 2) - np.sum(np.abs(rightV) ** 2)) / (
            np.sum(np.abs(leftV) ** 2) + np.sum(np.abs(rightV) ** 2)
        )
        if verbose:
            # Progress Reporting
            print("Chunk %s-%s success" % (idx_f, idx_t), flush=True)
    except Exception as e:
        # If chunk cannot be recovered print Error and return zero array
        #    (Prevents failure of a single chunk from ending phase retrieval)
        print(e, flush=True)
        asymm = np.nan
    return (asymm, idx_f, idx_t)
