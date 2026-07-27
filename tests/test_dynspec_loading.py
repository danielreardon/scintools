"""
Regression tests for psrflux dynamic-spectrum loading in scintools.dynspec.

These cover bugs found while investigating a report of a "misaligned"
secondary spectrum:

* ``remove_short_subs`` used ``<=`` (with a redundant, always-true
  ``sdt >= 0``), so once the remaining sub-integrations were uniformly
  spaced it kept deleting them instead of stopping.
* the channel bandwidth was computed as span/nchan rather than from the
  spacing between channel centres (span/(nchan-1)), mis-scaling the
  secondary-spectrum delay axis, which is built as ``tdel ~ 1/df``.
* non-uniform frequency channels (gaps / stitched sub-bands) were
  silently assumed uniform by the ACF and secondary-spectrum FFTs.
"""

import warnings

import numpy as np
import pytest

from scintools.dynspec import Dynspec


def write_psrflux(path, times_min, freqs_mhz, flux=None):
    """Write a minimal psrflux-format dynamic spectrum file.

    ``flux`` is indexed [channel_as_listed, subint]; if None, all ones.
    """
    with open(path, 'w') as fn:
        fn.write("# Dynamic spectrum computed by psrflux\n")
        fn.write("# MJD0: 59341.0\n")
        fn.write("# isub ichan time(min) freq(MHz) flux flux_err\n")
        for i, t in enumerate(times_min):
            for j, fr in enumerate(freqs_mhz):
                val = 1.0 if flux is None else flux[j, i]
                fn.write("{0} {1} {2:.6f} {3:.6f} {4:+.6e} {5:+.6e}\n"
                         .format(i, j, t, fr, val, 0.0))
    return str(path)


def load(path, **kwargs):
    """Load quietly (these files intentionally exercise warning paths)."""
    kwargs.setdefault('process', False)
    kwargs.setdefault('verbose', False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Dynspec(filename=path, **kwargs)


# ---------------------------------------------------------------------------
# remove_short_subs
# ---------------------------------------------------------------------------

def test_remove_short_subs_removes_only_the_short_first_subint(tmp_path):
    # 6 s first gap, then a uniform 30 s cadence. Exactly one leading
    # sub-integration is short and should be removed.
    times = [0.0, 0.1, 0.6, 1.1, 1.6, 2.1]
    freqs = [1400.0, 1399.0, 1398.0, 1397.0]
    path = write_psrflux(tmp_path / "short.dynspec", times, freqs)

    d = load(path, remove_short_subs=True)
    assert d.nsub == len(times) - 1
    assert d.dyn.shape[1] == len(times) - 1


def test_remove_short_subs_keeps_uniform_cadence_intact(tmp_path):
    # Perfectly uniform cadence: nothing should be removed.
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    freqs = [1400.0, 1399.0, 1398.0, 1397.0]
    path = write_psrflux(tmp_path / "uniform.dynspec", times, freqs)

    d = load(path, remove_short_subs=True)
    assert d.nsub == len(times)


def test_remove_short_subs_disabled_keeps_everything(tmp_path):
    times = [0.0, 0.1, 0.6, 1.1, 1.6, 2.1]
    freqs = [1400.0, 1399.0, 1398.0, 1397.0]
    path = write_psrflux(tmp_path / "short2.dynspec", times, freqs)

    d = load(path, remove_short_subs=False)
    assert d.nsub == len(times)


# ---------------------------------------------------------------------------
# channel bandwidth (centres, not span/nchan)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nchan", [4, 5, 16])
def test_df_matches_true_channel_spacing(tmp_path, nchan):
    spacing = 0.5  # MHz
    freqs = [1400.0 - i*spacing for i in range(nchan)]  # descending
    path = write_psrflux(tmp_path / "df{0}.dynspec".format(nchan),
                         [0.0, 0.5], freqs)

    d = load(path, remove_short_subs=False)
    # df must be the spacing between channel centres, not span/nchan
    assert d.df == pytest.approx(spacing, rel=1e-9)
    assert d.bw == pytest.approx(nchan*spacing, rel=1e-9)


def test_df_from_real_example_file():
    # The bundled J0437-4715 file is 400 MHz across 512 channels.
    path = ("scintools/examples/data/J0437-4715/"
            "p111220_074112.rf.pcm.dynspec")
    d = load(path)
    assert d.nchan == 512
    assert d.df == pytest.approx(400.0/512, rel=1e-9)
    assert np.all(np.diff(d.freqs) > 0)  # ascending


# ---------------------------------------------------------------------------
# non-uniform channels should warn (FFTs assume a uniform grid)
# ---------------------------------------------------------------------------

def test_non_uniform_channels_warn(tmp_path):
    # A sub-band excised: spacing goes 100, 200, 100, 100
    freqs = [1500.0, 1400.0, 1300.0, 1100.0, 1000.0]
    path = write_psrflux(tmp_path / "gap.dynspec", [0.0, 0.5], freqs)

    with pytest.warns(UserWarning, match="not uniformly spaced"):
        Dynspec(filename=path, process=False, verbose=False,
                remove_short_subs=False)


def test_uniform_channels_do_not_warn(tmp_path):
    freqs = [1400.0 - i*0.5 for i in range(8)]
    path = write_psrflux(tmp_path / "nogap.dynspec", [0.0, 0.5], freqs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Dynspec(filename=path, process=False, verbose=False,
                remove_short_subs=False)
    msgs = [str(w.message) for w in caught
            if "uniformly spaced" in str(w.message)]
    assert msgs == []


# ---------------------------------------------------------------------------
# descending-frequency files keep flux matched to (freq, time)
# ---------------------------------------------------------------------------

def test_descending_file_preserves_flux_freq_pairing(tmp_path):
    freqs = [1400.0, 1300.0, 1200.0, 1100.0]      # descending in the file
    times = [0.0, 0.5, 1.0]
    # flux[channel_as_listed, subint] with distinct, identifiable values
    flux = np.array([[j*100 + i for i in range(len(times))]
                     for j in range(len(freqs))], dtype=float)
    path = write_psrflux(tmp_path / "desc.dynspec", times, freqs, flux=flux)

    d = load(path, remove_short_subs=False)

    assert np.all(np.diff(d.freqs) > 0)           # stored ascending
    # lowest frequency (1100) was the last row in the file -> flux row 3
    assert d.freqs[0] == pytest.approx(1100.0)
    np.testing.assert_allclose(d.dyn[0], flux[3])
    # highest frequency (1400) was the first row in the file -> flux row 0
    assert d.freqs[-1] == pytest.approx(1400.0)
    np.testing.assert_allclose(d.dyn[-1], flux[0])


def test_ascending_file_preserves_flux_freq_pairing(tmp_path):
    freqs = [1100.0, 1200.0, 1300.0, 1400.0]      # ascending in the file
    times = [0.0, 0.5, 1.0]
    flux = np.array([[j*100 + i for i in range(len(times))]
                     for j in range(len(freqs))], dtype=float)
    path = write_psrflux(tmp_path / "asc.dynspec", times, freqs, flux=flux)

    d = load(path, remove_short_subs=False)

    assert np.all(np.diff(d.freqs) > 0)
    np.testing.assert_allclose(d.dyn[0], flux[0])     # 1100 -> file row 0
    np.testing.assert_allclose(d.dyn[-1], flux[3])    # 1400 -> file row 3


# ---------------------------------------------------------------------------
# the delay axis depends on df, so the fix propagates to calc_sspec
# ---------------------------------------------------------------------------

def test_delay_axis_uses_true_channel_spacing(tmp_path):
    spacing = 0.5  # MHz
    nchan, nsub = 16, 16
    freqs = [1400.0 - i*spacing for i in range(nchan)]
    times = [0.5*i for i in range(nsub)]
    rng = np.random.default_rng(0)
    flux = rng.standard_normal((nchan, nsub)) + 10.0
    path = write_psrflux(tmp_path / "sspec.dynspec", times, freqs, flux=flux)

    d = load(path, remove_short_subs=False)
    d.calc_sspec()

    # tdel = td / (nrfft * df); the maximum delay is set by 1/(2*df)
    assert d.tdel[0] == pytest.approx(0.0)
    assert np.max(d.tdel) < 1.0/(2*d.df) + 1e-9
    # spacing between delay bins reflects the true df
    nrfft = 2*len(d.tdel)
    np.testing.assert_allclose(np.diff(d.tdel)[0], 1.0/(nrfft*d.df),
                               rtol=1e-9)
