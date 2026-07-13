# Scintools Code Review

Scope: `scintools/dynspec.py`, `scintools/scint_models.py`, `scintools/scint_sim.py`,
`scintools/scint_utils.py`, `scintools/ththmod.py`. Review performed read-only against
the installed environment (numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.0; skimage/emcee/bilby absent).

## Summary

Overall the scientific core is reasonable, but there are several **confirmed, high-impact
breakages**, most of them triggered by ordinary default usage or by the move to modern NumPy:

1. **`ACF()` is completely broken on NumPy >= 2.0** because it uses the removed alias
   `np.complex_` (`scint_sim.py`). This also takes down the analytic 2D ACF model in
   `scint_models.scint_acf_model_2d` and the `method='acf2d'` fit path in `dynspec.py`.
2. **`Dynspec.fit_arc()` raises `TypeError` in its default (non-lamsteps) mode** because the
   default `constraint=[0, np.inf]` is a Python list that is divided by a float.
3. **`SimDyn.__init__` raises `AttributeError` immediately** (`self.header = self.header`),
   so importing a `Simulation` into the Dynspec framework via `SimDyn` never works.
4. A copy/paste slip in `get_scint_params` (`acf2d` cropping) corrupts the time-axis crop when
   `nscale` exceeds the frequency range.
5. Broad code duplication (SVD models, the chunk-mask block repeated ~6× in `ththmod.py`, window
   construction, NaN interpolation) and several 300–500 line functions make maintenance risky.

Counts: **10 confirmed bugs**, **5 plausible/suspected issues**.

## Bugs (most severe first)

### 1. `np.complex_` removed in NumPy 2.0 — `ACF` class unusable
- **File**: `scint_sim.py:589` and `scint_sim.py:634` (dtype=`np.complex_`)
- **Severity**: High · **Confidence**: Confirmed
- **What's wrong**: `np.complex_` was removed in NumPy 2.0.
- **Why it fails**: Verified — `ACF(nf=11, nt=11)` raises
  `AttributeError: 'np.complex_' was removed in the NumPy 2.0 release. Use 'np.complex128' instead.`
  Because `ACF` is imported and used by `scint_models.scint_acf_model_2d` (line 190) and by the
  `method='acf2d'` branch of `Dynspec.get_scint_params`, the analytic 2D ACF fit is dead on modern NumPy.
- **Fix**: Replace `np.complex_` with `np.complex128`. (Also `np.complex_` in `scint_sim.py` are the
  only two occurrences.)

### 2. `fit_arc` divides a list by a float in the default path
- **File**: `dynspec.py:1147-1148` (`constraint = constraint/(self.freq/ref_freq)**2`)
- **Severity**: High · **Confidence**: Confirmed
- **What's wrong**: The public default is `constraint=[0, np.inf]` (line 973), a Python list.
  When `lamsteps=False` (the default) the `if not lamsteps:` block divides `constraint` by a scalar.
- **Why it fails**: Verified — `[0, np.inf] / 1.0` raises
  `TypeError: unsupported operand type(s) for /: 'list' and 'float'`. So `dyn.fit_arc()` with defaults
  crashes; it only works if the caller passes `constraint` as an ndarray or uses `lamsteps=True`.
- **Fix**: Coerce once at the top, e.g. `constraint = np.array(constraint, dtype=float)`.

### 3. `SimDyn.__init__` reads `self.header` before it is set
- **File**: `dynspec.py:4283` (`self.header = self.header`)
- **Severity**: High · **Confidence**: Confirmed
- **What's wrong**: `self.header` is assigned from itself before ever being defined.
- **Why it fails**: A fresh `SimDyn` instance has no `header` attribute, so the RHS raises
  `AttributeError`. `SimDyn(sim)` therefore always fails.
- **Fix**: Set it from the simulation, e.g. `self.header = [self.name]` (mirroring `Simulation.header`).

### 4. Copy/paste error in `get_scint_params` acf2d cropping
- **File**: `dynspec.py:2766-2767` (inside `if ndnu > (self.bw / dnu):`)
- **Severity**: Medium · **Confidence**: Confirmed (by inspection)
- **What's wrong**: The frequency-overflow branch sets `tmin = 0; tmax = nf`, but it should set the
  **frequency** indices (`fmin = 0; fmax = nf`). The analogous time branch (lines 2754-2755) correctly
  sets `tmin/tmax`. The subsequent crop is `ydata_2d = ydata_centered[fmin:fmax, tmin:tmax]` (line 2775).
- **Why it fails**: When `nscale` exceeds the available frequency range, the time indices are clobbered
  (and with `nf`, a frequency count, on a time axis), while `fmin/fmax` keep their earlier centred values.
  The 2D fit is then cropped over the wrong region.
- **Fix**: Change to `fmin = 0; fmax = nf`.

### 5. `BasicDyn` mutable default args + `.size` on a list default
- **File**: `dynspec.py:4148` (`header=["BasicDyn"], times=[], freqs=[]`) and `dynspec.py:4195`
  (`if times.size == 0 or freqs.size == 0:`)
- **Severity**: Medium · **Confidence**: Confirmed
- **What's wrong**: (a) Mutable default arguments (`["BasicDyn"]`, `[]`, `[]`). (b) The guard calls
  `.size` on `times`/`freqs`, but the defaults are lists, which have no `.size`.
- **Why it fails**: Verified — `[].size` raises `AttributeError`. So `BasicDyn(dyn)` with default
  time/freq axes raises `AttributeError` instead of the intended `ValueError`. (Callers such as
  `Dynspec.__add__` pass ndarrays, so they avoid it, but the documented direct construction breaks.)
- **Fix**: Use `times=None`/`freqs=None` defaults and check `if times is None or len(times) == 0`.

### 6. `tau_sspec_model` / `dnu_sspec_model` signature mismatch
- **File**: `scint_models.py:218` and `:248` (defined with 3 params) vs `scint_models.py:281-282`
  (called with 4 args, passing `weights[0]`/`weights[1]`)
- **Severity**: Medium · **Confidence**: Confirmed (by inspection); currently latent
- **What's wrong**: `scint_sspec_model` calls `tau_sspec_model(params, xdata[0], ydata[0], weights[0])`
  but the callees accept only `(params, xdata, ydata)`. Calling `scint_sspec_model` would raise
  `TypeError: too many positional arguments`.
- **Why it doesn't currently bite**: The only caller (`dynspec.py:2937`) is commented out and the
  `sspec` method prints "This method doesn't work yet". Still a real defect if the path is revived.
- **Fix**: Give the `_sspec_model` functions a `weights` parameter (matching the acf models) and use it.

### 7. `except NameError` cannot catch a missing dict key
- **File**: `dynspec.py:4233-4241` (`MatlabDyn`), also relevant to `SimDyn`
- **Severity**: Low · **Confidence**: Confirmed
- **What's wrong**: `self.matfile['spi']` / `self.matfile['dlam']` raise `KeyError` when absent, but the
  handlers catch `NameError`.
- **Why it fails**: Verified — a missing dict key raises `KeyError`, which the `except NameError` clause
  does not catch, so the intended friendly `NameError('No variable named ...')` is never raised; the raw
  `KeyError` propagates instead.
- **Fix**: `except KeyError:`.

### 8. `calc_sspec` trapezoid branch checks the wrong attribute
- **File**: `dynspec.py:3657` (`if not hasattr(self, 'trap'):`)
- **Severity**: Low · **Confidence**: Confirmed
- **What's wrong**: The cached array is `self.trapdyn`, but the guard tests `self.trap`. Since `self.trap`
  is never set, the branch always recomputes `scale_dyn(scale='trapezoid')`. Functionally correct output
  but redundant work / dead cache.
- **Fix**: `if not hasattr(self, 'trapdyn'):`.

### 9. `plot_dyn` intensity-scaling mask loses the nonzero filter
- **File**: `dynspec.py:500-506`
- **Severity**: Low · **Confidence**: Confirmed
- **What's wrong**: The mask is written `dyn[is_valid(dyn)*np.array(np.abs(is_valid(dyn)) > 0)]`.
  `np.abs(is_valid(dyn)) > 0` is just `is_valid(dyn)` again, so the expression reduces to
  `is_valid(dyn)` and the "nonzero pixel" filter is silently dropped. Compare `plot_sspec` (line 787),
  which correctly uses `np.abs(sspec) > 0`.
- **Why it matters**: `vmin/vmax` for the dynamic-spectrum plot are computed over zero-filled pixels too,
  skewing the colour scale. Cosmetic only.
- **Fix**: Use `is_valid(dyn) * (np.abs(dyn) > 0)`.

### 10. `arc_power_curve` returns a broken/empty model
- **File**: `scint_models.py:287-297`
- **Severity**: Low · **Confidence**: Confirmed (dead/placeholder)
- **What's wrong**: `model = []` then `return (ydata - model) * weights`. `ydata - []` broadcasts to an
  empty array (or errors depending on shapes); the function cannot produce meaningful residuals.
- **Fix**: Either implement the template or remove the stub.

## Plausible / suspected issues

- **`effective_velocity_annual` may use `INC` before assignment** — `scint_models.py:532-555`.
  If none of `KIN`/`COSI`/`SINI` is in `params`, the code only prints a warning and never sets `INC`,
  then references `np.sin(INC)` at line 555 → `NameError`. Confidence: high that it errors, but only on
  a malformed parameter set.
- **`norm_sspec` logsteps path applies the wrong mask** — `dynspec.py:2116-2117`. `masklin` is computed
  but the masked array is built with `mask=mask` (the non-log mask). Looks like a copy/paste slip;
  `normSspeclin` should probably use `masklin`. Only affects `logsteps=True`.
- **`scint_velocity` error propagation** — `scint_utils.py:760-763`. `viss_err` adds `coeff_err`
  (a variance-like term) directly inside `np.sqrt(coeff**2*(...) + coeff_err)`; dimensionally this looks
  like it should be `coeff_err**2` (or `coeff_err` should already be a variance). Worth a domain check.
- **In-place mutation of caller weights** — `scint_models.py:81,105` (`weights[0] = 0`). `tau_acf_model`
  / `dnu_acf_model` zero the first weight of the array passed in by the caller. It is idempotent here so
  harmless in practice, but it mutates `get_scint_params`' `weights_t`/`weights_f` arrays as a side effect.
- **`np.float128` is not portable** — `scint_utils.py:543`, `dynspec.py:3982-3983`. It exists on this
  Linux host but is absent on Windows and Apple-silicon macOS, where these lines will raise
  `AttributeError`. Consider `np.longdouble`.

## Refactoring opportunities

1. **Duplicated `svd_model`** — `scint_utils.py:705` and `ththmod.py:18`. Same SVD core, but the utils
   version returns `(arr/|model|, model)` while the ththmod version returns only `model`. `dynspec.py`
   imports the utils one; ththmod uses its own. Consolidate into one function with a documented return
   contract to avoid drift.
2. **The chunk/mask block is copy-pasted ~6 times in `ththmod.py`** — `mosaic` (1515-1553), `rotMos`
   (1734-1769), `rotInit` (1815-1856), `rotDer` (1878-1919), `fullMos` (1948-1987), `fullMosGrad`
   (2044-2101), `fullMosHess` (2132-2239). The `mask_func` weighting logic is identical each time.
   Extract a helper `chunk_mask(cf, ct, ncf, nct, cwf, cwt)`.
3. **Window construction is duplicated** — `scint_utils.get_window` (810) vs the inline hanning/hamming/
   blackman/bartlett block in `dynspec.scale_dyn` (4086-4105). Call `get_window` in `scale_dyn`.
4. **NaN interpolation duplicated** — `scint_utils.interp_nan_2d` (769) vs the inline griddata "clean"
   block in `dynspec.calc_scattered_image` (3527-3548). Reuse the utility.
5. **Overly long functions** — `Dynspec.get_scint_params` (~2470-3156, ~500 lines), `Dynspec.fit_arc`
   (~970-1347), `Dynspec.norm_sspec` (~1920-2281), `ACF.calc_acf` in scint_sim. These would benefit from
   decomposition (e.g. separate 1D-init, approx-2D, analytic-2D, and plotting stages in
   `get_scint_params`).
6. **Dead / commented-out code** — the `sspec` method body in `get_scint_params` (2911-2944) is entirely
   commented and prints "doesn't work yet"; `scint_models.arc_power_curve` is a stub;
   `scint_utils.make_dynspec` (894) is an empty placeholder; various commented arrays in `scint_sim`.
7. **Unused / misleading parameters** — `Dynspec.trim_edges(..., remove_short_sub=True)` (259) never uses
   `remove_short_sub`; `auto_processing` forwards it (line 435). Either wire it up or drop it.
8. **Magic numbers / hardcoded constants** — the speed of light is redefined as a literal `299792458.0`
   in many places (`dynspec.py:904,1141,2033,3500`, `scint_sim.py:130`) alongside available
   `scipy.constants`/`astropy.constants`. The LSR solar motion `(11.1, 12.24, 7.25)` in
   `scint_utils.make_lsr` (343) and galactic radius/velocities in `differential_velocity` are hardcoded.
9. **`slow_FT` uses a hardcoded reference frequency** at mid-band (`scint_utils.py:682-684`), flagged in
   its own comment ("should change this").

## Deprecations / compatibility (modern numpy/scipy/matplotlib)

- **`np.complex_`** — removed in NumPy 2.0. Breaks `ACF` (see Bug 1). Use `np.complex128`.
  (`np.complex_dtype` scan: only `scint_sim.py:589,634`.)
- **`np.float128`** — not a portable alias (absent on Windows / Apple silicon). `scint_utils.py:543`,
  `dynspec.py:3982-3983`. Prefer `np.longdouble`.
- **`plt.colorbar` missing call parentheses** — `scint_sim.py:399` (`plt.colorbar` with no `()`), so no
  colorbar is ever drawn in `Simulation.plot_pulse`. Not a deprecation but a latent no-op.
- No usages of the fully-removed `np.int`/`np.float`/`np.bool` aliases were found in these modules
  (the earlier-numpy breakages are limited to `np.complex_` and the `np.float128` portability caveat).
- `scipy`/`matplotlib` APIs used (`convolve2d`, `medfilt`, `savgol_filter`, `RectBivariateSpline`,
  `pcolormesh(..., shading='auto')`, `eigsh`, `curve_fit`) are all current in the installed versions.
