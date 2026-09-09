# QPMix

**Quantum mixing software — fast multi-tone spectral domain analysis of SIS mixers**

QPMix simulates the quasiparticle tunneling currents in
Superconductor/Insulator/Superconductor (SIS) junctions. In radio astronomy
these junctions are used as heterodyne mixers at millimetre and
submillimetre wavelengths, and QPMix can be used to simulate their
behaviour, optimise their performance, and analyse measurements.

Because it works in the spectral domain with multiple tones (*multi-tone
spectral domain analysis*, MTSDA), it handles power saturation, higher-order
harmonics, sub-harmonic pumping, harmonic mixing and frequency
multiplication.

---

## Relationship to QMix

**QPMix is a derivative of [QMix](https://github.com/garrettj403/QMix) by
John Garrett and Ghassan Yassin**, released under the same GPL-3.0 licence.
The physics, the class and function names, and the equation numbering all
follow QMix and P. Kittara's 2002 DPhil thesis. QPMix re-engineers the
numerical methods, the packaging and the test suite.

If you use QPMix for published work, **please cite the original QMix
papers** — see [Citing](#citing) below.

What is new here:

| | QMix | QPMix |
|---|---|---|
| Phase-factor coefficients | Bessel functions + harmonic convolution | a single FFT — and more accurate |
| Response function | cubic spline over ~100 curvature-picked knots | dense uniform table, `O(1)` lookup |
| Response matrix | voltage array materialised, then interpolated | fused, thread-parallel, no temporaries |
| Current summation | nested loops, single-threaded | cache-blocked and parallel, plus an FFT-correlation path |
| Harmonic balance | Jacobian rebuilt every iteration | Broyden rank-1 updates, batched solve, line search |
| Number of tones | hard-capped at 4 (memory grows as `(2·num_b+1)^F`) | unlimited, via a common frequency grid (memory linear in `F`) |
| Measured-data analysis | Bessel loops, 15 bisection steps, a 20 301-point double loop | vectorised sums, Newton with an analytic derivative, one broadcast |
| numba | required | optional, with a vectorised NumPy fallback |
| Packaging | `setup.py`, conda `environment.yml` | `pyproject.toml`, `uv`, `src/` layout |
| Tests | 1 705 lines | 601 tests, per module, plus analytic validation |

A complete two-tone mixer simulation runs about **6× faster**, and the
individual stages are 3–160× faster — see [Performance](#performance).

---

## Performance

### At a glance

Every algorithm changed, what it cost before and after, and what that
bought. Symbols: `F` tones, `P` harmonics, `B` = `num_b`, `N` bias points,
`M` FFT length, `T` output frequencies, `V = (2(B+P)+1)^F`,
`k` ≈ 100 spline knots.

| Stage | QMix algorithm | QPMix algorithm | Complexity | Measured |
|---|---|---|---|---|
| **Phase-factor coefficients** `C_k` | Jacobi–Anger Bessel functions, then harmonic convolution | Fourier coefficients of the phase factor — **one FFT** | `O(F·P·B²·N)` + `O(F·P·B·N)` Bessels → **`O(F·N·M log M)`** | **2.8–8.2×** (grows with `P`) |
| **Response function** eval | cubic spline over ~100 curvature-picked knots | dense uniform table + Catmull–Rom, index arithmetic | `O(log k)` per point → **`O(1)` per point** | **46–162×** |
| **Response matrix** `R[k,l,…]` | build the `(2B+1)^F·N` voltage array, then interpolate it | fused — voltage in a register, interpolated on the spot | same order, ~3× less memory traffic | **18–79×** |
| **Current summation** (Eqn. 5.25) | nested loops per output frequency, single-threaded | cache-blocked + thread-parallel; **or** FFT cross-correlation | `O(T·(2B+1)^F·N)` → same, **or `O(V log V·N)`** (no `T`) | **3.4–13.9×** |
| **Harmonic balance** | finite-difference Jacobian rebuilt every iteration | Broyden rank-1 (Sherman–Morrison) + batched solve + line search | `2FP+1` solves/iteration → **1** after the first | **1.8–6.1×** |
| **Kramers–Kronig transform** | `scipy.signal.hilbert`, complex FFT, arbitrary length | real-input `rfft`/`irfft` on a 5-smooth length | same order, ~2× fewer flops | **2.8–13.6×** |
| **Gaussian smoothing** | `numpy.convolve`, always direct | FFT convolution for wide kernels, auto-selected | `O(N·W)` → **`O(N log N)`** | used in setup |
| **JIT strategy** | eager signatures, compiled at import | lazy compilation, cached on disk; numba **optional** | — | fast `import qpmix` |
| **Tone count** | one summation index per tone, capped at 4 | all tones on a common frequency grid, single index | `(2B+1)^F·N` → **`(2K+1)·N`** (exponential → linear in `F`) | **70×** at 4 tones; 24 tones now possible at all |

**Bottom line: a complete two-tone mixer simulation runs ~6× faster, and is
~8× closer to the exact answer.**

Two of these change the *exponent* rather than the constant factor; the
rest come from how the kernels touch memory.

### The two complexity changes

**Phase factors.** The coefficients satisfy

```
sum_k C_k e^{ikθ} = prod_p exp(i α_p sin(pθ − φ_p))
                  = exp(i sum_p α_p sin(pθ − φ_p))
```

so they are just the Fourier coefficients of a function costing one `sin`,
one `exp` and one FFT. No Bessel functions, no harmonic convolution — the
quadratic `B²` term disappears.

**Current summation.** With `P = C₁⊗C₂⊗…·R` and `Q = C₁⊗C₂⊗…`, Eqn. 5.25
is exactly a cross-correlation, `RS⁺(a,b,…) = Σ P[k,l,…]·conj(Q[k+a,l+b,…])`,
so one pair of zero-padded transforms yields *every* index tuple at once.

### Measured, case by case

16 threads, NumPy 2.5.3 / SciPy 1.18.1 / numba 0.67.0, Python 3.12.

JIT compilation excluded; best of three runs. Reproduce with
`uv run python benchmarks/bench_all.py`.

| Benchmark | QMix | QPMix | Speed-up |
|---|---|---|---|
| Response function, 10⁵ points | 8.2 ms | 0.11 ms | **71×** |
| Response function, 10⁶ points | 83.8 ms | 0.52 ms | **162×** |
| Response matrix, `f=2, nb=15, N=401` | 34.7 ms | 0.44 ms | **79×** |
| Response matrix, `f=3, nb=9, N=401` | 253.0 ms | 6.3 ms | 40× |
| Kramers–Kronig transform, 7 001 pts | 26.1 ms | 1.9 ms | 13.6× |
| Phase factors, `f=2, p=4, nb=15` | 7.19 ms | 0.87 ms | 8.2× |
| `qtcurrent`, `f=2, p=1, nb=15, N=401` | 40.4 ms | 2.91 ms | **13.9×** |
| `qtcurrent`, `f=3, p=1, nb=9, N=401` | 292.7 ms | 32.6 ms | 9.0× |
| **Harmonic balance, `f=2, p=1, N=401`** | 121.0 ms | 19.8 ms | **6.1×** |

The phase-factor speed-up grows with `num_p` (2.8× at `p=1` → 8.2× at
`p=4`) exactly as the `O(P·B²) → O(M log M)` analysis predicts, and
Broyden's advantage grows with `num_f × num_p`, which is what sets how many
current solves a full Jacobian costs.

### More accurate, not only faster

| Quantity | QMix | QPMix |
|---|---|---|
| Phase factors vs untruncated reference (`p=4`) | 2.1 × 10⁻⁹ | **7.8 × 10⁻¹⁶** |
| Response function vs high-accuracy reference | 1.0 × 10⁻⁴ | **3.1 × 10⁻⁵** |
| `qtcurrent` vs high-accuracy reference | 2.0 × 10⁻⁵ | **2.4 × 10⁻⁶** |

QMix truncates every intermediate convolution at ±`num_b`; the transform
never forms them. And because evaluation is now `O(1)` regardless of table
size, the response function can be tabulated ~500× more densely than QMix's
knot set *and* still be faster. This is why cross-validation against QMix
agrees to ~10⁻⁵ rather than to machine precision — the residual is QMix's
interpolation error.

### Architecture techniques

Branch-free `O(1)` table lookup (removes a data-dependent binary search, and
the monotonic bias sweep means the table is walked sequentially, so the
prefetcher hides the latency) · loop fusion in the response matrix (~240 MB
of temporaries never written, for three tones) · cache blocking over 128
bias points, with the block index doubling as the unit of thread
parallelism · contiguous bias axis innermost so inner loops vectorise ·
real-input `rfft`/`irfft` for the Hilbert transform · 5-smooth FFT lengths
via `next_fast_len` · lazy JIT with on-disk caching so `import qpmix` stays
fast · batched `numpy.linalg` and `einsum` in harmonic balance.

### Honest limitations

- **Building a response function is ~1.8× slower** (82 ms vs 46 ms). It is a
  one-off cost, and it is what pays for the 46–162× evaluation speed-up.
- **The FFT correlation path is not the default winner.** It computes every
  index tuple whether you asked for one or sixty, so it only pays off when
  many output frequencies are wanted at once — from ~12 tuples for two
  tones, reaching 2.9× for a full intermodulation sweep. `method="auto"`
  picks per call, and its choice is within 15% of optimal across the sweep
  in `benchmarks/bench_method.py`. Without numba it always wins, because
  the direct kernels are then interpreted loops.
- **Broyden and Newton converge to slightly different points** — both meet
  the requested `stop_rerror`, but they differ by roughly that tolerance.
  Use `jacobian="newton"` to reproduce QMix's iteration exactly.
- **Speed-ups are smallest for one tone, one harmonic** at small `npts`,
  where per-call Python overhead dominates. They grow with problem size.

### Many tones

QMix asserts `num_f in [1, 2, 3, 4]`, because the response matrix holds
`(2·num_b+1)^num_f · npts` complex values — 5.9 GB at four tones, 184 GB at
five. `qpmix.multitone` lifts the cap by placing every tone on a common
frequency grid, which collapses the multi-dimensional index to a single
one and makes memory grow *linearly* with the tone count:

| tones | `num_k` | grid memory | `qtcurrent` | multi-D would need |
|---|---|---|---|---|
| 4 | 414 | 5.1 MB | 16 ms | 797 MB |
| 8 | 972 | 11.9 MB | 60 ms | 101 483 GB |
| 16 | 2 520 | 30.8 MB | 256 ms | 1.7 × 10¹⁵ GB |
| 24 | 4 644 | 56.8 MB | 699 ms | 2.9 × 10²⁵ GB |

A full 16-tone harmonic balance takes 8.7 s — a quarter of what four tones
used to cost. The grid is **opt-in below five tones** (`method="grid"`),
because it also sums intermodulation products that the multi-dimensional
engine truncates at `±num_p`, so switching automatically would change
results rather than only run times.

```python
cct = qpmix.EmbeddingCircuit(16, 1, vb_npts=401)      # no longer capped at 4
for f in range(1, 17):
    cct.freq[f] = 0.20 + 0.02 * (f - 1)
...
i = qpmix.qtcurrent(vj, cct, resp, freqs, num_b=9)     # auto-selects the grid

from qpmix.multitone import ToneGrid
print(ToneGrid.from_circuit(cct, num_b=9).report(401))  # costs, before you commit
```

`benchmarks/study_continuum.py` answers the companion question — how many
tones before a band simply *is* a continuum. For a 20%-wide band of
random-phase tones at fixed total power, **five to thirteen tones already
reproduce the 31-tone answer** to the ensemble noise floor, while the cost
keeps climbing as `N^2.15`. The ceiling is set by cost, not physics.

Full analysis, including the complete complexity table, every measured
case, and a DC double-counting bug this work uncovered in the original
tuple bookkeeping, is in **[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)**.

---

## Install

QPMix uses [uv](https://docs.astral.sh/uv/). From a clone:

```bash
git clone https://github.com/xHexlabx/QPMix.git
cd QPMix

uv sync                    # create .venv and install everything, including dev tools
uv run pytest              # run the test suite
```

Or install it as a dependency:

```bash
uv add qpmix[all]          # with numba (JIT) and matplotlib
uv pip install qpmix       # NumPy/SciPy only
```

`numba` is optional. Without it every result is identical, but the current
kernels fall back to interpreted loops — QPMix automatically switches the
current summation to its vectorised FFT path in that case, which keeps
things usable. Install `qpmix[jit]` for the fast path.

Check what backend you got:

```python
>>> import qpmix
>>> qpmix.get_config()
{'jit_enabled': True, 'parallel': True, 'num_threads': 16, 'numba_version': '0.67.0'}
```

---

## Quick start

Simulate a 230 GHz SIS mixer pumped with 50 nW:

```python
import qpmix

# The Thevenin equivalent circuit the junction sees
cct = qpmix.EmbeddingCircuit(num_f=1, num_p=1, vb_npts=401, vb_max=1.5,
                             vgap=2.8e-3, rn=14.0)
cct.set_freq(230, units='GHz')
cct.zt[1, 1] = 0.3 - 0.3j
cct.set_available_power(50, units='nW')

# The junction's response function
resp = qpmix.RespFnPolynomial(50)

# Solve for the junction voltage, then the pumped I-V curve
vj = qpmix.harmonic_balance(cct, resp, num_b=15)
idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=15)
```

Two tones — an LO plus a weak RF signal — with the IF current at the
difference frequency:

```python
cct = qpmix.EmbeddingCircuit(num_f=2, num_p=1, vb_npts=401, vb_max=1.5,
                             vgap=2.8e-3, rn=14.0)
cct.set_freq(230, f=1, units='GHz')      # LO
cct.set_freq(232, f=2, units='GHz')      # RF
cct.zt[1, 1] = cct.zt[2, 1] = 0.3 - 0.3j
cct.set_available_power(50,   f=1, units='nW')
cct.set_available_power(0.05, f=2, units='nW')

resp = qpmix.RespFnPolynomial(50)
vj = qpmix.harmonic_balance(cct, resp, num_b=15)

f_if = round(float(cct.freq[2] - cct.freq[1]), 4)
i_if = qpmix.qtcurrent(vj, cct, resp, f_if, num_b=15)
```

---

## Analyzing measured data

`qpmix.exp` turns laboratory measurements into physics. Give it a DC I-V
curve, a pumped I-V curve and a hot/cold IF measurement, and it recovers the
junction parameters, the embedding circuit and the noise temperature:

```python
from qpmix.exp import DCData, PumpedData

dciv = DCData(dc_iv_array, dc_if_array)              # two-column arrays
pumped = PumpedData(pumped_iv_array, dciv,
                    if_hot_array, if_cold_array, freq=230.0)

dciv.vgap, dciv.rn, dciv.rsg      # gap voltage, normal and subgap resistance
pumped.zt, pumped.vt              # recovered embedding circuit
pumped.alpha, pumped.zw           # drive level and junction impedance
pumped.tn_best, pumped.g_db       # noise temperature and gain
```

The import pipeline corrects the voltage/current offset, removes a series
resistance, and filters the curve by rotating it so the gap transition is
not smeared. `check_offset` then *verifies* the correction worked, by
measuring how far the curve departs from its own point reflection — turning
a judgement made by eye into something a script can assert on.

### Two ways to recover the embedding circuit

| | voltage matching | current matching |
|---|---|---|
| fits | the load line from Tucker theory | the simulated pumped I-V curve |
| cost | closed form, ~20 ms | a harmonic balance per evaluation |
| needs | one tone, first photon step | nothing beyond the simulation |
| window | first photon step only | first step, full subgap, full range, or per-step |
| harmonics | fundamental only | fundamental plus higher harmonics |

```python
from qpmix.exp import recover_zemb_current_match

result = recover_zemb_current_match(
    resp, voltage, current, vph,
    method="full_subgap",      # or first_photon / full_range / photon_steps
    harmonics=2,               # fit Z_T at f_LO and 2*f_LO
    guesses="seeded",          # seed from the fast voltage-match answer
)
result.zt          # (Z_T at f_LO, Z_T at 2*f_LO), normalized to Rn
result.occurrences # how many independent starts agreed on this
```

Being independent, the two agreeing is evidence: on a real 183.6 GHz
measurement they agree to 0.04 Ω.

The residual surface has unphysical local minima, so the answer is the one
the independent starts *agree* on — after discarding negative resistances,
negative source voltages and runaway reactances — not simply the lowest
residual. On the real measurement, four of nine starts converged on
something unphysical with a *lower* residual than the accepted answer.

Measured against the same fit driven by QMix, on real data:

| run | `zt` | rms | time |
|---|---|---|---|
| QMix, 9 starts | 0.3612 − 0.5661j | 1.057e-3 | 145.2 s |
| QPMix, same 9 starts | 0.3608 − 0.5630j | 1.056e-3 | **25.1 s** (5.8×) |
| QPMix, seeded (3 starts) | 0.3608 − 0.5619j | 1.056e-3 | **5.6 s** (25.7×) |

`current_residual` exposes the quantity the fit minimises on its own, which
is how to ask whether the data actually constrains a parameter: vary it and
watch the residual. If it barely moves, the measurement does not determine
it, however confidently the optimiser reported a value. It is also the only
fair way to compare two fits, since each resamples the measured curve its
own way.

Impedance recovery follows the RF voltage-match method of Skalare (1989)
and Withington *et al.* (1995).

Three numerical pieces are reworked, and all three are faster *and* more
accurate:

| | QMix | QPMix | speed | accuracy |
|---|---|---|---|---|
| Tucker-theory sums | Python loop over Bessel orders | every order in one call; `J_{-n} = (-1)^n J_n` halves the work | **2.6–7.1×** | identical |
| Drive-level recovery | 15 fixed bisection steps | safeguarded Newton with an analytic derivative | **7.9–8.5×** | **~10¹⁰× closer** |
| Impedance error surface | 101×201 Python double loop | one broadcast, then an off-grid polish | **2.3–19.3×** | not tied to the grid |

Bisection over 15 steps cannot do better than `alpha_max / 2**15 ≈ 5e-5`;
the Newton iteration reaches machine precision.

### Testing without measured data

`qpmix.exp.simulate` builds synthetic measurements from *known* parameters
and then adds the distortions a real one has — offset, series resistance,
gain errors, a hysteretic double sweep, noise. The test suite is therefore a
round trip: feed in a junction fed by a known Thevenin source, and check the
analysis recovers it.

```python
from qpmix.exp.simulate import simulate_dciv, simulate_embedded_iv

dciv = DCData(simulate_dciv(vgap=2.8e-3, rn=14.0), verbose=False)
raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=230.0)
pumped = PumpedData(raw, dciv, freq=230.0, verbose=False)
# pumped.zt -> 0.4000-0.3000j,  pumped.vt -> 0.3503
```

That round trip closes to better than `6e-3` in both quantities, and the
drive level to `1e-9`.

---

## Units

Everything is normalized, as in QMix:

| quantity | normalized to |
|---|---|
| voltage | gap voltage `Vgap` |
| current | gap current `Igap = Vgap / Rn` |
| impedance | normal-state resistance `Rn` |
| frequency | gap frequency `Fgap = Vgap · e / h` |

`EmbeddingCircuit.set_freq` and `set_available_power` accept physical units
(`'GHz'`, `'mV'`, `'nW'`, `'dBm'`, …) and convert for you.

Indexing follows the published equations: tone index `f` runs 1…`num_f` and
harmonic index `p` runs 1…`num_p`, with index 0 reserved for DC.

---

## API

| | |
|---|---|
| `EmbeddingCircuit` | the Thevenin circuit, bias sweep and junction properties |
| `RespFn`, `RespFnFromIVData` | response function from measured DC I-V data |
| `RespFnPerfect`, `RespFnPolynomial`, `RespFnExponential` | response function from a model |
| `harmonic_balance` | solve for the junction voltage |
| `qtcurrent` | tunneling current at any set of frequencies |
| `interpolate_respfn` | pre-interpolate the response function to reuse across calls |
| `ToneGrid`, `qtcurrent_grid` | the common-grid engine for many tones (`qpmix.multitone`) |
| `qpmix.exp.DCData`, `PumpedData` | import and analyze measured data |
| `calculate_phase_factor_coeff` | phase-factor spectrum coefficients |
| `check_hb_error` | independently verify a harmonic balance solution |
| `read_circuit` | load a circuit from JSON or the legacy text format |
| `get_config`, `set_num_threads` | compute backend |

Most functions take `verbose=True` by default; pass `verbose=False` to
silence them.

---

## Testing and benchmarking

```bash
uv run pytest                                  # 601 tests
uv run pytest -m "not reference"               # skip QMix cross-validation
uv run pytest --cov=qpmix --cov-report=term    # with coverage
QPMIX_DISABLE_JIT=1 uv run pytest              # exercise the NumPy fallback

uv run python benchmarks/bench_all.py          # QMix vs QPMix, stage by stage
uv run python benchmarks/bench_method.py       # direct vs FFT cost model
uv run python benchmarks/bench_multitone.py    # scaling with the tone count
uv run python benchmarks/study_continuum.py    # how many tones make a continuum
uv run python benchmarks/bench_exp.py          # experimental-data analysis
```

The test suite is not only a comparison against QMix. The key checks are
against **Tucker theory** (Tucker and Feldman, *Rev. Mod. Phys.* **57**,
1055, 1985), which gives closed-form expressions for the single-tone
tunneling currents — an independent analytic ground truth. QPMix reproduces
them to 10⁻¹¹.

To run the QMix cross-validation tests, install the reference:

```bash
uv pip install QMix
```

Environment variables: `QPMIX_DISABLE_JIT=1` forces the pure-NumPy path,
`QPMIX_DISABLE_PARALLEL=1` keeps the JIT but runs single-threaded.

---

## Citing

QPMix is derivative work. Please cite the original QMix papers:

```bibtex
@article{Qmix1,
  author  = {J. D. Garrett and G. Yassin},
  title   = {{QMix: A Python package for simulating the quasiparticle
             tunneling currents in SIS junctions}},
  journal = {Journal of Open Source Software},
  year    = 2019, volume = 4, number = 35, pages = 1231,
  doi     = {10.21105/joss.01231},
}

@article{Qmix2,
  author  = {J. D. Garrett and B.-K. Tan and F. Boussaha and
             C. Chaumont and G. Yassin},
  title   = {{Simulating the Behavior of a 230-GHz SIS Mixer Using
             Multitone Spectral Domain Analysis}},
  journal = {IEEE Transactions on Terahertz Science and Technology},
  year    = 2019, volume = 9, number = 6, pages = {540--548},
  doi     = {10.1109/TTHZ.2019.2938993},
}
```

The theory behind the multi-tone spectral domain analysis is from:

- P. Kittara, *The Development of a 700 GHz SIS Mixer with Nb Finline
  Devices*, DPhil thesis, University of Oxford, 2002.
- S. Withington and E. Kollberg, *IEEE Trans. Microw. Theory Tech.* **37**,
  231 (1989).
- J. R. Tucker and M. J. Feldman, *Rev. Mod. Phys.* **57**, 1055 (1985).

See `CITATION.cff` for machine-readable citation metadata.

---

## Licence

GNU General Public License v3.0 or later — the same licence as QMix, which
QPMix derives from. See [`LICENSE`](LICENSE).
