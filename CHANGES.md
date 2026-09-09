# Changelog

## 0.1.0 — initial release

First release of QPMix, a re-engineered derivative of
[QMix 1.0.6](https://github.com/garrettj403/QMix) by John Garrett and
Ghassan Yassin, under the same GPL-3.0 licence. The physics and equation
numbering follow QMix; the numerical methods, packaging and tests are new.

### Algorithms

- **Phase-factor coefficients via FFT** (`qpmix.phase_factor`). The
  Jacobi-Anger expansion plus harmonic convolution — `O(num_p · num_b²)`
  with `num_p · num_b` Bessel evaluations per bias point — is replaced by a
  single `O(M log M)` transform. 2.8–8.2× faster, and *more* accurate: the
  transform never forms the intermediate convolutions that QMix truncates
  (7.8 × 10⁻¹⁶ vs 2.1 × 10⁻⁹ against an untruncated reference at
  `num_p = 4`). The original algorithm is kept as `method="direct"`.
- **Uniform-grid response function** (`qpmix.interp`, `qpmix.respfn`). The
  non-uniform cubic spline is replaced by a dense uniform table evaluated by
  index arithmetic, `O(1)` per point with no data-dependent branch.
  46–162× faster to evaluate, and ~3× more accurate than QMix against a
  high-accuracy reference.
- **Fused response-matrix construction** (`qpmix._respmat`). The shifted
  bias voltages are computed in registers and interpolated immediately
  rather than materialised. 18–79× faster.
- **FFT cross-correlation for the current summation** (`qtcurrent`,
  `method="fft"`). Eqn. 5.25 is a cross-correlation, so one pair of
  zero-padded transforms yields every index tuple at once, independent of
  how many output frequencies are requested. Up to 2.9× faster when a full
  intermodulation sweep is wanted; `method="auto"` picks per call.
- **Broyden harmonic balance** (`harmonic_balance`, `jacobian="broyden"`).
  Rank-one Sherman-Morrison updates replace rebuilding the finite-difference
  Jacobian every iteration, cutting `2·num_f·num_p + 1` current solves per
  iteration down to one. Up to 6.1× faster end-to-end. `jacobian="newton"`
  restores QMix's iteration.
- **Real-input, 5-smooth Kramers-Kronig transform** (`mathfn.kktrans`).
  `rfft`/`irfft` instead of the full complex analytic signal, on lengths
  rounded up by `next_fast_len`. 2.8–13.6× faster.
- **Adaptive damping** in harmonic balance: a step that increases the global
  residual is backtracked instead of applied.
- **FFT convolution for Gaussian smoothing** (`mathfn.filters`), selected
  automatically for wide kernels.

### Beyond four tones

- **`qpmix.multitone`**: a common-grid engine that lifts QMix's
  `assert num_f in [1, 2, 3, 4]`. Placing every tone on a shared frequency
  grid collapses the multi-dimensional summation index to a single one, so
  the response matrix holds `(2*num_k + 1) * npts` values instead of
  `(2*num_b + 1)**num_f * npts` — **linear in the tone count rather than
  exponential**. The phase factor is still one FFT, however many tones it
  contains, and the current summation reduces to the one-dimensional
  correlation the single-tone kernel already implements.
  - 70x faster than the multi-dimensional engine at four tones, 2.7x at
    three; 24 tones now run in 0.7 s where the direct method would have
    needed 2.9e25 GB.
  - A full 16-tone harmonic balance takes 8.7 s — a quarter of what four
    tones used to cost.
  - Opt-in below five tones (`method="grid"`), because the grid also sums
    intermodulation products that the multi-dimensional engine truncates at
    `±num_p`; switching automatically would change results, not just run
    times.
  - `ToneGrid` fits tones to a grid by rational approximation, reports the
    frequency error it introduces, refuses to allocate a grid that closely
    spaced tones would blow up, and takes a `max_num_k` budget as the cost
    knob (`max_denominator` is *not* monotonic in cost).
- `EmbeddingCircuit` no longer caps `num_f` at four.
- `benchmarks/bench_multitone.py` measures the scaling;
  `benchmarks/study_continuum.py` answers how many tones make a band a
  continuum — about 5-13 for a 20%-wide band at the ensemble noise floor,
  while cost keeps climbing as `N^2.15`.

### Fixed: DC double-counting in the tuple bookkeeping

Validating the grid engine against the photon-number sum rule
`Idc = sum_K |C_K|^2 Idc0(V0 + K*df)` exposed a bug inherited from QMix. At
zero output frequency both an index tuple `t` and its negation `-t` satisfy
the matching condition, but Eqn. 5.26 for `t` already contains the `-t`
contribution through `RS-(t) = RS+(-t)`, so the DC current was counted
twice. It needs `num_p >= 2` *and* commensurate tones to trigger, which is
why it had gone unnoticed; for three tones at 0.30/0.32/0.34 it is a 5%
error in the DC current.

QPMix now counts each `±` pair once and drops the quadrature term at DC,
where the current is real by construction. Both engines then agree with the
sum rule to 6e-13 and with each other to 1e-10.

### Architecture

- Cache-blocked current kernels (`_kernels.BLOCK`), with the block index
  doubling as the unit of thread-level parallelism. QMix's kernels are
  single-threaded.
- Contiguous bias axis innermost throughout, so inner loops are unit-stride
  and vectorise.
- Lazy JIT compilation with on-disk caching, so `import qpmix` stays fast.
  QMix compiles every kernel eagerly at import.
- Batched linear algebra in harmonic balance: one batched solve and one
  `einsum` instead of a Python loop over signals.
- **numba is optional.** Without it, results are identical and the current
  summation switches to its fully vectorised FFT path automatically.
  `QPMIX_DISABLE_JIT=1` and `QPMIX_DISABLE_PARALLEL=1` control the backend.

### Correctness and robustness

- `polynomial` I-V model rewritten to be overflow-free; the textbook form
  overflows float64 for orders above ~50.
- `numpy.seterr` global mutation replaced with scoped `numpy.errstate`, and
  a numerically stable logistic in the exponential I-V models.
- `assert` replaced with `ValueError`/`TypeError` for argument validation,
  so checks survive `python -O`.
- `perfect_kk` singularity at `v = ±1` filled by interpolation rather than a
  large sentinel, which would corrupt neighbouring interpolation stencils.
- `slope_span_n` rewritten: the span now shrinks symmetrically towards the
  edges instead of shadowing its own loop variable.
- Unknown keyword arguments to response-function classes are rejected.

### Packaging and API

- `pyproject.toml` with `hatchling`, `src/` layout, `uv`-based workflow,
  Python 3.10+.
- Type hints throughout; `ruff`-clean.
- Circuits round-trip through JSON as well as the legacy QMix text format.
- Unit conversion driven by lookup tables (`POWER_UNITS`, `FREQ_UNITS`).
- 376 tests across every module, including validation against Tucker theory
  and the photon-number sum rule (independent analytic ground truths) and
  optional cross-validation against QMix itself.

### Not included in this release

- `qmix.exp` — the experimental data-analysis subpackage (`exp_data`,
  `iv_data`, `if_data`, `clean_data`, `parameters`, `if_response`) has not
  been ported.
