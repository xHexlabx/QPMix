# Performance: what changed, and why

QPMix keeps QMix's physics and equation numbering but replaces the
numerical method behind each stage. This document gives the complexity
analysis first, then the measured numbers, then the reasoning behind each
change.

Everything below is reproducible:

```bash
uv run python benchmarks/bench_all.py       # the tables in §2
uv run python benchmarks/bench_method.py    # the direct/FFT cost model
```

---

## 1. Complexity summary

Symbols used throughout:

| symbol | meaning | typical |
|---|---|---|
| `F` | number of fundamental tones (`num_f`) | 1–4 |
| `P` | number of harmonics per tone (`num_p`) | 1–3 |
| `B` | phase-factor summation limit (`num_b`) | 9–20 |
| `N` | bias voltage points (`vb_npts`) | 201–801 |
| `K` | points in the response-function table | 1.4 × 10⁵ |
| `T` | index tuples needed = roughly the number of output frequencies | 2–60 |
| `M` | FFT length for the phase factors | 64–256 |
| `V` | padded correlation volume, `(2(B+P)+1)^F` | 10³–10⁴ |

### 1.1 At a glance

Every algorithm that changed, what it cost before and after, and what that
bought.

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

**Bottom line: a complete two-tone mixer simulation runs ~6× faster, and is
~8× closer to the exact answer.**

### 1.2 Per stage, in detail

| Stage | QMix | QPMix | Note |
|---|---|---|---|
| **Kramers–Kronig transform** (once) | `O(K·n log(K·n))` on an arbitrary length, complex FFT | `O(K·n log(K·n))` on a 5-smooth length, **real** FFT | same order, ~2× fewer flops, and a good transform length instead of an arbitrary one |
| **Response function evaluation** | `O(log k)` **per point** (binary search over `k ≈ 100` spline knots, FITPACK call) | `O(1)` **per point** (index arithmetic on a uniform table) | removes a data-dependent branch; vectorises and prefetches |
| **Phase-factor coefficients** `C_k` | `O(F·P·B·N)` Bessel evaluations **plus** `O(F·P·B²·N)` convolution | `O(F·N·M log M)` — one FFT | the Bessel functions and the convolution both disappear |
| **Response matrix** `R[k,l,…]` | build a voltage array of `(2B+1)^F·N` floats, then interpolate it | fused: `(2B+1)^F·N` table lookups, nothing else written | ~3× less memory traffic; thread-parallel |
| **Current summation** (Eqn. 5.25) | `O(T·(2B+1)^F·N)`, single-threaded | same order, cache-blocked and thread-parallel; **or** `O(V log V·N)` via FFT correlation, independent of `T` | `auto` picks whichever the cost model says is cheaper |
| **Harmonic balance** | `(2FP + 1)` current solves **per iteration** (Jacobian rebuilt every time) | `(2FP + 1)` on the first iteration, then **1** per iteration (Broyden rank-1 update) | asymptotically `(2FP+1)×` fewer solves |

### 1.3 The two changes that alter the exponent

Everything else is a constant-factor win. These two are not:

**Phase factors: `O(P·B²)` → `O(M log M)`.** The coefficients `C_k` satisfy

```
sum_k C_k e^{ikθ} = prod_p exp(i α_p sin(pθ − φ_p))
                  = exp(i sum_p α_p sin(pθ − φ_p))
```

so they are just the Fourier coefficients of a function costing one `sin`,
one `exp` and one FFT — no Bessel functions and no harmonic convolution.
The quadratic `B²` term is gone entirely, and `B` now only enters through
how many coefficients are read out.

**Current summation: `O(T·(2B+1)^F)` → `O(V log V)`.** Writing
`P = C₁⊗C₂⊗…·R` and `Q = C₁⊗C₂⊗…`, Eqn. 5.25 is exactly a
cross-correlation,

```
RS⁺(a,b,…) = sum_{k,l,…} P[k,l,…] · conj(Q[k+a, l+b, …])
```

so the correlation theorem gives every index tuple at once from one pair of
transforms. Zero-padding each axis to `2(B+P)+1` keeps the circular
wrap-around inside the padding, so the result is the *linear* correlation,
identical to QMix's to machine precision. The cost stops depending on `T`.

### 1.4 Accuracy, not just speed

Two of these changes make the answer **more** accurate, not just faster:

* **Phase factors.** QMix's recursion truncates every intermediate
  convolution at ±`B`. The transform never forms them. Measured against an
  effectively untruncated reference (`num_b = 90`) at `num_p = 4`:

  | method | error |
  |---|---|
  | QMix (truncated convolution) | 2.1 × 10⁻⁹ |
  | QPMix `method="direct"` (same algorithm) | 2.1 × 10⁻⁹ |
  | QPMix `method="fft"` (default) | **7.8 × 10⁻¹⁶** |

* **Response function.** Because evaluation is now `O(1)` regardless of
  table size, the table can be ~500× denser than QMix's knot set and still
  be faster. Against a high-accuracy reference (QMix with 6001 spline
  knots):

  | quantity | QMix | QPMix |
  |---|---|---|
  | response function `resp(v)` | 1.0 × 10⁻⁴ | **3.1 × 10⁻⁵** |
  | tunneling current `qtcurrent` | 2.0 × 10⁻⁵ | **2.4 × 10⁻⁶** |

  This is why cross-validation against QMix agrees to ~10⁻⁵ rather than to
  machine precision: the residual is QMix's interpolation error.

---

## 2. Measured speed-ups

Machine: 16 threads, NumPy 2.5.3, SciPy 1.18.1, numba 0.67.0, Python 3.12.
Every QPMix timing excludes JIT compilation (each case is run once to warm
the cache before being timed). Best of three runs.

### Kramers–Kronig transform

| case | QMix | QPMix | speed-up |
|---|---|---|---|
| `npts = 7 001` | 26.1 ms | 1.9 ms | **13.6×** |
| `npts = 14 001` | 34.6 ms | 12.5 ms | 2.8× |
| `npts = 70 001` | 371.1 ms | 74.9 ms | 5.0× |

### Response function

| case | QMix | QPMix | speed-up |
|---|---|---|---|
| build `RespFnPolynomial(50)` (one-off) | 46.0 ms | 82.2 ms | 0.6× |
| evaluate at 10⁵ points | 8.2 ms | 0.11 ms | **71×** |
| evaluate at 10⁶ points | 83.8 ms | 0.52 ms | **162×** |
| evaluate at 10⁷ points | 848.7 ms | 18.3 ms | **46×** |

Construction is the one place QPMix is slower: it tabulates 140 001 points
and Kramers–Kronig transforms them, where QMix fits ~100 spline knots. It
is paid once per response function, and it buys back two orders of
magnitude on every subsequent evaluation.

### Phase-factor coefficients (`num_f = 2`, `num_b = 15`, `npts = 401`)

| harmonics | QMix | QPMix | speed-up |
|---|---|---|---|
| `num_p = 1` | 1.67 ms | 0.60 ms | 2.8× |
| `num_p = 2` | 3.61 ms | 0.71 ms | 5.1× |
| `num_p = 3` | 5.44 ms | 0.81 ms | 6.8× |
| `num_p = 4` | 7.19 ms | 0.87 ms | **8.2×** |

The speed-up grows with `num_p` exactly as the `O(P·B²) → O(M log M)`
analysis predicts.

### Response matrix interpolation

| case | values | QMix | QPMix | speed-up |
|---|---|---|---|---|
| `num_f=1, num_b=15, npts=401` | 12 431 | 1.06 ms | 0.06 ms | 18× |
| `num_f=2, num_b=15, npts=401` | 385 361 | 34.7 ms | 0.44 ms | **79×** |
| `num_f=3, num_b=9, npts=401` | 2 750 459 | 253.0 ms | 6.3 ms | **40×** |

### Full `qtcurrent` call

| case | QMix | QPMix | speed-up |
|---|---|---|---|
| `num_f=1, num_p=1, num_b=15, npts=401` | 2.01 ms | 0.60 ms | 3.4× |
| `num_f=1, num_p=3, num_b=15, npts=401` | 4.20 ms | 0.74 ms | 5.7× |
| `num_f=2, num_p=1, num_b=15, npts=401` | 40.4 ms | 2.91 ms | **13.9×** |
| `num_f=2, num_p=2, num_b=15, npts=401` | 45.5 ms | 4.76 ms | 9.6× |
| `num_f=3, num_p=1, num_b=9, npts=401` | 292.7 ms | 32.6 ms | 9.0× |

### Harmonic balance (end-to-end simulation)

| case | QMix | QPMix Newton | QPMix Broyden |
|---|---|---|---|
| `num_f=1, num_p=1, npts=401` | 12.1 ms | 7.5 ms (1.6×) | 6.8 ms (**1.8×**) |
| `num_f=2, num_p=1, npts=401` | 121.0 ms | 41.1 ms (2.9×) | 19.8 ms (**6.1×**) |
| `num_f=2, num_p=2, npts=201` | 67.7 ms | 35.2 ms (1.9×) | 24.1 ms (**2.8×**) |

Broyden's advantage grows with `num_n = num_f × num_p`, because that is
what sets how many current solves a full Jacobian costs.

**Headline: a complete two-tone mixer simulation runs about 6× faster.**

---

## 3. Where the FFT correlation actually wins

The FFT path for the current summation is exact (it matches the direct path
to ~10⁻¹⁵) but it is **not** the default winner. Because it computes every
index tuple whether you asked for one or sixty, it only pays off when many
output frequencies are wanted at once:

| case | tuples | direct | FFT | winner |
|---|---|---|---|---|
| `f=2, p=1, nb=15`, 3 output freqs | 3 | 2.7 ms | 7.9 ms | direct |
| `f=2, p=2, nb=15`, 5 output freqs | 5 | 3.9 ms | 8.4 ms | direct |
| `f=2, p=2, nb=11`, 12 output freqs | 12 | 5.3 ms | 4.3 ms | FFT (1.2×) |
| `f=2, p=3, nb=11`, 24 output freqs | 24 | 10.6 ms | 5.8 ms | FFT (1.8×) |
| `f=2, p=4, nb=11`, 40 output freqs | 40 | 16.5 ms | 6.1 ms | FFT (2.7×) |
| `f=2, p=5, nb=11`, 60 output freqs | 60 | 22.8 ms | 7.9 ms | **FFT (2.9×)** |

`method="auto"` (the default) chooses with the cost model in
`_select_method`; across the sweep in `benchmarks/bench_method.py` its
choice is always within 15% of the faster path.

One case flips the default entirely: **without numba**, the direct kernels
are interpreted Python loops while the FFT path stays fully vectorised, so
`auto` always chooses the FFT. That is what keeps QPMix usable with no JIT
installed — the full test suite runs in 4.6 s without numba, versus 53 s
before the fallback was vectorised.

---

## 4. Computer-architecture techniques used

The complexity changes above are the algorithmic half. The rest of the
speed-up comes from how the kernels touch memory:

**Branch-free `O(1)` table lookup.** A spline over non-uniform knots costs
a binary search per evaluation — a data-dependent branch that the predictor
cannot learn and that blocks vectorisation. A uniform grid replaces it with
`i = floor((v − v₀)/dv)`. Since `qtcurrent` sweeps the bias voltage
monotonically, the table is then walked *sequentially*, so the hardware
prefetcher hides the memory latency entirely.

**Loop fusion.** `_respmat` computes the shifted voltage in a register and
interpolates it immediately, instead of materialising a
`(2B+1)^F × N` voltage array and reading it back. For three tones that is
~240 MB of temporaries never written.

**Cache blocking.** The current kernels walk the bias axis in blocks of 128
points (`_kernels.BLOCK`). The coefficient slices `C[k, block]`, which the
inner loops re-read for every `(k, l, …)` combination, then stay resident
in L2 while the much larger response matrix streams past exactly once.

**Thread-level parallelism.** The block index is also the unit of
parallelism, so each thread owns a disjoint slice of the output and no
synchronisation or reduction is needed. QMix's kernels are single-threaded.

**Contiguous inner axis.** The bias index is innermost in every array, so
inner loops are unit-stride and the JIT can emit SIMD.

**Real-input transforms.** The Kramers–Kronig transform only needs the
imaginary part of the analytic signal, so it goes through `rfft`/`irfft`
rather than building the full complex signal — half the arithmetic and half
the peak memory.

**5-smooth transform lengths.** Every FFT length is rounded up with
`next_fast_len`. `6801 × 50` has a prime factor of 2267; a few extra zeros
buy back an order of magnitude.

**Lazy JIT with on-disk caching.** QMix attaches eager signatures to every
kernel, so importing it pays full LLVM compilation before any user code
runs. QPMix compiles on first call and caches to disk, keeping
`import qpmix` fast.

**Batched linear algebra.** Harmonic balance solves `npts` independent
`2n × 2n` systems as one batched `numpy.linalg` call, and applies the
correction with a single `einsum` rather than a Python double loop over
signals.

---

## 5. Beyond four tones

QMix asserts `num_f in [1, 2, 3, 4]`.  The cap is not arbitrary: Kittara's
formulation carries one summation index per tone, so the interpolated
response function holds `(2*num_b + 1)**num_f * npts` complex values —
5.9 GB at four tones, 184 GB at five.

`qpmix.multitone` removes the cap by putting every tone on a **common
frequency grid**.  Tone `f` sits at `n_f * df` for integer `n_f`, and the
multi-dimensional index collapses to a single one:

```
sum_k C_k e^{ikθ} = exp( i * sum_f sum_p α_{f,p} sin(p n_f θ − φ_{f,p}) )
```

That is still one scalar function of θ, so still **one FFT**, however many
tones it contains.  The response function is then only needed at
`vb + k*df` for `|k| <= K = sum_f n_f * num_b_f`, and the current summation
becomes the same one-dimensional correlation the single-tone kernel already
implements.

### Complexity

| | multi-dimensional | common grid |
|---|---|---|
| response-matrix entries | `(2B+1)^F * N` | `(2K+1) * N` |
| growth in tone count `F` | **exponential** | **linear** |
| growth at fixed bandwidth | — | `K ~ F**2` (measured `N^2.15`) |
| set by | `num_b` | the grid multipliers `n_f` |

### Measured (`npts=401`, `num_b=9`, tones 0.02 apart)

Head to head, over the tone counts both engines support:

| tones | `num_k` | grid entries | direct entries | grid | direct | speed-up |
|---|---|---|---|---|---|---|
| 1 | 9 | 7 619 | 7 619 | 0.61 ms | 0.53 ms | 0.9× |
| 2 | 189 | 151 979 | 144 761 | 5.67 ms | 1.81 ms | 0.3× |
| 3 | 297 | 238 595 | 2 750 459 | 10.8 ms | 29.5 ms | **2.7×** |
| 4 | 414 | 332 429 | 52 258 721 | 16.2 ms | 1 142 ms | **70.7×** |

Past the wall, where there is no alternative:

| tones | `num_k` | grid memory | `qtcurrent` | multi-D would need |
|---|---|---|---|---|
| 6 | 675 | 8.3 MB | 36 ms | 281 GB |
| 8 | 972 | 11.9 MB | 60 ms | 101 483 GB |
| 12 | 1 674 | 20.5 MB | 137 ms | 1.3 × 10¹⁰ GB |
| 16 | 2 520 | 30.8 MB | 256 ms | 1.7 × 10¹⁵ GB |
| 24 | 4 644 | 56.8 MB | 699 ms | 2.9 × 10²⁵ GB |

Full simulations, not just one current evaluation:

| tones | harmonic balance (Newton) | (Broyden) |
|---|---|---|
| 4 | 33.5 s *(multi-dimensional)* | 14.6 s |
| 8 | 3.47 s *(grid)* | **1.12 s** |
| 12 | 9.08 s | 3.58 s |
| 16 | 22.9 s | **8.69 s** |

Sixteen tones now cost a quarter of what four used to.

### Why the grid is opt-in below five tones

`method="auto"` stays on the multi-dimensional engine at four tones and
below, even where the grid is 70× faster.  The two engines do not truncate
identically: with commensurate tones, index tuples beyond `±num_p` reach
the same output frequency, and the grid sums them all while the
multi-dimensional engine stops at `num_p`.  For three tones at
0.30/0.32/0.34 that is a 5% difference in the DC current.  The grid answer
is the more complete one — raising `num_p` makes the multi-dimensional
engine converge to it, to 1e-11 — but switching engines automatically would
change results, not just run times.  So it is a documented, benchmarked
opt-in:

```python
i = qpmix.qtcurrent(vj, cct, resp, freqs, num_b=9, method="grid")
```

`ToneGrid.report(npts)` prints both costs so the choice can be made from
numbers.

### A bug this uncovered

Validating the grid engine against the photon-number sum rule
`Idc = Σ_K |C_K|² Idc⁰(V₀ + K·df)` — which follows from `Σ_K |C_K|² = 1`
and holds for any number of tones — showed the multi-dimensional path
disagreeing by 4.8 × 10⁻².  The cause was in the tuple bookkeeping, and it
is inherited from QMix: at zero output frequency both `t` and `−t` satisfy
the matching condition, but Eqn. 5.26 for tuple `t` already contains the
`−t` contribution through `RS−(t) = RS+(−t)`, so the DC current was
double-counted.  It needs `num_p >= 2` *and* commensurate tones to trigger,
which is why it had gone unnoticed.

QPMix now counts each `±` pair once and drops the quadrature term at DC
(the current there is real by construction).  Both engines then agree with
the sum rule to 6 × 10⁻¹³ and with each other to 1 × 10⁻¹⁰.

### When *not* to use the grid

`K` is set by the multipliers, not the tone count.  An LO and an RF signal
5 MHz apart at 230 GHz need `n = (46000, 46001)` and `K = 1.4 × 10⁶` — far
worse than the two-dimensional method.  `ToneGrid.from_circuit` refuses to
allocate that and says so, rather than trying.  Use `max_num_k` to trade
frequency accuracy for cost:

```python
grid = ToneGrid.from_circuit(cct, num_b=9, max_num_k=20_000)
print(grid.frequency_error)
```

Note that `num_k` is *not* monotonic in `max_denominator` — a tighter cap
can force a rational approximation needing a finer grid — which is why
`max_num_k` exists as the cost knob.

### The continuum ceiling

`benchmarks/study_continuum.py` answers the other half of the question:
how many tones before a band *is* a continuum.  A band of fractional width
20% carrying fixed total power, split into `N` random-phase tones and
ensemble-averaged over 12 realizations, gives:

| tones | deviation from the `N=31` answer |
|---|---|
| 1 | 2.2 × 10⁻² |
| 3 | 6.8 × 10⁻³ |
| 5 | 2.6 × 10⁻³ |
| 13 | 1.4 × 10⁻³ |
| 21 | 1.8 × 10⁻³ |

against a statistical floor of 1.8 × 10⁻³ set by the ensemble size.  So
**about five to thirteen tones already reproduce the continuum** at this
bandwidth and drive level, while the cost keeps climbing as `N^2.15`.  The
ceiling is set by cost, not by physics.

Only the *random-phase* comb has a continuum limit, and the script is built
around that: each tone carries `α/√N`, so the RMS drive is fixed while the
amplitude distribution tends to a Gaussian by the central limit theorem.  A
single realization is one sample path and keeps changing with `N`; the
ensemble average is what converges.  The deterministic combs
(`--phase chirp`, `--phase flat`) are offered for comparison and
deliberately do not converge — they hold the RMS fixed while the crest
factor grows with `N`.

---

## 6. Honest limitations

* **Response-function construction is ~1.8× slower** than QMix. It is a
  one-off cost, and it is what pays for the evaluation speed-up.
* **The FFT correlation path is not the default winner** for typical
  simulations — see §3. It is kept because it wins decisively when a full
  intermodulation sweep is requested, and because it is the vectorised
  fallback when numba is absent.
* **Broyden and Newton converge to slightly different points** — both
  satisfy the requested `stop_rerror`, but they differ by roughly that
  tolerance. Use `jacobian="newton"` when you want to reproduce QMix's
  iteration exactly.
* **Speed-ups are smallest for one tone, one harmonic** at small `npts`,
  where per-call Python overhead dominates. They grow with problem size,
  which is the regime that matters.
* **The grid engine is slower than the multi-dimensional one at one and
  two tones** (0.9× and 0.3×), which is why `auto` does not use it there.
* **The grid engine needs commensurate tones.** Frequencies are fitted to a
  common grid by rational approximation, which introduces a frequency error
  that `ToneGrid.frequency_error` reports. Closely spaced tones force a
  fine grid and are better served by the multi-dimensional engine.
* **The continuum study measures one observable** (the pumped DC I-V curve)
  for one band shape. A different observable — mixer gain, noise
  temperature — may need more tones before it settles.
