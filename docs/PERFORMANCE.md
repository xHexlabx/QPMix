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

These are `qtcurrent` alone.  A full simulation also runs harmonic balance,
which dominates; see *Harmonic balance is what decides whether many tones
are practical* below for the end-to-end numbers.

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

### Harmonic balance is what decides whether many tones are practical

A recovery fit, or any simulation, spends its time in one place: a harmonic
balance followed by a current evaluation.  Harmonic balance is itself
`2·num_f·num_p + 1` current solves per Jacobian, so it dominates completely
— at four tones one harmonic balance costs about fifty times one
`qtcurrent` call.

That means making `qtcurrent` fast is not enough.  Until
`harmonic_balance` could reach the grid engine, the whole multi-tone
advantage was invisible: the response matrix was rebuilt multi-dimensionally
on every iteration however fast the summation was.  `harmonic_balance` now
takes the same `method` argument, and the picture changes completely.

One objective evaluation (harmonic balance + `qtcurrent`), real junction,
226 bias points, tones on a coarse comb:

| tones | `num_b` | QMix | QPMix multi-D | QPMix **grid** | grid vs QMix |
|---|---|---|---|---|---|
| 2 | 15 | 78.3 ms | 15.2 ms (5.1×) | 29.0 ms | 2.7× |
| 3 | 9 | 799 ms | 184 ms (4.3×) | **49.8 ms** | **16.0×** |
| 4 | 6 | 7 337 ms | 1 955 ms (3.8×) | **81.6 ms** | **89.9×** |
| 5 | 6 | *refuses* | 1.3 GB of matrix | 125 ms | — |
| 6 | 6 | *refuses* | 16.3 GB | 205 ms | — |

The multi-dimensional path alone is only 3.8–5.1× faster than QMix, and the
margin *shrinks* as tones are added — both engines are then bandwidth-bound
on the same exponentially growing array.  The grid engine is what breaks
that: its cost is nearly flat in the tone count (50 → 82 → 125 → 205 ms from
three to six tones), so the advantage grows without limit.

At realistic `num_b = 15`, five tones would need 96 GB of response matrix
and six would need 2 989 GB.  The grid engine does them in 125 ms and 205 ms.

### The grid engine depends on the frequencies, not the tone count

`num_k` is set by the grid multipliers.  Repeating the table above with the
*measured* LO frequencies (0.2639, 0.2879, … — arbitrary reals, so the
common grid is very fine) instead of a commensurate comb:

| tones | `num_b` | QPMix multi-D | QPMix grid |
|---|---|---|---|
| 2 | 15 | 13.7 ms | 1 019 ms |
| 3 | 9 | 169 ms | 1 615 ms |
| 4 | 6 | 1 851 ms | 3 697 ms |

The grid engine is 10–75× *slower* there.  So the choice is not "how many
tones" but "are the tones commensurate on a coarse grid": a harmonic comb or
an evenly channelised band, yes; an LO and an RF signal a few MHz apart, no.

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

## 6. Analyzing measured data

`qpmix.exp` is the port of `qmix.exp`.  Three parts of it do real numerical
work, and each is reworked.

| Stage | QMix | QPMix | Complexity | Measured |
|---|---|---|---|---|
| Tucker-theory sums | Python loop over `2*num_b+1` Bessel orders, interpolating the response once per order | every order in one call; the response ladder interpolated once; `J_{-n} = (-1)^n J_n` halves the Bessel work | same order, ~40x fewer calls | **2.6–7.1×** |
| Drive-level recovery | 15 fixed bisection steps | safeguarded Newton with the analytic derivative | 15 evaluations → ~5 | **7.9–8.5×** |
| Impedance error surface | 101×201 Python double loop | one broadcast expression | same order, no interpreter overhead | **2.3–19.3×** |

### Measured

| Benchmark | QMix | QPMix | Speed-up |
|---|---|---|---|
| pumped I-V curve, 401 points | 2.22 ms | 0.81 ms | 2.7× |
| AC current, 401 points | 5.90 ms | 0.83 ms | **7.1×** |
| AC current, 4001 points | 52.9 ms | 7.60 ms | 7.0× |
| recover alpha, 201 points | 19.0 ms | 2.40 ms | 7.9× |
| recover alpha, 2001 points | 167 ms | 19.7 ms | **8.5×** |
| error surface, 50 bias points | 154 ms | 7.97 ms | **19.3×** |
| error surface, 500 bias points | 216 ms | 95.4 ms | 2.3× |

### Accuracy: drive-level recovery

Bisection over 15 steps cannot resolve better than `alpha_max / 2**15`.  The
Newton iteration is not so limited:

| true `alpha` | QMix error | QPMix error |
|---|---|---|
| 0.2 | 2.28 × 10⁻⁵ | **1.1 × 10⁻¹⁶** |
| 0.8 | 2.27 × 10⁻⁵ | **4.4 × 10⁻¹⁶** |
| 1.5 | 2.29 × 10⁻⁵ | **1.1 × 10⁻¹⁵** |
| 2.2 | 2.28 × 10⁻⁵ | **5.7 × 10⁻¹⁴** |

The derivative is free because the neighbouring Bessel orders are already
needed for the AC current:

```
d/dalpha sum_n J_n(a)^2 I_n = sum_n J_n(a) (J_{n-1}(a) - J_{n+1}(a)) I_n
```

The pumped I-V curve is *not* monotonic in `alpha` everywhere — above the
gap it falls, and past the first Bessel maximum it turns over — so Newton
alone would run away.  A coarse scan locates the first upward crossing at
each bias point first, and the iteration is bracketed by bisection.

### Round-trip validation

`qpmix.exp.simulate` builds measurements from known parameters, so the
analysis can be tested by asking whether it recovers them.  Driving a
junction through a known Thevenin source and recovering it from the pumped
I-V curve alone closes to:

| quantity | error |
|---|---|
| embedding impedance `zt` | < 6 × 10⁻⁴ |
| embedding voltage `vt` | < 4 × 10⁻⁴ |
| drive level `alpha` | < 1 × 10⁻⁹ |
| fit residual | ~10⁻¹³ |

### Bugs found in the ported code

* **The normal-resistance fit range** defaults to `(3.5e-3, 5e3)` volts in
  QMix — five kilovolts — which contradicts its own documentation
  (`(3.5e-3, 4.5e-3)`).  In practice it fits from 3.5 mV to the end of the
  data rather than over the intended 1 mV window.
* **A flat shot-noise window** makes the Woody correction factor
  `5.8 / slope` infinite, which then silently turns every downstream IF
  power into NaN.  QPMix detects it and leaves the data in measured units.
* **Fixed-width local fits.**  The gap-voltage and subgap-resistance fits
  use hard-coded ±10 µV and ±100 µV windows, which collapse to one or two
  points on a coarsely resampled curve and make the polynomial fit
  singular.  QPMix widens the window until it holds enough samples.
* **A global `numpy.seterr`** at import time in `qmix.exp.if_data` disables
  divide and invalid warnings for the entire interpreter.

### An improvement worth its own note

The offset fit works by making the I-V curve overlap its own point
reflection.  Smoothing the data first costs almost nothing — the offset is a
bulk property, and Gaussian smoothing is symmetric so it commutes with the
reflection — but it changes the noise tolerance completely:

| current noise | QMix (unsmoothed) | QPMix |
|---|---|---|
| 0 | ~0 µV | < 1 µV |
| 5 × 10⁻⁴ | 70 µV | < 1 µV |
| 2 × 10⁻³ | 80 µV (total failure) | < 1 µV |
| 5 × 10⁻³ | 80 µV (total failure) | 11 µV |

against a true offset of 80 µV.

Note also a genuine degeneracy, independent of implementation: if the subgap
region is *perfectly ohmic*, a voltage offset and a current offset produce
identical distortions and no analysis can separate them.  Real junctions
have curvature near zero bias from leakage current, which breaks it — which
is why `qpmix.exp.simulate` defaults to an I-V model that includes it.

### Recovering the embedding circuit two ways

`qpmix.exp` offers both routes to the Thevenin equivalent source:

| | voltage matching (`zemb`) | current matching (`currentmatch`) |
|---|---|---|
| what is fitted | the load line through `(V_j, Z_j)` from Tucker theory | the *simulated pumped I-V curve* from full harmonic balance |
| cost | closed form, ~20-140 ms | one harmonic balance per objective evaluation |
| assumptions | one tone, no harmonics, invertible first photon step | none beyond the simulation itself |
| bias window | first photon step only | any: first step, full subgap, full range, or per-step |
| harmonics | fundamental only | fundamental plus higher harmonics |

They are independent, so agreeing is evidence. On synthetic data with a
known source they agree to `5e-3`; on the real 183.6 GHz measurement below
they agree to `|dz| = 4e-3` normalized (0.04 Ω).

### Measured on real data (183.6 GHz, `num_b = 100`, 226-point window)

All four runs fit exactly the same cubic-spline-resampled grid, and every
residual is re-evaluated on that grid, so the numbers are comparable:

| run | `zt` (normalized) | rms | evaluations | time |
|---|---|---|---|---|
| QMix, 9 Nelder-Mead starts | 0.3612 − 0.5661j | 1.057 × 10⁻³ | 4 089 | **145.2 s** |
| QPMix, same 9 starts | 0.3608 − 0.5630j | 1.056 × 10⁻³ | 4 052 | **25.1 s** (5.8×) |
| QPMix API, grid of 9 | 0.3608 − 0.5619j | 1.056 × 10⁻³ | 3 577 | 22.8 s |
| QPMix API, seeded (3 starts) | 0.3608 − 0.5619j | 1.056 × 10⁻³ | 781 | **5.6 s** (25.7×) |

Swapping the engine alone is **5.8×**; seeding the optimiser from the
closed-form voltage-match answer cuts nine starts to three for **25.7×**
overall, at the same answer. One objective evaluation costs 32.2 ms in QMix
and 5.9 ms in QPMix at `num_b = 100` — **5.4×**.

### Choosing a solution by agreement, not by residual

The residual surface has local minima and unphysical basins, so the lowest
residual is not automatically the answer. QPMix runs each start
independently, discards solutions that are not physically admissible
(negative resistance, negative source voltage, runaway reactance) and takes
the solution the surviving runs *agree* on, breaking ties by residual.

On the real measurement above, the nine starts resolve as:

| outcome | runs |
|---|---|
| `zt = 0.3676 − 0.6745j`, `vt = +0.360` — admissible | **5** |
| `zt = −0.2173 − 0.0274j` — negative resistance | 3 |
| same `zt`, `vt = −0.360` — negative source voltage | 1 |

Four of the nine converge on something unphysical with a *lower* residual
than the accepted answer. Taking the minimum would have returned one of
them.

### What the second harmonic is, and is not, recoverable from

Fitting a second-harmonic embedding impedance to all eighteen pumped
measurements answers a question a single fit cannot: is `zt2` a measurement,
or an artefact?

| | one harmonic | two harmonics |
|---|---|---|
| files where the nine starts agreed | **18 / 18** | **0 / 18** |
| Hot/Cold repeatability of `zt1` | 0.125 | 0.132 |
| Hot/Cold repeatability of `zt2` | — | **0.650** |

Hot and Cold are the same embedding circuit measured against different
blackbody loads, so the recovered impedance must repeat between them. That
check owes nothing to the optimiser. The fundamental repeats to 0.13
normalized; the second harmonic only to 0.65, against a magnitude `|zt2|` of
about 1.0–1.6 — a scatter of 50–65% of the value itself.

The data is not *blind* to the second harmonic. Shorting it (`zt2 = 0`)
raises the residual by a median factor of 3.7 over the first photon step, so
the fit does see something. But the minimum is too shallow and too broad for
the optimiser to locate, and the located value does not reproduce.

**A wider bias window makes it worse, not better** — which is the opposite
of the natural guess, since more photon steps ought to carry more
second-harmonic information:

| window | `rms(zt2=0) / rms(best)` | Hot/Cold `\|Δzt2\|` | Hot/Cold `\|Δzt1\|` (2H) |
|---|---|---|---|
| `first_photon` | 3.7 | 0.65 | 0.13 |
| `full_subgap` | **1.14** | 1.84 | 0.21 |
| `photon_steps` | 1.38 | 122 | 0.59 |

Over the full subgap the residual is dominated by the large-current regions,
where the second harmonic barely matters, so its relative weight *drops* to
almost nothing. And the extra free parameter then corrupts the part that
*was* well determined: over the full subgap the one-harmonic fit repeats
between Hot and Cold to 0.004, while the two-harmonic fit's fundamental
repeats only to 0.21.

The practical conclusion: fit one harmonic on the first photon step, treat a
recovered `zt2` as an upper-bound-shaped hint rather than a measurement, and
read the agreement figure before believing any of it.

### Two bugs this uncovered

`_admissible` bounded the reactance but nothing bounded the resistance from
above, so a poorly constrained higher-harmonic fit was free to return
`zt2 = 122 + 0.3j` and have it accepted. An embedding resistance a hundred
times the junction's normal resistance means the simplex ran away, not that
the circuit is unusual. Both are bounded now.

`harmonic_balance` printed its non-convergence warning unconditionally,
ignoring `verbose` — inherited from QMix. A fit that calls it four thousand
times then produces four thousand lines of warning. It is now gated on
`verbose`, and non-convergence is reported through `mode="x"` instead, which
`recover_zemb_current_match` counts and returns as `n_not_converged`.

---

## 7. Honest limitations

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
* **Offset recovery needs subgap curvature.** With a perfectly ohmic subgap
  region the voltage and current offsets are mathematically degenerate; see
  §6.
* **The automatic shot-noise window search is a heuristic**, inherited from
  QMix. It works on well-behaved data, but supplying `vshot` explicitly is
  more reliable when Josephson features are present.
* **Current matching is slow by construction** — seconds, against
  milliseconds for voltage matching — because every objective evaluation is
  a harmonic balance. Use voltage matching when its assumptions hold, and
  current matching when they do not.
* **A second-harmonic impedance is not recoverable from a DC I-V curve
  alone**, at least not from this data — see above. The one-harmonic fit is
  reliable; the two-harmonic one gives an answer that does not repeat.
* **`guesses="seeded"` trusts the voltage-match answer to be in the right
  basin.** It was on the data measured here, but it explores less than the
  nine-point grid; `guesses="grid"` remains the default for that reason.
* **Residuals from different fits are only comparable on a common grid.**
  Each recovery resamples the measured curve, so comparing the `err` fields
  of two results fitted differently is meaningless — re-evaluate both on one
  grid, as `notebooks/compare_qmix_vs_qpmix.py` does.
* **The plotting-heavy parts of `qmix.exp.exp_data` are not ported** — the
  multi-panel report figures and the file-hierarchy helpers. The three most
  useful plots are provided; the rest is presentation code that is easier to
  write against the returned arrays.
