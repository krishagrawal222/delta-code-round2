# k-Way Collision Probabilities: Vectorized Monte Carlo vs. Poisson Analysis

**Delta Code 2026, Round 2: Probability & Statistics** (Technozion, NIT Warangal)
Team Enthusiast · Krish Agrawal (krishagrawal222@gmail.com) · Code: `birthday.py` · Data: `results.csv`, `convergence.csv` · Figure: `plot_grid.png`

---

## 1. Problem and notation

`n` keys are drawn independently and uniformly from `{0, …, C-1}`. Let `L_j` be the number of keys in bucket `j`. We want

$$P_k(C,n) \;=\; \Pr\big(\max_j L_j \ge k\big),$$

the probability that some bucket receives at least `k` keys (`k = 2` is the classical birthday problem).

## 2. Mathematical derivation of the Poisson approximation

### 2.1 Exact reformulation as a "no-subset" event

For every `k`-subset `S` of the `n` keys let `I_S = 1{all keys in S fall in one bucket}`, and let

$$X \;=\; \sum_{|S|=k} I_S .$$

A bucket with load ≥ k contains at least one k-subset, and any k-subset whose keys coincide gives a bucket with load ≥ k. Therefore the events are **identical**:

$$\{\max_j L_j \ge k\} \;=\; \{X \ge 1\}.$$

### 2.2 Mean of X

Fix `S`. The first key of `S` lands anywhere; each of the other `k-1` keys must match it, each with probability `1/C`:

$$\Pr(I_S = 1) = C^{-(k-1)}, \qquad
\lambda \;:=\; \mathbb E[X] \;=\; \binom{n}{k}\,C^{-(k-1)} .$$

### 2.3 Poisson limit

If `X` were `Poisson(λ)` we would get `Pr(X ≥ 1) = 1 − e^{−λ}`, i.e.

$$\boxed{\,P_k(C,n) \;\approx\; 1-\exp\!\Big(-\frac{\binom{n}{k}}{C^{k-1}}\Big)\,}$$

**Why `X` is nearly Poisson (Chen-Stein).** Indicators `I_S, I_{S'}` for *disjoint* subsets are independent, and dependence arises only when the subsets overlap. Let `N(S)` be the set of subsets overlapping `S` (including `S`). The Arratia-Goldstein-Gordon bound gives

$$\Big|\Pr(X\ge1)-(1-e^{-\lambda})\Big| \;\le\; \min(1,\lambda^{-1})\,(b_1+b_2),$$

$$b_1=\sum_S\sum_{S'\in N(S)} p_S\,p_{S'}, \qquad
b_2=\sum_S\sum_{S'\in N(S),\,S'\ne S}\mathbb E[I_S I_{S'}].$$

Two subsets sharing `j` keys collide together with probability `C^{1-(2k-j)}`. The dominant overlap is `j = k-1`, which gives to leading order

$$b_1 \approx \lambda\,\frac{k\,(n/C)^{k-1}}{(k-1)!},\qquad b_2 \approx \lambda\,\frac{k\,n}{C}.$$

So the approximation is **asymptotically exact when `n ≪ C`** (equivalently, when `λ` stays bounded while `C → ∞`). The bound is a worst case: it is loose at moderate `n/C`, and the *actual* error is much smaller (Section 6).

### 2.4 Where the approximation is biased

* **k = 2.** Exactly, `P = 1 − ∏_{i<n}(1 − i/C)`, so the exponent is `n²/2C + n³/6C² + …` against Poisson's `n(n−1)/2C`. Poisson therefore *slightly under-estimates* `P`, by a relative factor of about `n/(3C)` in the exponent.
* **k ≥ 3.** `λ` counts *subsets*, so a bucket holding `m > k` keys contributes `C(m,k)` rather than 1. Poisson therefore *over-estimates* `P`, and the effect grows with `λ`. At small `λ` a bucket-occupancy argument gives a relative excess of order `k·n / ((k+1)·C)`. This is a heuristic leading-order estimate, but it matches the data (for `C=10⁴, k=3, n=292` it predicts about 2.2%, and the observed `p_poisson/p_ref` is 1.021).
* As `λ` grows the formula saturates at `1 − e^{−λ}` (e.g. 0.950 at `λ = 3`) while the true probability for `k ≥ 3` is lower (0.908 for `C=10⁴, k=4, n=2915`).

### 2.5 A refined reference formula (used as "ground truth" where no exact value exists)

Bucket loads are `Binomial(n, 1/C)`. With `q = Pr(Bin(n,1/C) ≥ k)` and treating buckets as independent,

$$P_k \approx 1-(1-q)^{C}.$$

This counts *buckets* rather than subsets and removes the overcounting above. It is validated against exact enumeration (`p_exact_small`, an EGF/polynomial-power solver cross-checked by an independent DP) and against the Monte Carlo itself: its value lies inside the 95% CI at **129 of 135** grid points (95.6%, matching the nominal rate). Exact values are used for `k = 2`.

---

## 3. Vectorized Monte Carlo design

### 3.1 Detection without Python loops

For a block `S` of shape `(rows, n)`:

1. `S.sort(axis=1)` sorts each row in place.
2. In a sorted row, some value occurs ≥ k times **iff** `S[i+k-1] == S[i]` for some `i`.
3. `(S[:, k-1:] == S[:, :n-k+1]).any(axis=1)` evaluates that test for all rows and positions in two array operations.

The code also implements an `np.diff` detector (k = 2) and a `bincount` detector (small `C`, using a row offset of `r·C`). The test suite checks that all three agree bit for bit.

### 3.2 Memory architecture

| Concern | Decision |
|---|---|
| Full matrix is `T·n` keys. For `T = 10⁶, n = 1000` that is 4 GB as `uint32`, 8 GB as `int64`. | **Chunking**: rows per block = `mem_budget / (n·(itemsize+1)·1.25)`. Peak memory stays near the budget (default 256 MB) for any `T`. |
| Key width | `uint32` whenever `C ≤ 2³²` (halves memory and bandwidth; `uint64` otherwise). |
| Layout | `rng.integers(..., size=(rows, n))` returns a **C-contiguous** array, so each row is one contiguous memory block and `sort(axis=1)` is cache-friendly. Sorting along `axis=0` would stride across rows. |
| Temporaries | In-place sort (no copy); the only temporary is a `bool` mask of `rows × (n-k+1)` bytes. |
| Reproducibility | One `SeedSequence([seed, C, n, k])` per grid point. The stream is consumed row by row, so **results are bit-identical for any chunk size** (checked in the self-test). |

Complexity is `O(T · n log n)` time and `O(mem_budget)` space.

### 3.3 Engines and measured performance (1 core, 3 GB RAM)

| n | T | NumPy | numba |
|---|---|---|---|
| 100 | 200,000 | 0.15 s | 1.07 s |
| 400 | 100,000 | 0.36 s | 2.53 s |
| 1500 | 20,000 | 0.33 s | 2.07 s |

The `numba` backend (`--backend numba`, parallel over blocks, fused generate-sort-scan) is **about 7× slower on this single-core machine**. NumPy's row-wise sort is SIMD-accelerated, while the numba kernel pays for scalar RNG calls. NumPy is therefore the default. numba can pay off on many-core machines (its `prange` scales with cores) and for fused early-exit logic, so it is kept as an option. Throughput of the NumPy engine is about 10⁸ keys/s per core.

### 3.4 Statistical machinery

For `x` hits in `T` trials, `scipy.stats` supplies the 95% **Clopper-Pearson** interval (Beta quantiles, exact and valid at `x = 0`) and the **Wilson** score interval; both are stored in `results.csv`.

---

## 4. Experimental design

* **Grid:** `C ∈ {10⁴, 10⁵, 10⁶, 10⁸, 2³²}`, `k ∈ {2, 3, 4}`, 9 log-spaced values of `n` per `(C, k)` from `λ ≈ 10⁻⁸` up to `λ ≈ 3` (capped at `n ≤ 4000`) → **135 points**. The smaller `C` values are included so that the full range of `P` (up to ≈ 1) is reachable, which lets us validate the Poisson formula in the bulk and not only in the tail.
* **Trials:** nominal `T = 10⁶`, reduced adaptively to `T = min(10⁶, 3·10⁸/n)` so that each point stays within a fixed compute budget (`--elems` raises the cap). The `T` actually used is recorded per row. In total 2.8·10¹⁰ keys were simulated in 277 s.
* **Convergence runs** (full `T = 10⁶`, chunks of 5,000 trials) at three difficulty levels: `p ≈ 10⁻²` (C=10⁶, k=2, n=142), `p ≈ 10⁻⁴` (C=10⁵, k=3, n=183), `p ≈ 10⁻⁶` (C=10⁵, k=3, n=41).
* **Validation (`--selftest`, 15 checks, all pass):** detector equivalence; the classical `365/23` case (0.5073); MC against the exact solver for four `(C,n,k)`; chunk invariance; the numba engine; CI utilities; the `n<k` edge case.

---

## 5. Rare-event failure threshold

For an estimate `p̂ = x/T` the variance is `σ² = p(1−p)/T`, so the relative standard error is `√((1−p)/(pT)) ≈ 1/√(pT)`. The relative half-width of the 95% interval is `1.96·√((1−p)/(pT))`. This gives three thresholds, which are shaded and marked in panel (e) of `plot_grid.png`:

| Threshold | Formula | T = 10⁶ | Meaning |
|---|---|---|---|
| `p_rel` (10% accuracy) | `z²/(T·0.1²)` | **3.8·10⁻⁴** | below this, estimates are worse than ±10% |
| `p_fail` (100% accuracy) | `z²/T` | **3.8·10⁻⁶** | the CI is as wide as the estimate itself |
| `p_zero` (expected 3 hits) | `3/T` | **3.0·10⁻⁶** | `Pr(0 hits) ≈ 5%`, and the estimate is often exactly 0 |

**Empirical confirmation (135 grid points).** Points are classed by hits: `reliable` (≥ 385 hits, relative half-width ≤ 10%), `noisy` (10–384), `failed` (< 10).

| Class | Points | True p range | Reference value inside CI |
|---|---|---|---|
| reliable | 39 | 3.6·10⁻⁴ to 0.95 | 35 / 39 |
| noisy | 22 | 9.9·10⁻⁶ to 1.9·10⁻³ | 20 / 22 |
| failed | 74 | 5.2·10⁻¹⁹ to 1.1·10⁻⁵ | 74 / 74 |

**60 of the 74 failed points returned exactly zero hits**: only the Clopper-Pearson upper bound (≈ 3.7·10⁻⁶ at `T = 10⁶`, the "rule of three") is available, and no sampling effort at this `T` can distinguish `p = 10⁻⁸` from `p = 10⁻¹⁸`. The class boundaries land where the theory says they should (reliable above ≈ 3.8·10⁻⁴, failed below ≈ 10⁻⁵).

**Convergence illustration** (`convergence.csv`, panel d). After 10⁶ trials:

| Target p | Config | Hits | p̂ | Reference | Rel. half-width (theory) |
|---|---|---|---|---|---|
| 10⁻² | C=10⁶, k=2, n=142 | 9,958 | 9.96·10⁻³ | 9.96·10⁻³ | 2% |
| 10⁻⁴ | C=10⁵, k=3, n=183 | 110 | 1.10·10⁻⁴ | 1.00·10⁻⁴ | 20% |
| 10⁻⁶ | C=10⁵, k=3, n=41 | 3 | 3.0·10⁻⁶ | 1.07·10⁻⁶ | 190% |

The last row is the breakdown in action: with only 3 hits the estimate is off by a factor of 2.8, and the running estimate jumps in discrete steps of `1/t`.

**Practical consequence.** The statement's headline regime (`C` up to `2³²`, `k ≥ 3`) lies entirely below `p_fail` for any feasible `n`: for `C = 2³², k = 3` even `n = 4000` gives `p ≈ 6·10⁻¹⁰` (every `C = 2³²`, `k ≥ 3` grid point returned zero hits). Only the analytic formula can answer there. Poisson is accurate in precisely that regime (relative error ≈ 7·10⁻⁷ at that point, in line with the `k·n/((k+1)C)` estimate), so **Monte Carlo covers `p ≳ 10⁻⁴` and Poisson covers `p ≲ 10⁻⁵`, with a complementary overlap in between.**

---

## 6. Accuracy of the Poisson formula

Panel (f) plots `|P_poisson − P_ref| / P_ref` against `λ`. Findings:

* In the rare regime (`λ ≲ 10⁻³`) the relative error is tiny and decays roughly linearly in `λ` (for `k = 2`, `C = 10⁶` it is about 1.5·10⁻⁵ at `λ = 10⁻³`).
* The error is **larger for smaller `C`** (compare `C = 10⁴` and `C = 10⁶`) and **larger for larger `k`**, as predicted by the `n/C` corrections.
* It peaks around `λ ≈ 0.5–1` (up to about 4% for `k = 3` and 12% for `k = 4` at `C = 10⁴`) and then shrinks as both curves saturate at 1.

Examples where the 95% CI separates Poisson from the truth (`T` as listed in `results.csv`):

| C | k | n | λ | MC (95% CP CI) | Poisson | Refined / exact |
|---|---|---|---|---|---|---|
| 10⁴ | 3 | 1218 | 3.0 | 0.9371 [0.9362, 0.9381] | 0.9504 | 0.9357 |
| 10⁴ | 4 | 2915 | 3.0 | 0.9088 [0.9071, 0.9106] | 0.9503 | 0.9076 |
| 10⁶ | 2 | 2450 | 3.0 | 0.9501 | 0.9502 | 0.9503 (exact) |

Poisson is excellent for `k = 2` at large `C` but visibly biased for `k ≥ 3` at moderate `λ` and small `C`. In total, Poisson lies inside the CI at 122 of 135 points; the 13 exceptions are all points with narrow CIs. At 6 of them the refined/exact reference is *also* outside the CI, so these are ordinary 5%-level sampling fluctuations (about 6.75 such misses are expected among 135 points). The other 7 are genuine Poisson bias (Section 2.4). One of the fluctuations looked striking (C=10⁶, k=3, n=718: 42 hits against ≈ 26 expected), so it was re-run with three fresh seeds and 3·10⁶ trials: p̂ = 6.4·10⁻⁵ with CI [5.6, 7.4]·10⁻⁵, which contains the reference 6.14·10⁻⁵.

---

## 7. Limitations and extensions

* The `T·n` cap means `T < 10⁶` for `n ≳ 300`; the exact `T` is in `results.csv`. A paper-scale run is `python birthday.py --elems 1e12` (hours on a laptop).
* Monte Carlo cannot certify probabilities below roughly `1/T`. Variance-reduction schemes (importance sampling that tilts keys toward a shared bucket, or multilevel splitting) would extend its reach, but were out of scope here, and the analytic route is cleaner in this regime.
* The refined formula assumes independent buckets. It is excellent empirically (129/135 inside CI) but is not an exact result for `k ≥ 3`.

## 8. Reproduce

```bash
pip install -r requirements.txt
python birthday.py --selftest      # 15 checks
python birthday.py                 # ~5 min on 1 core -> results.csv, convergence.csv, plot_grid.png
python birthday.py --benchmark     # numpy vs numba
```
