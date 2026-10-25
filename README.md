# k-Way Collision Probabilities: Vectorized Monte Carlo vs Poisson

**Delta Code 2026 @ Technozion (NIT Warangal), Round 2: Probability & Statistics**
**Team Enthusiast** · Krish Agrawal · krishagrawal222@gmail.com

Estimates `P(some bucket receives >= k of n uniform keys among C buckets)` with a fully
vectorized NumPy Monte Carlo, 95% Clopper-Pearson / Wilson confidence intervals, and compares it to
the Poisson approximation `1 - exp(-C(n,k) / C^(k-1))`. It also locates the rare-event regime where
empirical sampling breaks down and the analytic formula must take over.

## Files

| File | Purpose |
|---|---|
| `birthday.py` | Simulator, theory, CI code, experiment pipeline, plotting, self-tests, benchmark |
| `results.csv` | 135 grid points: empirical p-hat, CP and Wilson 95% CIs, Poisson / refined / exact values, regime label |
| `convergence.csv` | Running estimate and CI vs number of trials for p ~ 1e-2, 1e-4, 1e-6 |
| `plot_grid.png` | 6-panel log-log figure: P vs n (k = 2, 3, 4), convergence, rare-event breakdown, Poisson accuracy |
| `report.md` | Poisson derivation (Chen-Stein), memory/vectorization notes, failure-threshold analysis |
| `requirements.txt` | numpy, scipy, matplotlib (numba optional) |

## Quick start

```bash
pip install -r requirements.txt
python birthday.py --selftest     # 15 correctness checks (exact solver, detector equivalence, CIs, ...)
python birthday.py                # full pipeline, ~5 min on 1 core
python birthday.py --plot-only    # redraw plot_grid.png from the CSVs
python birthday.py --benchmark    # numpy vs numba timing
```

Useful flags: `--T` (nominal trials, default 1,000,000), `--elems` (cap on T*n keys per point, default
3e8; raise it for full-T runs at large n), `--mem-mb` (chunk memory budget), `--backend {numpy,numba}`,
`--seed`, `--quick` (smoke test).

## How it works

1. **Detection with no Python loop.** Draw a C-contiguous `(rows, n)` block of keys in `[0, C)`, sort each
   row in place, and test `S[:, k-1:] == S[:, :n-k+1]` (a value occurs >= k times iff keys `i` and `i+k-1`
   match in the sorted row). `np.diff` and `bincount` detectors are included as cross-checks.
2. **Bounded memory.** The block size is derived from a memory budget, so peak memory is independent of `T`.
   Keys are stored as `uint32` for `C <= 2^32`. Results are bit-identical for any chunk size.
3. **Statistics.** `scipy.stats` Beta quantiles give Clopper-Pearson intervals (valid at 0 hits), plus Wilson.
4. **Theory.** Poisson (`1 - e^{-lambda}`, `lambda = C(n,k)/C^(k-1)`), a refined bucket-occupancy
   approximation, exact `k = 2`, and an exact small-case solver used for testing.

## Headline results (see `report.md`)

* Monte Carlo is reliable (<= 10% relative CI half-width) for `p >~ 3.8e-4` at `T = 1e6`, and fails
  (CI as wide as the estimate) below `p ~ 3.8e-6`. 60 of 135 grid points returned exactly zero hits.
* For `C` up to `2^32` and `k >= 3`, every feasible `n` lies in the failure zone, so only the analytic
  Poisson formula can answer there, and it is accurate in that regime (relative error ~ 1e-6).
* In the bulk, Poisson is excellent for `k = 2` at large `C` but over-estimates for `k >= 3` at moderate
  `lambda` and small `C` (e.g. 0.950 vs 0.908 at `lambda = 3`, `k = 4`, `C = 1e4`).
