#!/usr/bin/env python3
"""
birthday.py -- Vectorized Monte Carlo for generalized k-way birthday collisions
===============================================================================
Delta Code 2026 (Technozion, NIT Warangal) -- Round 2, Probability & Statistics.
Team Enthusiast | Krish Agrawal | krishagrawal222@gmail.com

Problem
-------
Throw n keys uniformly into C buckets. Estimate
        P(Collision >= k) = P( some bucket receives >= k keys )
and compare with the Poisson approximation
        P ~= 1 - exp(-lambda),   lambda = binom(n, k) / C**(k-1).

Contents
--------
 1. Theory     : Poisson (as in the problem statement), a refined Poisson-occupancy
                 approximation, exact k=2, and an exact small-case solver (for tests).
 2. Simulation : fully vectorized NumPy engine (sort + shifted compare, no Python loop
                 over trials), chunked to a fixed memory budget; optional numba engine.
 3. Statistics : Clopper-Pearson and Wilson 95% confidence intervals (scipy.stats).
 4. Pipeline   : parameter grid -> results.csv, convergence runs -> convergence.csv,
                 6-panel log-log figure -> plot_grid.png.

Usage
-----
    python birthday.py --selftest                 # correctness checks
    python birthday.py --benchmark                # numpy vs numba speed
    python birthday.py                            # full pipeline (defaults)
    python birthday.py --T 1000000 --elems 3e8    # paper-scale T (needs time/cores)
    python birthday.py --plot-only                # redraw figure from CSVs
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from scipy import stats

try:  # numba is optional: the NumPy engine is always available
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

ALPHA = 0.05                      # -> 95% confidence intervals
Z975 = stats.norm.ppf(1 - ALPHA / 2)


# =============================================================================
# 1. THEORY
# =============================================================================
def log_lambda(C: float, n: int, k: int) -> float:
    """log of lambda = binom(n,k) / C^(k-1)  (stable for huge C, n)."""
    if n < k:
        return -math.inf
    log_binom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_binom - (k - 1) * math.log(C)


def poisson_lambda(C: float, n: int, k: int) -> float:
    """Expected number of k-subsets of keys that all share one bucket."""
    ll = log_lambda(C, n, k)
    return 0.0 if ll == -math.inf else math.exp(ll)


def p_poisson(C: float, n: int, k: int) -> float:
    """Problem-statement approximation: 1 - exp(-C(n,k)/C^(k-1)).  (expm1: no cancellation)"""
    return -math.expm1(-poisson_lambda(C, n, k))


def p_poisson_refined(C: float, n: int, k: int) -> float:
    """
    Poisson-occupancy approximation. Bucket load ~ Binomial(n, 1/C); with
    q = P(load >= k), treat the C buckets as independent:
            P ~= 1 - (1 - q)^C.
    Much more accurate than the k-subset Poisson when lambda is not small, because
    it counts *buckets* with >= k keys instead of k-subsets (which overcounts when a
    bucket holds more than k keys).
    """
    if n < k:
        return 0.0
    q = stats.binom.sf(k - 1, n, 1.0 / C)
    return -math.expm1(C * math.log1p(-q)) if q < 1 else 1.0


def p_exact_k2(C: float, n: int) -> float:
    """Exact birthday probability 1 - prod_{i<n}(1 - i/C), via log1p-sum (accurate for tiny p)."""
    if n < 2:
        return 0.0
    if n > C:
        return 1.0
    return -math.expm1(float(np.log1p(-np.arange(n, dtype=np.float64) / C).sum()))


def p_exact_small(C: int, n: int, k: int) -> float:
    """
    Exact P(max load >= k) for small (C, n) via exponential generating functions:
        P(max load < k) = n!/C^n * [x^n] ( sum_{j<k} x^j / j! )^C .
    Polynomial power by repeated squaring, truncated at degree n.  Used only for tests.
    """
    base = np.zeros(n + 1)
    for j in range(min(k, n + 1)):
        base[j] = 1.0 / math.factorial(j)
    result = np.zeros(n + 1)
    result[0] = 1.0
    e = C
    while e:
        if e & 1:
            result = np.convolve(result, base)[: n + 1]
        base = np.convolve(base, base)[: n + 1]
        e >>= 1
    log_p_lt = math.lgamma(n + 1) - n * math.log(C) + math.log(result[n]) if result[n] > 0 else -math.inf
    return 1.0 - math.exp(log_p_lt)


def p_reference(C: float, n: int, k: int) -> float:
    """Best available 'ground truth': exact for k=2, refined Poisson-occupancy otherwise."""
    return p_exact_k2(C, n) if k == 2 else p_poisson_refined(C, n, k)


def n_for_lambda(C: float, k: int, target: float, n_max: int = 10**9) -> int:
    """Smallest n with lambda(C,n,k) >= target (bisection on a monotone function)."""
    lo, hi = k, k
    while hi < n_max and poisson_lambda(C, hi, k) < target:
        hi *= 2
    hi = min(hi, n_max)
    while lo < hi:
        mid = (lo + hi) // 2
        if poisson_lambda(C, mid, k) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# =============================================================================
# 2. SIMULATION ENGINES
# =============================================================================
def key_dtype(C: int):
    """Narrowest unsigned dtype that holds [0, C): uint32 for C <= 2^32 (halves memory vs int64)."""
    return np.uint32 if C <= 2**32 else np.uint64


def detect_sorted_runs(S: np.ndarray, k: int) -> np.ndarray:
    """
    Core vectorized detector.  Sorts every row in place (axis=1, C-contiguous => each
    row is one contiguous memory block) and checks S[:, i+k-1] == S[:, i].  In a sorted
    row that holds iff keys i..i+k-1 are equal, i.e. some value occurs >= k times.
    Pure array ops: no Python loop over trials.  Returns a boolean flag per row.
    """
    S.sort(axis=1)
    if S.shape[1] < k:
        return np.zeros(S.shape[0], dtype=bool)
    return (S[:, k - 1:] == S[:, : S.shape[1] - k + 1]).any(axis=1)


def detect_diff_k2(S: np.ndarray) -> np.ndarray:
    """k=2 variant using np.diff (adjacent equal keys after sorting)."""
    S.sort(axis=1)
    return (np.diff(S.astype(np.int64), axis=1) == 0).any(axis=1)


def detect_bincount(S: np.ndarray, C: int, k: int) -> np.ndarray:
    """
    bincount variant for small C: offset row r by r*C so one flat np.bincount counts
    all rows at once, then reshape to (rows, C) and take the row-wise max load.
    Memory is rows*C counters, so only usable when C is small (used as a cross-check).
    """
    R = S.shape[0]
    flat = (S.astype(np.int64) + (np.arange(R, dtype=np.int64) * C)[:, None]).ravel()
    counts = np.bincount(flat, minlength=R * C).reshape(R, C)
    return counts.max(axis=1) >= k


def rows_per_chunk(n: int, dtype, mem_bytes: float) -> int:
    """Rows that fit the budget: key block + boolean mask (+ slack for temporaries)."""
    per_row = n * (np.dtype(dtype).itemsize + 1) * 1.25
    return max(1, int(mem_bytes // per_row))


def run_mc_numpy(C: int, n: int, k: int, T: int, seed: int = 0,
                 mem_bytes: float = 256e6, chunk_rows: int | None = None,
                 trace: bool = False):
    """
    Vectorized Monte Carlo.  Generates a (rows x n) C-contiguous block of uniform
    integers in [0, C) per chunk, so peak memory is O(mem_bytes) rather than O(T*n).
    Returns (hits, T, trace) where trace = [(cumulative_T, cumulative_hits), ...].
    """
    if n < k:
        return 0, T, []
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(C), int(n), int(k)]))
    dt = key_dtype(C)
    R = min(T, chunk_rows or rows_per_chunk(n, dt, mem_bytes))
    hits = done = 0
    tr = []
    while done < T:
        r = min(R, T - done)
        S = rng.integers(0, C, size=(r, n), dtype=dt)      # C-contiguous by construction
        hits += int(detect_sorted_runs(S, k).sum())
        done += r
        if trace:
            tr.append((done, hits))
    return hits, T, tr


if HAVE_NUMBA:
    @njit(cache=True)
    def _nb_block(n, C, k, trials, seed):
        np.random.seed(seed)
        buf = np.empty(n, np.int64)
        hits = 0
        for _ in range(trials):
            for i in range(n):
                buf[i] = np.random.randint(0, C)
            buf.sort()
            for i in range(n - k + 1):
                if buf[i + k - 1] == buf[i]:
                    hits += 1
                    break
        return hits

    @njit(parallel=True, cache=True)
    def _nb_hits(n, C, k, blocks, trials_per_block, seed):
        out = np.zeros(blocks, np.int64)
        for b in prange(blocks):
            out[b] = _nb_block(n, C, k, trials_per_block, seed + b)
        return out


def run_mc_numba(C: int, n: int, k: int, T: int, seed: int = 0, blocks: int = 64):
    """JIT engine: one fused loop per trial (generate, sort, scan), parallel over blocks.
    O(n) memory per thread, deterministic per block seed. Returns (hits, T_used, [])."""
    if not HAVE_NUMBA:
        raise RuntimeError("numba is not installed")
    if n < k:
        return 0, T, []
    per = max(1, T // blocks)
    s = (seed * 1_000_003 + n * 101 + k * 7) % (2**31 - 1 - blocks)
    hits = int(_nb_hits(n, int(C), k, blocks, per, s).sum())
    return hits, per * blocks, []


# =============================================================================
# 3. STATISTICS
# =============================================================================
def clopper_pearson(x: int, T: int, alpha: float = ALPHA):
    """Exact (conservative) binomial interval via Beta quantiles.  Valid even when x = 0."""
    lo = 0.0 if x == 0 else float(stats.beta.ppf(alpha / 2, x, T - x + 1))
    hi = 1.0 if x == T else float(stats.beta.ppf(1 - alpha / 2, x + 1, T - x))
    return lo, hi


def wilson(x: int, T: int, alpha: float = ALPHA):
    """Wilson score interval."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / T
    denom = 1 + z * z / T
    centre = (p + z * z / (2 * T)) / denom
    half = z * math.sqrt(p * (1 - p) / T + z * z / (4 * T * T)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rel_half_width(p: float, T: int) -> float:
    """Relative half-width of the 95% normal interval: 1.96 * sqrt((1-p)/(p T))."""
    return Z975 * math.sqrt((1 - p) / (p * T)) if p > 0 else math.inf


def p_failure_thresholds(T: int, rel: float = 0.10):
    """
    Rare-event thresholds for sample size T (95% level):
      p_zero  ~ 3/T          : expected hits ~3 => P(0 hits) ~ 5%; estimate frequently exactly 0
      p_fail  = z^2/T        : relative half-width reaches 100% (estimate no better than its size)
      p_rel   = z^2/(T rel^2): relative half-width falls to `rel` (e.g. 10%)
    """
    return {"p_zero": 3.0 / T, "p_fail": Z975**2 / T, "p_rel": Z975**2 / (T * rel**2)}


# =============================================================================
# 4. PIPELINE
# =============================================================================
FIELDS = ["C", "k", "n", "T", "hits", "p_hat", "cp_low", "cp_high", "wilson_low", "wilson_high",
          "lambda", "p_poisson", "p_refined", "p_exact", "p_ref", "rel_err_poisson",
          "rel_half_width", "poisson_in_cp", "regime", "backend", "seconds"]


def classify(hits: int) -> str:
    """Reliability of an empirical estimate from the number of observed hits."""
    if hits < 10:
        return "failed"        # < 10 events: relative error > ~60%, often exactly zero
    if hits < 385:
        return "noisy"         # relative 95% half-width between 10% and ~60%
    return "reliable"          # relative 95% half-width <= 10%


def build_grid(n_max: int, Cs, ks, points: int = 9):
    """For each (C,k) choose n log-spaced from lambda~1e-8 up to lambda~3 (or n_max)."""
    grid = []
    for C in Cs:
        for k in ks:
            n_hi = min(n_max, n_for_lambda(C, k, 3.0, n_max))
            n_lo = min(n_for_lambda(C, k, 1e-8, n_max), max(k, n_hi // 4))
            ns = sorted({int(round(v)) for v in np.geomspace(max(n_lo, k), max(n_hi, k + 1), points)})
            grid += [(C, n, k) for n in ns if n >= k]
    return grid


def evaluate_point(C, n, k, T_nominal, elems_budget, backend, seed, mem_bytes):
    T = int(min(T_nominal, max(2_000, elems_budget // n)))
    t0 = time.perf_counter()
    if backend == "numba":
        hits, T, _ = run_mc_numba(C, n, k, T, seed)
    else:
        hits, T, _ = run_mc_numpy(C, n, k, T, seed, mem_bytes)
    dt = time.perf_counter() - t0
    p_hat = hits / T
    cp_lo, cp_hi = clopper_pearson(hits, T)
    w_lo, w_hi = wilson(hits, T)
    lam = poisson_lambda(C, n, k)
    pp, pr = p_poisson(C, n, k), p_poisson_refined(C, n, k)
    pe = p_exact_k2(C, n) if k == 2 else float("nan")
    pref = pe if k == 2 else pr
    return {
        "C": C, "k": k, "n": n, "T": T, "hits": hits, "p_hat": p_hat,
        "cp_low": cp_lo, "cp_high": cp_hi, "wilson_low": w_lo, "wilson_high": w_hi,
        "lambda": lam, "p_poisson": pp, "p_refined": pr, "p_exact": pe, "p_ref": pref,
        "rel_err_poisson": (pp - p_hat) / p_hat if p_hat > 0 else float("nan"),
        "rel_half_width": (cp_hi - cp_lo) / (2 * p_hat) if p_hat > 0 else float("inf"),
        "poisson_in_cp": bool(cp_lo <= pp <= cp_hi), "regime": classify(hits),
        "backend": backend, "seconds": round(dt, 3),
    }


def run_convergence(C, k, ns_targets, T, seed, chunk_rows=5_000):
    """Running estimate p_hat(t) with CP bands for configs at p ~ 1e-2, 1e-4, 1e-6."""
    rows = []
    for tag, target in ns_targets:
        n = n_for_lambda(C, k, target) if k != 2 else n_for_lambda(C, 2, target)
        hits, _, tr = run_mc_numpy(C, n, k, T, seed, chunk_rows=chunk_rows, trace=True)
        for t, h in tr:
            lo, hi = clopper_pearson(h, t)
            rows.append({"C": C, "k": k, "n": n, "tag": tag, "t": t, "hits": h, "p_hat": h / t,
                         "cp_low": lo, "cp_high": hi, "p_ref": p_reference(C, n, k)})
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items() if k in fields})


def read_csv(path):
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = int(v)
                except ValueError:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = {"True": True, "False": False}.get(v, v)
            out.append(d)
        return out


# ------------------------------------------------------------------ plotting
def make_plot_grid(results, conv, path, T_nominal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Cs = sorted({r["C"] for r in results})
    cmap = plt.get_cmap("viridis")
    col = {C: cmap(i / max(1, len(Cs) - 1) * 0.9) for i, C in enumerate(Cs)}
    lab = lambda C: ("2^32" if C == 2**32 else f"1e{int(round(math.log10(C)))}")
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"k-way collision probability: Monte Carlo vs Poisson  "
                 f"(Clopper-Pearson 95% CI; nominal T = {T_nominal:,})", fontsize=14, y=0.995)

    # (a-c) P vs n for k = 2,3,4
    for j, k in enumerate((2, 3, 4)):
        a = ax[0, j]
        for C in Cs:
            pts = [r for r in results if r["C"] == C and r["k"] == k]
            if not pts:
                continue
            nmin, nmax = min(r["n"] for r in pts), max(r["n"] for r in pts)
            ng = np.unique(np.geomspace(max(k, nmin * 0.8), nmax * 1.25, 120).astype(int))
            a.plot(ng, [p_poisson(C, int(n), k) for n in ng], color=col[C], lw=1.6)
            if k == 2:
                a.plot(ng, [p_exact_k2(C, int(n)) for n in ng], color=col[C], lw=1.0, ls=":", alpha=.9)
            else:
                a.plot(ng, [p_poisson_refined(C, int(n), k) for n in ng], color=col[C], lw=1.0, ls="--", alpha=.9)
            ok = [r for r in pts if r["hits"] > 0]
            zero = [r for r in pts if r["hits"] == 0]
            if ok:
                y = np.array([r["p_hat"] for r in ok])
                a.errorbar([r["n"] for r in ok], y,
                           yerr=[y - [r["cp_low"] for r in ok], [r["cp_high"] for r in ok] - y],
                           fmt="o", ms=4, color=col[C], capsize=2, label=f"C={lab(C)}")
            else:
                a.plot([], [], "o", color=col[C], label=f"C={lab(C)}")
            if zero:
                a.plot([r["n"] for r in zero], [r["cp_high"] for r in zero], "v", mfc="none",
                       color=col[C], ms=6)
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-10, 3)
        a.set_xlabel("n (keys)"); a.set_ylabel("P(collision >= k)")
        a.set_title(f"k = {k}   (solid: Poisson, {'dotted: exact' if k == 2 else 'dashed: refined'}; "
                    f"v = 0 hits, upper CI)", fontsize=9.5)
        a.grid(alpha=.3, which="both"); a.legend(fontsize=7.5, loc="lower right")

    # (d) convergence
    a = ax[1, 0]
    tags = sorted({r["tag"] for r in conv})
    cc = plt.get_cmap("tab10")
    for i, tag in enumerate(tags):
        rr = [r for r in conv if r["tag"] == tag]
        t = np.array([r["t"] for r in rr]); ph = np.array([r["p_hat"] for r in rr])
        lo = np.array([r["cp_low"] for r in rr]); hi = np.array([r["cp_high"] for r in rr])
        a.fill_between(t, np.where(lo > 0, lo, 1e-12), hi, color=cc(i), alpha=.18)
        a.plot(t[ph > 0], ph[ph > 0], color=cc(i), lw=1.4,
               label=f"{tag}: C={lab(rr[0]['C'])}, n={rr[0]['n']}, k={rr[0]['k']}")
        a.axhline(rr[0]["p_ref"], color=cc(i), ls="--", lw=.9)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-8, 1)
    a.set_xlabel("trials t"); a.set_ylabel("running estimate p_hat(t)")
    a.set_title("Empirical convergence (dashed: reference value; band: CP 95% CI)", fontsize=9.5)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=7.5, loc="lower left")

    # (e) rare-event variance breakdown
    a = ax[1, 1]
    pg = np.geomspace(1e-9, 1, 400)
    for T, ls in ((T_nominal, "-"), (T_nominal // 10, "--")):
        a.plot(pg, [Z975 * math.sqrt((1 - p) / (p * T)) for p in pg], "k", ls=ls, lw=1.6,
               label=f"theory 1.96*sqrt((1-p)/(pT)), T={T:,}")
    th = p_failure_thresholds(T_nominal)
    a.axvspan(1e-9, th["p_fail"], color="red", alpha=.12, label=f"failure zone p < {th['p_fail']:.1e}")
    a.axvline(th["p_rel"], color="green", ls=":", label=f"10% rel. error at p = {th['p_rel']:.1e}")
    a.axhline(1.0, color="gray", lw=.7)
    for C in Cs:
        pts = [r for r in results if r["C"] == C and r["p_ref"] > 0]
        a.plot([r["p_ref"] for r in pts], [(r["cp_high"] - r["cp_low"]) / (2 * r["p_ref"]) for r in pts],
               "o", ms=3.5, color=col[C], alpha=.85)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-9, 1); a.set_ylim(1e-3, 1e5)
    a.set_xlabel("true probability p"); a.set_ylabel("relative half-width of 95% CI")
    a.set_title("Rare-event breakdown: MC relative error vs p (dots: grid runs)", fontsize=9.5)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=7.5, loc="upper right")

    # (f) bias of the Poisson approximation
    a = ax[1, 2]
    ls_for = {1e4: "--", 1e6: "-"}
    kc = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    for k in (2, 3, 4):
        for C in (10**4, 10**6):
            ng = np.unique(np.geomspace(k, 3e5, 500).astype(np.int64))
            ng = ng[ng < C] if k == 2 else ng
            lam = np.array([poisson_lambda(C, int(n), k) for n in ng])
            m = (lam > 1e-6) & (lam < 10)
            if m.sum() < 3:
                continue
            ref = np.array([p_reference(C, int(n), k) for n in ng[m]])
            pp = np.array([p_poisson(C, int(n), k) for n in ng[m]])
            a.plot(lam[m], np.abs(pp - ref) / ref, color=kc[k], ls=ls_for[C], lw=1.5,
                   label=f"k={k}, C={lab(C)}")
    emp = [r for r in results if r["hits"] >= 385 and r["lambda"] > 0]
    a.plot([r["lambda"] for r in emp], [abs(r["p_poisson"] - r["p_hat"]) / r["p_hat"] for r in emp],
           "k.", ms=5, label="MC (>=385 hits)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("lambda = C(n,k)/C^(k-1)"); a.set_ylabel("|P_poisson - P_ref| / P_ref")
    a.set_title("Accuracy of the Poisson formula vs lambda (ref: exact k=2, refined k>=3)", fontsize=9.5)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ self-test
def selftest() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    print("Self-test")
    rng = np.random.default_rng(1)
    # (1) three detectors agree on identical data
    S = rng.integers(0, 40, size=(5000, 12), dtype=np.uint32)
    a = detect_sorted_runs(S.copy(), 3)
    b = detect_bincount(S.copy(), 40, 3)
    check("sorted-run detector == bincount detector (k=3)", np.array_equal(a, b))
    a2 = detect_sorted_runs(S.copy(), 2)
    check("sorted-run detector == np.diff detector (k=2)", np.array_equal(a2, detect_diff_k2(S.copy())))
    # (2) exact solver sanity: classic birthday, n=23, C=365
    check("exact_small(365,23,2) ~ 0.5073", abs(p_exact_small(365, 23, 2) - 0.5072972) < 1e-6)
    check("exact_k2(365,23) == exact_small", abs(p_exact_k2(365, 23) - p_exact_small(365, 23, 2)) < 1e-12)
    # (3) MC vs exact solver. A 95% CI would fail by chance 1 time in 20, so the automated
    #     test uses a 4-sigma z-test (false-alarm rate ~6e-5); the CI is printed for information.
    for (C, n, k) in [(365, 23, 2), (50, 15, 3), (20, 12, 4), (100, 25, 3)]:
        h, T, _ = run_mc_numpy(C, n, k, 400_000, seed=3)
        lo, hi = clopper_pearson(h, T)
        pe = p_exact_small(C, n, k)
        z = (h / T - pe) / math.sqrt(pe * (1 - pe) / T)
        check(f"MC vs exact C={C} n={n} k={k}: exact={pe:.4f}, CI=[{lo:.4f},{hi:.4f}], z={z:+.2f}", abs(z) < 4)
    # (4) chunking invariance: the RNG stream is consumed identically, so hits are identical
    h1, T1, _ = run_mc_numpy(1000, 60, 2, 100_000, seed=5, chunk_rows=100_000)
    h2, T2, _ = run_mc_numpy(1000, 60, 2, 100_000, seed=5, chunk_rows=777)
    check("chunk size does not change the result (bit-identical hits)", h1 == h2)
    pe = p_exact_k2(1000, 60)
    z = (h1 / T1 - pe) / math.sqrt(pe * (1 - pe) / T1)
    check(f"chunked run consistent with exact k=2 (z={z:+.2f})", abs(z) < 4)
    # (5) numba engine statistically consistent
    if HAVE_NUMBA:
        h, T, _ = run_mc_numba(365, 23, 2, 200_000, seed=2)
        lo, hi = clopper_pearson(h, T)
        check("numba engine: classic birthday in CI", lo <= p_exact_small(365, 23, 2) <= hi)
    # (6) CI utilities
    lo, hi = clopper_pearson(0, 10**6)
    check("CP upper bound at 0/1e6 ~ 3.69e-6 (rule of three)", abs(hi - 3.689e-6) < 1e-8)
    wl, wh = wilson(50, 100)
    check("Wilson(50/100) symmetric about 0.5", abs((wl + wh) / 2 - 0.5) < 1e-9)
    # (7) Poisson formula limit behaviour
    check("Poisson ~ exact for k=2, C=2^32, n=1000", abs(p_poisson(2**32, 1000, 2) / p_exact_k2(2**32, 1000) - 1) < 1e-3)
    check("edge case n<k gives 0", p_poisson(10**6, 2, 3) == 0.0 and run_mc_numpy(10**6, 2, 3, 10)[0] == 0)
    print("ALL PASSED" if ok else "SOME TESTS FAILED")
    return ok


def benchmark():
    print("Benchmark: C=1e6, k=3")
    for n, T in ((100, 200_000), (400, 100_000), (1500, 20_000)):
        t0 = time.perf_counter(); run_mc_numpy(10**6, n, 3, T); t_np = time.perf_counter() - t0
        msg = f"  n={n:5d} T={T:7d}  numpy {t_np:6.2f}s"
        if HAVE_NUMBA:
            run_mc_numba(10**6, n, 3, 64)  # warm-up / compile
            t0 = time.perf_counter(); run_mc_numba(10**6, n, 3, T); t_nb = time.perf_counter() - t0
            msg += f" | numba {t_nb:6.2f}s | speed-up x{t_np / t_nb:.2f}"
        print(msg)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, default=1_000_000, help="nominal number of trials per point")
    ap.add_argument("--elems", type=float, default=3e8,
                    help="cap on T*n random keys per grid point (adaptive T; raise for paper-scale runs)")
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--mem-mb", type=float, default=256, help="memory budget for one chunk (MB)")
    ap.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--conv-out", default="convergence.csv")
    ap.add_argument("--plot", default="plot_grid.png")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.benchmark:
        return benchmark()
    if args.plot_only:
        make_plot_grid(read_csv(args.out), read_csv(args.conv_out), args.plot, args.T)
        return print(f"wrote {args.plot}")

    Cs = [10**4, 10**5, 10**6, 10**8, 2**32]
    ks = [2, 3, 4]
    if args.quick:
        Cs, args.elems, args.T = [10**4, 10**6], 2e6, 20_000
    grid = build_grid(args.n_max, Cs, ks)
    print(f"{len(grid)} grid points | backend={args.backend} | T<= {args.T:,} | elems cap {args.elems:.1e}")
    results, t_all = [], time.perf_counter()
    for i, (C, n, k) in enumerate(grid, 1):
        r = evaluate_point(C, n, k, args.T, args.elems, args.backend, args.seed, args.mem_mb * 1e6)
        results.append(r)
        print(f"[{i:3d}/{len(grid)}] C={C:<11d} k={k} n={n:<5d} T={r['T']:<8d} hits={r['hits']:<7d} "
              f"p_hat={r['p_hat']:.3e}  poisson={r['p_poisson']:.3e}  ref={r['p_ref']:.3e}  "
              f"{r['regime']:<8s} {r['seconds']:.1f}s", flush=True)
    write_csv(args.out, results, FIELDS)
    print(f"wrote {args.out} ({time.perf_counter() - t_all:.0f}s total)")

    Tc = 20_000 if args.quick else args.T
    conv = run_convergence(10**6, 2, [("p~1e-2", 1e-2 * 1.0)], Tc, args.seed)  # k=2 : p ~ 1e-2
    for tag, tgt in (("p~1e-4", 1e-4), ("p~1e-6", 1e-6)):
        conv += run_convergence(10**5, 3, [(tag, tgt)], Tc, args.seed)
    write_csv(args.conv_out, conv, ["C", "k", "n", "tag", "t", "hits", "p_hat", "cp_low", "cp_high", "p_ref"])
    print(f"wrote {args.conv_out}")
    make_plot_grid(results, conv, args.plot, args.T)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
