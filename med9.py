"""
med9.py
-------
Core implementation of the Med-k jump detection family (Khashanah, Chen, Buckle
& Hawkes, 2025, JRSSC), generalised for k in {3,5,7,9,11}, plus the three
classical benchmark detectors used in the paper: RV-BV (Barndorff-Nielsen &
Shephard), the ABD test (Andersen, Bollerslev & Diebold, 2007a) and the LM
test (Lee & Mykland, 2008).

All functions operate on a single trading "day" (a 1-D numpy array of
log-returns for that day) and are combined at the dataset level by the
pipeline in `pipeline.py`.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# 1. Scaling constants c_k
# ---------------------------------------------------------------------------
# The paper hard-codes c_k for k = 3,5,7,9 (1.41936, 1.62360, 1.74332, 1.82184)
# but gives no closed form (footnote: "we thank Assad Jalali for his initial
# calculation of these factors"). We re-derive them (and extend to k=11) by
# Monte-Carlo: c_k = 1 / E[ median(|Z_1|,...,|Z_k|)^2 ] for Z_i ~ iid N(0,1),
# which is exactly the unbiasedness condition stated in the paper.

def compute_ck_monte_carlo(k_values, n_sim=4_000_000, seed=42):
    """Monte-Carlo estimate of the Med-k scaling constants c_k."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in k_values:
        n_blocks = n_sim // k
        z = rng.standard_normal((n_blocks, k))
        med = np.median(np.abs(z), axis=1)
        out[k] = 1.0 / np.mean(med ** 2)
    return out

# Values reported explicitly in Khashanah et al. (2025), Section 3, used to
# validate the Monte-Carlo re-derivation above.
PAPER_CK = {3: 1.41936, 5: 1.62360, 7: 1.74332, 9: 1.82184}


# ---------------------------------------------------------------------------
# 2. Med-k local volatility / realized volatility / standardized returns
# ---------------------------------------------------------------------------

def rolling_median_sq(returns, k):
    """Vectorised centred rolling median of |r| over a window of k (odd),
    returned only for the interior indices (k+1)/2 .. M-(k-1)/2 (1-indexed in
    the paper => 0-indexed here as (k-1)//2 .. M-1-(k-1)//2)."""
    r = np.abs(np.asarray(returns, dtype=float))
    M = len(r)
    half = (k - 1) // 2
    if M < k:
        return np.array([]), np.array([], dtype=int)
    s = pd.Series(r)
    med = s.rolling(window=k, center=True).median().to_numpy()
    idx = np.arange(half, M - half)
    return med[idx] ** 2, idx


def med_k_rv(returns, k, ck):
    """Return (MedkRV, LVk_full) where LVk_full is aligned to `returns`
    (NaN outside the valid interior window)."""
    M = len(returns)
    med_sq, idx = rolling_median_sq(returns, k)
    LVk_full = np.full(M, np.nan)
    if len(idx):
        LVk_full[idx] = ck * med_sq
        MedkRV = (M / (M + 1 - k)) * np.nansum(LVk_full)
    else:
        MedkRV = np.nan
    return MedkRV, LVk_full


def standardized_returns(returns, MedkRV):
    """r*_{t,j} = r_{t,j} / sqrt(MedkRV / M)"""
    M = len(returns)
    if not np.isfinite(MedkRV) or MedkRV <= 0:
        return np.full(M, np.nan)
    denom = np.sqrt(MedkRV / M)
    return np.asarray(returns, dtype=float) / denom


def detect_jumps_medk(returns, k, ck, threshold):
    """Full Med-k pipeline for one day. Returns dict with MedkRV, r_star,
    jump flags (bool array) and jump count."""
    MedkRV, LVk = med_k_rv(returns, k, ck)
    r_star = standardized_returns(returns, MedkRV)
    jumps = np.abs(r_star) > threshold
    jumps = np.where(np.isnan(r_star), False, jumps)
    return {
        "MedkRV": MedkRV,
        "LVk": LVk,
        "r_star": r_star,
        "jumps": jumps,
        "n_jumps": int(np.nansum(jumps)),
    }


# ---------------------------------------------------------------------------
# 3. Benchmark 1: RV - BV (Barndorff-Nielsen & Shephard) — a DAY-level test
# ---------------------------------------------------------------------------
MU1 = np.sqrt(2 / np.pi)


def rv_bv_stat(returns):
    """Barndorff-Nielsen & Shephard (2004,2006) day-level jump test.
    Returns (RV, BV, TQ, Z)."""
    r = np.asarray(returns, dtype=float)
    M = len(r)
    RV = np.sum(r ** 2)
    BV = (MU1 ** -2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    # tri-power quarticity
    from scipy.special import gamma
    mu_43_inv3 = (2 ** (2 / 3)) * gamma(7 / 6) / gamma(1 / 2)
    TQ = M * (mu_43_inv3 ** -3) * np.sum(
        (np.abs(r[2:]) ** (4 / 3)) * (np.abs(r[1:-1]) ** (4 / 3)) * (np.abs(r[:-2]) ** (4 / 3))
    )
    denom = np.sqrt(max((MU1 ** -4 + 2 * MU1 ** -2 - 5) * TQ / (M * BV ** 2), 1e-16))
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.sqrt(M) * (np.log(max(RV, 1e-16)) - np.log(max(BV, 1e-16))) / denom
    return RV, BV, TQ, Z


# ---------------------------------------------------------------------------
# 4. Benchmark 2: ABD test (Andersen, Bollerslev & Diebold, 2007a)
# ---------------------------------------------------------------------------

def abd_stat(returns):
    """|r_j| / sqrt(BV/M) for the whole day. Returns array aligned to
    `returns`."""
    r = np.asarray(returns, dtype=float)
    M = len(r)
    BV = (MU1 ** -2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])) if M > 1 else np.nan
    if not np.isfinite(BV) or BV <= 0:
        return np.full(M, np.nan)
    return r / np.sqrt(BV / M)


def abd_shrink_stat(returns, threshold, alpha=0.3, max_iter=3):
    """ABD variant with shrinkage of large returns before recomputing BV, as
    described in Section 6.2.1 of the paper (replace r_j by alpha*r_j if it
    exceeds the cut-off, then recompute BV)."""
    r = np.asarray(returns, dtype=float).copy()
    M = len(r)
    for _ in range(max_iter):
        BV = (MU1 ** -2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])) if M > 1 else np.nan
        if not np.isfinite(BV) or BV <= 0:
            return np.full(M, np.nan)
        stat = r / np.sqrt(BV / M)
        mask = np.abs(stat) > threshold
        if not mask.any():
            break
        r = np.where(mask, alpha * r, r)
    return stat


# ---------------------------------------------------------------------------
# 5. Benchmark 3: LM test (Lee & Mykland, 2008)
# ---------------------------------------------------------------------------

def lm_stat(returns_full_series, K):
    """Lee & Mykland (2008) local test computed on a single, continuous
    (multi-day) 1-D array of returns using a trailing window of K prior
    returns (as in the paper's cross-day LM implementation, Section 6.2.2).
    Vectorised with a pandas rolling mean of |r_i|*|r_i-1| (bi-power form),
    strictly using only PAST returns (shift(1) before rolling) so sigma2(t)
    never uses r_t itself -> no look-ahead. Returns array aligned to
    `returns_full_series` (NaN where the trailing window is incomplete)."""
    r = pd.Series(np.asarray(returns_full_series, dtype=float))
    abs_r = r.abs()
    prod = abs_r * abs_r.shift(1)          # prod[i] = |r_i| * |r_i-1|
    # sigma2 at time i uses the K prior "prod" values, i.e. prod[i-K..i-1]
    sigma2 = prod.shift(1).rolling(window=K, min_periods=K).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        L = r / np.sqrt(sigma2)
    return L.to_numpy()
