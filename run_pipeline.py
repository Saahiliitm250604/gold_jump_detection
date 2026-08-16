"""
run_pipeline.py
----------------
End-to-end driver. Produces every table/figure used in the notebook and the
PDF report. Run once; all downstream artifacts (notebook, report) read from
tables/ and figures/.
"""
import sys, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

import data_prep, med9, evaluation, simulation

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

T0 = time.time()
LOG = {}

def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
log("loading data")
df = data_prep.load_raw("data.csv")
df, counts = data_prep.filter_sessions(df, min_intervals=700)
LOG["n_rows"] = len(df)
LOG["n_days"] = df["date"].nunique()
LOG["date_range"] = [str(df["Date-Time"].min()), str(df["Date-Time"].max())]

# ---------------------------------------------------------------------------
# 2. Re-derive c_k scaling constants (Monte Carlo) and validate vs paper
# ---------------------------------------------------------------------------
log("deriving c_k")
ck_map = med9.compute_ck_monte_carlo([3, 5, 7, 9, 11], n_sim=6_000_000, seed=42)
ck_table = pd.DataFrame({
    "k": list(ck_map.keys()),
    "c_k_monte_carlo": [round(v, 5) for v in ck_map.values()],
    "c_k_paper": [med9.PAPER_CK.get(k, np.nan) for k in ck_map.keys()],
})
ck_table.to_csv("tables/ck_validation.csv", index=False)
log(ck_table.to_string())

# ---------------------------------------------------------------------------
# 3. Precompute r* for every k (needed for grid search + downstream use)
# ---------------------------------------------------------------------------
log("precomputing r* per k")
rstar_by_k = {}
for k in [3, 5, 7, 9, 11]:
    jumps, rstar = evaluation.run_medk_all_days(df, k, ck_map[k], threshold=1e9)  # threshold irrelevant, just want rstar
    rstar_by_k[k] = rstar
df["r_star_9"] = rstar_by_k[9]

# ---------------------------------------------------------------------------
# 4. Grid search over k x threshold vs ground truth
# ---------------------------------------------------------------------------
log("grid search")
k_values = [3, 5, 7, 9, 11]
thr_values = [3.0, 3.5, 4.0, 4.5, 5.0]
grid_rows = []
y_true = df["Dummy Bivariate"].to_numpy()
for k in k_values:
    rstar = rstar_by_k[k]
    for thr in thr_values:
        jumps = np.where(np.isnan(rstar), False, np.abs(rstar) > thr)
        m = evaluation.score_against_truth(y_true, jumps, np.abs(rstar))
        m.update({"k": k, "threshold": thr, "method": f"Med{k}"})
        grid_rows.append(m)
grid = pd.DataFrame(grid_rows)
grid.to_csv("tables/medk_grid_1min.csv", index=False)
best_f1 = grid.sort_values("f1", ascending=False).iloc[0]
best_auc = grid.sort_values("roc_auc", ascending=False).iloc[0]
LOG["best_f1_config"] = best_f1[["k", "threshold", "f1", "precision", "recall"]].to_dict()
LOG["best_auc_config"] = best_auc[["k", "threshold", "roc_auc"]].to_dict()

# heatmap figure
pivot_f1 = grid.pivot(index="k", columns="threshold", values="f1")
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(pivot_f1.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(pivot_f1.columns))); ax.set_xticklabels(pivot_f1.columns)
ax.set_yticks(range(len(pivot_f1.index))); ax.set_yticklabels(pivot_f1.index)
ax.set_xlabel("threshold n"); ax.set_ylabel("k")
ax.set_title("F1 vs (k, threshold) against synthetic ground truth")
for i in range(pivot_f1.shape[0]):
    for j in range(pivot_f1.shape[1]):
        ax.text(j, i, f"{pivot_f1.values[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
fig.colorbar(im, ax=ax, label="F1")
fig.tight_layout()
fig.savefig("figures/fig01_grid_heatmap.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Benchmark comparison: Med9 vs RV-BV vs ABD vs LM
# ---------------------------------------------------------------------------
log("benchmark comparison")
jumps9 = np.where(np.isnan(rstar_by_k[9]), False, np.abs(rstar_by_k[9]) > 4.0)
m9 = evaluation.score_against_truth(y_true, jumps9, np.abs(rstar_by_k[9]))

jumpsA, statA = evaluation.run_abd_all_days(df, 4.0)
mA = evaluation.score_against_truth(y_true, jumpsA, np.abs(statA))

jumpsAs, statAs = evaluation.run_abd_all_days(df, 4.0, alpha=0.3)
mAs = evaluation.score_against_truth(y_true, jumpsAs, np.abs(statAs))

L = med9.lm_stat(df["Log Returns"].to_numpy(), K=194)
jumpsL = np.where(np.isnan(L), False, np.abs(L) > 4.0)
mL = evaluation.score_against_truth(y_true, jumpsL, np.abs(L))

bns_rows = []
for date, g in df.groupby("date", sort=False):
    r = g["Log Returns"].to_numpy()
    RV, BV, TQ, Z = med9.rv_bv_stat(r)
    bns_rows.append({"date": date, "Z": Z, "day_true": g["Dummy Bivariate"].max()})
bns = pd.DataFrame(bns_rows)
bns["sig"] = np.abs(bns["Z"]) > 1.96
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
mB = {
    "precision": precision_score(bns["day_true"], bns["sig"], zero_division=0),
    "recall": recall_score(bns["day_true"], bns["sig"], zero_division=0),
    "f1": f1_score(bns["day_true"], bns["sig"], zero_division=0),
    "roc_auc": roc_auc_score(bns["day_true"], np.abs(bns["Z"])),
    "n_pred_jumps": int(bns["sig"].sum()),
    "n_true_jumps": int(bns["day_true"].sum()),
}
bns.to_csv("tables/rvbv_daylevel.csv", index=False)

bench = pd.DataFrame([
    {"method": "Med9 (thr=4)", **m9},
    {"method": "ABD (thr=4, no shrink)", **mA},
    {"method": "ABD (thr=4, shrink a=0.3)", **mAs},
    {"method": "LM (K=194, thr=4)", **mL},
    {"method": "RV-BV (day-level, 5%)", **mB},
])
bench.to_csv("tables/benchmark_comparison.csv", index=False)
log(bench.to_string())

# ---------------------------------------------------------------------------
# 6. Clustering-robustness simulation (Med9 vs ABD under contagion)
# ---------------------------------------------------------------------------
log("clustering simulation")
sim_rows = []
for gamma in [0.0, 0.5, 1.0, 2.0, 4.0]:
    returns, truth = simulation.build_clustered_dataset(
        n_days=800, M=194, lam0=0.03, gamma=gamma, beta=5.0,
        base_sigma=0.0007, jump_sigma_mult=6.0, seed=7)
    n_true = truth.sum()
    med_flags = np.zeros_like(truth)
    abd_flags = np.zeros_like(truth)
    for d in range(returns.shape[0]):
        r = returns[d]
        res = med9.detect_jumps_medk(r, 9, ck_map[9], threshold=4.101)
        med_flags[d] = res["jumps"]
        s = med9.abd_shrink_stat(r, threshold=3.914, alpha=0.3)
        abd_flags[d] = np.where(np.isnan(s), False, np.abs(s) > 3.914)
    adj = truth & (np.roll(truth, 1, axis=1) | np.roll(truth, -1, axis=1))
    sim_rows.append(dict(
        gamma=gamma, n_true=int(n_true),
        pct_clustered=round(adj.sum() / n_true, 4),
        recall_med9=round((med_flags & truth).sum() / n_true, 4),
        recall_abd=round((abd_flags & truth).sum() / n_true, 4),
        med9_detected=int(med_flags.sum()), abd_detected=int(abd_flags.sum()),
    ))
sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv("tables/clustering_simulation.csv", index=False)
log(sim_df.to_string())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sim_df["gamma"], sim_df["recall_med9"], "o-", label="Med9", color="#1f77b4")
ax.plot(sim_df["gamma"], sim_df["recall_abd"], "o-", label="ABD (shrink)", color="#d62728")
ax.set_xlabel("Hawkes self-excitation strength (gamma)")
ax.set_ylabel("Recall of injected jumps")
ax.set_title("Detection recall under simulated jump clustering")
ax.legend(); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/fig02_clustering_robustness.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Adaptive vs fixed threshold
# ---------------------------------------------------------------------------
log("adaptive threshold")
fixed_jumps = (np.abs(df["r_star_9"]) > 4.0).to_numpy()
target_rate = fixed_jumps.mean()

date_list = df["date"].to_numpy()
unique_dates = sorted(df["date"].unique())
daily_abs = df.groupby("date")["r_star_9"].apply(lambda s: np.abs(s.to_numpy()))

window = 60
thr_by_day = {}
buf = deque(maxlen=window)
for d in unique_dates:
    if len(buf) == 0:
        thr_by_day[d] = 4.0
    else:
        pool = np.concatenate(list(buf))
        thr_by_day[d] = np.nanquantile(pool, 1 - target_rate)
    buf.append(daily_abs[d])

adaptive_thr_arr = df["date"].map(thr_by_day).to_numpy()
adaptive_jumps = (np.abs(df["r_star_9"]).to_numpy() > adaptive_thr_arr)

adapt_rows = []
for name, flags in [("fixed_n4", fixed_jumps), ("adaptive_rolling60", adaptive_jumps)]:
    adapt_rows.append({
        "method": name, "n_pred_jumps": int(flags.sum()),
        "precision": precision_score(y_true, flags, zero_division=0),
        "recall": recall_score(y_true, flags, zero_division=0),
        "f1": f1_score(y_true, flags, zero_division=0),
    })
adapt_df = pd.DataFrame(adapt_rows)
adapt_df.to_csv("tables/adaptive_vs_fixed.csv", index=False)
log(adapt_df.to_string())

years = pd.to_datetime(df["date"].astype(str)).dt.year
year_stab = pd.DataFrame({
    "year": years, "date": df["date"], "fixed": fixed_jumps, "adaptive": adaptive_jumps
}).groupby("year").apply(lambda g: pd.Series({
    "days": g["date"].nunique(),
    "fixed_jumps_per_day": g["fixed"].sum() / g["date"].nunique(),
    "adaptive_jumps_per_day": g["adaptive"].sum() / g["date"].nunique(),
}), include_groups=False)
year_stab.to_csv("tables/threshold_stability_by_year.csv")
log(year_stab.to_string())

fig, ax = plt.subplots(figsize=(6, 4))
year_stab[["fixed_jumps_per_day", "adaptive_jumps_per_day"]].plot(kind="bar", ax=ax)
ax.set_ylabel("Avg. jumps / day"); ax.set_title("Detection-rate stability: fixed vs adaptive threshold")
fig.tight_layout()
fig.savefig("figures/fig03_adaptive_stability.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Jump analytics (Med9, thr=4)
# ---------------------------------------------------------------------------
log("jump analytics")
df["jump9"] = fixed_jumps
df["minute_of_day"] = df["Date-Time"].dt.hour * 60 + df["Date-Time"].dt.minute
intraday = df.groupby("minute_of_day")["jump9"].sum()
intraday.to_csv("tables/intraday_jump_distribution.csv")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(intraday.index / 60, intraday.values, lw=0.8)
ax.set_xlabel("Hour of day (UTC-ish exchange clock)"); ax.set_ylabel("Jump count")
ax.set_title("Intraday distribution of Med9 jumps (gold, k=9, thr=4)")
fig.tight_layout()
fig.savefig("figures/fig04_intraday_jumps.png")
plt.close(fig)

daily_jump_counts = df.groupby("date")["jump9"].sum()
daily_jump_counts.to_csv("tables/daily_jump_counts.csv")

def acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x ** 2)
    return np.array([np.sum(x[:n - h] * x[h:]) / denom for h in range(nlags + 1)])

acf_vals = acf(daily_jump_counts.values, 20)
fig, ax = plt.subplots(figsize=(7, 4))
ax.stem(range(len(acf_vals)), acf_vals)
ci = 1.96 / np.sqrt(len(daily_jump_counts))
ax.axhline(ci, color="grey", ls="--", lw=0.8); ax.axhline(-ci, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
ax.set_title("Autocorrelation of daily Med9 jump counts")
fig.tight_layout()
fig.savefig("figures/fig05_jump_count_acf.png")
plt.close(fig)

up = ((df["r_star_9"] > 4.0)).sum()
down = ((df["r_star_9"] < -4.0)).sum()
LOG["jump_direction"] = {"up": int(up), "down": int(down)}

# ---------------------------------------------------------------------------
# 9. ML predictive model: elevated future 15-min RV following a jump signal
# ---------------------------------------------------------------------------
log("building ML features")
d = df.reset_index(drop=True).copy()
d["ret"] = d["Log Returns"]
d["ret2"] = d["ret"] ** 2
d["abs_ret"] = d["ret"].abs()

# future 15-min realized variance (sum of next 15 squared returns), NOT including current
fut_rv15 = d["ret2"].iloc[::-1].rolling(15).sum().iloc[::-1].shift(-1)
d["fut_rv15"] = fut_rv15
elevated_cut = d["fut_rv15"].quantile(0.90)
d["y_elevated"] = (d["fut_rv15"] > elevated_cut).astype(int)

# causal features (only past/current info)
d["jump9_lag1"] = d["jump9"].shift(1).fillna(0)
d["jump_count_30"] = d["jump9"].shift(1).rolling(30, min_periods=5).sum()
d["rv_past_15"] = d["ret2"].shift(1).rolling(15, min_periods=5).sum()
d["rv_past_60"] = d["ret2"].shift(1).rolling(60, min_periods=10).sum()
d["abs_rstar9"] = d["r_star_9"].abs()
d["order_imb_ma5"] = d["Order Imbalance"].shift(1).rolling(5, min_periods=1).mean()
d["buzz_ma15"] = d["buzz"].shift(1).rolling(15, min_periods=1).mean()
d["sentiment_ma15"] = d["sentiment"].shift(1).rolling(15, min_periods=1).mean()
d["spread"] = d["Proportional Effective Spread"]
d["illiquidity"] = d["Ammiihud Illiquidity"]
d["depth"] = d["Depth"]
d["force_ma5"] = d["Force"].shift(1).rolling(5, min_periods=1).mean()
d["obv_chg"] = d["OBV"].diff()
d["D_lag1"] = d["D"].shift(1)
d["hour"] = d["Date-Time"].dt.hour
d["minute"] = d["Date-Time"].dt.minute

feature_cols = [
    "jump9_lag1", "jump_count_30", "rv_past_15", "rv_past_60", "abs_rstar9",
    "order_imb_ma5", "buzz_ma15", "sentiment_ma15", "spread", "illiquidity",
    "depth", "force_ma5", "obv_chg", "D_lag1", "hour", "minute",
]
model_df = d.dropna(subset=feature_cols + ["y_elevated"]).reset_index(drop=True)
model_df.to_parquet("tables/ml_model_frame.parquet") if False else None
model_df[["Date-Time"] + feature_cols + ["y_elevated", "jump9"]].to_csv("tables/ml_features_sample.csv", index=False)

LOG["ml_n_rows"] = len(model_df)
LOG["ml_positive_rate"] = float(model_df["y_elevated"].mean())
LOG["ml_positive_rate_given_jump"] = float(model_df.loc[model_df["jump9_lag1"] == 1, "y_elevated"].mean())
log(f"ML frame rows={len(model_df)}, base positive rate={LOG['ml_positive_rate']:.4f}, "
    f"positive rate | recent jump={LOG['ml_positive_rate_given_jump']:.4f}")

with open("tables/run_log.json", "w") as f:
    json.dump(LOG, f, indent=2, default=str)

log("PART 1 pipeline complete (ML model training happens in ml_pipeline.py)")
