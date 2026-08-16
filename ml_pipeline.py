"""
ml_pipeline.py
--------------
Causal ML reformulation: predict whether realized volatility over the NEXT
15 minutes will be "elevated" (top decile of its unconditional distribution),
using only information available up to and including the current minute
(jump signals + rolling volatility/liquidity/sentiment features).

Validation design:
  * Walk-forward (expanding-window), split by calendar year -> no shuffling,
    no random train/test split, so the model is always tested strictly on
    the future relative to its training data.
  * Class imbalance (target positive rate ~10%) handled two ways for
    comparison: (a) class_weight='balanced', (b) TS-safe SMOTE/ADASYN-style
    oversampling applied only inside each training fold, after the temporal
    split, so no test-fold information is ever used to build synthetic
    training points (see src/ts_smote.py for the leakage argument).
"""
import sys, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix, roc_curve)

import data_prep, med9, evaluation, ts_smote

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}")

# ---------------------------------------------------------------------------
# Rebuild the feature frame (kept identical to run_pipeline.py Section 9)
# ---------------------------------------------------------------------------
log("loading data + rebuilding features")
df = data_prep.load_raw("data.csv")
df, counts = data_prep.filter_sessions(df, min_intervals=700)
ck9 = med9.PAPER_CK[9]

rstar = np.full(len(df), np.nan)
pos = 0
for date, g in df.groupby("date", sort=False):
    idx = np.arange(pos, pos + len(g)); pos += len(g)
    r = g["Log Returns"].to_numpy()
    MedkRV, _ = med9.med_k_rv(r, 9, ck9)
    rstar[idx] = med9.standardized_returns(r, MedkRV)
df["r_star_9"] = rstar
df["jump9"] = np.abs(df["r_star_9"]) > 4.0

d = df.reset_index(drop=True).copy()
d["ret2"] = d["Log Returns"] ** 2
fut_rv15 = d["ret2"].iloc[::-1].rolling(15).sum().iloc[::-1].shift(-1)
d["fut_rv15"] = fut_rv15
elevated_cut = d["fut_rv15"].quantile(0.90)
d["y_elevated"] = (d["fut_rv15"] > elevated_cut).astype(int)

d["jump9_lag1"] = d["jump9"].shift(1).fillna(0).astype(int)
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
d["year"] = d["Date-Time"].dt.year

feature_cols = [
    "jump9_lag1", "jump_count_30", "rv_past_15", "rv_past_60", "abs_rstar9",
    "order_imb_ma5", "buzz_ma15", "sentiment_ma15", "spread", "illiquidity",
    "depth", "force_ma5", "obv_chg", "D_lag1", "hour", "minute",
]
model_df = d.dropna(subset=feature_cols + ["y_elevated"]).reset_index(drop=True)
log(f"model frame: {len(model_df)} rows, positive rate={model_df['y_elevated'].mean():.4f}")

# ---------------------------------------------------------------------------
# Walk-forward folds (expanding window, split by calendar year)
# ---------------------------------------------------------------------------
years = sorted(model_df["year"].unique())
folds = []
for i in range(1, len(years)):
    train_years = years[:i]
    test_year = years[i]
    folds.append((train_years, test_year))
log(f"walk-forward folds: {folds}")

def get_split(train_years, test_year):
    train = model_df[model_df["year"].isin(train_years)]
    test = model_df[model_df["year"] == test_year]
    return train, test

# ---------------------------------------------------------------------------
# Run all (model x imbalance-strategy) combinations across folds
# ---------------------------------------------------------------------------
strategies = ["none", "class_weight", "smote", "adasyn"]
model_names = ["logreg", "histgb"]

def make_model(name, strategy):
    if name == "logreg":
        cw = "balanced" if strategy == "class_weight" else None
        return LogisticRegression(max_iter=200, class_weight=cw, n_jobs=None)
    else:
        cw = "balanced" if strategy == "class_weight" else None
        # HistGradientBoostingClassifier doesn't take class_weight; approximate
        # via sample_weight instead (handled at fit time).
        return HistGradientBoostingClassifier(max_depth=6, max_iter=150, learning_rate=0.08,
                                               random_state=0)

results = []
roc_curves = {}  # keep one representative fold's ROC curve per (model,strategy)

for fi, (train_years, test_year) in enumerate(folds):
    train, test = get_split(train_years, test_year)
    Xtr_raw, ytr = train[feature_cols].to_numpy(), train["y_elevated"].to_numpy()
    Xte_raw, yte = test[feature_cols].to_numpy(), test["y_elevated"].to_numpy()

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s, Xte_s = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    for strat in strategies:
        if strat == "smote":
            Xtr, ytr_use = ts_smote.smote_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=fi)
        elif strat == "adasyn":
            Xtr, ytr_use = ts_smote.adasyn_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=fi)
        else:
            Xtr, ytr_use = Xtr_s, ytr

        for mname in model_names:
            model = make_model(mname, strat)
            if mname == "histgb" and strat == "class_weight":
                pos_w = (ytr_use == 0).sum() / max((ytr_use == 1).sum(), 1)
                sw = np.where(ytr_use == 1, pos_w, 1.0)
                model.fit(Xtr, ytr_use, sample_weight=sw)
            else:
                model.fit(Xtr, ytr_use)
            proba = model.predict_proba(Xte_s)[:, 1]
            pred = (proba >= 0.5).astype(int)
            row = {
                "fold": fi, "train_years": ",".join(map(str, train_years)), "test_year": test_year,
                "model": mname, "strategy": strat,
                "n_train": len(ytr_use), "train_pos_rate": float(np.mean(ytr_use)),
                "precision": precision_score(yte, pred, zero_division=0),
                "recall": recall_score(yte, pred, zero_division=0),
                "f1": f1_score(yte, pred, zero_division=0),
                "roc_auc": roc_auc_score(yte, proba),
                "pr_auc": average_precision_score(yte, proba),
            }
            results.append(row)
            key = (mname, strat)
            if key not in roc_curves and fi == len(folds) - 1:
                fpr, tpr, _ = roc_curve(yte, proba)
                roc_curves[key] = (fpr, tpr, row["roc_auc"])
            log(f"fold{fi} {mname:7s} {strat:12s} F1={row['f1']:.3f} AUC={row['roc_auc']:.3f} PR-AUC={row['pr_auc']:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv("tables/ml_results_by_fold.csv", index=False)

summary = results_df.groupby(["model", "strategy"])[["precision", "recall", "f1", "roc_auc", "pr_auc"]].mean().reset_index()
summary.to_csv("tables/ml_results_summary.csv", index=False)
log("\n" + summary.to_string())

# ---------------------------------------------------------------------------
# Bootstrap CI for F1 on the final (largest) fold, best model
# ---------------------------------------------------------------------------
best_row = summary.sort_values("f1", ascending=False).iloc[0]
best_model_name, best_strategy = best_row["model"], best_row["strategy"]
log(f"bootstrapping CI for best config: {best_model_name} / {best_strategy}")

train_years, test_year = folds[-1]
train, test = get_split(train_years, test_year)
Xtr_raw, ytr = train[feature_cols].to_numpy(), train["y_elevated"].to_numpy()
Xte_raw, yte = test[feature_cols].to_numpy(), test["y_elevated"].to_numpy()
scaler = StandardScaler().fit(Xtr_raw)
Xtr_s, Xte_s = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
if best_strategy == "smote":
    Xtr, ytr_use = ts_smote.smote_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=99)
elif best_strategy == "adasyn":
    Xtr, ytr_use = ts_smote.adasyn_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=99)
else:
    Xtr, ytr_use = Xtr_s, ytr
best_model = make_model(best_model_name, best_strategy)
if best_model_name == "histgb" and best_strategy == "class_weight":
    pos_w = (ytr_use == 0).sum() / max((ytr_use == 1).sum(), 1)
    sw = np.where(ytr_use == 1, pos_w, 1.0)
    best_model.fit(Xtr, ytr_use, sample_weight=sw)
else:
    best_model.fit(Xtr, ytr_use)
proba_test = best_model.predict_proba(Xte_s)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

rng = np.random.default_rng(0)
n = len(yte)
boot_f1, boot_auc = [], []
for _ in range(500):
    idx = rng.integers(0, n, n)
    if yte[idx].sum() == 0 or yte[idx].sum() == n:
        continue
    boot_f1.append(f1_score(yte[idx], pred_test[idx], zero_division=0))
    boot_auc.append(roc_auc_score(yte[idx], proba_test[idx]))
ci = {
    "model": best_model_name, "strategy": best_strategy,
    "f1_mean": float(np.mean(boot_f1)), "f1_ci_lo": float(np.percentile(boot_f1, 2.5)), "f1_ci_hi": float(np.percentile(boot_f1, 97.5)),
    "auc_mean": float(np.mean(boot_auc)), "auc_ci_lo": float(np.percentile(boot_auc, 2.5)), "auc_ci_hi": float(np.percentile(boot_auc, 97.5)),
}
with open("tables/ml_bootstrap_ci.json", "w") as f:
    json.dump(ci, f, indent=2)
log(f"bootstrap CI: {ci}")

# confusion matrix + ROC curve figures for the best config
cm = confusion_matrix(yte, pred_test)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Not elevated", "Elevated"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Not elevated", "Elevated"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix: {best_model_name}/{best_strategy}\n(test year {test_year})")
fig.tight_layout()
fig.savefig("figures/fig07_confusion_matrix.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
for (mname, strat), (fpr, tpr, auc) in roc_curves.items():
    ax.plot(fpr, tpr, label=f"{mname}/{strat} (AUC={auc:.3f})", lw=1.3)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC curves — final fold (test year {folds[-1][1]})")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/fig08_roc_curves.png")
plt.close(fig)

# imbalance strategy comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
piv = summary.pivot(index="strategy", columns="model", values="f1")
piv.plot(kind="bar", ax=ax)
ax.set_ylabel("F1 (avg over folds)"); ax.set_title("Imbalance-handling strategy comparison")
fig.tight_layout()
fig.savefig("figures/fig06_imbalance_strategy_comparison.png")
plt.close(fig)

# feature importance (histgb, best strategy) via permutation-free approx: use built-in if available
try:
    from sklearn.inspection import permutation_importance
    imp_model = make_model("histgb", best_strategy) if best_strategy != "smote" else make_model("histgb", "none")
    imp_model.fit(Xtr_s, ytr)
    pi = permutation_importance(imp_model, Xte_s[:20000], yte[:20000], n_repeats=3, random_state=0, n_jobs=1)
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    fi_df.to_csv("tables/ml_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
    ax.set_xlabel("Permutation importance (drop in ROC-AUC)")
    ax.set_title("Feature importance (HistGB)")
    fig.tight_layout()
    fig.savefig("figures/fig09_feature_importance.png")
    plt.close(fig)
except Exception as e:
    log(f"feature importance skipped: {e}")

log("ML pipeline complete")
