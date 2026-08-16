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
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix, roc_curve,
                              precision_score, recall_score)
from sklearn.inspection import permutation_importance

import ts_smote

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}")

FEATURE_COLS = ["jump9_lag1", "jump_count_30", "rv_past_15", "rv_past_60", "abs_rstar9",
                "order_imb_ma5", "buzz_ma15", "sentiment_ma15", "spread", "illiquidity",
                "depth", "force_ma5", "obv_chg", "D_lag1", "hour", "minute"]

model_df = pd.read_pickle("model_df_cache.pkl")
years = sorted(model_df["year"].unique())
folds = [(years[:i], years[i]) for i in range(1, len(years))]
train_years, test_year = folds[-1]  # final, largest fold: train 2021-2023, test 2024
log(f"final fold: train={train_years} test={test_year}")

train = model_df[model_df["year"].isin(train_years)]
test = model_df[model_df["year"] == test_year]
Xtr_raw, ytr = train[FEATURE_COLS].to_numpy(), train["y_elevated"].to_numpy()
Xte_raw, yte = test[FEATURE_COLS].to_numpy(), test["y_elevated"].to_numpy()
scaler = StandardScaler().fit(Xtr_raw)
Xtr_s, Xte_s = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

summary = pd.read_csv("tables/ml_results_summary.csv")
best_row = summary.sort_values("f1", ascending=False).iloc[0]
best_model_name, best_strategy = best_row["model"], best_row["strategy"]
log(f"best config overall (avg F1 across folds): {best_model_name}/{best_strategy}")

def make_model(name, strategy):
    if name == "logreg":
        cw = "balanced" if strategy == "class_weight" else None
        return LogisticRegression(max_iter=200, class_weight=cw)
    return HistGradientBoostingClassifier(max_depth=6, max_iter=150, learning_rate=0.08, random_state=0)

def get_train(strategy, seed=99):
    if strategy == "smote":
        return ts_smote.smote_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=seed)
    elif strategy == "adasyn":
        return ts_smote.adasyn_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=seed)
    return Xtr_s, ytr

Xtr, ytr_use = get_train(best_strategy)
best_model = make_model(best_model_name, best_strategy)
if best_model_name == "histgb" and best_strategy == "class_weight":
    pos_w = (ytr_use == 0).sum() / max((ytr_use == 1).sum(), 1)
    sw = np.where(ytr_use == 1, pos_w, 1.0)
    best_model.fit(Xtr, ytr_use, sample_weight=sw)
else:
    best_model.fit(Xtr, ytr_use)
proba_test = best_model.predict_proba(Xte_s)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)
log(f"final-fold test: F1={f1_score(yte,pred_test):.3f} AUC={roc_auc_score(yte,proba_test):.3f}")

# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
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
    "test_year": int(test_year), "n_test": int(n),
}
with open("tables/ml_bootstrap_ci.json", "w") as f:
    json.dump(ci, f, indent=2)
log(f"bootstrap CI: F1={ci['f1_mean']:.3f} [{ci['f1_ci_lo']:.3f},{ci['f1_ci_hi']:.3f}]  "
    f"AUC={ci['auc_mean']:.3f} [{ci['auc_ci_lo']:.3f},{ci['auc_ci_hi']:.3f}]")

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# ROC curves: compare strategies for histgb on final fold
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
for strat in ["none", "class_weight", "smote", "adasyn"]:
    Xtr_s2, ytr2 = get_train(strat)
    m = make_model("histgb", strat)
    if strat == "class_weight":
        pos_w = (ytr2 == 0).sum() / max((ytr2 == 1).sum(), 1)
        sw = np.where(ytr2 == 1, pos_w, 1.0)
        m.fit(Xtr_s2, ytr2, sample_weight=sw)
    else:
        m.fit(Xtr_s2, ytr2)
    p = m.predict_proba(Xte_s)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    auc = roc_auc_score(yte, p)
    ax.plot(fpr, tpr, label=f"histgb/{strat} (AUC={auc:.3f})", lw=1.3)
    log(f"roc curve strat={strat} auc={auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC curves (HistGB) — final fold (test year {test_year})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig08_roc_curves.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Imbalance strategy comparison bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
piv = summary.pivot(index="strategy", columns="model", values="f1")
piv = piv.loc[["none", "class_weight", "smote", "adasyn"]]
piv.plot(kind="bar", ax=ax)
ax.set_ylabel("F1 (avg over 3 walk-forward folds)")
ax.set_title("Imbalance-handling strategy comparison")
ax.set_xticklabels(piv.index, rotation=0)
fig.tight_layout()
fig.savefig("figures/fig06_imbalance_strategy_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Feature importance (permutation, HistGB no-resample, subsample of test)
# ---------------------------------------------------------------------------
imp_model = make_model("histgb", "none")
imp_model.fit(Xtr_s, ytr)
sub = min(20000, len(yte))
pi = permutation_importance(imp_model, Xte_s[:sub], yte[:sub], n_repeats=3, random_state=0, n_jobs=1)
fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
fi_df.to_csv("tables/ml_feature_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
ax.set_xlabel("Permutation importance (drop in ROC-AUC)")
ax.set_title("Feature importance (HistGB, no resampling)")
fig.tight_layout()
fig.savefig("figures/fig09_feature_importance.png")
plt.close(fig)
log(fi_df.to_string())

log("done")
