import sys, time, json, argparse
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import ts_smote

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}")

FEATURE_COLS = ["jump9_lag1", "jump_count_30", "rv_past_15", "rv_past_60", "abs_rstar9",
                "order_imb_ma5", "buzz_ma15", "sentiment_ma15", "spread", "illiquidity",
                "depth", "force_ma5", "obv_chg", "D_lag1", "hour", "minute"]

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--strategies", type=str, default="none,class_weight,smote,adasyn")
args = parser.parse_args()

model_df = pd.read_pickle("model_df_cache.pkl")
years = sorted(model_df["year"].unique())
folds = [(years[:i], years[i]) for i in range(1, len(years))]
train_years, test_year = folds[args.fold]
log(f"fold {args.fold}: train={train_years} test={test_year}")

train = model_df[model_df["year"].isin(train_years)]
test = model_df[model_df["year"] == test_year]
Xtr_raw, ytr = train[FEATURE_COLS].to_numpy(), train["y_elevated"].to_numpy()
Xte_raw, yte = test[FEATURE_COLS].to_numpy(), test["y_elevated"].to_numpy()
scaler = StandardScaler().fit(Xtr_raw)
Xtr_s, Xte_s = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

def make_model(name, strategy):
    if name == "logreg":
        cw = "balanced" if strategy == "class_weight" else None
        return LogisticRegression(max_iter=200, class_weight=cw)
    return HistGradientBoostingClassifier(max_depth=6, max_iter=150, learning_rate=0.08, random_state=0)

results = []
roc_data = {}
strategies = args.strategies.split(",")
for strat in strategies:
    if strat == "smote":
        Xtr, ytr_use = ts_smote.smote_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=args.fold)
    elif strat == "adasyn":
        Xtr, ytr_use = ts_smote.adasyn_oversample(Xtr_s, ytr, target_ratio=0.3, random_state=args.fold)
    else:
        Xtr, ytr_use = Xtr_s, ytr

    for mname in ["logreg", "histgb"]:
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
            "fold": args.fold, "train_years": ",".join(map(str, train_years)), "test_year": int(test_year),
            "model": mname, "strategy": strat,
            "n_train": len(ytr_use), "train_pos_rate": float(np.mean(ytr_use)),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "pr_auc": average_precision_score(yte, proba),
        }
        results.append(row)
        log(f"{mname:7s} {strat:12s} F1={row['f1']:.3f} AUC={row['roc_auc']:.3f} PR-AUC={row['pr_auc']:.3f}")
        if strat in ("none", "smote"):
            np.save(f"fold{args.fold}_{mname}_{strat}_proba.npy", proba)
            np.save(f"fold{args.fold}_{mname}_{strat}_ytrue.npy", yte)

out_df = pd.DataFrame(results)
out_path = "tables/ml_results_by_fold.csv"
try:
    existing = pd.read_csv(out_path)
    existing = existing[~((existing["fold"] == args.fold) & (existing["strategy"].isin(strategies)))]
    out_df = pd.concat([existing, out_df], ignore_index=True)
except FileNotFoundError:
    pass
out_df.to_csv(out_path, index=False)
log(f"saved -> {out_path} ({len(out_df)} rows total)")
