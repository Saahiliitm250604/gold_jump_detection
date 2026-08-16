"""
evaluation.py
-------------
Runs Med-k (and the ABD benchmark) across every session in the dataset and
scores the resulting jump flags against the synthetic ground-truth jump
indicator ('Dummy Bivariate') supplied with the data set. This ground truth
is what lets us go beyond the original paper (which had no labelled jumps to
validate against) and report real Precision / Recall / F1 / ROC-AUC numbers
for the parameter-tuning research questions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import med9


def run_medk_all_days(df, k, ck, threshold, return_col="Log Returns", date_col="date"):
    """Vectorised-by-day Med-k pass across the whole dataframe. Returns
    (jump_flags, r_star) numpy arrays aligned to df.index order."""
    n = len(df)
    jump_flags = np.zeros(n, dtype=bool)
    r_star = np.full(n, np.nan)
    pos = 0
    for _, g in df.groupby(date_col, sort=False):
        idx = np.arange(pos, pos + len(g))
        pos += len(g)
        r = g[return_col].to_numpy()
        res = med9.detect_jumps_medk(r, k, ck, threshold)
        jump_flags[idx] = res["jumps"]
        r_star[idx] = res["r_star"]
    return jump_flags, r_star


def run_abd_all_days(df, threshold, return_col="Log Returns", date_col="date", alpha=None):
    n = len(df)
    jump_flags = np.zeros(n, dtype=bool)
    stat = np.full(n, np.nan)
    pos = 0
    for _, g in df.groupby(date_col, sort=False):
        idx = np.arange(pos, pos + len(g))
        pos += len(g)
        r = g[return_col].to_numpy()
        s = med9.abd_shrink_stat(r, threshold, alpha=alpha) if alpha is not None else med9.abd_stat(r)
        stat[idx] = s
        jump_flags[idx] = np.where(np.isnan(s), False, np.abs(s) > threshold)
    return jump_flags, stat


def score_against_truth(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "n_true_jumps": int(y_true.sum()),
        "n_pred_jumps": int(y_pred.sum()),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        finite = np.isfinite(y_score)
        if finite.sum() > 0 and y_true[finite].sum() > 0 and y_true[finite].sum() < finite.sum():
            out["roc_auc"] = roc_auc_score(y_true[finite], np.nan_to_num(y_score[finite]))
        else:
            out["roc_auc"] = np.nan
    return out


def grid_search_medk(df, k_values, threshold_values, ck_map, truth_col="Dummy Bivariate"):
    rows = []
    y_true = df[truth_col].to_numpy()
    for k in k_values:
        ck = ck_map[k]
        for thr in threshold_values:
            jumps, r_star = run_medk_all_days(df, k, ck, thr)
            m = score_against_truth(y_true, jumps, np.abs(r_star))
            m.update({"k": k, "threshold": thr, "method": "Med" + str(k)})
            rows.append(m)
    return pd.DataFrame(rows)
