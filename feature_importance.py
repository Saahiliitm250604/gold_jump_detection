import sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}")

FEATURE_COLS = ["jump9_lag1", "jump_count_30", "rv_past_15", "rv_past_60", "abs_rstar9",
                "order_imb_ma5", "buzz_ma15", "sentiment_ma15", "spread", "illiquidity",
                "depth", "force_ma5", "obv_chg", "D_lag1", "hour", "minute"]

model_df = pd.read_pickle("model_df_cache.pkl")
years = sorted(model_df["year"].unique())
folds = [(years[:i], years[i]) for i in range(1, len(years))]
train_years, test_year = folds[-1]

train = model_df[model_df["year"].isin(train_years)]
test = model_df[model_df["year"] == test_year]
Xtr_raw, ytr = train[FEATURE_COLS].to_numpy(), train["y_elevated"].to_numpy()
Xte_raw, yte = test[FEATURE_COLS].to_numpy(), test["y_elevated"].to_numpy()
scaler = StandardScaler().fit(Xtr_raw)
Xtr_s, Xte_s = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

log("fitting histgb (no resampling)")
imp_model = HistGradientBoostingClassifier(max_depth=6, max_iter=150, learning_rate=0.08, random_state=0)
imp_model.fit(Xtr_s, ytr)
log("fit done, computing permutation importance on subsample")
sub = 12000
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
