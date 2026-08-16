import sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import data_prep, med9

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}")

log("loading data")
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
keep_cols = ["Date-Time", "year"] + feature_cols + ["y_elevated", "jump9"]
model_df[keep_cols].to_pickle("model_df_cache.pkl")
log(f"cached {len(model_df)} rows, positive rate={model_df['y_elevated'].mean():.4f}")
print("FEATURE_COLS=" + ",".join(feature_cols))
