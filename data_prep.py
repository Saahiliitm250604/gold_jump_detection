"""
data_prep.py
------------
Loads the 1-minute gold dataset, builds trading-day sessions, and provides
a resample-to-k-minute helper (log-returns are additive so k-min returns are
obtained by summing consecutive 1-min log returns).
"""
import numpy as np
import pandas as pd

RAW_COLS = [
    "Date-Time", "Open", "High", "Low", "Last", "No. Trades", "Depth",
    "Log Returns", "Realized Variance", "Proportional Effective Spread",
    "Ammiihud Illiquidity", "Order Imbalance", "buzz", "sentiment",
    "Cumulative Returns", "Force", "OBV", "D", "Dummy Bivariate",
]


def load_raw(path):
    df = pd.read_csv(path, parse_dates=["Date-Time"], dayfirst=True)
    df = df.sort_values("Date-Time").reset_index(drop=True)
    df["date"] = df["Date-Time"].dt.date
    return df


def filter_sessions(df, min_intervals=700):
    """Drop sessions (calendar days) with too few 1-min observations to be
    considered a full trading session (mirrors the paper's exclusion of
    half-days / holiday-truncated days)."""
    counts = df.groupby("date").size()
    good_days = counts[counts >= min_intervals].index
    out = df[df["date"].isin(good_days)].copy()
    return out, counts


def resample_returns(df, minutes=2):
    """Aggregate 1-min log returns into `minutes`-min log returns within each
    session by summing consecutive 1-min returns (log-returns are additive).
    Ground-truth jump flag ('Dummy Bivariate') is aggregated with `max` (a
    jump anywhere in the bucket marks the bucket as a jump minute), matching
    common practice for coarsening high-frequency jump labels."""
    def agg_day(g):
        g = g.reset_index(drop=True)
        n_bins = len(g) // minutes
        g = g.iloc[: n_bins * minutes]
        bin_id = np.repeat(np.arange(n_bins), minutes)
        out = pd.DataFrame({
            "Date-Time": g["Date-Time"].groupby(bin_id).last().values,
            "Log Returns": g["Log Returns"].groupby(bin_id).sum().values,
            "Dummy Bivariate": g["Dummy Bivariate"].groupby(bin_id).max().values,
            "Order Imbalance": g["Order Imbalance"].groupby(bin_id).sum().values,
            "buzz": g["buzz"].groupby(bin_id).sum().values,
            "sentiment": g["sentiment"].groupby(bin_id).mean().values,
            "No. Trades": g["No. Trades"].groupby(bin_id).sum().values,
            "date": g["date"].iloc[0],
        })
        return out
    return df.groupby("date", group_keys=False).apply(agg_day).reset_index(drop=True)
