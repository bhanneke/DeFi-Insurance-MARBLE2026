#!/usr/bin/env python3
"""Realized premium rates on Nexus Mutual, used as the empirical benchmark
in Section 6 of the MARBLE 2026 paper.

Input: nexus_combined_data.csv, an export of on-chain Nexus Mutual cover
data (V1: one row per cover; V2: one row per staking-pool allocation, so
premiums must be summed per cover_id before computing rates -- naive
per-row rates understate multi-pool covers and summing cover_start_usd per
row double-counts volume). The export is not committed to this repository;
see README.md in this directory.

Rate definition: annualized premium = premium_usd / cover_start_usd
                                      / (cover_period_days / 365).

Reference output (10,893 covers, 2019-07-12 to 2025-06-12):
  median 2.60%, mean 4.21%, 5-95% [1.30%, 13.40%],
  cover-amount-weighted mean 3.76%, total cover volume $5.86B.
"""
from pathlib import Path

import numpy as np
import pandas as pd

df = pd.read_csv(Path(__file__).parent / "nexus_combined_data.csv", low_memory=False)
print(f"rows: {len(df)} | unique cover ids: {df['cover_id'].nunique()} "
      f"| versions: {df['nexus_version'].value_counts().to_dict()}")

g = df.groupby(["nexus_version", "cover_id"]).agg(
    premium_usd=("premium_usd", "sum"),
    cover_usd=("cover_start_usd", "first"),
    period=("cover_period", "first"),
    start=("cover_start_time", "first"),
).reset_index()
g = g[(g.cover_usd > 0) & (g.premium_usd > 0) & (g.period > 0)]
g["rate"] = g.premium_usd / g.cover_usd / (g.period / 365.0)
g = g[g.rate < 1.0]  # drop >100%/yr as data errors

print(f"covers: {len(g)} | {g.start.min()[:10]} to {g.start.max()[:10]}")
print(f"annualized premium rate: median {100*g.rate.median():.2f}%  "
      f"mean {100*g.rate.mean():.2f}%  "
      f"5-95% [{100*g.rate.quantile(.05):.2f}%, {100*g.rate.quantile(.95):.2f}%]")
print(f"cover-amount-weighted mean: {100*np.average(g.rate, weights=g.cover_usd):.2f}%")
print(f"total cover volume: ${g.cover_usd.sum()/1e9:.2f}B")
