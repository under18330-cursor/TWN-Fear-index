"""Offline smoke test: runs the full scoring pipeline on synthetic data.

No network access required. Generates ~3 years of daily data with a
deliberate "melt-up then crash" pattern in the middle so you can see the
composite score swing from extreme_greed to extreme_fear and check that
the overheat/oversold flags fire where expected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from twn_fear_greed.index import TWNFearGreedIndex

RNG = np.random.default_rng(42)
N = 756
dates = pd.bdate_range("2023-01-02", periods=N)

# Base random walk for TAIEX, with a melt-up (days 500-560) then a crash (560-590).
returns = RNG.normal(0.0003, 0.008, N)
returns[500:560] += 0.006   # euphoric melt-up
returns[560:590] -= 0.02    # sharp crash
close = 17000 * np.cumprod(1 + returns)
index_price = pd.DataFrame({"close": close}, index=dates)

new_highs = np.clip(RNG.poisson(15, N) + (returns * 3000).astype(int), 0, None)
new_lows = np.clip(RNG.poisson(15, N) - (returns * 3000).astype(int), 0, None)
new_high_low = pd.DataFrame({"new_highs": new_highs, "new_lows": new_lows}, index=dates)

adv_vol = np.clip(3000 + returns * 200000 + RNG.normal(0, 300, N), 100, None)
dec_vol = np.clip(3000 - returns * 200000 + RNG.normal(0, 300, N), 100, None)
breadth = pd.DataFrame({"advancing_volume": adv_vol, "declining_volume": dec_vol}, index=dates)

put_vol = np.clip(50000 - returns * 3_000_000 + RNG.normal(0, 3000, N), 1000, None)
call_vol = np.clip(50000 + returns * 3_000_000 + RNG.normal(0, 3000, N), 1000, None)
options = pd.DataFrame({"put_volume": put_vol, "call_volume": call_vol}, index=dates)

realized_vol = pd.Series(returns, index=dates).rolling(10, min_periods=1).std() * np.sqrt(252) * 100
taiex_vix = (realized_vol + RNG.normal(0, 1, N)).clip(lower=8).rename("taiex_vix")
vix = taiex_vix.to_frame()

bond_close = 100 * np.cumprod(1 + RNG.normal(0.00005, 0.0008, N))
bond_index = pd.DataFrame({"close": bond_close}, index=dates)

margin_balance = 200_000 * np.cumprod(1 + returns * 1.5 + RNG.normal(0, 0.002, N))
margin = pd.DataFrame({"margin_balance": margin_balance}, index=dates)

net_buy_sell = (returns * 5_000_000 + RNG.normal(0, 200_000, N))
futures_net_oi = np.clip((returns * 8000 + RNG.normal(0, 500, N)).cumsum() * 0.05, -20000, 20000)
foreign = pd.DataFrame({"net_buy_sell": net_buy_sell, "futures_net_oi": futures_net_oi}, index=dates)

raw = {
    "index_price": index_price,
    "new_high_low": new_high_low,
    "breadth": breadth,
    "options": options,
    "vix": vix,
    "bond_index": bond_index,
    "margin": margin,
    "foreign": foreign,
}

idx = TWNFearGreedIndex(lookback=252)
result = idx.compute(raw)

print("=== Latest reading ===")
print(result.iloc[-1][["composite", "label", "overheat_flag", "oversold_flag"]])

print("\n=== Composite score around the melt-up (day 555-565) ===")
print(result["composite"].iloc[555:565])

print("\n=== Composite score around the crash (day 585-595) ===")
print(result["composite"].iloc[585:595])

print("\n=== Any overheat/oversold streak flags fired? ===")
print(result[["overheat_flag", "oversold_flag", "sentiment_collapse", "sentiment_spike"]].any())

out_path = Path(__file__).with_name("example_output.csv")
result.to_csv(out_path)
print(f"\nFull history written to {out_path}")
