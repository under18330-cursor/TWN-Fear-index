"""Raw (pre-normalization) indicator calculations.

Each function takes one or more tidy DataFrames (indexed by trading date,
ascending) and returns a single pd.Series of the RAW indicator value —
percentile normalization happens later in `scoring.py`. Keeping this split
means the raw-value math can be tested independently of how the number
later gets turned into a 0-100 score.

Expected input schemas (all indexed by a DatetimeIndex named "date"):

  index_price   : close                              (TAIEX 收盤價)
  new_high_low  : new_highs, new_lows                 (52週新高/新低家數)
  breadth       : advancing_volume, declining_volume  (漲跌家數對應成交量)
  options       : put_volume, call_volume             (台指選擇權量)
  vix           : taiex_vix                           (台指選擇權波動率指數)
  bond_index    : close                               (公債/貨幣市場基金指數)
  margin        : margin_balance                      (融資餘額，新台幣)
  foreign       : net_buy_sell, futures_net_oi         (外資現貨買賣超金額, 台指期未平倉多空淨口數)
"""

from __future__ import annotations

import pandas as pd


def momentum(index_price: pd.DataFrame, ma_window: int = 125) -> pd.Series:
    """% deviation of TAIEX close from its trailing moving average."""
    close = index_price["close"].astype(float)
    ma = close.rolling(ma_window, min_periods=ma_window // 2).mean()
    return ((close - ma) / ma * 100.0).rename("momentum")


def strength(new_high_low: pd.DataFrame) -> pd.Series:
    """Net new-52-week-highs as a share of highs+lows activity."""
    highs = new_high_low["new_highs"].astype(float)
    lows = new_high_low["new_lows"].astype(float)
    total = (highs + lows).replace(0.0, pd.NA)
    return ((highs - lows) / total * 100.0).rename("strength")


def breadth(breadth_df: pd.DataFrame, cumulative_window: int = 20) -> pd.Series:
    """Rolling advancing-vs-declining volume ratio (log scale, like McClellan-style breadth)."""
    adv = breadth_df["advancing_volume"].astype(float)
    dec = breadth_df["declining_volume"].astype(float)
    net = (adv - dec).rolling(cumulative_window, min_periods=1).sum()
    total = (adv + dec).rolling(cumulative_window, min_periods=1).sum().replace(0.0, pd.NA)
    return (net / total * 100.0).rename("breadth")


def put_call_ratio(options: pd.DataFrame) -> pd.Series:
    """TXO put volume / call volume. Higher = more hedging demand = more fear."""
    put = options["put_volume"].astype(float)
    call = options["call_volume"].astype(float).replace(0.0, pd.NA)
    return (put / call).rename("put_call")


def volatility(vix: pd.DataFrame, ma_window: int = 50) -> pd.Series:
    """TAIEX VIX relative to its own trailing average (>0 = spiking fear)."""
    v = vix["taiex_vix"].astype(float)
    ma = v.rolling(ma_window, min_periods=ma_window // 2).mean()
    return ((v - ma) / ma * 100.0).rename("volatility")


def safe_haven_demand(
    index_price: pd.DataFrame,
    bond_index: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Equity outperformance vs. bonds/money-market over `window` trading days."""
    equity_ret = index_price["close"].astype(float).pct_change(window)
    bond_ret = bond_index["close"].astype(float).pct_change(window)
    return ((equity_ret - bond_ret) * 100.0).rename("safe_haven")


def margin_sentiment(margin: pd.DataFrame, window: int = 20) -> pd.Series:
    """% change in margin (融資) balance over `window` trading days."""
    bal = margin["margin_balance"].astype(float)
    return (bal.pct_change(window) * 100.0).rename("margin")


def foreign_positioning(foreign: pd.DataFrame, cumulative_window: int = 20) -> pd.Series:
    """Blend of cumulative foreign net cash buying and net futures positioning.

    Both legs are pre-scaled by the caller's data source to comparable
    magnitudes (e.g. NT$ billions and thousands of contracts); here we
    just sum trailing cumulative flow, then z-like normalize is left to
    `scoring.percentile_score`.
    """
    net_cash = foreign["net_buy_sell"].astype(float).rolling(cumulative_window, min_periods=1).sum()
    net_futures = foreign["futures_net_oi"].astype(float)
    return (net_cash + net_futures).rename("foreign")


def compute_all_raw_indicators(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run every indicator function against a raw-data bundle and align on date.

    Sources are optional: a source your data provider doesn't offer (e.g.
    FinLab has no options/VIX/bond dataset) can simply be omitted from
    `raw`, and that indicator is skipped rather than raising. Downstream,
    `scoring.composite_score` renormalizes weights over whatever
    indicators actually produced a score, so missing sources don't bias
    the composite toward "neutral" -- they're excluded, not zeroed.
    """
    builders = {
        "momentum": lambda: momentum(raw["index_price"]),
        "strength": lambda: strength(raw["new_high_low"]),
        "breadth": lambda: breadth(raw["breadth"]),
        "put_call": lambda: put_call_ratio(raw["options"]),
        "volatility": lambda: volatility(raw["vix"]),
        "safe_haven": lambda: safe_haven_demand(raw["index_price"], raw["bond_index"]),
        "margin": lambda: margin_sentiment(raw["margin"]),
        "foreign": lambda: foreign_positioning(raw["foreign"]),
    }
    series = {}
    for name, build in builders.items():
        try:
            series[name] = build()
        except KeyError:
            continue
    if not series:
        raise ValueError("raw data bundle did not contain any known indicator sources")
    return pd.concat(series.values(), axis=1)
