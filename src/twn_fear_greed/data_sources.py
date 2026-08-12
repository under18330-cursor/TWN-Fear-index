"""Fetchers for TWSE / TAIFEX open data.

These hit real network endpoints and are NOT covered by the unit tests in
this repo (see examples/run_example.py for an offline, synthetic-data
smoke test of the scoring pipeline instead). Endpoint paths and field
names are consolidated here as constants so a future exchange API change
only requires editing this file, not indicators.py or scoring.py.

All endpoints below are free, public, and require no API key:
  TWSE OpenAPI   https://openapi.twse.com.tw/
  TAIFEX OpenAPI https://openapi.taifex.com.tw/
"""

from __future__ import annotations

import pandas as pd
import requests

TWSE_BASE = "https://openapi.twse.com.tw/v1"
TAIFEX_BASE = "https://openapi.taifex.com.tw/v1"

ENDPOINTS = {
    # TAIEX 每日收盤指數
    "index_price": f"{TWSE_BASE}/indices/TAIEX",
    # 個股日成交資訊（用來自行推算漲跌家數/量、52週新高新低）
    "daily_quotes": f"{TWSE_BASE}/exchangeReport/MI_INDEX",
    # 三大法人買賣金額統計表（外資買賣超）
    "institutional_investors": f"{TWSE_BASE}/fund/T86",
    # 融資融券餘額
    "margin_balance": f"{TWSE_BASE}/exchangeReport/MI_MARGN",
    # 台指選擇權每日交易資訊（Put/Call 量）
    "taiex_options": f"{TAIFEX_BASE}/DailyMarketReportOpt",
    # 台指選擇權波動率指數
    "taiex_vix": f"{TAIFEX_BASE}/OptVixIndex",
    # 期貨三大法人未平倉（台指期外資淨部位）
    "futures_institutional_oi": f"{TAIFEX_BASE}/OiFutuvsOpt",
}

DEFAULT_TIMEOUT = 15


def _get_json(url: str) -> list[dict]:
    resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_index_price() -> pd.DataFrame:
    """TAIEX daily close. Columns: close (indexed by date)."""
    data = _get_json(ENDPOINTS["index_price"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["Date"])
    df["close"] = pd.to_numeric(df["ClosingIndex"], errors="coerce")
    return df.set_index("date")[["close"]].sort_index()


def fetch_new_high_low() -> pd.DataFrame:
    """Placeholder aggregation: derive 52-week new-high/new-low counts from
    daily per-stock quotes. TWSE's raw feed is per-stock, so this requires
    a rolling 52-week max/min computed once per symbol and then aggregated
    cross-sectionally per day -- left as a project-specific ETL step since
    it depends on how much per-symbol history you choose to retain locally.
    """
    raise NotImplementedError(
        "Aggregate 52-week new-high/new-low counts from a locally stored "
        "per-symbol price history; TWSE's OpenAPI only exposes one day of "
        "per-stock quotes at a time."
    )


def fetch_breadth() -> pd.DataFrame:
    """Same caveat as fetch_new_high_low: advancing/declining volume needs
    a locally accumulated per-symbol daily history, not a single endpoint.
    """
    raise NotImplementedError(
        "Aggregate advancing/declining volume from a locally stored "
        "per-symbol daily quotes history."
    )


def fetch_options() -> pd.DataFrame:
    """TXO put/call volume. Columns: put_volume, call_volume (indexed by date)."""
    data = _get_json(ENDPOINTS["taiex_options"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["Date"])
    is_put = df["CallPut"].str.contains("Put", case=False, na=False)
    daily = (
        df.assign(volume=pd.to_numeric(df["Volume"], errors="coerce"))
        .groupby(["date", is_put.rename("is_put")])["volume"]
        .sum()
        .unstack(fill_value=0)
    )
    daily.columns = ["call_volume", "put_volume"]
    return daily.sort_index()


def fetch_vix() -> pd.DataFrame:
    """TAIEX option volatility index. Columns: taiex_vix (indexed by date)."""
    data = _get_json(ENDPOINTS["taiex_vix"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["Date"])
    df["taiex_vix"] = pd.to_numeric(df["Value"], errors="coerce")
    return df.set_index("date")[["taiex_vix"]].sort_index()


def fetch_margin() -> pd.DataFrame:
    """融資餘額. Columns: margin_balance (indexed by date)."""
    data = _get_json(ENDPOINTS["margin_balance"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["Date"])
    df["margin_balance"] = pd.to_numeric(df["MarginPurchaseBalance"], errors="coerce")
    return df.set_index("date")[["margin_balance"]].sort_index()


def fetch_foreign() -> pd.DataFrame:
    """外資現貨買賣超 + 台指期外資未平倉淨口數. Columns: net_buy_sell, futures_net_oi."""
    cash = _get_json(ENDPOINTS["institutional_investors"])
    cash_df = pd.DataFrame(cash)
    cash_df["date"] = pd.to_datetime(cash_df["Date"])
    cash_df["net_buy_sell"] = pd.to_numeric(
        cash_df["ForeignInvestorsExcludingDealersNetBuySell"], errors="coerce"
    )
    cash_df = cash_df.set_index("date")[["net_buy_sell"]]

    fut = _get_json(ENDPOINTS["futures_institutional_oi"])
    fut_df = pd.DataFrame(fut)
    fut_df["date"] = pd.to_datetime(fut_df["Date"])
    fut_df["futures_net_oi"] = pd.to_numeric(
        fut_df["ForeignInvestorsNetOI"], errors="coerce"
    )
    fut_df = fut_df.set_index("date")[["futures_net_oi"]]

    return cash_df.join(fut_df, how="outer").sort_index()


def fetch_bond_index() -> pd.DataFrame:
    """Bond / money-market proxy for the safe-haven-demand indicator.

    TWSE OpenAPI doesn't publish a ready-made bond index; use a local bond
    ETF/fund NAV series (e.g. a Taiwan government bond ETF) as the proxy
    and adapt this function to your chosen data source.
    """
    raise NotImplementedError(
        "Supply a bond/money-market index/ETF NAV series as the safe-haven proxy."
    )


def fetch_all_raw_data(start: str | None = None) -> dict[str, pd.DataFrame]:
    """Fetch everything indicators.compute_all_raw_indicators expects.

    Note: fetch_new_high_low, fetch_breadth, and fetch_bond_index require
    project-specific local data (see their docstrings) and will raise
    NotImplementedError until wired up.
    """
    raw = {
        "index_price": fetch_index_price(),
        "new_high_low": fetch_new_high_low(),
        "breadth": fetch_breadth(),
        "options": fetch_options(),
        "vix": fetch_vix(),
        "bond_index": fetch_bond_index(),
        "margin": fetch_margin(),
        "foreign": fetch_foreign(),
    }
    if start:
        raw = {k: v.loc[start:] for k, v in raw.items()}
    return raw
