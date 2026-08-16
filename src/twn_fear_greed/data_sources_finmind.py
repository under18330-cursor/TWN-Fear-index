"""FinMind (https://github.com/FinMind/FinMind) data connector.

Fills two gaps neither FinLab nor the free TWSE/TAIFEX connectors
(data_sources.py) can: TAIEX option (TXO) Put/Call volume (FinLab has no
options dataset at all -- see data_sources_finlab.py) and the
safe-haven-demand indicator's bond leg (TWSE/TAIFEX's own free endpoints
have no bond series; Taiwan's long-duration Treasury ETFs are TPEx-listed,
not TWSE, and FinMind's TaiwanStockPrice dataset covers both markets
under one dataset name -- no separate TPEx integration needed). This
talks to FinMind's public v4 REST API directly with `requests` instead of
the `FinMind` pip package: that package pulls in the `ta` indicator
library as a dependency, which failed to build in this sandbox, and isn't
needed just to call a couple of REST endpoints.

Unlike the free TWSE/TAIFEX classic endpoints (data_sources.py), FinMind
takes a real start_date/end_date range in one request -- a 3-year pull is
one call, not one per trading day.

Authentication
---------------
FinMind tokens are issued by FinMind's own account system and are NOT
the same credential as a FinLab API key -- a FinLab key will not
authenticate against FinMind's API. Register/generate a token at
FinMind's own site, then:

    export FINMIND_API_TOKEN="..."

As always, never hardcode the token in source, and treat any token
that has been pasted into a chat/ticket as compromised -- rotate it.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_TIMEOUT = 15

DATASETS = {
    "option_daily": "TaiwanOptionDaily",
    "margin": "TaiwanStockMarginPurchaseShortSale",
}


def _get_dataframe(
    dataset: str,
    data_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    token = token or os.environ.get("FINMIND_API_TOKEN")
    if not token:
        raise RuntimeError(
            "No FinMind API token found. Set the FINMIND_API_TOKEN "
            "environment variable (a token issued by FinMind itself, not "
            "your FinLab key) or pass token= explicitly."
        )
    params = {"dataset": dataset, "token": token}
    if data_id:
        params["data_id"] = data_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    resp = requests.get(FINMIND_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 200:
        raise RuntimeError(f"FinMind API error: {payload.get('msg', payload)}")
    return pd.DataFrame(payload["data"])


def _aggregate_put_call(raw_options: pd.DataFrame, session: str | None = None) -> pd.DataFrame:
    """Pure transform: raw TaiwanOptionDaily rows -> daily put/call volume.

    Kept separate from the network call so this logic is unit-testable
    offline (see tests/test_data_sources_finmind.py).

    TaiwanOptionDaily returns one row per (date, contract, strike, side,
    trading_session), where trading_session is "position" (the regular
    daytime session) or "after_market" (the night session). session=None
    sums both into a whole-day figure; pass "position" to restrict the
    ratio to regular-hours flow.
    """
    if raw_options.empty:
        return pd.DataFrame(columns=["call_volume", "put_volume"])

    df = raw_options.copy()
    if session is not None:
        df = df[df["trading_session"].astype(str) == session]
        if df.empty:
            return pd.DataFrame(columns=["call_volume", "put_volume"])
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    is_put = df["call_put"].astype(str).str.upper().eq("PUT")

    daily = (
        df.assign(is_put=is_put)
        .groupby(["date", "is_put"])["volume"]
        .sum()
        .unstack(fill_value=0)
    )
    daily = daily.rename(columns={False: "call_volume", True: "put_volume"})
    for col in ("call_volume", "put_volume"):
        if col not in daily.columns:
            daily[col] = 0.0
    return daily[["call_volume", "put_volume"]].sort_index()


def fetch_put_call_ratio(
    start: str | None = None,
    data_id: str = "TXO",
    token: str | None = None,
    session: str | None = None,
) -> pd.DataFrame:
    """Daily TAIEX option put/call volume. Columns: call_volume, put_volume."""
    raw_options = _get_dataframe(
        DATASETS["option_daily"], data_id=data_id, start_date=start, token=token
    )
    return _aggregate_put_call(raw_options, session=session)


def fetch_put_call_ratio_history(
    start: str,
    end: str | None = None,
    data_id: str = "TXO",
    token: str | None = None,
    session: str | None = None,
) -> pd.DataFrame:
    """Real daily TAIEX option put/call volume over a multi-year range.

    TaiwanOptionDaily is tick-granular (date x contract x strike x side):
    a single trading day is already ~6,500 rows, so one request spanning
    years (~4.7M rows for 3 years, extrapolated) times out and isn't
    practical to hold in memory anyway -- fetched one calendar month at a
    time instead, aggregating each month down to its daily put/call
    volume immediately (`_aggregate_put_call`) and discarding the raw
    rows, so peak memory stays at "one month of raw ticks", not "three
    years of them". ~36 requests for 3 years, a couple seconds each.
    Columns: call_volume, put_volume.
    """
    start_ts = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()
    months = pd.date_range(start_ts, end_ts, freq="MS")
    if months.empty or months[0] != start_ts:
        months = months.insert(0, start_ts)

    frames = []
    for month_start in months:
        month_end = min(month_start + pd.offsets.MonthEnd(0), end_ts)
        raw = _get_dataframe(
            DATASETS["option_daily"],
            data_id=data_id,
            start_date=month_start.strftime("%Y-%m-%d"),
            end_date=month_end.strftime("%Y-%m-%d"),
            token=token,
            timeout=60,
        )
        frames.append(_aggregate_put_call(raw, session=session))

    if not frames:
        return pd.DataFrame(columns=["call_volume", "put_volume"]).rename_axis("date")
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result.loc[start:end] if end else result.loc[start:]


def augment_with_options(
    raw: dict[str, pd.DataFrame],
    start: str | None = None,
    token: str | None = None,
    session: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Take a raw-data bundle (e.g. from data_sources_finlab.fetch_all_raw_data)
    and add/replace its "options" entry with FinMind's put/call data, so
    the put_call indicator -- otherwise skipped for lack of a source --
    gets computed too.
    """
    augmented = dict(raw)
    augmented["options"] = fetch_put_call_ratio(start=start, token=token, session=session)
    return augmented


DEFAULT_BOND_ETF = "00679B"  # 元大美債20年 -- Yuanta 20+ Year US Treasury Bond ETF, TPEx


def _bond_index_from_price(raw_price: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: raw TaiwanStockPrice rows for a bond ETF -> the
    bond_index shape indicators.safe_haven_demand expects. Columns: close.
    """
    if raw_price.empty:
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    df = raw_price.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("date")[["close"]].sort_index()


def fetch_bond_index(
    start: str | None = None,
    data_id: str = DEFAULT_BOND_ETF,
    token: str | None = None,
) -> pd.DataFrame:
    """Daily close of a Taiwan-listed long-duration bond ETF, as the
    safe-haven-demand indicator's bond leg (see indicators.safe_haven_demand).

    Defaults to 00679B (元大美債20年, Yuanta 20+ Year US Treasury Bond ETF,
    TPEx-listed) -- the standard long-duration Treasury proxy for a
    "stocks vs. bonds" flight-to-safety signal, and liquid (tens of
    millions of shares/day) so its price series shouldn't have stale gaps.
    Pass a different `data_id` for a different bond ETF (e.g. 00687B,
    國泰20年美債, a comparable alternative). Columns: close.
    """
    raw_price = _get_dataframe("TaiwanStockPrice", data_id=data_id, start_date=start, token=token)
    return _bond_index_from_price(raw_price)


def augment_with_bond_index(
    raw: dict[str, pd.DataFrame],
    start: str | None = None,
    token: str | None = None,
    data_id: str = DEFAULT_BOND_ETF,
) -> dict[str, pd.DataFrame]:
    """Take a raw-data bundle and add/replace its "bond_index" entry with
    a FinMind-sourced bond ETF close series, so the safe_haven indicator
    -- otherwise skipped for lack of a free source -- gets computed too.
    """
    augmented = dict(raw)
    augmented["bond_index"] = fetch_bond_index(start=start, token=token, data_id=data_id)
    return augmented


def _margin_balance_from_stock(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: one stock's raw TaiwanStockMarginPurchaseShortSale
    rows -> its daily margin balance. Columns: margin_balance (as
    FinMind reports MarginPurchaseTodayBalance -- board lots, same
    convention as the TWSE-derived per-stock figures elsewhere in this
    project; only relative day-to-day change is used downstream, so the
    exact unit doesn't matter as long as it's consistent).
    """
    if raw.empty:
        return pd.DataFrame(columns=["margin_balance"]).rename_axis("date")
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["margin_balance"] = pd.to_numeric(df["MarginPurchaseTodayBalance"], errors="coerce")
    return df.set_index("date")[["margin_balance"]].sort_index()


def fetch_margin_history_for_stock(
    stock_id: str,
    start: str,
    end: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """One stock's real daily margin balance over a date range -- a
    single request, since (unlike TaiwanOptionDaily) this dataset is
    small enough per stock not to need month-chunking. Columns:
    margin_balance. See `_aggregate_margin_history` for combining many
    stocks into a market-wide total, and the module docstring / project
    notes for why this is fetched one stock at a time: FinMind's
    market-wide query (no data_id) 400s on the free tier ("Your level is
    register") -- a paid Backer-or-above tier unlocks it in one request;
    the free-tier alternative is one request per stock, paced under the
    600 req/hr limit (~1,100 TWSE-listed common stocks means budgeting a
    couple of hours for a full backfill).
    """
    raw = _get_dataframe(DATASETS["margin"], data_id=stock_id, start_date=start, end_date=end, token=token)
    return _margin_balance_from_stock(raw)


def _aggregate_margin_history(per_stock: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sum many stocks' margin_balance series (as returned by
    `fetch_margin_history_for_stock`) into one market-wide daily total.
    A stock missing data for a given date contributes 0 for that date
    (not excluded), so a handful of failed/skipped stocks in a large
    backfill don't bias the total down disproportionately more on some
    days than others -- they're just a small, roughly constant
    undercount throughout. Columns: margin_balance.
    """
    frames = [df["margin_balance"] for df in per_stock.values() if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["margin_balance"]).rename_axis("date")
    combined = pd.concat(frames, axis=1)
    total = combined.sum(axis=1, skipna=True).sort_index()
    return total.to_frame("margin_balance")
