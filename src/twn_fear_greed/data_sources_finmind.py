"""FinMind (https://github.com/FinMind/FinMind) data connector.

Fills the one gap FinLab left open: TAIEX option (TXO) Put/Call volume --
FinLab has no options dataset at all (see data_sources_finlab.py). This
talks to FinMind's public v4 REST API directly with `requests` instead of
the `FinMind` pip package: that package pulls in the `ta` indicator
library as a dependency, which failed to build in this sandbox, and isn't
needed just to call two REST endpoints.

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
}


def _get_dataframe(dataset: str, data_id: str | None = None, start_date: str | None = None, token: str | None = None) -> pd.DataFrame:
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

    resp = requests.get(FINMIND_BASE, params=params, timeout=DEFAULT_TIMEOUT)
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
