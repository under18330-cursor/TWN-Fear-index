"""FinLab (https://finlab.finance/) data connector.

FinLab gives full historical per-stock matrices (not just a single day's
snapshot like the raw TWSE OpenAPI), which is what actually makes the
52-week new-high/new-low and market-breadth indicators computable -- with
the plain TWSE OpenAPI those required a locally maintained per-symbol
history (see data_sources.py's NotImplementedError notes).

FinLab does NOT publish an options or implied-volatility dataset, so:
  - `put_call` has no FinLab source; omit "options" from the raw bundle
    and it is skipped automatically (see indicators.compute_all_raw_indicators).
  - `volatility` falls back to a realized-volatility proxy computed from
    TAIEX's own daily returns instead of the (implied-vol) TAIEX VIX.
FinLab also has no government-bond index, so the safe-haven-demand
indicator uses a listed bond ETF's price series (e.g. 00679B) as the proxy.

Authentication
---------------
Never hardcode the API token in source. Set it as an environment variable:

    export FINLAB_API_KEY="..."

and call `login()` once per process before any `fetch_*` function. If you
were ever given a token in a chat message or ticket, treat it as
compromised and rotate it from the FinLab dashboard -- anything pasted
into a conversation should not be trusted as a long-lived secret.
"""

from __future__ import annotations

import os

import pandas as pd

try:
    import finlab
    from finlab import data as finlab_data
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'finlab' package is required for this module. Install it with "
        "`pip install finlab`."
    ) from exc

DEFAULT_NEW_HIGH_LOW_WINDOW = 252  # ~52 trading weeks
DEFAULT_BOND_ETF_ID = "00679B"  # 元大美债20年 -- swap for whatever proxy you prefer

_logged_in = False


def login(api_token: str | None = None) -> None:
    """Authenticate with FinLab. Reads FINLAB_API_KEY from the environment
    if `api_token` isn't passed explicitly. Safe to call more than once.
    """
    global _logged_in
    token = api_token or os.environ.get("FINLAB_API_KEY")
    if not token:
        raise RuntimeError(
            "No FinLab API token found. Set the FINLAB_API_KEY environment "
            "variable or pass api_token explicitly."
        )
    finlab.login(token)
    _logged_in = True


def _ensure_login() -> None:
    if not _logged_in:
        login()


def _slice_start(df: pd.DataFrame, start: str | None) -> pd.DataFrame:
    return df.loc[start:] if start else df


def fetch_index_price(start: str | None = None) -> pd.DataFrame:
    """TAIEX daily close. Columns: close (indexed by date)."""
    _ensure_login()
    raw = finlab_data.get("taiex_total_index:收盤指數")
    close = raw.iloc[:, 0].rename("close").to_frame()
    return _slice_start(close, start)


def fetch_new_high_low(
    start: str | None = None,
    window: int = DEFAULT_NEW_HIGH_LOW_WINDOW,
) -> pd.DataFrame:
    """Cross-sectional count of stocks making a `window`-day new high/low.

    Built from the full per-stock close-price matrix, since FinLab (unlike
    the raw TWSE OpenAPI) keeps history rather than only today's snapshot.
    """
    _ensure_login()
    close = finlab_data.get("price:收盤價")
    rolling_max = close.rolling(window, min_periods=window // 2).max()
    rolling_min = close.rolling(window, min_periods=window // 2).min()
    new_highs = (close >= rolling_max).sum(axis=1).rename("new_highs")
    new_lows = (close <= rolling_min).sum(axis=1).rename("new_lows")
    result = pd.concat([new_highs, new_lows], axis=1)
    return _slice_start(result, start)


def fetch_breadth(start: str | None = None) -> pd.DataFrame:
    """Market-wide advancing vs. declining share volume."""
    _ensure_login()
    close = finlab_data.get("price:收盤價")
    volume = finlab_data.get("price:成交股數")
    change = close.diff()
    advancing_volume = volume.where(change > 0, 0.0).sum(axis=1).rename("advancing_volume")
    declining_volume = volume.where(change < 0, 0.0).sum(axis=1).rename("declining_volume")
    result = pd.concat([advancing_volume, declining_volume], axis=1)
    return _slice_start(result, start)


def fetch_volatility_proxy(start: str | None = None, window: int = 20) -> pd.DataFrame:
    """Annualized realized volatility of TAIEX returns, used as a stand-in
    for an implied-vol index since FinLab has no options/VIX dataset.
    Column name is kept as `taiex_vix` so it plugs into
    `indicators.volatility()` unchanged.
    """
    _ensure_login()
    close = finlab_data.get("taiex_total_index:收盤指數").iloc[:, 0]
    realized_vol = close.pct_change().rolling(window, min_periods=window // 2).std() * (252**0.5) * 100
    result = realized_vol.rename("taiex_vix").to_frame()
    return _slice_start(result, start)


def fetch_margin(start: str | None = None) -> pd.DataFrame:
    """Market-wide 融資券總餘額 (all-stock margin balance total)."""
    _ensure_login()
    raw = finlab_data.get("margin_balance:融資券總餘額")
    balance = raw.iloc[:, 0].rename("margin_balance").to_frame()
    return _slice_start(balance, start)


def fetch_foreign(start: str | None = None) -> pd.DataFrame:
    """Market-wide foreign cash net-buy/sell (summed across all stocks) plus
    net futures positioning (summed across all TAIFEX index-futures products).
    """
    _ensure_login()
    cash = finlab_data.get("institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)")
    net_buy_sell = cash.sum(axis=1).rename("net_buy_sell")

    futures = finlab_data.get("futures_institutional_investors_trading_summary:多空未平倉口數淨額")
    futures_net_oi = futures.sum(axis=1).rename("futures_net_oi")

    result = pd.concat([net_buy_sell, futures_net_oi], axis=1)
    return _slice_start(result, start)


def fetch_bond_index(start: str | None = None, etf_id: str = DEFAULT_BOND_ETF_ID) -> pd.DataFrame:
    """Bond ETF close price, used as the safe-haven-demand proxy since
    FinLab has no standalone government-bond index.
    """
    _ensure_login()
    close = finlab_data.get("price:收盤價")
    if etf_id not in close.columns:
        raise KeyError(f"Bond ETF '{etf_id}' not found in price:收盤價; pick a listed bond ETF id.")
    bond_close = close[etf_id].rename("close").to_frame()
    return _slice_start(bond_close, start)


def fetch_all_raw_data(start: str | None = None, include_bond: bool = True) -> dict[str, pd.DataFrame]:
    """Assemble the raw-data bundle expected by
    indicators.compute_all_raw_indicators. "options" is intentionally
    omitted (no FinLab source) so put_call is skipped automatically.
    """
    raw = {
        "index_price": fetch_index_price(start),
        "new_high_low": fetch_new_high_low(start),
        "breadth": fetch_breadth(start),
        "vix": fetch_volatility_proxy(start),
        "margin": fetch_margin(start),
        "foreign": fetch_foreign(start),
    }
    if include_bond:
        try:
            raw["bond_index"] = fetch_bond_index(start)
        except KeyError:
            pass
    return raw
