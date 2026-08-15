"""Fetchers for TWSE / TAIFEX open data.

These hit real network endpoints and are exercised here only through their
pure transform helpers (`_*_from_*`), which take API-shaped fixtures and
are unit-tested in tests/test_data_sources.py; the network wrappers
themselves are not (see examples/run_example.py for an offline,
synthetic-data smoke test of the scoring pipeline instead).

All endpoints below are free, public, and require no API key:
  TWSE OpenAPI     https://openapi.twse.com.tw/   (openapi.twse.com.tw)
  TAIFEX OpenAPI   https://openapi.taifex.com.tw/
  TWSE classic     https://www.twse.com.tw/rwd/   (the JSON backend behind
                   www.twse.com.tw's own report pages, e.g. the page at
                   twse.com.tw/zh/indices/taiex/mi-5min-hist.html calls
                   /rwd/zh/TAIEX/MI_5MINS_HIST -- found by reading that
                   page's `data-api` attribute)

The openapi.* gateways are pure *snapshot* APIs: most endpoints return only
the current trading day, a couple return a short rolling window (10
trading days / ~1 month), and none accept a `date` query parameter to page
through history (confirmed by probing -- passing `date=` is silently
ignored). The classic www.twse.com.tw/rwd endpoints are different: several
of them DO honor a `date=YYYYMMDD` param and return the whole month
containing that date, which is what makes real multi-year history
achievable at all here (see `fetch_index_price_history`). Where no
endpoint of either kind supports history, building the composite scorer's
~252-day lookback still means calling `fetch_all_raw_data` once per
trading day and persisting the result yourself -- this module only
fetches, it does not accumulate.

Coverage against what indicators.py needs:
  index_price   -- real multi-year history via `fetch_index_price_history`
                   (classic MI_5MINS_HIST, one request per month); the
                   quick `fetch_index_price` (openapi, ~10 trading days,
                   one request) is kept for a fast last-few-days read.
  options       -- real, ~1 month (PutCallRatio; official TXO put/call, no
                   need to hand-aggregate raw option legs)
  margin        -- real, today only (MI_MARGN is per-stock; summed here
                   into one market-wide total, in board lots (張), not NT$).
                   The classic marginTrading/MI_MARGN endpoint that might
                   offer history returns empty for every date+selectType
                   combination tried -- left on the openapi snapshot.
  foreign       -- net_buy_sell (外資及陸資現貨買賣超, excluding the
                   proprietary foreign-dealer arm, NT$) has real multi-year
                   history via `fetch_foreign_cash_history` (classic
                   fund/BFI82U, one request per trading DAY -- there's no
                   month-batch for this report, so budget minutes for a
                   multi-year `start`). futures_net_oi (TAIEX futures,
                   foreign+China net open interest, contracts, from
                   TAIFEX's institutional-futures breakdown) has no
                   historical query at any layer found -- always today
                   only, filled with 0.0 rather than NaN on backfilled
                   rows so the real cash leg still drives the indicator.
  vix           -- real for the last ~4 months via `fetch_vix_history`
                   (TAIFEX's own CBOE-formula index, the average over the
                   final minute before close -- not in openapi.taifex.com.tw's
                   REST catalog, but published as one bulk file per month
                   off the download links on taifex.com.tw/cht/7/vixMinNew;
                   older months exist as single-day files at
                   /cht/7/getVixData, see `fetch_taifex_vix_close`, but
                   there's no bulk archive reaching further back than
                   ~4 months either way). `fetch_vix` overlays this real
                   window on top of `realized_volatility` (derived from
                   index_price, so it covers the full history for free)
                   and lets the real data win wherever both exist.
  new_high_low,
  breadth       -- both real, full history via
                   `fetch_market_breadth_and_strength_history` (classic
                   MI_INDEX with type=ALLNOTIND, the same full-market
                   bulk table behind www.twse.com.tw's "每日收盤行情" page --
                   one request per trading day returns every listed
                   instrument, filtered down to ordinary common stocks by
                   code shape). No free source gives this pre-aggregated,
                   so it's computed locally from the accumulated per-day
                   snapshots -- new_high_low needs a real trailing
                   252-day window before its first non-NaN value, same as
                   any other rolling indicator.
  bond_index    -- still needs an external bond/ETF NAV series.
"""

from __future__ import annotations

import re
import time

import pandas as pd
import requests

TWSE_BASE = "https://openapi.twse.com.tw/v1"
TAIFEX_BASE = "https://openapi.taifex.com.tw/v1"
TWSE_CLASSIC_BASE = "https://www.twse.com.tw/rwd/zh"

ENDPOINTS = {
    # TAIEX 近 10 個交易日開高低收 (openapi, 無 date 參數, 固定回傳最近窗口)
    "index_price_hist": f"{TWSE_BASE}/indicesReport/MI_5MINS_HIST",
    # TAIEX 開高低收 (舊版網站, 接受 date=YYYYMMDD, 回傳當月整月資料)
    "index_price_classic": f"{TWSE_CLASSIC_BASE}/TAIEX/MI_5MINS_HIST",
    # 集中市場融資融券餘額 (個股別, 當日快照; 加總得大盤融資餘額)
    "margin_balance": f"{TWSE_BASE}/exchangeReport/MI_MARGN",
    # 三大法人買賣金額統計表 (舊版網站, 當日快照, 已依身份別分行, NT$)
    "foreign_cash": f"{TWSE_CLASSIC_BASE}/fund/BFI82U",
    # 每日收盤行情 (舊版網站, 接受 date=YYYYMMDD, 單日全市場所有證券一次回傳)
    "market_index": f"{TWSE_CLASSIC_BASE}/afterTrading/MI_INDEX",
    # 臺指選擇權 Put/Call 比 (官方計算, 近 ~1 個月)
    "put_call_ratio": f"{TAIFEX_BASE}/PutCallRatio",
    # 三大法人-區分各期貨契約-依日期 (當日快照; 篩「臺股期貨」x「外資及陸資」)
    "institutional_futures": f"{TAIFEX_BASE}/MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate",
    # 臺指選擇權波動率指數逐日檔案 (舊版網站, 每個交易日一支檔案, 含收盤前1分鐘均值)
    "vix_daily_file": "https://www.taifex.com.tw/cht/7/getVixData",
    # 前3個月每日收盤之臺指選擇權波動率指數 (舊版網站, 每個月一支靜態檔, 同樣含收盤前1分鐘均值;
    # 只保留最近 ~4 個月, 更早的月份回 200 但內容是一頁 404 錯誤頁)
    "vix_monthly_file": "https://www.taifex.com.tw/file/taifex/Dailydownload/vix/log2data",
}

DEFAULT_TIMEOUT = 15
# Some TWSE endpoints (observed on the classic www.twse.com.tw/rwd family)
# behave differently or reject requests carrying the default python-requests
# User-Agent; a browser-like one avoids that without doing anything else.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _get_json(url: str, params: dict | None = None, retries: int = 3) -> list[dict] | dict:
    """A handful of these fetchers loop hundreds of requests in a row for
    a multi-year backfill (`fetch_index_price_history`,
    `fetch_foreign_cash_history`, `fetch_vix_history`); at that volume a
    transient blip -- a timeout, a connection reset, an occasional
    non-JSON response body -- is a when, not an if. Retried with a short
    linear backoff rather than propagating immediately and losing the
    whole loop over one bad request.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc


def _get_bytes(url: str, params: dict | None = None, retries: int = 3) -> bytes:
    """Same retry treatment as `_get_json`, for the file endpoints (the
    VIX daily/monthly files) that need raw bytes rather than parsed JSON.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc


def _num(series: pd.Series) -> pd.Series:
    """TWSE/TAIFEX numeric fields arrive as strings, occasionally with
    thousands separators or blank for N/A -- clean and coerce."""
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _parse_roc_date(series: pd.Series) -> pd.Series:
    """TWSE dates on this API are 7-digit ROC strings: YYY MM DD, e.g.
    "1150803" = ROC year 115 (= 1911 + 115 = 2026), month 08, day 03.
    TAIFEX's endpoints use ordinary 8-digit AD dates and don't need this.
    """
    s = series.astype(str).str.zfill(7)
    year = s.str.slice(0, 3).astype(int) + 1911
    month = s.str.slice(3, 5)
    day = s.str.slice(5, 7)
    return pd.to_datetime(year.astype(str) + month + day, format="%Y%m%d")


def _parse_roc_slash_date(series: pd.Series) -> pd.Series:
    """The classic www.twse.com.tw/rwd endpoints format ROC dates with
    slashes instead: "115/08/03" rather than openapi.twse.com.tw's plain
    "1150803". Same ROC-year-plus-1911 conversion either way.
    """
    parts = series.astype(str).str.split("/", expand=True)
    year = parts[0].astype(int) + 1911
    return pd.to_datetime(
        year.astype(str) + "-" + parts[1] + "-" + parts[2], format="%Y-%m-%d"
    )


# ---------------------------------------------------------------------------
# Pure transforms: API-shaped list[dict] -> tidy DataFrame. No I/O, so these
# are what tests/test_data_sources.py exercises directly.
# ---------------------------------------------------------------------------

def _index_price_from_hist(raw: list[dict]) -> pd.DataFrame:
    """MI_5MINS_HIST rows -> close, indexed by date. Columns: close."""
    if not raw:
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    df = pd.DataFrame(raw)
    df["date"] = _parse_roc_date(df["Date"])
    df["close"] = _num(df["ClosingIndex"])
    return df.set_index("date")[["close"]].sort_index()


def _put_call_from_ratio(raw: list[dict]) -> pd.DataFrame:
    """PutCallRatio rows -> put_volume, call_volume, indexed by date."""
    if not raw:
        return pd.DataFrame(columns=["put_volume", "call_volume"]).rename_axis("date")
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df["put_volume"] = _num(df["PutVolume"])
    df["call_volume"] = _num(df["CallVolume"])
    return df.set_index("date")[["put_volume", "call_volume"]].sort_index()


def _margin_balance_from_stockwise(raw: list[dict], as_of: pd.Timestamp) -> pd.DataFrame:
    """MI_MARGN rows (one per listed stock, no date field of its own) ->
    a single market-wide total for `as_of`, summing 融資今日餘額 (board
    lots) across every stock. Columns: margin_balance.
    """
    if not raw:
        return pd.DataFrame(columns=["margin_balance"]).rename_axis("date")
    df = pd.DataFrame(raw)
    total = _num(df["融資今日餘額"]).sum(skipna=True)
    return pd.DataFrame({"margin_balance": [total]}, index=pd.DatetimeIndex([as_of], name="date"))


def _foreign_futures_oi_from_institutional(
    raw: list[dict],
    as_of: pd.Timestamp,
    contract_code: str = "臺股期貨",
    item: str = "外資及陸資",
) -> pd.DataFrame:
    """MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate
    rows -> net_buy_sell (always 0.0 -- see module docstring, this free API
    has no spot-equity flow endpoint) and futures_net_oi (real, contracts).
    Unlike MI_MARGN, this endpoint's rows carry their own Date field, so
    that's used as the index; `as_of` is only a fallback for the
    no-match case, where there's no row to read a date from.
    """
    if not raw:
        return pd.DataFrame(columns=["net_buy_sell", "futures_net_oi"]).rename_axis("date")
    df = pd.DataFrame(raw)
    match = df[(df["ContractCode"] == contract_code) & (df["Item"] == item)]
    if match.empty:
        return pd.DataFrame(
            {"net_buy_sell": [0.0], "futures_net_oi": [float("nan")]},
            index=pd.DatetimeIndex([as_of], name="date"),
        )
    net_oi = _num(match["OpenInterest(Net)"]).sum(skipna=True)
    row_date = pd.to_datetime(match["Date"].iloc[0], format="%Y%m%d")
    return pd.DataFrame(
        {"net_buy_sell": [0.0], "futures_net_oi": [net_oi]},
        index=pd.DatetimeIndex([row_date], name="date"),
    )


def _index_price_from_classic_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Classic TAIEX/MI_5MINS_HIST rows -> close, indexed by date.

    Each row is [ROC-slash date, open, high, low, close], e.g.
    ["115/08/03", "42,780.42", "43,784.19", "42,780.42", "43,386.41"].
    A month with no trading data (e.g. a future month) comes back as an
    empty `data` list, which the caller passes straight through here.
    Columns: close.
    """
    if not rows:
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    df = pd.DataFrame(rows, columns=["date_str", "open", "high", "low", "close"])
    df["date"] = _parse_roc_slash_date(df["date_str"])
    df["close"] = _num(df["close"])
    return df.set_index("date")[["close"]].sort_index()


def _foreign_cash_from_bfi82u(
    payload: dict, label: str = "外資及陸資(不含外資自營商)"
) -> pd.DataFrame:
    """BFI82U's parsed JSON payload (the `{"date": ..., "fields": [...],
    "data": [[...], ...]}` shape, not just its `data` list) -> the net
    NT$ buy/sell amount for `label` on the payload's own date. Excludes
    the proprietary foreign-dealer arm by default, matching how "外資買超
    /賣超" is normally reported. Columns: net_buy_sell (NT$).
    """
    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=["net_buy_sell"]).rename_axis("date")
    fields = payload["fields"]
    df = pd.DataFrame(rows, columns=fields)
    match = df[df[fields[0]] == label]
    net = _num(match[fields[3]]).sum(skipna=True) if not match.empty else float("nan")
    as_of = pd.to_datetime(str(payload["date"]), format="%Y%m%d")
    return pd.DataFrame({"net_buy_sell": [net]}, index=pd.DatetimeIndex([as_of], name="date"))


COMMON_STOCK_CODE = re.compile(r"[1-9][0-9]{3}")  # 4 digits, not "00"-prefixed


def _market_snapshot_from_mi_index(payload: dict) -> pd.DataFrame:
    """MI_INDEX (classic www.twse.com.tw/rwd, type=ALLNOTIND) full-market
    daily table -> one row per ordinary common stock.

    A single day's response bundles ~31,000 rows across every instrument
    class TWSE lists (common stocks, ETFs incl. actively-managed ones,
    warrants, TDRs, ...); `COMMON_STOCK_CODE` keeps only the ~1,100 plain
    4-digit codes not prefixed "00" -- ETFs/warrants/TDRs use different
    code shapes and would otherwise swamp the real stocks ~30-to-1 in any
    aggregate. The table itself isn't at a fixed index across days (seen
    at index 8, but that's not guaranteed), so it's located by field
    names instead. The direction column arrives as an HTML fragment
    (`<p style= color:red>+</p>` for up, `color:green>-</p>` for down,
    plain `<p>X</p>`/`<p> </p>` for halted/unchanged) rather than a plain
    sign. Columns: close, volume, is_up, is_down (both False if flat or
    halted). Empty if the date had no trading or the table wasn't found.
    """
    empty = pd.DataFrame(columns=["close", "volume", "is_up", "is_down"]).rename_axis("code")
    target = next(
        (t for t in (payload.get("tables") or []) if "證券代號" in (t.get("fields") or []) and "收盤價" in (t.get("fields") or [])),
        None,
    )
    if target is None or not target.get("data"):
        return empty

    fields = target["fields"]
    idx = {name: i for i, name in enumerate(fields)}
    rows = []
    for r in target["data"]:
        code = r[idx["證券代號"]]
        if not COMMON_STOCK_CODE.fullmatch(code):
            continue
        direction = r[idx["漲跌(+/-)"]] or ""
        rows.append((code, r[idx["收盤價"]], r[idx["成交股數"]], "red" in direction, "green" in direction))
    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=["code", "close", "volume", "is_up", "is_down"]).set_index("code")
    df["close"] = _num(df["close"])
    df["volume"] = _num(df["volume"])
    return df


def _aggregate_breadth_and_strength(
    daily_snapshots: dict[pd.Timestamp, pd.DataFrame], new_high_low_window: int = 252
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a {date: per-stock snapshot} bundle (each snapshot shaped like
    `_market_snapshot_from_mi_index`'s output) into the two market-wide
    series indicators.py wants.

    breadth: advancing_volume / declining_volume, a same-day aggregate --
    just sums each stock's traded volume into the up or down bucket.

    new_high_low: new_highs / new_lows, which needs history -- a stock
    only counts as a new high on a day its close is >= its own trailing
    `new_high_low_window`-trading-day max (analogously for lows). The
    first `new_high_low_window - 1` days of whatever range was fetched
    can't have this computed yet and come back NaN, same as any other
    rolling-window indicator short on lookback -- this isn't a bug, it's
    the real 252-day requirement biting on however much history was
    actually pulled.
    """
    if not daily_snapshots:
        empty_b = pd.DataFrame(columns=["advancing_volume", "declining_volume"]).rename_axis("date")
        empty_h = pd.DataFrame(columns=["new_highs", "new_lows"]).rename_axis("date")
        return empty_b, empty_h

    dates = sorted(daily_snapshots)
    breadth_rows = []
    for d in dates:
        snap = daily_snapshots[d]
        adv = snap.loc[snap["is_up"], "volume"].sum()
        dec = snap.loc[snap["is_down"], "volume"].sum()
        breadth_rows.append((d, adv, dec))
    breadth = pd.DataFrame(breadth_rows, columns=["date", "advancing_volume", "declining_volume"]).set_index("date")

    close_matrix = pd.DataFrame({d: daily_snapshots[d]["close"] for d in dates}).T
    close_matrix.index.name = "date"
    rolling_max = close_matrix.rolling(window=new_high_low_window, min_periods=new_high_low_window).max()
    rolling_min = close_matrix.rolling(window=new_high_low_window, min_periods=new_high_low_window).min()
    is_new_high = (close_matrix >= rolling_max) & rolling_max.notna()
    is_new_low = (close_matrix <= rolling_min) & rolling_min.notna()
    has_lookback = rolling_max.notna().any(axis=1)

    new_highs = is_new_high.sum(axis=1).where(has_lookback, other=float("nan"))
    new_lows = is_new_low.sum(axis=1).where(has_lookback, other=float("nan"))
    new_high_low = pd.DataFrame({"new_highs": new_highs, "new_lows": new_lows})

    return breadth, new_high_low


def realized_volatility(index_price: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Annualized realized volatility of TAIEX daily returns.

    A fallback/backfill for `fetch_taifex_vix_close`, which is real but
    only reaches one trading day per request: `sqrt(252) *
    rolling_std(returns)`, scaled to roughly the same ~10-30 magnitude as
    the implied-vol index so it's comparable in the `volatility`
    indicator's percentile ranking. This is REALIZED (backward-looking)
    vol, not implied vol -- it won't spike ahead of an event the way the
    real options-derived index does. Columns: taiex_vix.
    """
    close = index_price["close"].astype(float)
    returns = close.pct_change()
    vol = returns.rolling(window, min_periods=max(2, window // 2)).std() * (252 ** 0.5) * 100.0
    return vol.to_frame("taiex_vix")


def _vix_close_from_taifex_file(raw_bytes: bytes, date: pd.Timestamp) -> pd.DataFrame:
    """TAIFEX's per-day VIX file (tab-separated, CP950-encoded, one row
    every 15 seconds through the trading day) ends with a summary row
    labeled "Last 1 min AVG" -- the average TAIEX-option VIX over the
    final minute before close, which TAIFEX has published since
    2020-11-23. That row is what this pulls out, as the day's reading.
    `date` is stamped by the caller (the file has no explicit ISO date
    field of its own to read back). Columns: taiex_vix. Empty if the file
    has no such row (e.g. `filesname` wasn't a real trading day).
    """
    text = raw_bytes.decode("cp950", errors="ignore")
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("\t") if c.strip()]
        if len(cols) >= 2 and cols[1].lower().startswith("last 1 min avg"):
            value = pd.to_numeric(cols[-1], errors="coerce")
            return pd.DataFrame({"taiex_vix": [value]}, index=pd.DatetimeIndex([date], name="date"))
    return pd.DataFrame(columns=["taiex_vix"]).rename_axis("date")


def _vix_from_monthly_file(raw_bytes: bytes) -> pd.DataFrame:
    """The vixDaily3MNew bulk file: same shape as the per-day file but one
    row per TRADING DAY instead of per 15 seconds, e.g.
    "20260814\\t13450000\\t\\t\\t30.22\\t\\t30.23" -- date, the 13:45:00
    snapshot, then the same "previous 1-minute average" figure as the
    per-day file's closing row (here the last column, confirmed against
    a day present in both). A month with no file yet (or too old --
    only ~4 months are kept) comes back as a 200-status Chinese 404 page
    rather than a real error, so that's what's actually being detected
    and rejected here, not just "empty". Columns: taiex_vix.
    """
    text = raw_bytes.decode("cp950", errors="ignore")
    rows = []
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("\t") if c.strip()]
        if len(cols) < 2 or not cols[0].isdigit() or len(cols[0]) != 8:
            continue  # skips the header row and the 404 page's HTML/prose
        rows.append((cols[0], cols[-1]))
    if not rows:
        return pd.DataFrame(columns=["taiex_vix"]).rename_axis("date")
    df = pd.DataFrame(rows, columns=["date_str", "value"])
    df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%d")
    df["taiex_vix"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")[["taiex_vix"]].sort_index()


# ---------------------------------------------------------------------------
# Network wrappers
# ---------------------------------------------------------------------------

def fetch_index_price() -> pd.DataFrame:
    """TAIEX daily close, last ~10 trading days, one request. Columns: close.

    Fast path for "just the last few days" -- for real multi-year history
    use `fetch_index_price_history` instead.
    """
    return _index_price_from_hist(_get_json(ENDPOINTS["index_price_hist"]))


def fetch_index_price_history(start: str, end: str | None = None) -> pd.DataFrame:
    """TAIEX daily close from `start` through `end` (default: today), real
    history -- one request per calendar month via the classic
    www.twse.com.tw/rwd endpoint, which (unlike openapi.twse.com.tw) honors
    a `date=` param and returns the whole month containing it. Columns:
    close. ~36 requests for 3 years; be considerate of TWSE's server and
    don't loop this in a tight retry.
    """
    start_ts = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()
    months = pd.date_range(start_ts, end_ts, freq="MS")
    if months.empty or months[0] != start_ts:
        months = months.insert(0, start_ts)

    frames = []
    for month_start in months:
        payload = _get_json(
            ENDPOINTS["index_price_classic"],
            params={"date": month_start.strftime("%Y%m%d"), "response": "json"},
        )
        rows = payload.get("data") if isinstance(payload, dict) else None
        frames.append(_index_price_from_classic_rows(rows or []))

    if not frames:
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result.loc[start:end] if end else result.loc[start:]


def fetch_market_breadth_and_strength_history(
    start: str,
    end: str | None = None,
    new_high_low_window: int = 252,
    chunk_size: int = 40,
    cooldown_seconds: float = 45.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real market breadth AND 52-week new-high/new-low counts, backed by
    one MI_INDEX (classic www.twse.com.tw/rwd, type=ALLNOTIND) request per
    trading day -- the full-market bulk table behind www.twse.com.tw's own
    "每日收盤行情" page, found the same way as the other classic endpoints
    (reading that page's data-api). Bundles both indicators because
    they're aggregated from the same expensive per-day pull; call this
    instead of `fetch_breadth`/`fetch_new_high_low` separately if you
    need both, to avoid fetching ~750 days of full-market data twice.

    Same rate-limit profile assumed as `fetch_foreign_cash_history` (same
    host, same classic /rwd family, not independently reconfirmed here):
    chunked with a cooldown between chunks, one cooldown-and-retry for a
    single failing day before it's dropped rather than aborting the whole
    range. A dropped day is simply absent from the accumulated series --
    the rolling `new_high_low_window` count is over however many days
    actually came back, not strictly that many calendar trading days.

    Returns (breadth, new_high_low); see `_aggregate_breadth_and_strength`
    for exactly what each contains and why the first stretch of
    new_high_low is NaN (not enough trailing history yet).
    """
    dates = pd.bdate_range(start, end or pd.Timestamp.now())
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    for i, d in enumerate(dates):
        if i > 0 and i % chunk_size == 0:
            time.sleep(cooldown_seconds)
        params = {"date": d.strftime("%Y%m%d"), "type": "ALLNOTIND", "response": "json"}
        try:
            payload = _get_json(ENDPOINTS["market_index"], params=params)
        except (requests.RequestException, ValueError):
            time.sleep(cooldown_seconds)
            try:
                payload = _get_json(ENDPOINTS["market_index"], params=params)
            except (requests.RequestException, ValueError):
                continue
        snap = _market_snapshot_from_mi_index(payload)
        if not snap.empty:
            snapshots[d] = snap
        time.sleep(0.2)
    return _aggregate_breadth_and_strength(snapshots, new_high_low_window=new_high_low_window)


def fetch_new_high_low(start: str | None = None) -> pd.DataFrame:
    """52-week new-high/new-low counts, market-wide. Columns: new_highs,
    new_lows.

    Needs `start`: this fundamentally requires trailing per-stock price
    history, which no free snapshot endpoint gives without actually
    fetching that history day by day -- see
    `fetch_market_breadth_and_strength_history` (call it directly, not
    this function, if you also need `fetch_breadth`: both come off the
    same pull and calling each separately fetches it twice).
    """
    if not start:
        raise NotImplementedError(
            "Needs a start= date to backfill trailing price history -- "
            "call fetch_market_breadth_and_strength_history(start) directly "
            "if you also need fetch_breadth, to avoid fetching twice."
        )
    _, new_high_low = fetch_market_breadth_and_strength_history(start)
    return new_high_low


def fetch_breadth(start: str | None = None) -> pd.DataFrame:
    """Advancing/declining volume, market-wide. Columns: advancing_volume,
    declining_volume. Needs `start` -- see `fetch_new_high_low`.
    """
    if not start:
        raise NotImplementedError(
            "Needs a start= date -- see fetch_new_high_low."
        )
    breadth, _ = fetch_market_breadth_and_strength_history(start)
    return breadth


def fetch_options() -> pd.DataFrame:
    """TXO put/call volume via TAIFEX's own official ratio endpoint, last
    ~1 month. Columns: put_volume, call_volume (indexed by date).
    """
    return _put_call_from_ratio(_get_json(ENDPOINTS["put_call_ratio"]))


def fetch_taifex_vix_close(date: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """One real, official TAIEX-option VIX reading -- see
    `_vix_close_from_taifex_file` for what it actually is (the last-minute
    average, not a full-day close). Defaults to the most recent probable
    trading day if `date` isn't given. Empty if that day turns out not to
    have a file (e.g. a market holiday `_last_probable_trading_day` can't
    detect, since it only knows about weekends).
    """
    as_of = _last_probable_trading_day(pd.Timestamp(date) if date else None)
    content = _get_bytes(ENDPOINTS["vix_daily_file"], params={"filesname": as_of.strftime("%Y%m%d")})
    return _vix_close_from_taifex_file(content, as_of)


def fetch_vix_history(start: str, end: str | None = None) -> pd.DataFrame:
    """Real TAIFEX VIX (the same last-1-min-avg reading as
    `fetch_taifex_vix_close`), one request per calendar month via the
    bulk file behind vixDaily3MNew -- but unlike `fetch_index_price_history`,
    this archive only actually holds the ~4 most recent months; any
    earlier month's request 200s with a Chinese "content not found" page
    instead of erroring, which `_vix_from_monthly_file` detects and
    quietly turns into an empty month rather than bad data. There is no
    free source for VIX further back than that -- combine with
    `realized_volatility` for the rest of the history, which `fetch_vix`
    does automatically.
    """
    start_ts = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()
    months = pd.date_range(start_ts, end_ts, freq="MS")
    if months.empty or months[0] != start_ts:
        months = months.insert(0, start_ts)

    frames = []
    for month_start in months:
        url = f"{ENDPOINTS['vix_monthly_file']}/{month_start.strftime('%Y%m')}new.txt"
        frames.append(_vix_from_monthly_file(_get_bytes(url)))

    if not frames:
        return pd.DataFrame(columns=["taiex_vix"]).rename_axis("date")
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result.loc[start:end] if end else result.loc[start:]


def fetch_vix(index_price: pd.DataFrame | None = None, real_lookback_months: int = 4) -> pd.DataFrame:
    """Realized volatility of TAIEX returns (see `realized_volatility`) as
    the base series, overlaid with TAIFEX's real closing-minute VIX
    (`fetch_vix_history`) for the most recent `real_lookback_months`
    months where that archive actually has data -- typically 1-4 requests,
    since months without a real file just come back empty rather than
    erroring. Real values take priority over the realized-vol estimate
    wherever both exist. Fetches index_price itself if not supplied.
    """
    if index_price is None:
        index_price = fetch_index_price()
    vix = realized_volatility(index_price)
    if real_lookback_months > 0:
        start = (pd.Timestamp.now() - pd.DateOffset(months=real_lookback_months)).strftime("%Y-%m-%d")
        try:
            real = fetch_vix_history(start)
        except requests.RequestException:
            real = pd.DataFrame(columns=["taiex_vix"])
        if not real.empty:
            vix = real.combine_first(vix)
    return vix.sort_index()


_WEEKDAY_ONLY = pd.tseries.offsets.CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri")


def _last_probable_trading_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Roll a calendar date back to the most recent Mon-Fri (itself,
    if it already is one). Doesn't know about TWSE public holidays
    (there's no free calendar endpoint for that either) -- only fixes
    the weekend case, which is the common one for a script run outside
    trading hours.
    """
    now = (now or pd.Timestamp.now()).normalize()
    return _WEEKDAY_ONLY.rollback(now)


def fetch_margin() -> pd.DataFrame:
    """融資餘額, summed market-wide across all listed stocks, most recent
    weekday only (see `_last_probable_trading_day` -- a market holiday
    that falls on a weekday isn't detected, so its row would repeat the
    prior real session under a wrong date label). Columns: margin_balance
    (board lots, not NT$).
    """
    as_of = _last_probable_trading_day()
    return _margin_balance_from_stockwise(_get_json(ENDPOINTS["margin_balance"]), as_of)


def fetch_foreign_cash_history(
    start: str,
    end: str | None = None,
    chunk_size: int = 40,
    cooldown_seconds: float = 45.0,
) -> pd.DataFrame:
    """Real 外資及陸資 net buy/sell (NT$), one BFI82U request per calendar
    day -- unlike the TAIEX/VIX series, there is no month-batch file for
    this report at any layer found, so a multi-year `start` means one
    request per weekday and takes minutes, not seconds (roughly 250
    requests/year).

    BFI82U's server rate-limits at roughly 50 requests within a short
    rolling window regardless of the pacing between individual requests
    -- confirmed empirically (the same block at request #50 whether paced
    at ~0.15s or ~0.5s apart): a WAF page served with a 307 status that
    only fails once the body is parsed as JSON, and which a 30s pause was
    enough to clear in testing. Requests are issued in chunks of
    `chunk_size` (comfortably under 50) separated by a `cooldown_seconds`
    pause (kept with margin over the observed 30s). If a single request
    still fails despite that pacing, one extra cooldown-and-retry is
    attempted before giving up on that one day and moving on -- a
    multi-year fetch shouldn't abort over one bad day the way the first
    version of this function did.

    Non-trading weekdays come back empty from `_foreign_cash_from_bfi82u`
    and are silently skipped, same as always. Columns: net_buy_sell.
    """
    dates = pd.bdate_range(start, end or pd.Timestamp.now())
    frames = []
    for i, d in enumerate(dates):
        if i > 0 and i % chunk_size == 0:
            time.sleep(cooldown_seconds)
        params = {"dayDate": d.strftime("%Y%m%d"), "type": "day", "response": "json"}
        try:
            payload = _get_json(ENDPOINTS["foreign_cash"], params=params)
        except (requests.RequestException, ValueError):
            time.sleep(cooldown_seconds)
            try:
                payload = _get_json(ENDPOINTS["foreign_cash"], params=params)
            except (requests.RequestException, ValueError):
                continue
        frame = _foreign_cash_from_bfi82u(payload)
        if not frame.empty:
            frames.append(frame)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=["net_buy_sell"]).rename_axis("date")
    result = pd.concat(frames).sort_index()
    return result[~result.index.duplicated(keep="last")]


def fetch_foreign(start: str | None = None) -> pd.DataFrame:
    """外資現貨買賣超金額 (NT$, via BFI82U) + 台指期未平倉多空淨口數 (contracts,
    via TAIFEX). Columns: net_buy_sell, futures_net_oi.

    futures_net_oi has no historical query at any layer found -- TAIFEX's
    institutional-futures endpoint ignores date params entirely and
    always reflects the latest session, unlike BFI82U -- so it's only
    ever real for the most recent trading day, `start` or not.

    Without `start`: most recent session for both legs. BFI82U returns
    an explicit "no data" response for a non-trading day rather than the
    prior session's figures, so a weekend run walks back day by day
    until it finds one (capped at 7 calendar days). The two legs' dates
    aren't forced to match -- a lag on either side surfaces as NaN on
    that leg for that row.

    With `start`: net_buy_sell is backfilled for the whole range via
    `fetch_foreign_cash_history` (budget several minutes). Since
    futures_net_oi can't be backfilled the same way, older rows get 0.0
    there rather than NaN, so the real cash leg still drives the
    indicator on those days instead of the whole row going NaN -- same
    convention as the always-0.0 net_buy_sell fallback this replaces
    for the no-`start` case.
    """
    futures_oi = _foreign_futures_oi_from_institutional(
        _get_json(ENDPOINTS["institutional_futures"]), _last_probable_trading_day()
    )

    if start:
        cash = fetch_foreign_cash_history(start)
        if cash.empty:
            return pd.DataFrame(columns=["net_buy_sell", "futures_net_oi"]).rename_axis("date")
        result = cash.copy()
        result["futures_net_oi"] = 0.0
        if not futures_oi.empty:
            latest = futures_oi.index[0]
            latest_oi = futures_oi["futures_net_oi"].iloc[0]
            if latest in result.index:
                result.loc[latest, "futures_net_oi"] = latest_oi
            else:
                result.loc[latest, ["net_buy_sell", "futures_net_oi"]] = [float("nan"), latest_oi]
        return result.sort_index()

    cash = pd.DataFrame(columns=["net_buy_sell"]).rename_axis("date")
    probe = _last_probable_trading_day()
    for _ in range(7):
        payload = _get_json(
            ENDPOINTS["foreign_cash"],
            params={"dayDate": probe.strftime("%Y%m%d"), "type": "day", "response": "json"},
        )
        cash = _foreign_cash_from_bfi82u(payload)
        if not cash.empty:
            break
        probe = _last_probable_trading_day(probe - pd.Timedelta(days=1))

    return cash.join(futures_oi["futures_net_oi"], how="outer")


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
    """Fetch everything indicators.compute_all_raw_indicators can use from
    this free API. bond_index still has no free implementation and is
    silently omitted rather than raising -- compute_all_raw_indicators
    already tolerates missing keys and renormalizes weights over
    whatever's present, so add it to `raw` yourself once you have a
    source (e.g. a bond ETF NAV series).

    `start`, if given:
      - switches index_price (and so vix, which is derived from it) from
        the ~10-day snapshot to the real multi-month history via
        `fetch_index_price_history` -- one extra request per month.
      - switches foreign's net_buy_sell leg to a real day-by-day backfill
        via `fetch_foreign_cash_history` -- one request per trading day,
        several minutes for a multi-year start (see its docstring for the
        rate-limit pacing this involves).
      - fetches new_high_low and breadth for the first time at all, via
        one shared `fetch_market_breadth_and_strength_history` call
        (again one request per trading day, several minutes).
    Without `start`, new_high_low and breadth are omitted (they have no
    meaningful "just today" reading -- new_high_low needs trailing
    history by definition), same as bond_index. margin and options'
    windows (today / ~1 month) don't change with or without `start`;
    see the module docstring for why.
    """
    index_price = fetch_index_price_history(start) if start else fetch_index_price()
    raw: dict[str, pd.DataFrame] = {
        "index_price": index_price,
        "options": fetch_options(),
        "vix": fetch_vix(index_price),
        "margin": fetch_margin(),
        "foreign": fetch_foreign(start=start) if start else fetch_foreign(),
    }
    if start:
        breadth, new_high_low = fetch_market_breadth_and_strength_history(start)
        if not breadth.empty:
            raw["breadth"] = breadth
        if not new_high_low.empty:
            raw["new_high_low"] = new_high_low
    try:
        raw["bond_index"] = fetch_bond_index()
    except NotImplementedError:
        pass
    if start:
        raw = {k: v.loc[start:] for k, v in raw.items()}
    return raw
