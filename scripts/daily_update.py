"""Daily incremental update for the TWN Fear & Greed ledger.

Meant to run once a day (scheduled ~17:00 local time, after TWSE's
same-day settlement data is typically published). Idempotent: if the
target trading day's row already exists in data/ledger_history.json,
this exits immediately without hitting any network endpoint.

All state lives under data/ (gitignored, local-only, not committed):

  ledger_history.json      one scored row per trading day -- exactly
                            what the published artifact/table shows.
  raw_index_price.csv, raw_options.csv, raw_vix.csv, raw_margin.csv,
  raw_foreign.csv, raw_bond_index.csv
                            persisted raw indicator inputs (trailing
                            window only), so the composite score's
                            rolling percentile rank can be recomputed
                            each day without re-fetching years of
                            history.
  mi_index_snapshots.pkl   per-day per-stock closing-price snapshots,
                            needed to roll the 252-trading-day
                            new-high/low and breadth window forward one
                            day at a time without re-pulling the whole
                            window every run.

FinMind token: read from the FINMIND_API_TOKEN env var, falling back to
data/.finmind_token (a local, gitignored plaintext file) if unset. Used
first for options (put/call) and bond_index. Both legs have keyless
fallbacks that kick in when the token is missing or rejected, or when
FinMind simply doesn't have the target day yet: options -> TAIFEX's own
pcRatioDown CSV download (then the openapi ratio snapshot as a last
resort), bond_index -> TPEx's per-security monthly table. Neither leg
goes NaN just because the token died -- that happened silently for seven
weeks in Sep-Nov 2026 before the fallbacks existed.

Gaps: this script only ever scores "the latest weekday". A day it never
ran for (or ran too early for) stays missing -- use
scripts/backfill_ledger.py to fill or recompute a date range.

Endpoint choices matter for same-day freshness: openapi.twse.com.tw's
snapshot feeds (index_price_hist, TAIFEX's put/call ratio) were observed
still one session behind at ~18:00 local, while the classic
www.twse.com.tw/rwd monthly TAIEX file and FinMind's options dataset
already carried the day -- so those are what the append path uses.

Margin comes from the classic, dated MI_MARGN endpoint (date=target),
not the openapi snapshot: that snapshot carries no date and was found to
still show the PRIOR session's total late into the evening (every
daily-run day Sep-Nov 2026 had repeated the previous day's balance), so
using it meant silently scoring yesterday's margin under today's date.
If the dated endpoint hasn't published the target day yet, margin is
left out of that day's row (weights renormalize) and the gap is
reported, rather than filled with a stale number.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from twn_fear_greed import data_sources as ds
from twn_fear_greed import data_sources_finmind as fm
from twn_fear_greed.index import TWNFearGreedIndex

DATA_DIR = REPO_ROOT / "data"
TOKEN_FILE = DATA_DIR / ".finmind_token"
# Calendar days of raw history kept. The longest chain is momentum: a
# 125-day MA feeding a 252-day percentile rank = ~377 trading days
# before the rank is over a full sample; 600 calendar days covers that
# with margin without letting the per-stock snapshot pickle grow forever.
TRAILING_DAYS = 600
LOOKBACK = 252

SCORE_COLS = {
    "momentum": "momentum_score",
    "strength": "strength_score",
    "breadth": "breadth_score",
    "put_call": "put_call_score",
    "volatility": "volatility_score",
    "safe_haven": "safe_haven_score",
    "margin": "margin_score",
    "foreign": "foreign_score",
}


def _describe(exc: Exception, token: str | None) -> str:
    """One-line error text safe to print: requests' HTTPError repr embeds
    the full URL, token query-string and all, so blank it out first.
    """
    text = f"{type(exc).__name__}: {exc}"
    return text.replace(token, "***") if token else text


def _rnd(v, nd=1):
    return None if pd.isna(v) else round(float(v), nd)


def ledger_entry(result: pd.DataFrame, day: pd.Timestamp) -> dict:
    """One ledger_history.json row for `day` out of TWNFearGreedIndex's
    result frame (joined with taiex_close). Shared with backfill_ledger.py
    so both scripts write byte-identical rows for the same inputs.
    """
    row = result.loc[day]
    entry = {"date": day.strftime("%Y-%m-%d"), "taiex": _rnd(row.get("taiex_close"), 0)}
    for short, col in SCORE_COLS.items():
        entry[short] = _rnd(row[col]) if col in result.columns else None
    entry["composite"] = _rnd(row["composite"])
    entry["label"] = row["label"]
    return entry


def fetch_snapshot(day: pd.Timestamp) -> pd.DataFrame:
    """Per-stock MI_INDEX closing snapshot for one trading day (empty on a
    holiday or before TWSE has published it)."""
    payload = ds._get_json(
        ds.ENDPOINTS["market_index"],
        params={"date": day.strftime("%Y%m%d"), "type": "ALLNOTIND", "response": "json"},
    )
    return ds._market_snapshot_from_mi_index(payload)


def _finmind_token() -> str | None:
    token = os.environ.get("FINMIND_API_TOKEN")
    if token:
        return token
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"raw_{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def _save_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(DATA_DIR / f"raw_{name}.csv")


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New non-NaN values win; NaNs in `new` never erase a prior value.
    fetch_foreign() in particular can return a partial row for the prior
    session (futures leg only) that would otherwise clobber its full row.
    """
    if new.empty:
        return old
    if old.empty:
        return new.sort_index()
    return new.combine_first(old).sort_index()


def _trim(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df.index >= cutoff]


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    target = ds._last_probable_trading_day()

    ledger_path = DATA_DIR / "ledger_history.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []
    if ledger and ledger[-1]["date"] == target.strftime("%Y-%m-%d"):
        print(f"{target.date()} already in ledger_history.json -- nothing to do.")
        return 0

    token = _finmind_token()
    if not token:
        print(
            "NOTE: no FinMind token found (env FINMIND_API_TOKEN or data/.finmind_token) "
            "-- options and bond_index will use the keyless TAIFEX / TPEx sources."
        )

    print(f"target trading day: {target.date()}")
    cutoff = target - pd.Timedelta(days=TRAILING_DAYS)

    recent_start = (target - pd.Timedelta(days=35)).strftime("%Y-%m-%d")

    print("index_price (classic monthly TAIEX file) + vix...")
    index_price = _merge(_load_csv("index_price"), ds.fetch_index_price_history(recent_start))
    vix = ds.fetch_vix(index_price, real_lookback_months=4)

    options = _load_csv("options")
    fresh = pd.DataFrame()
    if token:
        print("options (FinMind put/call, recent window)...")
        try:
            fresh = fm.fetch_put_call_ratio_history(recent_start, token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  FinMind options failed ({_describe(exc, token)}).")
    if fresh.empty or target not in fresh.index:
        print("options (TAIFEX pcRatioDown download, recent window)...")
        try:
            fresh = _merge(fresh, ds.fetch_put_call_ratio_history_taifex(recent_start))
        except Exception as exc:  # noqa: BLE001
            print(f"  TAIFEX download failed ({_describe(exc, token)}); falling back to the openapi ratio snapshot.")
            fresh = _merge(fresh, ds.fetch_options())
    options = _merge(options, fresh)

    print("margin (classic MI_MARGN, dated, market-wide)...")
    margin_today = ds.fetch_margin_history(dates=[target])
    if margin_today.empty:
        print(f"  margin for {target.date()} not published yet -- scoring today without it.")
    margin = _merge(_load_csv("margin"), margin_today)

    print("foreign (BFI82U + TAIFEX futures OI, today)...")
    foreign = _merge(_load_csv("foreign"), ds.fetch_foreign())
    # TAIFEX's OI feed often isn't out yet at 17:00; same 0.0 convention
    # as fetch_foreign(start=...) so the real cash leg still scores the day.
    if "futures_net_oi" in foreign.columns:
        foreign["futures_net_oi"] = foreign["futures_net_oi"].fillna(0.0)

    # Fetched over the recent window, not just the target day, so a stretch
    # of failed runs heals itself the next time either source answers.
    bond_index = _load_csv("bond_index")
    fresh = pd.DataFrame()
    if token:
        print("bond_index (FinMind 00679B close, recent window)...")
        try:
            fresh = fm.fetch_bond_index(start=recent_start, token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  FinMind bond_index failed ({_describe(exc, token)}).")
    if fresh.empty or target not in fresh.index:
        print("bond_index (TPEx 00679B monthly close)...")
        try:
            fresh = _merge(fresh, ds.fetch_bond_index_tpex(recent_start))
        except Exception as exc:  # noqa: BLE001 - keep going without this leg
            print(f"  TPEx bond_index failed ({_describe(exc, token)}); keeping prior history only.")
    bond_index = _merge(bond_index, fresh)

    print("breadth / new_high_low (MI_INDEX per-stock snapshot, today)...")
    snapshots_path = DATA_DIR / "mi_index_snapshots.pkl"
    snapshots = pickle.loads(snapshots_path.read_bytes()) if snapshots_path.exists() else {}
    if target not in snapshots:
        snap = fetch_snapshot(target)
        if not snap.empty:
            snapshots[target] = snap
        else:
            print(f"  no MI_INDEX snapshot for {target.date()} yet (holiday or not published).")
    snapshots = {d: s for d, s in snapshots.items() if d >= cutoff}
    breadth, new_high_low = ds._aggregate_breadth_and_strength(snapshots, new_high_low_window=LOOKBACK)

    index_price = _trim(index_price, cutoff)
    vix = _trim(vix, cutoff)
    options = _trim(options, cutoff)
    margin = _trim(margin, cutoff)
    foreign = _trim(foreign, cutoff)
    bond_index = _trim(bond_index, cutoff)

    raw = {
        "index_price": index_price,
        "options": options,
        "vix": vix,
        "margin": margin,
        "foreign": foreign,
        "breadth": breadth,
        "new_high_low": new_high_low,
        "bond_index": bond_index,
    }
    raw = {k: v for k, v in raw.items() if not v.empty}

    result = TWNFearGreedIndex(lookback=LOOKBACK).compute(raw)
    result = result.join(index_price.rename(columns={"close": "taiex_close"}))

    if target not in result.index:
        print(f"no computed row for {target.date()} -- check upstream data availability. Nothing saved.")
        return 1

    row = result.loc[target]

    # No TAIEX close means the market was shut (holiday) or the close isn't
    # published yet. Either way skip the day entirely: snapshot feeds like
    # margin carry no date, so they'd otherwise score a stale number here.
    if pd.isna(row.get("taiex_close")):
        print(f"no TAIEX close for {target.date()} -- treating as market holiday. Nothing saved.")
        return 1

    entry = ledger_entry(result, target)
    ledger.append(entry)

    _save_csv("index_price", index_price)
    _save_csv("vix", vix)
    _save_csv("options", options)
    _save_csv("margin", margin)
    _save_csv("foreign", foreign)
    _save_csv("bond_index", bond_index)
    snapshots_path.write_bytes(pickle.dumps(snapshots))
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")

    print(f"\nappended {entry['date']}: composite={entry['composite']} label={entry['label']}")
    print(f"ledger_history.json now has {len(ledger)} rows ({ledger[0]['date']} -> {ledger[-1]['date']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
