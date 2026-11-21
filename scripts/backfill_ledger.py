"""Backfill or recompute ledger rows for a date range.

daily_update.py only ever scores "the latest weekday", so any day it
didn't run for -- or ran before the exchange had published -- is simply
absent from data/ledger_history.json forever, and a day whose inputs
were incomplete at the time (e.g. a dead FinMind token leaving
bond_index sparse) keeps the score it got. This script repairs both:

    .venv\\Scripts\\python.exe scripts\\backfill_ledger.py --start 2026-09-15
    .venv\\Scripts\\python.exe scripts\\backfill_ledger.py --start 2026-09-15 --end 2026-11-04 --dry-run
    .venv\\Scripts\\python.exe scripts\\backfill_ledger.py --start 2026-09-15 --only-missing

For every TAIEX trading day in [--start, --end] (end defaults to the
latest weekday) it fetches whatever raw inputs are missing locally,
recomputes the whole trailing window exactly the way daily_update.py
does, then upserts one row per trading day: missing days are added, and
existing days are rewritten if any field changed (pass --only-missing to
leave existing rows alone). Rows outside the range are never touched.

Sources, in the order tried (same as daily_update.py):
  index_price   classic monthly TAIEX file (one request per month)
  vix           TAIFEX monthly VIX files for the last ~4 months, realized
                vol from index_price before that
  options       FinMind TaiwanOptionDaily (token) -> TAIFEX pcRatioDown
  margin        classic MI_MARGN per day, missing days only
  foreign       BFI82U per day for the cash leg, missing days only; the
                futures-OI leg has no history and is 0.0 on backfilled
                days (same convention as daily_update.py)
  bond_index    FinMind TaiwanStockPrice 00679B (token) -> TPEx monthly
  breadth /     MI_INDEX per-stock snapshot per day, missing days only
  new_high_low

The three per-day TWSE pulls share www.twse.com.tw's ~50-requests-per-
window rate limit; each is chunked with cooldowns and a pause separates
them, so a two-month gap takes a few minutes, not seconds.

Recomputing a window shifts the percentile ranks of the days already in
it slightly (they're ranked against a fuller sample), which is why the
default rewrites existing rows in range: the ledger should read as one
consistent series, not as "whatever each day saw at 17:00".
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pandas as pd

import daily_update as du
from twn_fear_greed import data_sources as ds
from twn_fear_greed import data_sources_finmind as fm
from twn_fear_greed.index import TWNFearGreedIndex

HOST_PAUSE_SECONDS = 30.0  # between consecutive per-day loops against www.twse.com.tw


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first date to (re)score, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="last date to (re)score, YYYY-MM-DD (default: latest weekday)")
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="add rows for trading days absent from the ledger; leave existing rows untouched",
    )
    ap.add_argument("--dry-run", action="store_true", help="fetch and compute, print the diff, save nothing")
    return ap.parse_args(argv)


def _fetch_snapshots(dates: list[pd.Timestamp], chunk_size: int = 20, cooldown_seconds: float = 45.0) -> dict:
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for i, d in enumerate(dates):
        if i > 0 and i % chunk_size == 0:
            time.sleep(cooldown_seconds)
        try:
            snap = du.fetch_snapshot(d)
        except Exception as exc:  # noqa: BLE001 - one bad day shouldn't abort the range
            print(f"  MI_INDEX {d.date()} failed ({type(exc).__name__}); skipped.")
            continue
        if snap.empty:
            print(f"  MI_INDEX {d.date()}: no snapshot (holiday or not published).")
        else:
            out[d] = snap
        time.sleep(0.2)
    return out


def _host_pause(requests_just_made: int) -> None:
    if requests_just_made > 0:
        time.sleep(HOST_PAUSE_SECONDS)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize() if args.end else ds._last_probable_trading_day()
    if end < start:
        print(f"--end {end.date()} is before --start {start.date()}")
        return 2

    token = du._finmind_token()
    du.DATA_DIR.mkdir(exist_ok=True)
    ledger_path = du.DATA_DIR / "ledger_history.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []

    cutoff = end - pd.Timedelta(days=du.TRAILING_DAYS)
    if start < cutoff:
        print(
            f"NOTE: --start {start.date()} is before the trailing raw window ({cutoff.date()}); "
            f"rows before that can't be scored and the range is clamped."
        )
        start = cutoff
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    print(f"backfill range: {start_s} -> {end_s}" + (" (dry run)" if args.dry_run else ""))

    print("index_price (classic monthly TAIEX files)...")
    index_price = du._merge(du._load_csv("index_price"), ds.fetch_index_price_history(start_s, end_s))
    trading_days = [d for d in index_price.index if start <= d <= end]
    if not trading_days:
        print("  no TAIEX trading days in range -- nothing to do.")
        return 1
    print(f"  {len(trading_days)} trading days in range ({trading_days[0].date()} -> {trading_days[-1].date()})")

    print("vix (TAIFEX monthly files + realized vol)...")
    vix = ds.fetch_vix(index_price, real_lookback_months=4)

    options = du._load_csv("options")
    missing = [d for d in trading_days if d not in options.index]
    if missing:
        fresh = pd.DataFrame()
        if token:
            print(f"options (FinMind TaiwanOptionDaily, {len(missing)} missing days)...")
            try:
                fresh = fm.fetch_put_call_ratio_history(start_s, end_s, token=token)
            except Exception as exc:  # noqa: BLE001
                print(f"  FinMind options failed ({du._describe(exc, token)}).")
        if fresh.empty or any(d not in fresh.index for d in missing):
            print("options (TAIFEX pcRatioDown download, 30-day slices)...")
            try:
                fresh = du._merge(fresh, ds.fetch_put_call_ratio_history_taifex(start_s, end_s))
            except Exception as exc:  # noqa: BLE001
                print(f"  TAIFEX download failed ({du._describe(exc, token)}).")
        options = du._merge(options, fresh)
    else:
        print("options: nothing missing")

    margin = du._load_csv("margin")
    missing = [d for d in trading_days if d not in margin.index]
    print(f"margin (classic MI_MARGN per day, {len(missing)} missing days)...")
    if missing:
        margin = du._merge(margin, ds.fetch_margin_history(dates=missing))
    _host_pause(len(missing))

    foreign = du._load_csv("foreign")
    missing = [
        d for d in trading_days
        if d not in foreign.index or pd.isna(foreign.loc[d, "net_buy_sell"])
    ]
    print(f"foreign cash (BFI82U per day, {len(missing)} missing days)...")
    if missing:
        foreign = du._merge(foreign, ds.fetch_foreign_cash_history(dates=missing))
    if "futures_net_oi" in foreign.columns:
        foreign["futures_net_oi"] = foreign["futures_net_oi"].fillna(0.0)
    _host_pause(len(missing))

    bond_index = du._load_csv("bond_index")
    missing = [d for d in trading_days if d not in bond_index.index]
    if missing:
        fresh = pd.DataFrame()
        if token:
            print(f"bond_index (FinMind 00679B, {len(missing)} missing days)...")
            try:
                fresh = fm.fetch_bond_index(start=start_s, token=token)
            except Exception as exc:  # noqa: BLE001
                print(f"  FinMind bond_index failed ({du._describe(exc, token)}).")
        if fresh.empty or any(d not in fresh.index for d in missing):
            print("bond_index (TPEx 00679B monthly)...")
            try:
                fresh = du._merge(fresh, ds.fetch_bond_index_tpex(start_s, end_s))
            except Exception as exc:  # noqa: BLE001
                print(f"  TPEx bond_index failed ({du._describe(exc, token)}).")
        bond_index = du._merge(bond_index, fresh)
    else:
        print("bond_index: nothing missing")

    snapshots_path = du.DATA_DIR / "mi_index_snapshots.pkl"
    snapshots = pickle.loads(snapshots_path.read_bytes()) if snapshots_path.exists() else {}
    missing = [d for d in trading_days if d not in snapshots]
    print(f"breadth / new_high_low (MI_INDEX snapshots, {len(missing)} missing days)...")
    if missing:
        snapshots.update(_fetch_snapshots(missing))
    snapshots = {d: s for d, s in snapshots.items() if d >= cutoff}
    breadth, new_high_low = ds._aggregate_breadth_and_strength(snapshots, new_high_low_window=du.LOOKBACK)

    index_price = du._trim(index_price, cutoff)
    vix = du._trim(vix, cutoff)
    options = du._trim(options, cutoff)
    margin = du._trim(margin, cutoff)
    foreign = du._trim(foreign, cutoff)
    bond_index = du._trim(bond_index, cutoff)

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

    result = TWNFearGreedIndex(lookback=du.LOOKBACK).compute(raw)
    result = result.join(index_price.rename(columns={"close": "taiex_close"}))

    by_date = {r["date"]: i for i, r in enumerate(ledger)}
    added: list[str] = []
    changed: list[tuple[str, float | None, float | None]] = []
    unchanged: list[str] = []
    skipped: list[str] = []
    for d in trading_days:
        key = d.strftime("%Y-%m-%d")
        if d not in result.index or pd.isna(result.loc[d].get("taiex_close")):
            skipped.append(key)
            continue
        entry = du.ledger_entry(result, d)
        if key in by_date:
            if args.only_missing:
                continue
            old = ledger[by_date[key]]
            if old == entry:
                unchanged.append(key)
            else:
                changed.append((key, old.get("composite"), entry["composite"]))
                ledger[by_date[key]] = entry
        else:
            added.append(key)
            ledger.append(entry)
    ledger.sort(key=lambda r: r["date"])

    print()
    print(f"added    {len(added)}: {', '.join(added) or '-'}")
    print(f"changed  {len(changed)}:")
    for key, before, after in changed:
        print(f"  {key}: composite {before} -> {after}")
    if not changed:
        print("  -")
    print(f"unchanged {len(unchanged)}, skipped (no computed row) {len(skipped)}: {', '.join(skipped) or '-'}")
    if args.dry_run:
        print("\ndry run -- nothing saved.")
        return 0

    du._save_csv("index_price", index_price)
    du._save_csv("vix", vix)
    du._save_csv("options", options)
    du._save_csv("margin", margin)
    du._save_csv("foreign", foreign)
    du._save_csv("bond_index", bond_index)
    snapshots_path.write_bytes(pickle.dumps(snapshots))
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
    print(f"\nledger_history.json now has {len(ledger)} rows ({ledger[0]['date']} -> {ledger[-1]['date']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
