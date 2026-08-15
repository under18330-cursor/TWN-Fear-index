"""Live data-source checker: hit the real APIs and verify what comes back.

The fetchers in `data_sources.py` / `data_sources_finlab.py` /
`data_sources_finmind.py` are not covered by the unit tests, because they
need the network (and, for FinLab/FinMind, credentials). This script is
the manual counterpart: run it against the live endpoints and it reports,
per source, whether the request succeeded AND whether the payload still
has the field names and value ranges the fetchers assume.

The point is the second half. A fetcher that raises is easy to notice; a
fetcher that silently produces garbage -- because the exchange renamed a
field, or returns dates in ROC calendar form (1130815) that
`pd.to_datetime` happily misparses -- is not. So every check validates
the parsed result, not just the HTTP status.

Usage
-----
    python scripts/check_data_sources.py                # all sources
    python scripts/check_data_sources.py --source twse  # one group
    python scripts/check_data_sources.py --pipeline     # also run the index end-to-end

Credentials are read from the environment, never passed on the command
line: FINLAB_API_KEY, FINMIND_API_TOKEN. Sources whose credential is
absent are reported as SKIP, not FAIL.

Exit code is 0 only if nothing FAILed (SKIPs are fine).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OK, FAIL, SKIP = "OK", "FAIL", "SKIP"

# Any parsed trading date outside this range means the raw field was
# misinterpreted (ROC-vs-AD calendar, epoch seconds, swapped D/M...).
MIN_PLAUSIBLE_DATE = pd.Timestamp("1990-01-01")


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        mark = {OK: "PASS", FAIL: "FAIL", SKIP: "skip"}[self.status]
        out = [f"[{mark}] {self.name}: {self.detail}"]
        out += [f"         - {n}" for n in self.notes]
        return "\n".join(out)


def _describe(df: pd.DataFrame) -> str:
    if df.empty:
        return "0 rows"
    return (
        f"{len(df)} rows, {df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}, "
        f"cols={list(df.columns)}"
    )


def _check_frame(name: str, df: pd.DataFrame, expected_cols: list[str]) -> Result:
    """Validate a fetcher's output against the schema indicators.py expects."""
    notes: list[str] = []

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        return Result(
            name, FAIL,
            f"missing column(s) {missing}; got {list(df.columns)}",
            ["indicators.py will KeyError and silently skip this sub-indicator"],
        )

    if df.empty:
        return Result(name, FAIL, "fetch returned 0 rows")

    if not isinstance(df.index, pd.DatetimeIndex):
        return Result(name, FAIL, f"index is {type(df.index).__name__}, expected DatetimeIndex")

    oldest, newest = df.index.min(), df.index.max()
    today = pd.Timestamp(datetime.now().date())
    if oldest < MIN_PLAUSIBLE_DATE or newest > today + pd.Timedelta(days=1):
        return Result(
            name, FAIL,
            f"date index out of plausible range ({oldest:%Y-%m-%d}..{newest:%Y-%m-%d})",
            ["likely a calendar-format mismatch (ROC dates parsed as AD?)"],
        )

    staleness = (today - newest).days
    if staleness > 7:
        notes.append(f"latest row is {staleness} days old -- endpoint may be stale or discontinued")

    for col in expected_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        na_share = values.isna().mean()
        if na_share == 1.0:
            return Result(name, FAIL, f"column '{col}' is entirely non-numeric/NaN after parsing")
        if na_share > 0.1:
            notes.append(f"column '{col}' is {na_share:.0%} NaN")

    if not df.index.is_monotonic_increasing:
        notes.append("index is not sorted ascending; indicators.py assumes ascending dates")
    if df.index.has_duplicates:
        notes.append(f"index has {df.index.duplicated().sum()} duplicate date(s)")

    tail = df[expected_cols].tail(1).to_dict("records")[0]
    notes.append(f"latest: {newest:%Y-%m-%d} {tail}")
    return Result(name, OK, _describe(df), notes)


def _run(name: str, fetch, expected_cols: list[str], verbose: bool) -> Result:
    try:
        df = fetch()
    except NotImplementedError as exc:
        return Result(name, SKIP, f"not implemented: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001 -- this script's job is to report every failure mode
        if verbose:
            traceback.print_exc()
        return Result(name, FAIL, f"{type(exc).__name__}: {exc}", _diagnose(exc))
    return _check_frame(name, df, expected_cols)


def _diagnose(exc: Exception) -> list[str]:
    """Turn common transport failures into an actionable next step."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "403" in text and "connect" in text:
        return ["blocked by an egress/network policy, not by the data provider -- "
                "run this from a network that allows the host"]
    if "proxy" in text or "tunnel" in text:
        return ["the HTTPS proxy rejected the connection; check HTTPS_PROXY and its allowlist"]
    if "certificate" in text or "ssl" in text:
        return ["TLS verification failed; point REQUESTS_CA_BUNDLE at your proxy's CA bundle"]
    if "timeout" in text or "timed out" in text:
        return ["request timed out; the exchange endpoints are slow at market close"]
    if "401" in text or "unauthor" in text or "token" in text:
        return ["credential rejected -- check the env var holds a current, non-expired token"]
    return []


def check_twse(verbose: bool) -> list[Result]:
    from twn_fear_greed import data_sources as ds

    return [
        _run("TWSE  index_price", ds.fetch_index_price, ["close"], verbose),
        _run("TWSE  margin", ds.fetch_margin, ["margin_balance"], verbose),
        _run("TWSE  foreign", ds.fetch_foreign, ["net_buy_sell", "futures_net_oi"], verbose),
        _run("TWSE  new_high_low", ds.fetch_new_high_low, ["new_highs", "new_lows"], verbose),
        _run("TWSE  breadth", ds.fetch_breadth, ["advancing_volume", "declining_volume"], verbose),
        _run("TWSE  bond_index", ds.fetch_bond_index, ["close"], verbose),
    ]


def check_taifex(verbose: bool) -> list[Result]:
    from twn_fear_greed import data_sources as ds

    results = [
        _run("TAIFEX options", ds.fetch_options, ["put_volume", "call_volume"], verbose),
        _run("TAIFEX vix", ds.fetch_vix, ["taiex_vix"], verbose),
    ]
    results.append(_check_put_call_single_class(verbose))
    return results


def _check_put_call_single_class(verbose: bool) -> Result:
    """fetch_options assigns column names positionally after unstacking a
    boolean, which only holds if the day's rows contain BOTH puts and
    calls. Verify that assumption against what the endpoint actually
    returned rather than trusting it.
    """
    from twn_fear_greed import data_sources as ds

    name = "TAIFEX options put/call split"
    try:
        rows = pd.DataFrame(ds._get_json(ds.ENDPOINTS["taiex_options"]))
    except Exception as exc:  # noqa: BLE001
        if verbose:
            traceback.print_exc()
        return Result(name, SKIP, f"could not re-fetch raw rows ({type(exc).__name__})")

    if "CallPut" not in rows.columns:
        return Result(
            name, FAIL,
            f"no 'CallPut' field in payload; got {list(rows.columns)}",
            ["fetch_options() classifies puts by this field name"],
        )

    labels = sorted(rows["CallPut"].dropna().unique().tolist())
    is_put = rows["CallPut"].str.contains("Put", case=False, na=False)
    if is_put.nunique() < 2:
        return Result(
            name, FAIL,
            f"payload contains only one side (CallPut values: {labels})",
            ["fetch_options() renames unstacked columns positionally to "
             "['call_volume', 'put_volume'] and raises ValueError on a one-sided day"],
        )
    return Result(name, OK, f"both sides present (CallPut values: {labels})")


def check_finlab(start: str | None, verbose: bool) -> list[Result]:
    if not os.environ.get("FINLAB_API_KEY"):
        return [Result("FinLab", SKIP, "FINLAB_API_KEY not set")]
    try:
        from twn_fear_greed import data_sources_finlab as fl
    except ImportError as exc:
        return [Result("FinLab", SKIP, f"{exc}")]

    try:
        fl.login()
    except Exception as exc:  # noqa: BLE001
        if verbose:
            traceback.print_exc()
        return [Result("FinLab login", FAIL, f"{type(exc).__name__}: {exc}", _diagnose(exc))]

    return [
        Result("FinLab login", OK, "authenticated"),
        _run("FinLab index_price", lambda: fl.fetch_index_price(start), ["close"], verbose),
        _run("FinLab new_high_low", lambda: fl.fetch_new_high_low(start), ["new_highs", "new_lows"], verbose),
        _run("FinLab breadth", lambda: fl.fetch_breadth(start), ["advancing_volume", "declining_volume"], verbose),
        _run("FinLab vix (realized-vol proxy)", lambda: fl.fetch_volatility_proxy(start), ["taiex_vix"], verbose),
        _run("FinLab margin", lambda: fl.fetch_margin(start), ["margin_balance"], verbose),
        _run("FinLab foreign", lambda: fl.fetch_foreign(start), ["net_buy_sell", "futures_net_oi"], verbose),
        _run("FinLab bond_index", lambda: fl.fetch_bond_index(start), ["close"], verbose),
    ]


def check_finmind(start: str | None, verbose: bool) -> list[Result]:
    if not os.environ.get("FINMIND_API_TOKEN"):
        return [Result("FinMind", SKIP, "FINMIND_API_TOKEN not set (separate credential from FinLab)")]
    from twn_fear_greed import data_sources_finmind as fm

    return [
        _run("FinMind put/call", lambda: fm.fetch_put_call_ratio(start=start),
             ["put_volume", "call_volume"], verbose),
    ]


def run_pipeline(start: str | None, verbose: bool) -> list[Result]:
    """End-to-end smoke test on live data: fetch -> indicators -> composite."""
    if not os.environ.get("FINLAB_API_KEY"):
        return [Result("pipeline", SKIP, "needs FINLAB_API_KEY for a full raw bundle")]

    from twn_fear_greed.index import TWNFearGreedIndex

    try:
        from twn_fear_greed import data_sources_finlab as fl

        fl.login()
        raw = fl.fetch_all_raw_data(start=start)
        if os.environ.get("FINMIND_API_TOKEN"):
            from twn_fear_greed import data_sources_finmind as fm

            raw = fm.augment_with_options(raw, start=start)
        result = TWNFearGreedIndex().compute(raw)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            traceback.print_exc()
        return [Result("pipeline", FAIL, f"{type(exc).__name__}: {exc}", _diagnose(exc))]

    if result.empty:
        return [Result("pipeline", FAIL, "computed an empty result frame")]

    scored = [c for c in result.columns if c.endswith("_score")]
    latest = result.iloc[-1]
    notes = [
        f"sub-indicators scored: {len(scored)} -> {sorted(scored)}",
        f"latest {result.index[-1]:%Y-%m-%d}: composite={latest['composite']:.1f} ({latest['label']})",
    ]
    if len(scored) < 8:
        notes.append(f"{8 - len(scored)} sub-indicator(s) had no data source; weights were renormalized")
    if not (0 <= latest["composite"] <= 100):
        return [Result("pipeline", FAIL, f"composite out of range: {latest['composite']}", notes)]
    return [Result("pipeline", OK, _describe(result), notes)]


GROUPS = {"twse", "taifex", "finlab", "finmind"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="all", help=f"one of: all, {', '.join(sorted(GROUPS))}")
    parser.add_argument("--start", default=None, help="start date for the keyed sources, e.g. 2024-01-01")
    parser.add_argument("--pipeline", action="store_true", help="also run the full index on live data")
    parser.add_argument("-v", "--verbose", action="store_true", help="print tracebacks for failures")
    args = parser.parse_args()

    if args.source != "all" and args.source not in GROUPS:
        parser.error(f"--source must be 'all' or one of {sorted(GROUPS)}")
    wanted = GROUPS if args.source == "all" else {args.source}

    results: list[Result] = []
    if "twse" in wanted:
        results += check_twse(args.verbose)
    if "taifex" in wanted:
        results += check_taifex(args.verbose)
    if "finlab" in wanted:
        results += check_finlab(args.start, args.verbose)
    if "finmind" in wanted:
        results += check_finmind(args.start, args.verbose)
    if args.pipeline:
        results += run_pipeline(args.start, args.verbose)

    for r in results:
        print(r.line())

    counts = {s: sum(r.status == s for r in results) for s in (OK, FAIL, SKIP)}
    print(f"\n{counts[OK]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
