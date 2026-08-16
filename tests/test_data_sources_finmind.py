import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

import twn_fear_greed.data_sources_finmind as fm
from twn_fear_greed.data_sources_finmind import (
    _aggregate_margin_history,
    _aggregate_put_call,
    _bond_index_from_price,
    _margin_balance_from_stock,
)


def _raw(dates, sides, volumes, sessions=None):
    """Build a TaiwanOptionDaily-shaped frame.

    Column names here mirror the live v4 response exactly -- the volume
    column is "volume", not "trading_volume". Getting this wrong is what
    made the connector pass its tests while failing against the real API.
    """
    data = {"date": dates, "call_put": sides, "volume": volumes}
    if sessions is not None:
        data["trading_session"] = sessions
    return pd.DataFrame(data)


def test_aggregate_put_call_sums_volume_by_side_and_date():
    raw = _raw(
        ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"],
        ["call", "put", "put", "call"],
        ["100", "40", "10", "200"],
    )
    result = _aggregate_put_call(raw)
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 100
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 50
    assert result.loc[pd.Timestamp("2024-01-03"), "call_volume"] == 200
    assert result.loc[pd.Timestamp("2024-01-03"), "put_volume"] == 0


def test_aggregate_put_call_handles_uppercase_and_mixed_case():
    raw = _raw(["2024-01-02", "2024-01-02"], ["CALL", "Put"], ["5", "7"])
    result = _aggregate_put_call(raw)
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 5
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 7


def test_aggregate_put_call_empty_input():
    raw = pd.DataFrame(columns=["date", "call_put", "volume"])
    result = _aggregate_put_call(raw)
    assert result.empty
    assert list(result.columns) == ["call_volume", "put_volume"]


def test_aggregate_put_call_missing_side_gets_zero_column():
    raw = _raw(["2024-01-02"], ["call"], ["100"])
    result = _aggregate_put_call(raw)
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 100
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 0


def test_aggregate_put_call_rejects_legacy_column_name():
    """A frame using the old assumed schema must fail loudly, not silently."""
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "call_put": ["call"],
            "trading_volume": ["100"],
        }
    )
    with pytest.raises(KeyError):
        _aggregate_put_call(raw)


def test_aggregate_put_call_combines_both_sessions_by_default():
    raw = _raw(
        ["2024-01-02"] * 4,
        ["call", "put", "call", "put"],
        ["100", "40", "30", "20"],
        sessions=["position", "position", "after_market", "after_market"],
    )
    result = _aggregate_put_call(raw)
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 130
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 60


def test_aggregate_put_call_can_restrict_to_one_session():
    raw = _raw(
        ["2024-01-02"] * 4,
        ["call", "put", "call", "put"],
        ["100", "40", "30", "20"],
        sessions=["position", "position", "after_market", "after_market"],
    )
    result = _aggregate_put_call(raw, session="position")
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 100
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 40


def test_aggregate_put_call_unknown_session_yields_empty():
    raw = _raw(["2024-01-02"], ["call"], ["100"], sessions=["position"])
    result = _aggregate_put_call(raw, session="nonexistent")
    assert result.empty
    assert list(result.columns) == ["call_volume", "put_volume"]


def test_bond_index_from_price_keeps_only_close_indexed_by_date():
    raw = pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-02"],
            "stock_id": ["00679B", "00679B"],
            "open": [27.07, 27.10],
            "close": [27.05, 27.11],
        }
    )
    result = _bond_index_from_price(raw)
    assert list(result.columns) == ["close"]
    assert result.loc[pd.Timestamp("2026-07-01"), "close"] == 27.05
    assert result.loc[pd.Timestamp("2026-07-02"), "close"] == 27.11
    assert result.index.is_monotonic_increasing


def test_bond_index_from_price_empty_input():
    result = _bond_index_from_price(pd.DataFrame(columns=["date", "close"]))
    assert result.empty
    assert list(result.columns) == ["close"]


def test_fetch_put_call_ratio_history_chunks_by_calendar_month(monkeypatch):
    calls = []

    def fake_get_dataframe(dataset, data_id=None, start_date=None, end_date=None, token=None, timeout=15):
        calls.append((start_date, end_date))
        # one distinct row per month so the concatenated result is checkable
        day = start_date  # first day of that month's chunk
        return _raw([day], ["call"], ["10"])

    monkeypatch.setattr(fm, "_get_dataframe", fake_get_dataframe)

    result = fm.fetch_put_call_ratio_history("2026-06-15", "2026-08-14")

    # three calendar months touched: Jun, Jul, Aug -- each capped to the actual range
    assert calls == [
        ("2026-06-01", "2026-06-30"),
        ("2026-07-01", "2026-07-31"),
        ("2026-08-01", "2026-08-14"),
    ]
    # the June chunk's row lands on 2026-06-01, before the requested start
    # (2026-06-15) -- final .loc[start:end] trim correctly drops it, same
    # as fetch_index_price_history's analogous trim
    assert list(result.index) == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")]
    assert list(result.columns) == ["call_volume", "put_volume"]


def test_fetch_put_call_ratio_history_empty_range_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(fm, "_get_dataframe", lambda *a, **k: pd.DataFrame(columns=["date", "call_put", "volume"]))
    result = fm.fetch_put_call_ratio_history("2026-08-01", "2026-08-14")
    assert result.empty
    assert list(result.columns) == ["call_volume", "put_volume"]


def test_margin_balance_from_stock_reads_today_balance():
    raw = pd.DataFrame(
        {
            "date": ["2026-08-03", "2026-08-04"],
            "stock_id": ["2330", "2330"],
            "MarginPurchaseTodayBalance": [30003, 30500],
        }
    )
    result = _margin_balance_from_stock(raw)
    assert list(result.columns) == ["margin_balance"]
    assert result.loc[pd.Timestamp("2026-08-03"), "margin_balance"] == 30003
    assert result.loc[pd.Timestamp("2026-08-04"), "margin_balance"] == 30500


def test_margin_balance_from_stock_empty_input():
    result = _margin_balance_from_stock(pd.DataFrame(columns=["date", "MarginPurchaseTodayBalance"]))
    assert result.empty
    assert list(result.columns) == ["margin_balance"]


def test_aggregate_margin_history_sums_across_stocks():
    per_stock = {
        "2330": pd.DataFrame({"margin_balance": [100.0, 110.0]}, index=pd.DatetimeIndex(["2026-08-03", "2026-08-04"])),
        "2317": pd.DataFrame({"margin_balance": [50.0, 60.0]}, index=pd.DatetimeIndex(["2026-08-03", "2026-08-04"])),
    }
    result = _aggregate_margin_history(per_stock)
    assert result.loc[pd.Timestamp("2026-08-03"), "margin_balance"] == 150.0
    assert result.loc[pd.Timestamp("2026-08-04"), "margin_balance"] == 170.0


def test_aggregate_margin_history_missing_stock_day_contributes_zero_not_nan():
    per_stock = {
        "2330": pd.DataFrame({"margin_balance": [100.0]}, index=pd.DatetimeIndex(["2026-08-03"])),
        # 2317 only has a later date -- no overlap with 2330 on 2026-08-03
        "2317": pd.DataFrame({"margin_balance": [50.0]}, index=pd.DatetimeIndex(["2026-08-04"])),
    }
    result = _aggregate_margin_history(per_stock)
    assert result.loc[pd.Timestamp("2026-08-03"), "margin_balance"] == 100.0
    assert result.loc[pd.Timestamp("2026-08-04"), "margin_balance"] == 50.0


def test_aggregate_margin_history_empty_input():
    result = _aggregate_margin_history({})
    assert result.empty
    assert list(result.columns) == ["margin_balance"]
