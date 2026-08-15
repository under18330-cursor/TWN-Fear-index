import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from twn_fear_greed.data_sources_finmind import _aggregate_put_call


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
