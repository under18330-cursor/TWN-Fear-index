import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from twn_fear_greed.data_sources_finmind import _aggregate_put_call


def test_aggregate_put_call_sums_volume_by_side_and_date():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"],
            "call_put": ["call", "put", "put", "call"],
            "trading_volume": ["100", "40", "10", "200"],
        }
    )
    result = _aggregate_put_call(raw)
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 100
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 50
    assert result.loc[pd.Timestamp("2024-01-03"), "call_volume"] == 200
    assert result.loc[pd.Timestamp("2024-01-03"), "put_volume"] == 0


def test_aggregate_put_call_handles_uppercase_and_mixed_case():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "call_put": ["CALL", "Put"],
            "trading_volume": ["5", "7"],
        }
    )
    result = _aggregate_put_call(raw)
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 5
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 7


def test_aggregate_put_call_empty_input():
    raw = pd.DataFrame(columns=["date", "call_put", "trading_volume"])
    result = _aggregate_put_call(raw)
    assert result.empty
    assert list(result.columns) == ["call_volume", "put_volume"]


def test_aggregate_put_call_missing_side_gets_zero_column():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "call_put": ["call"],
            "trading_volume": ["100"],
        }
    )
    result = _aggregate_put_call(raw)
    assert result.loc[pd.Timestamp("2024-01-02"), "call_volume"] == 100
    assert result.loc[pd.Timestamp("2024-01-02"), "put_volume"] == 0
