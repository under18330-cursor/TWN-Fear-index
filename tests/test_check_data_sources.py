"""Offline tests for the live-checker's validation logic.

scripts/check_data_sources.py only earns its keep if its checks actually
fire on bad data -- a checker that returns PASS for a misparsed date
index is worse than no checker. These tests feed it deliberately broken
frames; no network involved.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_data_sources",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "check_data_sources.py"),
)
checker = importlib.util.module_from_spec(_SPEC)
# @dataclass resolves annotations via sys.modules, so register before exec.
sys.modules["check_data_sources"] = checker
_SPEC.loader.exec_module(checker)


def _good_frame(rows: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=rows, name="date")
    return pd.DataFrame({"close": range(rows)}, index=idx, dtype=float)


def test_healthy_frame_passes():
    result = checker._check_frame("x", _good_frame(), ["close"])
    assert result.status == checker.OK


def test_missing_column_fails():
    df = _good_frame().rename(columns={"close": "ClosingIndex"})
    result = checker._check_frame("x", df, ["close"])
    assert result.status == checker.FAIL
    assert "missing column" in result.detail


def test_empty_frame_fails():
    result = checker._check_frame("x", pd.DataFrame(columns=["close"]), ["close"])
    assert result.status == checker.FAIL


def test_roc_dates_parsed_as_ad_are_rejected():
    """'1130815' (ROC) through pd.to_datetime lands in year 1130 -- the
    exact silent-corruption case the date-range check exists for."""
    idx = pd.to_datetime(["1130813", "1130814", "1130815"], format="%Y%m%d")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    result = checker._check_frame("x", df, ["close"])
    assert result.status == checker.FAIL
    assert "out of plausible range" in result.detail


def test_future_dates_are_rejected():
    idx = pd.bdate_range(start=pd.Timestamp.now().normalize() + pd.Timedelta(days=30), periods=5)
    df = pd.DataFrame({"close": [1.0] * 5}, index=idx)
    assert checker._check_frame("x", df, ["close"]).status == checker.FAIL


def test_non_numeric_column_fails():
    df = _good_frame()
    df["close"] = "--"
    result = checker._check_frame("x", df, ["close"])
    assert result.status == checker.FAIL
    assert "non-numeric" in result.detail


def test_non_datetime_index_fails():
    df = _good_frame().reset_index(drop=True)
    assert checker._check_frame("x", df, ["close"]).status == checker.FAIL


def test_stale_data_passes_but_is_flagged():
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=60), periods=10)
    df = pd.DataFrame({"close": [1.0] * 10}, index=idx)
    result = checker._check_frame("x", df, ["close"])
    assert result.status == checker.OK
    assert any("days old" in n for n in result.notes)


def test_unsorted_and_duplicate_dates_are_flagged():
    df = _good_frame(10)
    df = pd.concat([df, df.iloc[[0]]])
    result = checker._check_frame("x", df, ["close"])
    assert result.status == checker.OK
    assert any("duplicate" in n for n in result.notes)
    assert any("not sorted" in n for n in result.notes)


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Tunnel connection failed: 403 Forbidden", "egress"),
        ("certificate verify failed", "CA bundle"),
        ("Read timed out", "timed out"),
        ("401 Unauthorized", "credential"),
    ],
)
def test_diagnose_maps_transport_failures_to_advice(message, expected):
    notes = checker._diagnose(RuntimeError(message))
    assert notes and expected in notes[0]
