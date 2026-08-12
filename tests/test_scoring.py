import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from twn_fear_greed.scoring import (
    composite_score,
    extreme_flags,
    label_for_score,
    percentile_score,
)


def _series(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_percentile_score_monotonic_increasing_series_is_near_100_at_end():
    s = _series(range(1, 101))
    scores = percentile_score(s, lookback=100)
    assert scores.iloc[-1] == pytest.approx(100.0)
    assert scores.iloc[0] == pytest.approx(100.0)  # single-sample window


def test_percentile_score_invert_flips_ranking():
    s = _series(range(1, 101))
    normal = percentile_score(s, lookback=100)
    inverted = percentile_score(s, lookback=100, invert=True)
    assert inverted.iloc[-1] == pytest.approx(100.0 - normal.iloc[-1])


def test_percentile_score_handles_nan():
    values = list(range(1, 51)) + [np.nan] * 5 + list(range(51, 61))
    s = _series(values)
    scores = percentile_score(s, lookback=60)
    assert scores.iloc[50:55].isna().all()
    assert not scores.iloc[-1] != scores.iloc[-1]  # last value not NaN


def test_composite_score_equal_weights_matches_manual_average():
    dates = pd.bdate_range("2024-01-01", periods=3)
    scores = pd.DataFrame(
        {"momentum": [10.0, 50.0, 90.0], "strength": [30.0, 50.0, 70.0]},
        index=dates,
    )
    weights = {"momentum": 0.5, "strength": 0.5}
    result = composite_score(scores, weights)
    assert result.iloc[0] == pytest.approx(20.0)
    assert result.iloc[1] == pytest.approx(50.0)
    assert result.iloc[2] == pytest.approx(80.0)


def test_composite_score_renormalizes_when_indicator_missing():
    dates = pd.bdate_range("2024-01-01", periods=1)
    scores = pd.DataFrame({"momentum": [80.0], "strength": [np.nan]}, index=dates)
    weights = {"momentum": 0.5, "strength": 0.5}
    result = composite_score(scores, weights)
    assert result.iloc[0] == pytest.approx(80.0)


def test_label_for_score_boundaries():
    assert label_for_score(0) == "extreme_fear"
    assert label_for_score(19.9) == "extreme_fear"
    assert label_for_score(20) == "fear"
    assert label_for_score(39.9) == "fear"
    assert label_for_score(40) == "neutral"
    assert label_for_score(59.9) == "neutral"
    assert label_for_score(60) == "greed"
    assert label_for_score(79.9) == "greed"
    assert label_for_score(80) == "extreme_greed"
    assert label_for_score(100) == "extreme_greed"
    assert label_for_score(float("nan")) == "unknown"


def test_extreme_flags_overheat_streak():
    composite = _series([50] * 5 + [85] * 6 + [50] * 5)
    flags = extreme_flags(composite, streak_len=5)
    assert flags["overheat_flag"].iloc[9]  # 5th consecutive day at 85
    assert not flags["overheat_flag"].iloc[8]
    assert flags["overheat_streak"].iloc[10] == 6


def test_extreme_flags_sentiment_collapse():
    values = [75] * 3 + [40] * 3 + [50] * 5
    composite = _series(values)
    flags = extreme_flags(composite, collapse_window=10, collapse_drop=30.0)
    assert flags["sentiment_collapse"].iloc[3]
