"""Normalization and composite-scoring utilities.

These functions are pure (no I/O, no network) so they can be unit-tested
and reused regardless of where the raw indicator values come from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK = 756  # ~3 trading years
MIN_SAMPLES_FOR_CONFIDENCE = 60  # ~3 trading months

DEFAULT_WEIGHTS = {
    "momentum": 0.15,
    "strength": 0.15,
    "breadth": 0.15,
    "put_call": 0.12,
    "volatility": 0.15,
    "safe_haven": 0.10,
    "margin": 0.10,
    "foreign": 0.08,
}

# Indicators where a higher raw value means MORE fear (so the percentile
# rank must be inverted before it represents "greed").
INVERTED_INDICATORS = {"put_call", "volatility"}


def percentile_score(
    series: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    invert: bool = False,
) -> pd.Series:
    """Map each value to its trailing percentile rank, scaled to 0-100.

    score[t] = percentile of value[t] within value[t-lookback+1 : t]
    """
    series = series.astype(float)

    def _rank_last(window: np.ndarray) -> float:
        last = window[-1]
        if np.isnan(last):
            return np.nan
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            return np.nan
        return (valid <= last).sum() / len(valid) * 100.0

    scores = series.rolling(window=lookback, min_periods=1).apply(_rank_last, raw=True)
    if invert:
        scores = 100.0 - scores
    return scores.rename(series.name)


def sample_confidence(series: pd.Series, lookback: int = DEFAULT_LOOKBACK) -> pd.Series:
    """True once enough trailing history exists for a stable percentile rank."""
    counts = series.notna().rolling(window=lookback, min_periods=1).count()
    return counts >= MIN_SAMPLES_FOR_CONFIDENCE


def composite_score(
    scores: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Weighted average of per-indicator 0-100 scores, renormalized for
    whatever subset of indicators has data on a given day (missing
    indicators are excluded rather than treated as zero).
    """
    weights = weights or DEFAULT_WEIGHTS
    cols = [c for c in scores.columns if c in weights]
    w = pd.Series({c: weights[c] for c in cols})

    weighted_sum = scores[cols].mul(w, axis=1).sum(axis=1, skipna=True)
    active_weight = scores[cols].notna().mul(w, axis=1).sum(axis=1)
    active_weight = active_weight.replace(0.0, np.nan)

    return (weighted_sum / active_weight).rename("composite")


def label_for_score(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score < 20:
        return "extreme_fear"
    if score < 40:
        return "fear"
    if score < 60:
        return "neutral"
    if score < 80:
        return "greed"
    return "extreme_greed"


def extreme_flags(
    composite: pd.Series,
    streak_len: int = 5,
    collapse_window: int = 10,
    collapse_drop: float = 30.0,
) -> pd.DataFrame:
    """Persistence and rapid-reversal flags computed from the composite score."""
    overheat = composite >= 80
    oversold = composite <= 20

    def _consecutive_true_streak(flag: pd.Series) -> pd.Series:
        group_id = (flag != flag.shift()).cumsum()
        streak = flag.groupby(group_id).cumcount() + 1
        return streak.where(flag, 0)

    overheat_streak = _consecutive_true_streak(overheat)
    oversold_streak = _consecutive_true_streak(oversold)

    rolling_max = composite.rolling(collapse_window, min_periods=1).max()
    rolling_min = composite.rolling(collapse_window, min_periods=1).min()

    sentiment_collapse = (rolling_max - composite) >= collapse_drop
    sentiment_collapse &= rolling_max >= 70
    sentiment_spike = (composite - rolling_min) >= collapse_drop
    sentiment_spike &= rolling_min <= 30

    return pd.DataFrame(
        {
            "overheat_streak": overheat_streak.astype(int),
            "oversold_streak": oversold_streak.astype(int),
            "overheat_flag": overheat_streak >= streak_len,
            "oversold_flag": oversold_streak >= streak_len,
            "sentiment_collapse": sentiment_collapse.fillna(False),
            "sentiment_spike": sentiment_spike.fillna(False),
        }
    )
