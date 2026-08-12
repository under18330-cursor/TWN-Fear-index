"""TWNFearGreedIndex: end-to-end orchestration of the composite index."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import compute_all_raw_indicators
from .scoring import (
    DEFAULT_LOOKBACK,
    DEFAULT_WEIGHTS,
    INVERTED_INDICATORS,
    composite_score,
    extreme_flags,
    label_for_score,
    percentile_score,
    sample_confidence,
)


@dataclass
class TWNFearGreedIndex:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    lookback: int = DEFAULT_LOOKBACK

    def compute(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """raw: dict of tidy DataFrames as documented in indicators.py.

        Returns a DataFrame indexed by date with per-indicator scores,
        the composite score, its label, and extreme/reversal flags.
        """
        raw_values = compute_all_raw_indicators(raw)

        scores = {}
        confidence = {}
        for name in raw_values.columns:
            invert = name in INVERTED_INDICATORS
            scores[name] = percentile_score(raw_values[name], self.lookback, invert=invert)
            confidence[name] = sample_confidence(raw_values[name], self.lookback)

        score_df = pd.DataFrame(scores)
        confidence_df = pd.DataFrame(confidence).add_suffix("_confident")

        composite = composite_score(score_df, self.weights)
        labels = composite.apply(label_for_score).rename("label")
        flags = extreme_flags(composite)

        result = pd.concat(
            [score_df.add_suffix("_score"), composite, labels, flags, confidence_df],
            axis=1,
        )
        return result

    def latest(self, raw: dict[str, pd.DataFrame]) -> dict:
        """Convenience: compute the full history and return just the most recent row as a dict."""
        result = self.compute(raw)
        if result.empty:
            return {}
        last = result.iloc[-1]
        return {"date": result.index[-1], **last.to_dict()}
