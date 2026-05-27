"""
scorer.py — Ranking logic for Edgeware stock screener.

Stocks that survived all filters are scored 0–100 using min-max normalisation
within the surviving cohort, then weighted:

    30 %  PEG attractiveness          (lower PEG → higher score)
    25 %  Revenue acceleration mag.   (higher acceleration → higher score)
    20 %  Gross margin expansion mag. (higher expansion → higher score)
    15 %  Insider buying conviction   (higher net $ bought → higher score)
    10 %  Relative P/E discount       (lower P/E vs sector → higher score)

Returns the list sorted by composite score descending, with per-component
scores attached to each dict for transparency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Score weights (must sum to 1.0)
WEIGHTS: Dict[str, float] = {
    "peg": 0.30,
    "revenue_accel": 0.25,
    "gross_margin": 0.20,
    "insider": 0.15,
    "pe_relative": 0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _minmax(
    values: List[Optional[float]],
    invert: bool = False,
    default_on_tie: float = 50.0,
) -> List[float]:
    """
    Min-max normalise *values* to [0, 100].

    None values are treated as the worst possible (0 after normalisation, or
    100 if inverted).  If all values are identical the cohort is assigned
    *default_on_tie* uniformly.

    Args:
        values:         Raw metric values, may contain None.
        invert:         If True, lower raw value → higher normalised score.
        default_on_tie: Score returned when all values are identical.
    """
    # Replace None with NaN so numpy ignores them cleanly
    arr = np.array(
        [v if v is not None else np.nan for v in values],
        dtype=float,
    )

    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return [default_on_tie] * len(values)

    min_v, max_v = np.nanmin(arr), np.nanmax(arr)

    if max_v == min_v:
        return [default_on_tie] * len(values)

    normalized = (arr - min_v) / (max_v - min_v) * 100.0

    if invert:
        normalized = 100.0 - normalized

    # Replace NaN (originally None) with 0 (worst score)
    worst = 100.0 if invert else 0.0
    normalized = np.where(np.isnan(normalized), worst, normalized)

    return normalized.tolist()


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_stocks(passed_stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score and rank a list of stocks that have survived all filters.

    Mutates each dict in-place, adding:
        "score"            — weighted composite score 0–100
        "score_components" — dict with per-dimension sub-scores 0–100

    Returns the same list sorted by "score" descending.
    """
    if not passed_stocks:
        return []

    n = len(passed_stocks)
    logger.info("Scoring %d stocks …", n)

    # ------------------------------------------------------------------
    # Extract raw values for each scoring dimension
    # ------------------------------------------------------------------
    pegs: List[Optional[float]] = [s.get("peg_ratio") for s in passed_stocks]

    accels: List[Optional[float]] = [
        s.get("revenue_acceleration") for s in passed_stocks
    ]

    gm_expansions: List[Optional[float]] = [
        s.get("gm_expansion") for s in passed_stocks
    ]

    # Insider buying: clip negative values to 0 (shouldn't happen post-filter)
    insiders: List[Optional[float]] = [
        max(float(s.get("net_insider_buying") or 0), 0.0) for s in passed_stocks
    ]

    # Relative P/E: pe / sector_median_pe  (lower = better)
    pe_rels: List[Optional[float]] = []
    for s in passed_stocks:
        pe = s.get("pe_ratio")
        med = s.get("sector_median_pe")
        if pe and med and med > 0:
            pe_rels.append(pe / med)
        else:
            pe_rels.append(None)  # treated as worst in normalisation

    # ------------------------------------------------------------------
    # Normalise each dimension
    # ------------------------------------------------------------------
    peg_scores = _minmax(pegs, invert=True)          # lower PEG → higher score
    accel_scores = _minmax(accels, invert=False)      # higher accel → higher score
    gm_scores = _minmax(gm_expansions, invert=False)  # higher expansion → higher score
    insider_scores = _minmax(insiders, invert=False)  # higher buying → higher score
    pe_scores = _minmax(pe_rels, invert=True)         # lower ratio → higher score

    # ------------------------------------------------------------------
    # Compute weighted composite score
    # ------------------------------------------------------------------
    for i, stock in enumerate(passed_stocks):
        components = {
            "peg_score": round(peg_scores[i], 2),
            "accel_score": round(accel_scores[i], 2),
            "gm_score": round(gm_scores[i], 2),
            "insider_score": round(insider_scores[i], 2),
            "pe_score": round(pe_scores[i], 2),
        }

        composite = (
            WEIGHTS["peg"] * peg_scores[i]
            + WEIGHTS["revenue_accel"] * accel_scores[i]
            + WEIGHTS["gross_margin"] * gm_scores[i]
            + WEIGHTS["insider"] * insider_scores[i]
            + WEIGHTS["pe_relative"] * pe_scores[i]
        )

        stock["score"] = round(composite, 2)
        stock["score_components"] = components

        logger.debug(
            "%s score=%.1f  peg=%.1f accel=%.1f gm=%.1f ins=%.1f pe=%.1f",
            stock.get("ticker"),
            composite,
            peg_scores[i],
            accel_scores[i],
            gm_scores[i],
            insider_scores[i],
            pe_scores[i],
        )

    # Sort descending by composite score
    passed_stocks.sort(key=lambda s: s["score"], reverse=True)

    logger.info(
        "Top scorer: %s (%.1f)  |  #10: %s (%.1f)",
        passed_stocks[0]["ticker"],
        passed_stocks[0]["score"],
        passed_stocks[min(9, n - 1)]["ticker"],
        passed_stocks[min(9, n - 1)]["score"],
    )

    return passed_stocks
