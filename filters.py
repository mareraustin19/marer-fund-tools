"""
filters.py — Individual screening filter functions for Edgeware.

Each filter accepts a stock data dict and returns either:
  • (bool, float | None)  — pass/fail + a magnitude used for scoring / watch flags
  • bool                  — pass/fail only (no magnitude needed downstream)

Thresholds are parameterised so they can be overridden in tests.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter 1 — PEG ratio
# ---------------------------------------------------------------------------

def filter_peg_ratio(
    data: Dict[str, Any],
    threshold: float = 1.5,
) -> Tuple[bool, Optional[float]]:
    """
    Pass when PEG ratio is below *threshold* (default 1.5).

    Returns (passed, peg_value).  peg_value is None when data is missing.
    A zero or negative PEG is treated as missing (data artefact).
    """
    peg = data.get("peg_ratio")
    if peg is None or peg <= 0:
        logger.debug("%s: PEG data missing / invalid (%s)", data.get("ticker"), peg)
        return False, None
    return peg < threshold, peg


# ---------------------------------------------------------------------------
# Filter 2 — Revenue growth acceleration (QoQ)
# ---------------------------------------------------------------------------

def filter_revenue_acceleration(
    data: Dict[str, Any],
) -> Tuple[bool, Optional[float]]:
    """
    Pass when the most-recent quarter-over-quarter revenue growth rate
    exceeds the prior quarter's QoQ growth rate (i.e., acceleration).

    Returns (passed, acceleration) where acceleration = recent_qoq − prior_qoq.
    Requires at least 3 quarters of revenue data.
    """
    quarters = data.get("revenue_quarters", [])

    if len(quarters) < 3:
        logger.debug(
            "%s: insufficient quarterly revenue data (%d quarters)",
            data.get("ticker"),
            len(quarters),
        )
        return False, None

    try:
        # yfinance returns newest first → q0 most recent, q1 prior, q2 two-prior
        q0, q1, q2 = float(quarters[0]), float(quarters[1]), float(quarters[2])
    except (TypeError, ValueError) as exc:
        logger.debug("%s: revenue quarter parse error: %s", data.get("ticker"), exc)
        return False, None

    if q1 == 0 or q2 == 0:
        return False, None

    recent_qoq = (q0 - q1) / abs(q1)
    prior_qoq = (q1 - q2) / abs(q2)
    acceleration = recent_qoq - prior_qoq

    return recent_qoq > prior_qoq, acceleration


# ---------------------------------------------------------------------------
# Filter 3 — Gross margin expansion (YoY)
# ---------------------------------------------------------------------------

def filter_gross_margin_expansion(
    data: Dict[str, Any],
) -> Tuple[bool, Optional[float]]:
    """
    Pass when the current fiscal-year gross margin exceeds the prior year.

    Returns (passed, expansion_pp) where expansion is the percentage-point delta.
    """
    gm_current = data.get("gross_margin_current")
    gm_prior = data.get("gross_margin_prior")

    if gm_current is None or gm_prior is None:
        logger.debug(
            "%s: gross margin data missing (current=%s, prior=%s)",
            data.get("ticker"),
            gm_current,
            gm_prior,
        )
        return False, None

    expansion = gm_current - gm_prior
    return gm_current > gm_prior, expansion


# ---------------------------------------------------------------------------
# Filter 4 — Market capitalisation
# ---------------------------------------------------------------------------

def filter_market_cap(
    data: Dict[str, Any],
    min_cap: float = 2_000_000_000,    # $2 B
    max_cap: float = 50_000_000_000,   # $50 B
) -> bool:
    """Pass when market cap is between *min_cap* and *max_cap* (inclusive)."""
    mc = data.get("market_cap")
    if mc is None:
        logger.debug("%s: market_cap data missing", data.get("ticker"))
        return False
    return min_cap <= mc <= max_cap


# ---------------------------------------------------------------------------
# Filter 5 — Analyst coverage
# ---------------------------------------------------------------------------

def filter_analyst_coverage(
    data: Dict[str, Any],
    max_analysts: int = 20,
) -> bool:
    """Pass when the number of analysts with price targets is below *max_analysts*."""
    count = data.get("analyst_count")
    if count is None:
        # Missing analyst count is ambiguous; treat as fail to avoid false positives
        logger.debug("%s: analyst_count data missing", data.get("ticker"))
        return False
    return int(count) < max_analysts


# ---------------------------------------------------------------------------
# Filter 6 — Insider buying (net)
# ---------------------------------------------------------------------------

def filter_insider_buying(
    data: Dict[str, Any],
) -> Tuple[bool, float]:
    """
    Pass when the net insider dollar flow (buys − sells) is positive
    over the trailing 90 days.

    Returns (passed, net_dollar_value).
    """
    net = float(data.get("net_insider_buying", 0.0) or 0.0)
    return net > 0, net


# ---------------------------------------------------------------------------
# Filter 7 — P/E ratio vs sector median
# ---------------------------------------------------------------------------

def filter_pe_vs_sector(
    data: Dict[str, Any],
    sector_median_pe: Dict[str, float],
    multiplier: float = 2.0,
) -> Tuple[bool, Optional[float]]:
    """
    Pass when the stock's P/E is no more than *multiplier* × its sector median P/E.

    *sector_median_pe* maps sector name → median P/E across the full universe.

    Returns (passed, pe_ratio_vs_sector_median).
    If sector data is unavailable the stock passes through (conservative choice).
    """
    pe = data.get("pe_ratio")
    sector = data.get("sector")

    if pe is None or pe <= 0:
        logger.debug("%s: P/E data missing / invalid (%s)", data.get("ticker"), pe)
        return False, None

    median_pe = sector_median_pe.get(sector) if sector else None
    if median_pe is None or median_pe <= 0:
        # No sector comparison possible — pass through
        logger.debug(
            "%s: no sector median P/E for '%s', passing through filter 7",
            data.get("ticker"),
            sector,
        )
        return True, None

    ratio = pe / median_pe
    return ratio <= multiplier, ratio


# ---------------------------------------------------------------------------
# Watch-flag helper (near-miss detection)
# ---------------------------------------------------------------------------

WATCH_TOLERANCE = 0.10  # 10 % of threshold


def compute_watch_flags(
    data: Dict[str, Any],
    sector_median_pe: Dict[str, float],
) -> list[str]:
    """
    Return a list of human-readable strings describing filters the stock nearly
    missed (i.e., passed but within 10 % of the threshold).
    """
    flags: list[str] = []
    ticker = data.get("ticker", "?")

    # PEG near 1.5
    peg = data.get("peg_ratio")
    if peg and 1.5 * (1 - WATCH_TOLERANCE) <= peg < 1.5:
        flags.append(f"PEG near limit: {peg:.2f} (threshold 1.50)")

    # Market cap near lower bound ($2 B ± 10 %)
    mc = data.get("market_cap", 0) or 0
    if 2e9 <= mc <= 2e9 * (1 + WATCH_TOLERANCE):
        flags.append(f"Market cap near lower bound: ${mc/1e9:.2f}B (min $2.00B)")
    # Market cap near upper bound ($50 B ± 10 %)
    if 50e9 * (1 - WATCH_TOLERANCE) <= mc <= 50e9:
        flags.append(f"Market cap near upper bound: ${mc/1e9:.2f}B (max $50.00B)")

    # Analyst count near 20
    ac = data.get("analyst_count")
    if ac is not None and 20 * (1 - WATCH_TOLERANCE) <= ac < 20:
        flags.append(f"Analyst coverage near limit: {ac} analysts (max 19)")

    # P/E vs sector near 2×
    pe_rel = data.get("pe_ratio_vs_sector")
    if pe_rel is not None and 2.0 * (1 - WATCH_TOLERANCE) <= pe_rel <= 2.0:
        flags.append(f"P/E vs sector near 2× limit: {pe_rel:.2f}×")

    # Revenue acceleration very small
    rev_accel = data.get("revenue_acceleration")
    if rev_accel is not None and 0 < rev_accel < 0.01:  # < 1 pp acceleration
        flags.append(f"Revenue acceleration marginal: +{rev_accel:.2%}")

    # Gross margin expansion very small
    gm_exp = data.get("gm_expansion")
    if gm_exp is not None and 0 < gm_exp < 0.005:  # < 0.5 pp expansion
        flags.append(f"Gross margin expansion marginal: +{gm_exp:.2%}")

    # Net insider buying very small
    net_buy = data.get("net_insider_buying", 0) or 0
    if 0 < net_buy < 100_000:  # < $100 K
        flags.append(f"Insider buying minimal: ${net_buy:,.0f}")

    if flags:
        logger.debug("%s: watch flags: %s", ticker, flags)

    return flags
