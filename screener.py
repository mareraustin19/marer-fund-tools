"""
screener.py — Main pipeline orchestrator for Edgeware.

Run directly:
    python screener.py

The pipeline:
  1. Fetch the S&P 500 universe from Wikipedia
  2. Fetch financial data via yfinance (skips bad tickers gracefully)
  3. Fetch insider transactions from SEC EDGAR Form 4
  4. Compute sector-level median P/E ratios
  5. Apply the 7 screening filters sequentially
  6. Score and rank survivors
  7. Generate a markdown report for the top 10
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

# Load .env before importing anything that reads env vars
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup (file + console)
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)-20s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("edgeware.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edgeware modules
# ---------------------------------------------------------------------------

from data import get_insider_transactions, get_sp500_tickers, get_stock_data
from filters import (
    compute_watch_flags,
    filter_analyst_coverage,
    filter_gross_margin_expansion,
    filter_insider_buying,
    filter_market_cap,
    filter_pe_vs_sector,
    filter_peg_ratio,
    filter_revenue_acceleration,
)
from report import generate_report
from scorer import score_stocks

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

MAX_FAIL_RATE = 0.20  # abort if more than 20% of fetches fail


def _compute_sector_median_pe(
    all_stocks: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute the median trailing P/E ratio per GICS sector across all stocks
    with valid P/E data.  Used for Filter 7.
    """
    sector_pes: Dict[str, list] = defaultdict(list)
    for s in all_stocks:
        pe = s.get("pe_ratio")
        sector = s.get("sector")
        if pe and pe > 0 and sector:
            sector_pes[sector].append(pe)

    medians = {
        sector: float(np.median(pes))
        for sector, pes in sector_pes.items()
        if pes
    }
    logger.info("Sector median P/E: %s", medians)
    return medians


def run_pipeline(
    *,
    universe_override: Optional[List[Dict[str, str]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Execute the full Edgeware screening pipeline.

    Args:
        universe_override: If provided, use this list instead of fetching
                           the S&P 500 from Wikipedia.  Useful for testing.

    Returns:
        Top-10 ranked stocks (or fewer if fewer survived), or None on fatal error.
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("Edgeware — pipeline started at %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1 — Universe                                                    #
    # ------------------------------------------------------------------ #
    universe = universe_override or get_sp500_tickers()
    if not universe:
        logger.error("Empty universe — aborting")
        return None
    logger.info("Universe: %d tickers", len(universe))

    # ------------------------------------------------------------------ #
    # Step 2 — Fetch financial data                                        #
    # ------------------------------------------------------------------ #
    all_stocks: List[Dict[str, Any]] = []
    failed_fetch: List[str] = []

    total = len(universe)
    for idx, entry in enumerate(universe, start=1):
        ticker = entry["ticker"]
        logger.info("[%d/%d] Fetching data for %s …", idx, total, ticker)

        stock = get_stock_data(ticker)
        if stock is None:
            failed_fetch.append(ticker)
            fail_rate = len(failed_fetch) / idx
            if fail_rate > MAX_FAIL_RATE and idx >= 50:
                logger.warning(
                    "Fetch failure rate %.0f%% exceeds %.0f%% safety threshold "
                    "but pipeline will continue.",
                    fail_rate * 100,
                    MAX_FAIL_RATE * 100,
                )
            continue

        # Enrich with Wikipedia sector when yfinance returns 'Unknown'
        if stock.get("sector") in (None, "Unknown", ""):
            stock["sector"] = entry.get("sector", "Unknown")
        if not stock.get("company_name") or stock["company_name"] == ticker:
            stock["company_name"] = entry.get("company", ticker)

        all_stocks.append(stock)

    fail_rate = len(failed_fetch) / total if total else 0
    logger.info(
        "Data fetch: %d OK, %d failed (%.1f%%)",
        len(all_stocks),
        len(failed_fetch),
        fail_rate * 100,
    )
    if failed_fetch:
        logger.info("Failed tickers: %s", ", ".join(failed_fetch[:30]))

    if not all_stocks:
        logger.error("No usable stock data retrieved — aborting")
        return None

    # ------------------------------------------------------------------ #
    # Step 3 — Insider transactions (EDGAR Form 4)                        #
    # ------------------------------------------------------------------ #
    logger.info("Fetching insider transactions from SEC EDGAR …")
    for idx, stock in enumerate(all_stocks, start=1):
        ticker = stock["ticker"]
        logger.info("[%d/%d] EDGAR Form 4 for %s …", idx, len(all_stocks), ticker)
        insider = get_insider_transactions(ticker, days=90)
        stock["net_insider_buying"] = insider["net_buying"]
        stock["insider_transactions"] = insider["transactions"]

    # ------------------------------------------------------------------ #
    # Step 4 — Sector median P/E                                          #
    # ------------------------------------------------------------------ #
    sector_median_pe = _compute_sector_median_pe(all_stocks)
    for stock in all_stocks:
        stock["sector_median_pe"] = sector_median_pe.get(stock.get("sector"))

    # ------------------------------------------------------------------ #
    # Step 5 — Apply filters                                               #
    # ------------------------------------------------------------------ #
    filter_stats: Dict[str, int] = {
        "passed_f1_peg": 0,
        "passed_f2_revenue": 0,
        "passed_f3_gm": 0,
        "passed_f4_mktcap": 0,
        "passed_f5_analysts": 0,
        "passed_f6_insider": 0,
        "passed_all": 0,
    }
    eliminated_at: Dict[str, int] = defaultdict(int)

    passed: List[Dict[str, Any]] = []

    for stock in all_stocks:
        ticker = stock["ticker"]

        # ── Filter 1: PEG ──────────────────────────────────────────────
        f1_pass, peg_val = filter_peg_ratio(stock)
        if not f1_pass:
            eliminated_at["F1 PEG"] += 1
            continue
        filter_stats["passed_f1_peg"] += 1

        # ── Filter 2: Revenue acceleration ────────────────────────────
        f2_pass, rev_accel = filter_revenue_acceleration(stock)
        if not f2_pass:
            eliminated_at["F2 Revenue Accel"] += 1
            continue
        stock["revenue_acceleration"] = rev_accel
        filter_stats["passed_f2_revenue"] += 1

        # ── Filter 3: Gross margin expansion ──────────────────────────
        f3_pass, gm_expansion = filter_gross_margin_expansion(stock)
        if not f3_pass:
            eliminated_at["F3 Gross Margin"] += 1
            continue
        stock["gm_expansion"] = gm_expansion
        filter_stats["passed_f3_gm"] += 1

        # ── Filter 4: Market cap ───────────────────────────────────────
        if not filter_market_cap(stock):
            eliminated_at["F4 Market Cap"] += 1
            continue
        filter_stats["passed_f4_mktcap"] += 1

        # ── Filter 5: Analyst coverage ─────────────────────────────────
        if not filter_analyst_coverage(stock):
            eliminated_at["F5 Analysts"] += 1
            continue
        filter_stats["passed_f5_analysts"] += 1

        # ── Filter 6: Insider buying ───────────────────────────────────
        f6_pass, net_buy = filter_insider_buying(stock)
        if not f6_pass:
            eliminated_at["F6 Insider"] += 1
            continue
        filter_stats["passed_f6_insider"] += 1

        # ── Filter 7: P/E vs sector median ────────────────────────────
        f7_pass, pe_ratio_vs_sector = filter_pe_vs_sector(stock, sector_median_pe)
        if not f7_pass:
            eliminated_at["F7 P/E Sector"] += 1
            continue

        stock["pe_ratio_vs_sector"] = pe_ratio_vs_sector
        stock["watch_flags"] = compute_watch_flags(stock, sector_median_pe)
        filter_stats["passed_all"] += 1
        passed.append(stock)
        logger.info("✓ %s passed all 7 filters", ticker)

    # Log funnel
    logger.info("-" * 50)
    logger.info("Filter funnel (from %d stocks with valid data):", len(all_stocks))
    for stage, count in filter_stats.items():
        logger.info("  %s: %d", stage, count)
    logger.info("Eliminated at each filter:")
    for filt, count in sorted(eliminated_at.items()):
        logger.info("  %s: %d eliminated", filt, count)
    logger.info("-" * 50)

    if not passed:
        logger.warning("No stocks survived all 7 filters — report will be empty")
        generate_report(
            [],
            date=start_time,
            pipeline_stats={
                "universe": f"S&P 500 ({len(universe)} tickers)",
                "passed": 0,
                "failed_fetch": len(failed_fetch),
            },
        )
        return []

    # ------------------------------------------------------------------ #
    # Step 6 — Score and rank                                             #
    # ------------------------------------------------------------------ #
    ranked = score_stocks(passed)
    top_10 = ranked[:10]

    logger.info("Top 10:")
    for i, s in enumerate(top_10, 1):
        logger.info("  #%d  %-6s  %.1f/100", i, s["ticker"], s["score"])

    # ------------------------------------------------------------------ #
    # Step 7 — Generate report                                            #
    # ------------------------------------------------------------------ #
    report_path = generate_report(
        top_10,
        date=start_time,
        pipeline_stats={
            "universe": f"S&P 500 ({len(universe)} tickers)",
            "passed": len(passed),
            "failed_fetch": len(failed_fetch),
        },
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info(
        "Pipeline complete in %.0fs — report: %s",
        elapsed,
        report_path,
    )
    logger.info("=" * 70)

    return top_10


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_pipeline()
    if result is None:
        sys.exit(1)
    sys.exit(0)
