"""
report.py — Markdown report generation and Claude API thesis generation for Edgeware.

Exports:
    generate_report(top_stocks, date, pipeline_stats) → Path
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # must run before importing anthropic or reading ANTHROPIC_API_KEY

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

REPORT_DIR = Path(os.getenv("REPORT_DIR", "reports"))
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_THESIS_TOKENS = 350

# ---------------------------------------------------------------------------
# Claude API — investment thesis generation
# ---------------------------------------------------------------------------


def _build_thesis_prompt(stock: Dict[str, Any]) -> str:
    """Construct the prompt sent to Claude for thesis generation."""

    def _pct(v: Optional[float], decimals: int = 1) -> str:
        return f"{v * 100:.{decimals}f}%" if v is not None else "N/A"

    def _bn(v: Optional[float]) -> str:
        return f"${v / 1e9:.2f}B" if v is not None else "N/A"

    def _m(v: Optional[float]) -> str:
        return f"${v / 1e6:.1f}M" if v is not None else "N/A"

    lines = [
        f"Company: {stock['company_name']} ({stock['ticker']})",
        f"Sector: {stock.get('sector', 'N/A')}",
        f"Market Cap: {_bn(stock.get('market_cap'))}",
        f"P/E Ratio: {stock.get('pe_ratio', 'N/A')}",
        f"PEG Ratio: {stock.get('peg_ratio', 'N/A')}",
        f"Sector Median P/E: {round(stock.get('sector_median_pe') or 0, 1)}",
        f"P/E vs Sector: {round(stock.get('pe_ratio_vs_sector') or 0, 2)}x",
        f"Revenue Acceleration (QoQ delta): {_pct(stock.get('revenue_acceleration'), 2)}",
        f"Gross Margin (Current FY): {_pct(stock.get('gross_margin_current'), 1)}",
        f"Gross Margin Expansion YoY: {_pct(stock.get('gm_expansion'), 2)}",
        f"Net Insider Buying (90d): {_m(stock.get('net_insider_buying'))}",
        f"Analyst Coverage: {stock.get('analyst_count', 'N/A')} analysts",
        f"Composite Edgeware Score: {stock.get('score', 0):.1f}/100",
    ]

    metrics_block = "\n".join(f"  • {line}" for line in lines)

    return f"""You are a concise, data-driven equity analyst writing for a sophisticated audience.
Write exactly ONE paragraph (4–5 sentences, no bullet points) explaining why {stock['company_name']} ({stock['ticker']})
is an interesting investment opportunity based strictly on the quantitative evidence below.

Be specific: reference the actual numbers. Do NOT pad with generic disclaimers.
Focus on what is distinctive: accelerating revenue, margin expansion, insider conviction,
or valuation discount relative to sector peers.

Quantitative Evidence:
{metrics_block}

Write the thesis paragraph now (plain text, no markdown):"""


def generate_thesis(
    client: anthropic.Anthropic,
    stock: Dict[str, Any],
) -> str:
    """Call Claude and return a one-paragraph investment thesis string."""
    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_THESIS_TOKENS,
            messages=[
                {"role": "user", "content": _build_thesis_prompt(stock)}
            ],
        )
        thesis = message.content[0].text.strip()
        logger.debug("%s: thesis generated (%d chars)", stock["ticker"], len(thesis))
        return thesis
    except Exception as exc:
        logger.error("%s: Claude thesis generation failed: %s", stock["ticker"], exc)
        return (
            f"{stock['company_name']} passed all Edgeware screening criteria "
            f"(thesis generation failed: {exc})."
        )


# ---------------------------------------------------------------------------
# Markdown formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%" if v is not None else "—"


def _fmt_bn(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    return f"${v / 1e6:.0f}M"


def _fmt_m(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def _fmt_x(v: Optional[float]) -> str:
    return f"{v:.2f}×" if v is not None else "—"


def _metrics_table(stock: Dict[str, Any]) -> str:
    """Render a markdown metrics table for a single stock."""
    pe = stock.get("pe_ratio")
    peg = stock.get("peg_ratio")
    sector_pe = stock.get("sector_median_pe")
    pe_vs = stock.get("pe_ratio_vs_sector")
    gm_cur = stock.get("gross_margin_current")
    gm_prior = stock.get("gross_margin_prior")
    gm_exp = stock.get("gm_expansion")
    rev_accel = stock.get("revenue_acceleration")
    net_buy = stock.get("net_insider_buying")
    ac = stock.get("analyst_count")

    rows = [
        ("Market Cap", _fmt_bn(stock.get("market_cap"))),
        ("P/E Ratio (Trailing)", f"{pe:.1f}" if pe else "—"),
        ("Sector Median P/E", f"{sector_pe:.1f}" if sector_pe else "—"),
        ("P/E vs Sector", _fmt_x(pe_vs)),
        ("PEG Ratio", f"{peg:.2f}" if peg else "—"),
        ("Gross Margin (Current FY)", _fmt_pct(gm_cur)),
        ("Gross Margin (Prior FY)", _fmt_pct(gm_prior)),
        ("Gross Margin Expansion YoY", _fmt_pct(gm_exp, 2)),
        ("Revenue Acceleration (QoQ Δ)", _fmt_pct(rev_accel, 2)),
        ("Net Insider Buying (90d)", _fmt_m(net_buy)),
        ("Analyst Coverage", f"{ac}" if ac is not None else "—"),
    ]

    header = "| Metric | Value |\n|--------|-------|\n"
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return header + body


def _score_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for the Edgeware composite score."""
    filled = round(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"`[{bar}]` **{score:.1f}/100**"


def _stock_section(rank: int, stock: Dict[str, Any], thesis: str) -> str:
    """Build the full markdown section for one ranked stock."""
    ticker = stock["ticker"]
    name = stock.get("company_name", ticker)
    sector = stock.get("sector", "—")
    score = stock.get("score", 0)
    watch = stock.get("watch_flags", [])
    comps = stock.get("score_components", {})

    lines: list[str] = [
        f"---\n",
        f"### #{rank} — {ticker} | {name}",
        f"**Sector:** {sector} &nbsp;|&nbsp; **Edgeware Score:** {_score_bar(score)}\n",
        "#### Key Metrics\n",
        _metrics_table(stock),
        "",
        "#### Score Breakdown\n",
        f"| Dimension | Weight | Sub-score |",
        f"|-----------|--------|-----------|",
        f"| PEG Attractiveness | 30% | {comps.get('peg_score', 0):.1f} |",
        f"| Revenue Acceleration | 25% | {comps.get('accel_score', 0):.1f} |",
        f"| Gross Margin Expansion | 20% | {comps.get('gm_score', 0):.1f} |",
        f"| Insider Buying Conviction | 15% | {comps.get('insider_score', 0):.1f} |",
        f"| Relative P/E Discount | 10% | {comps.get('pe_score', 0):.1f} |",
        "",
        "#### Investment Thesis\n",
        f"> {thesis}",
        "",
    ]

    if watch:
        lines.append("#### ⚠️ Watch Flags\n")
        for flag in watch:
            lines.append(f"- {flag}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report entry point
# ---------------------------------------------------------------------------


def generate_report(
    top_stocks: List[Dict[str, Any]],
    date: Optional[datetime] = None,
    pipeline_stats: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Generate the full Edgeware markdown report.

    Args:
        top_stocks:     Ranked list (up to 10) from scorer.score_stocks().
        date:           Report date; defaults to today.
        pipeline_stats: Optional summary dict from the screener pipeline.

    Returns:
        Path to the generated report file.
    """
    if date is None:
        date = datetime.now()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{date.strftime('%Y-%m-%d')}.md"

    # ------------------------------------------------------------------
    # Initialise Claude client
    # ------------------------------------------------------------------
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — theses will be placeholder text")
    client = anthropic.Anthropic(api_key=api_key) if api_key else None

    # ------------------------------------------------------------------
    # Build report header
    # ------------------------------------------------------------------
    stats = pipeline_stats or {}
    universe_size = stats.get("universe", "S&P 500")
    passed_count = stats.get("passed", len(top_stocks))
    failed_count = stats.get("failed_fetch", 0)

    header_lines = [
        f"# 📊 Edgeware Stock Screener — {date.strftime('%B %d, %Y')}",
        "",
        f"*Weekly automated screen of the {universe_size} universe.*  ",
        f"*Stocks passing all 7 filters: **{passed_count}** · "
        f"Data-fetch failures: **{failed_count}***",
        "",
        "## Screening Criteria Applied",
        "",
        "| # | Filter | Threshold |",
        "|---|--------|-----------|",
        "| 1 | PEG Ratio | < 1.5 |",
        "| 2 | Revenue Growth Acceleration | Most-recent QoQ growth > prior QoQ |",
        "| 3 | Gross Margin Expansion | Current FY > Prior FY |",
        "| 4 | Market Cap | $2 B – $50 B |",
        "| 5 | Analyst Coverage | < 20 analysts |",
        "| 6 | Insider Activity | No significant net insider selling (> −$500K over 90 days) |",
        "| 7 | Relative P/E | ≤ 2× GICS sector median |",
        "",
        "## Scoring Weights",
        "",
        "| Dimension | Weight |",
        "|-----------|--------|",
        "| PEG Attractiveness | 30% |",
        "| Revenue Acceleration Magnitude | 25% |",
        "| Gross Margin Expansion Magnitude | 20% |",
        "| Insider Buying Conviction ($ value) | 15% |",
        "| Relative P/E Discount to Sector | 10% |",
        "",
        "---",
        "",
        "## Top 10 Ranked Stocks",
        "",
    ]

    # ------------------------------------------------------------------
    # Build each stock section (with Claude thesis)
    # ------------------------------------------------------------------
    stock_sections: list[str] = []
    for rank, stock in enumerate(top_stocks[:10], start=1):
        ticker = stock["ticker"]
        logger.info("Generating thesis for #%d %s …", rank, ticker)

        if client:
            thesis = generate_thesis(client, stock)
        else:
            thesis = (
                f"{stock.get('company_name', ticker)} passed all Edgeware filters. "
                "(Thesis generation skipped — ANTHROPIC_API_KEY not set.)"
            )

        stock_sections.append(_stock_section(rank, stock, thesis))

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    footer_lines = [
        "---",
        "",
        "## Methodology Notes",
        "",
        "- Financial data sourced from **yfinance** (Yahoo Finance).",
        "- Insider transaction data scraped from **SEC EDGAR Form 4** filings.",
        "- Sector median P/E computed across all S&P 500 constituents with valid data.",
        "- Scores are relative within the surviving cohort; they are not comparable "
        "across different run dates.",
        "- ⚠️ Watch flags indicate a stock passed but was within 10% of a filter threshold.",
        "",
        f"*Report generated by Edgeware on {date.strftime('%Y-%m-%d %H:%M:%S')}.*",
        "*This is not investment advice.*",
        "",
    ]

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    full_report = (
        "\n".join(header_lines)
        + "\n"
        + "\n".join(stock_sections)
        + "\n"
        + "\n".join(footer_lines)
    )

    report_path.write_text(full_report, encoding="utf-8")
    logger.info("Report written → %s", report_path)
    return report_path
