"""
data.py — All yfinance and SEC EDGAR data fetching for Edgeware.

Provides:
  get_sp500_tickers()        → list of {ticker, company, sector} dicts
  get_stock_data(ticker)     → dict of financial metrics (or None on failure)
  get_insider_transactions() → {net_buying, transactions} for a ticker
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EDGAR rate-limiting: SEC allows max 10 req/s; we stay well under that
# ---------------------------------------------------------------------------
_EDGAR_MIN_INTERVAL = 0.12  # seconds between EDGAR requests
_last_edgar_request: float = 0.0

EDGAR_HEADERS = {
    "User-Agent": os.getenv(
        "EDGAR_USER_AGENT",
        "Edgeware Stock Screener edgeware@edgeware.local",
    ),
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html, application/xml",
}

# Cached CIK map: ticker (upper) → int CIK
_cik_map: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edgar_get(url: str, **kwargs) -> Optional[requests.Response]:
    """Rate-limited GET wrapper for all EDGAR requests."""
    global _last_edgar_request
    elapsed = time.monotonic() - _last_edgar_request
    if elapsed < _EDGAR_MIN_INTERVAL:
        time.sleep(_EDGAR_MIN_INTERVAL - elapsed)
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20, **kwargs)
        _last_edgar_request = time.monotonic()
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.debug("EDGAR request failed %s: %s", url, exc)
        _last_edgar_request = time.monotonic()
        return None


def _load_cik_map() -> Dict[str, int]:
    """
    Fetch the EDGAR company_tickers.json once and return a ticker→CIK dict.
    Results are cached in the module-level _cik_map.
    """
    global _cik_map
    if _cik_map:
        return _cik_map
    logger.info("Loading EDGAR CIK map …")
    resp = _edgar_get("https://www.sec.gov/files/company_tickers.json")
    if resp is None:
        logger.error("Failed to load EDGAR CIK map")
        return {}
    data = resp.json()
    _cik_map = {
        entry["ticker"].upper(): int(entry["cik_str"])
        for entry in data.values()
    }
    logger.info("Loaded %d CIK entries", len(_cik_map))
    return _cik_map


def _safe_float(value) -> Optional[float]:
    """Return float or None for any value (handles NaN, None, etc.)."""
    try:
        f = float(value)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None


def _find_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Return the first matching row from a DataFrame by index label."""
    for label in candidates:
        if label in df.index:
            return df.loc[label]
    return None


# ---------------------------------------------------------------------------
# S&P 500 universe  — three-tier fetch
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Finviz rate-limiting — 1–3 s random delay between all requests
# ---------------------------------------------------------------------------

_FINVIZ_MIN_DELAY = 1.0
_FINVIZ_MAX_DELAY = 3.0

_FINVIZ_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    # Note: deliberately omit Accept-Encoding — sending 'gzip, deflate, br' causes
    # Finviz to return a stripped 40 KB page that lacks the snapshot table.
    # Without the header, requests negotiates encoding itself and receives the full page.
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _finviz_delay() -> None:
    """Sleep a random 1–3 s to respect Finviz rate limits."""
    time.sleep(random.uniform(_FINVIZ_MIN_DELAY, _FINVIZ_MAX_DELAY))


# ---------------------------------------------------------------------------
# S&P 500 / S&P 400 universe constants
# ---------------------------------------------------------------------------

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
# iShares IVV holdings CSV (may require a browser session; used as last attempt in tier 2)
IVV_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
)

# Open-data S&P 500 CSV maintained by the 'datasets' project on GitHub
# Format: Symbol,Name,Sector (no metadata header rows, pure CSV)
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
    "/main/data/constituents.csv"
)

# Hardcoded S&P 500 constituents (~Q2 2025).
# Used only when all live sources fail. Missing a handful of recent adds/drops
# is acceptable — yfinance returns None for invalid tickers gracefully.
_SP500_HARDCODED_TICKERS: List[str] = [
    # A
    "A",    "AAL",  "AAPL", "ABBV", "ABNB", "ABT",  "ACGL", "ACN",  "ADBE", "ADI",
    "ADM",  "ADP",  "ADSK", "AEE",  "AEP",  "AES",  "AFL",  "AIG",  "AIZ",  "AJG",
    "AKAM", "ALB",  "ALGN", "ALL",  "ALLE", "AMAT", "AMCR", "AMD",  "AME",  "AMGN",
    "AMP",  "AMT",  "AMZN", "ANET", "AON",  "AOS",  "APA",  "APD",  "APH",  "APO",
    "APTV", "ARE",  "ATO",  "AVB",  "AVGO", "AVY",  "AWK",  "AXON", "AXP",  "AZO",
    # B
    "BA",   "BAC",  "BALL", "BAX",  "BBY",  "BDX",  "BEN",  "BF-B", "BG",   "BIIB",
    "BK",   "BKNG", "BKR",  "BLDR", "BLK",  "BMY",  "BR",   "BRK-B","BSX",  "BWA",
    "BX",   "BXP",
    # C
    "C",    "CAG",  "CAH",  "CARR", "CAT",  "CB",   "CBOE", "CBRE", "CCI",  "CCL",
    "CDNS", "CDW",  "CE",   "CEG",  "CF",   "CFG",  "CHD",  "CHRW", "CHTR", "CI",
    "CINF", "CL",   "CLX",  "CMA",  "CMCSA","CME",  "CMG",  "CMI",  "CMS",  "CNC",
    "CNP",  "COF",  "COO",  "COP",  "COR",  "COST", "CPAY", "CPB",  "CPRT", "CPT",
    "CRL",  "CRM",  "CRWD", "CSCO", "CSGP", "CSX",  "CTAS", "CTLT", "CTSH", "CTVA",
    "CVS",  "CVX",  "CZR",
    # D
    "D",    "DAL",  "DAY",  "DD",   "DE",   "DECK", "DELL", "DEI",  "DG",   "DGX",
    "DHI",  "DHR",  "DIS",  "DLR",  "DLTR", "DOC",  "DOV",  "DOW",  "DPZ",  "DRI",
    "DTE",  "DUK",  "DVA",  "DVN",  "DXCM",
    # E
    "EA",   "EBAY", "ECL",  "ED",   "EFX",  "EG",   "EIX",  "EL",   "ELV",  "EMN",
    "EMR",  "ENPH", "EOG",  "EPAM", "EQIX", "EQR",  "EQT",  "ES",   "ESS",  "ETN",
    "ETR",  "EVRG", "EW",   "EXC",  "EXPD", "EXPE", "EXR",
    # F
    "F",    "FANG", "FAST", "FCX",  "FDS",  "FDX",  "FE",   "FFIV", "FI",   "FICO",
    "FIS",  "FITB", "FLT",  "FMC",  "FOX",  "FOXA", "FRT",  "FSLR", "FTNT", "FTV",
    # G
    "GD",   "GE",   "GEHC", "GEN",  "GEV",  "GILD", "GIS",  "GL",   "GLW",  "GM",
    "GNRC", "GOOG", "GOOGL","GPC",  "GPN",  "GRMN", "GS",   "GWW",
    # H
    "HAL",  "HAS",  "HBAN", "HCA",  "HD",   "HES",  "HIG",  "HII",  "HLT",  "HOLX",
    "HON",  "HPE",  "HPQ",  "HRL",  "HSIC", "HST",  "HSY",  "HUBB", "HUM",  "HWM",
    # I
    "IBM",  "ICE",  "IDXX", "IEX",  "IFF",  "ILMN", "INCY", "INTC", "INTU", "INVH",
    "IP",   "IPG",  "IQV",  "IR",   "IRM",  "ISRG", "IT",   "ITW",  "IVZ",
    # J
    "J",    "JBHT", "JCI",  "JKHY", "JNJ",  "JNPR", "JPM",
    # K
    "K",    "KDP",  "KEY",  "KEYS", "KHC",  "KIM",  "KLAC", "KMB",  "KMI",  "KMX",
    "KO",   "KR",   "KVUE",
    # L
    "L",    "LDOS", "LEN",  "LH",   "LHX",  "LIN",  "LKQ",  "LLY",  "LMT",  "LNT",
    "LOW",  "LRCX", "LULU", "LUV",  "LVS",  "LW",   "LYB",  "LYV",
    # M
    "MA",   "MAA",  "MAR",  "MAS",  "MCD",  "MCHP", "MCK",  "MCO",  "MDLZ", "MDT",
    "MET",  "META", "MGM",  "MHK",  "MKC",  "MKTX", "MLM",  "MMC",  "MMM",  "MNST",
    "MO",   "MOH",  "MOS",  "MPC",  "MPWR", "MRK",  "MRNA", "MRO",  "MS",   "MSCI",
    "MSFT", "MSI",  "MTB",  "MTD",  "MU",
    # N
    "NCLH", "NEE",  "NEM",  "NFLX", "NI",   "NKE",  "NOC",  "NOW",  "NRG",  "NSC",
    "NTAP", "NTRS", "NUE",  "NVDA", "NVR",  "NWS",  "NWSA", "NXPI",
    # O
    "O",    "ODFL", "OKE",  "OMC",  "ON",   "ORCL", "ORLY", "OTIS", "OXY",
    # P
    "PAYC", "PAYX", "PCAR", "PCG",  "PEG",  "PEP",  "PFE",  "PFG",  "PG",   "PGR",
    "PH",   "PHM",  "PKG",  "PLD",  "PLTR", "PM",   "PNC",  "PNR",  "PNW",  "PODD",
    "POOL", "PPG",  "PPL",  "PRU",  "PSA",  "PSX",  "PTC",  "PWR",  "PYPL",
    # Q
    "QCOM",
    # R
    "RCL",  "REG",  "REGN", "RF",   "RJF",  "RL",   "RMD",  "ROK",  "ROL",  "ROP",
    "ROST", "RSG",  "RTX",  "RVTY",
    # S
    "SBAC", "SBUX", "SCHW", "SHW",  "SJM",  "SLB",  "SMCI", "SNA",  "SNPS", "SO",
    "SOLV", "SPG",  "SPGI", "SRE",  "STE",  "STLD", "STT",  "STX",  "STZ",  "SWK",
    "SWKS", "SYF",  "SYK",  "SYY",
    # T
    "T",    "TAP",  "TDG",  "TDY",  "TECH", "TEL",  "TFC",  "TFX",  "TGT",  "TJX",
    "TMO",  "TMUS", "TPR",  "TRGP", "TRMB", "TROW", "TRV",  "TSCO", "TSLA", "TSN",
    "TT",   "TTWO", "TXN",  "TXT",  "TYL",
    # U
    "UAL",  "UBER", "UDR",  "UHS",  "ULTA", "UNH",  "UNP",  "UPS",  "URI",  "USB",
    # V
    "V",    "VICI", "VLO",  "VLTO", "VMC",  "VRSK", "VRSN", "VRTX", "VST",  "VTR",
    "VTRS", "VZ",
    # W
    "WAB",  "WAT",  "WBA",  "WBD",  "WDC",  "WEC",  "WELL", "WFC",  "WHR",  "WM",
    "WMB",  "WMT",  "WRB",  "WRK",  "WST",  "WTW",  "WY",   "WYNN",
    # X
    "XEL",  "XOM",  "XYL",
    # Y
    "YUM",
    # Z
    "ZBH",  "ZBRA", "ZTS",
]


def _fetch_sp500_wikipedia() -> List[Dict[str, str]]:
    """
    Tier 1 — Wikipedia via pd.read_html with a browser User-Agent.
    Returns list of {ticker, company, sector} or [] on any failure.
    """
    from io import StringIO

    try:
        resp = requests.get(
            SP500_WIKI_URL,
            headers={"User-Agent": _BROWSER_UA},
            timeout=30,
        )
        resp.raise_for_status()
        # pandas 2.x requires a file-like object, not a raw HTML string
        tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
        if not tables:
            raise ValueError("constituents table not found in Wikipedia HTML")
        df = tables[0]
        result: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            # Column names vary slightly by Wikipedia edit; try common variants
            ticker = str(row.get("Symbol", row.get("Ticker", ""))).strip().replace(".", "-")
            company = str(row.get("Security", row.get("Company", ""))).strip()
            sector = str(row.get("GICS Sector", row.get("Sector", ""))).strip()
            if ticker and ticker.lower() not in ("nan", ""):
                result.append({"ticker": ticker, "company": company, "sector": sector})
        if len(result) < 400:
            raise ValueError(f"Only {len(result)} tickers parsed — table may have changed")
        logger.info("Tier 1 (Wikipedia): loaded %d S&P 500 tickers", len(result))
        return result
    except Exception as exc:
        logger.warning("Tier 1 (Wikipedia) S&P 500 fetch failed: %s", exc)
        return []


def _fetch_sp500_csv() -> List[Dict[str, str]]:
    """
    Tier 2 — CSV-based fallback; tries two sources in order:

    2a. GitHub open-data CSV (datasets/s-and-p-500-companies)
        Format: Symbol,Name,Sector  — no metadata rows, plain CSV.

    2b. iShares IVV holdings CSV
        Has metadata rows before the real header; we scan for the line
        containing both 'Ticker' and 'Name'.
        Note: iShares may gate this behind a browser session, in which
        case the response body is HTML and this attempt silently fails.

    Returns list of {ticker, company, sector} or [] if both sources fail.
    """
    from io import StringIO

    # ── 2a: GitHub datasets CSV ──────────────────────────────────────────
    try:
        resp = requests.get(
            SP500_CSV_URL,
            headers={"User-Agent": _BROWSER_UA},
            timeout=20,
        )
        resp.raise_for_status()
        if resp.text.lstrip().startswith("<"):
            raise ValueError("Got HTML instead of CSV from GitHub URL")
        df = pd.read_csv(StringIO(resp.text))
        result: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            # Column names: Symbol, Name, Sector
            ticker = str(row.get("Symbol", row.get("Ticker", ""))).strip().replace(".", "-")
            if not ticker or ticker.lower() in ("nan", ""):
                continue
            company = str(row.get("Name", row.get("Security", ""))).strip()
            sector = str(row.get("Sector", row.get("GICS Sector", ""))).strip()
            result.append({"ticker": ticker, "company": company, "sector": sector})
        if len(result) < 400:
            raise ValueError(f"Only {len(result)} tickers from GitHub CSV — unexpected format")
        logger.info("Tier 2a (GitHub CSV): loaded %d S&P 500 tickers", len(result))
        return result
    except Exception as exc:
        logger.warning("Tier 2a (GitHub CSV) fetch failed: %s", exc)

    # ── 2b: iShares IVV holdings CSV ─────────────────────────────────────
    try:
        resp = requests.get(
            IVV_HOLDINGS_URL,
            headers={
                "User-Agent": _BROWSER_UA,
                "Referer": "https://www.ishares.com/us/products/239726/",
                "Accept": "text/csv,*/*",
            },
            timeout=30,
        )
        resp.raise_for_status()
        if resp.text.lstrip().startswith("<"):
            raise ValueError("Got HTML instead of CSV from IVV URL (session gate)")

        lines = resp.text.splitlines()
        header_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if "Ticker" in line and "Name" in line:
                header_idx = i
                break
        if header_idx is None:
            raise ValueError("Could not find header row in IVV CSV")

        df = pd.read_csv(
            StringIO("\n".join(lines[header_idx:])),
            on_bad_lines="skip",
        )
        result = []
        for _, row in df.iterrows():
            ticker = str(row.get("Ticker", "")).strip().replace(".", "-")
            if not ticker or ticker.upper() in ("NAN", "-", "CASH_USD", "USD", ""):
                continue
            asset_class = str(row.get("Asset Class", "Equity")).strip().lower()
            if asset_class not in ("equity", ""):
                continue
            company = str(row.get("Name", "")).strip()
            sector = str(row.get("Sector", "")).strip()
            result.append({"ticker": ticker, "company": company, "sector": sector})
        if len(result) < 400:
            raise ValueError(f"Only {len(result)} equity tickers parsed from IVV CSV")
        logger.info("Tier 2b (IVV CSV): loaded %d S&P 500 tickers", len(result))
        return result
    except Exception as exc:
        logger.warning("Tier 2b (IVV CSV) fetch failed: %s", exc)

    return []


def _sp500_hardcoded() -> List[Dict[str, str]]:
    """
    Tier 3 — static fallback list compiled ~Q2 2025.
    Sector/company fields are left blank; yfinance enriches them during fetch.
    """
    logger.warning(
        "Tier 3 (hardcoded): using static S&P 500 list — "
        "may not reflect the latest index additions/deletions"
    )
    return [{"ticker": t, "company": "", "sector": ""} for t in _SP500_HARDCODED_TICKERS]


def get_sp500_tickers() -> List[Dict[str, str]]:
    """
    Return the S&P 500 constituent universe as a list of
    {ticker, company, sector} dicts, trying sources in order:

      1. Wikipedia  (pd.read_html + browser User-Agent)
      2. CSV fallback — GitHub open-data CSV, then iShares IVV holdings CSV
      3. Hardcoded list of ~500 tickers embedded in this module

    The pipeline can therefore always start even when both live sources
    are unreachable (e.g. in a restricted network environment).
    """
    for fetcher in (_fetch_sp500_wikipedia, _fetch_sp500_csv):
        result = fetcher()
        if result:
            return result
    return _sp500_hardcoded()


# ---------------------------------------------------------------------------
# S&P 400 MidCap universe
# ---------------------------------------------------------------------------

def get_sp400_tickers() -> List[Dict[str, str]]:
    """
    Fetch the S&P 400 MidCap constituent list from Wikipedia using
    pd.read_html with a browser User-Agent (same approach as SP500).

    Returns a list of {ticker, company, sector} dicts, or [] on failure.
    """
    from io import StringIO

    try:
        resp = requests.get(
            SP400_WIKI_URL,
            headers={"User-Agent": _BROWSER_UA},
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
        if not tables:
            raise ValueError("constituents table not found on S&P 400 Wikipedia page")
        df = tables[0]
        result: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            ticker = str(row.get("Symbol", row.get("Ticker", ""))).strip().replace(".", "-")
            company = str(row.get("Security", row.get("Company", ""))).strip()
            sector = str(row.get("GICS Sector", row.get("Sector", ""))).strip()
            if ticker and ticker.lower() not in ("nan", ""):
                result.append({"ticker": ticker, "company": company, "sector": sector})
        if len(result) < 300:
            raise ValueError(
                f"Only {len(result)} SP400 tickers parsed — table may have changed"
            )
        logger.info("S&P 400 (Wikipedia): loaded %d tickers", len(result))
        return result
    except Exception as exc:
        logger.warning("S&P 400 Wikipedia fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Finviz deep-universe screener
# ---------------------------------------------------------------------------

FINVIZ_SCREENER_URL = "https://finviz.com/screener.ashx"
# v=111: Overview view (Ticker, Company, Sector, Industry, Country, MCap, P/E, …)
# f=cap_smallover: market cap > ~$300 M
_FINVIZ_SCREENER_PARAMS = {"v": "111", "f": "cap_smallover"}


def _parse_finviz_screener_page(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract {ticker, company, sector} rows from a parsed Finviz screener page.
    Ticker links have class 'screener-link-primary'; the view-111 column order is:
    No. | Ticker | Company | Sector | Industry | Country | MCap | P/E | Price | …
    """
    result: List[Dict[str, str]] = []
    for link in soup.find_all("a", class_="screener-link-primary"):
        ticker = link.text.strip()
        if not ticker:
            continue
        row_el = link.find_parent("tr")
        if not row_el:
            result.append({"ticker": ticker, "company": "", "sector": ""})
            continue
        cells = row_el.find_all("td")
        company = cells[2].text.strip() if len(cells) > 2 else ""
        sector  = cells[3].text.strip() if len(cells) > 3 else ""
        result.append({"ticker": ticker, "company": company, "sector": sector})
    return result


def _parse_finviz_total(soup: BeautifulSoup) -> int:
    """Parse the 'Total: N' count displayed at the top of a Finviz screener page."""
    for node in soup.find_all(string=re.compile(r"Total:\s*\d+")):
        m = re.search(r"Total:\s*(\d[\d,]*)", node)
        if m:
            return int(m.group(1).replace(",", ""))
    return 0


def get_finviz_universe() -> List[Dict[str, str]]:
    """
    Scrape the Finviz screener for all stocks with market cap ≥ ~$300 M
    (cap_smallover filter) and return the complete list.

    Intended for quarterly --deep runs only.  Applies 1–3 s delays between
    page requests.  Continues past individual page failures so a single
    blocked page does not abort the entire scrape.

    Returns list of unique {ticker, company, sector} dicts.
    """
    all_tickers: List[Dict[str, str]] = []
    seen: set = set()

    try:
        # First page also reveals the total result count
        resp = requests.get(
            FINVIZ_SCREENER_URL,
            params={**_FINVIZ_SCREENER_PARAMS, "r": "1"},
            headers=_FINVIZ_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        total = _parse_finviz_total(soup)
        logger.info("Finviz screener: %d total stocks reported", total)

        for row in _parse_finviz_screener_page(soup):
            if row["ticker"] not in seen:
                seen.add(row["ticker"])
                all_tickers.append(row)

        # Paginate (20 rows per page; r= is the 1-based row offset)
        row_offset = 21
        while row_offset <= max(total, 1):
            _finviz_delay()
            try:
                resp = requests.get(
                    FINVIZ_SCREENER_URL,
                    params={**_FINVIZ_SCREENER_PARAMS, "r": str(row_offset)},
                    headers=_FINVIZ_HEADERS,
                    timeout=30,
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                page_rows = _parse_finviz_screener_page(soup)

                if not page_rows:
                    logger.info(
                        "Finviz screener: no rows at r=%d — reached end", row_offset
                    )
                    break

                new = sum(
                    1
                    for r in page_rows
                    if r["ticker"] not in seen
                    and (seen.add(r["ticker"]) or True)  # type: ignore[func-returns-value]
                    and all_tickers.append(r) is None  # type: ignore[func-returns-value]
                )
                logger.info(
                    "Finviz screener r=%d: +%d new tickers (total %d)",
                    row_offset,
                    new,
                    len(all_tickers),
                )
            except Exception as exc:
                logger.warning(
                    "Finviz screener page r=%d failed: %s — skipping", row_offset, exc
                )
            row_offset += 20

    except Exception as exc:
        logger.error("Finviz screener initial request failed: %s", exc)

    logger.info("Finviz deep universe: %d unique tickers scraped", len(all_tickers))
    return all_tickers


# ---------------------------------------------------------------------------
# yfinance stock data
# ---------------------------------------------------------------------------

def _get_quarterly_revenue(stock: yf.Ticker) -> List[float]:
    """
    Extract quarterly revenue values (newest first) from yfinance.
    Tries multiple attribute names for cross-version compatibility.
    Returns up to 4 values, or [] on failure.
    """
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            df = getattr(stock, attr, None)
            if df is None or df.empty:
                continue
            row = _find_row(
                df,
                ["Total Revenue", "TotalRevenue", "Revenue", "Net Revenue"],
            )
            if row is not None:
                values = [_safe_float(v) for v in row.dropna().values]
                # yfinance sorts columns newest→oldest (left→right)
                return [v for v in values if v is not None][:4]
        except Exception as exc:
            logger.debug("quarterly revenue error: %s", exc)
    return []


def _get_annual_gross_margin(
    stock: yf.Ticker,
    info: Dict,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (current_gross_margin, prior_year_gross_margin).
    Tries income statement first; falls back to info['grossMargins'].
    """
    for attr in ("income_stmt", "financials"):
        try:
            df = getattr(stock, attr, None)
            if df is None or df.empty or df.shape[1] < 2:
                continue
            rev_row = _find_row(df, ["Total Revenue", "TotalRevenue", "Revenue"])
            gp_row = _find_row(df, ["Gross Profit", "GrossProfit"])
            if rev_row is None or gp_row is None:
                continue

            def _margin(col_idx: int) -> Optional[float]:
                rev = _safe_float(rev_row.iloc[col_idx])
                gp = _safe_float(gp_row.iloc[col_idx])
                if rev and gp and rev != 0:
                    return gp / rev
                return None

            current = _margin(0)
            prior = _margin(1)
            if current is not None:
                return current, prior
        except Exception as exc:
            logger.debug("gross margin error: %s", exc)

    # Fallback to info dict
    return _safe_float(info.get("grossMargins")), None


FINVIZ_QUOTE_URL = "https://finviz.com/quote.ashx"


def get_finviz_data(ticker: str) -> Dict[str, Any]:
    """
    Scrape the Finviz quote page for a single ticker.

    Primary source for: pe_ratio, peg_ratio, eps_growth_next_5y.
    analyst_count is always returned as None (not in Finviz free snapshot).

    Any field that cannot be parsed is returned as None so the caller
    can fall back to yfinance for that specific field.
    Applies the standard 1–3 s Finviz delay before making the request.
    """
    empty: Dict[str, Any] = {
        "pe_ratio": None,
        "peg_ratio": None,
        "eps_growth_next_5y": None,
        "analyst_count": None,
    }

    def _parse_fv(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        s = val.strip()
        if s in ("-", "", "N/A"):
            return None
        try:
            if s.endswith("%"):
                return float(s[:-1]) / 100
            if s.upper().endswith("B"):
                return float(s[:-1]) * 1e9
            if s.upper().endswith("M"):
                return float(s[:-1]) * 1e6
            if s.upper().endswith("K"):
                return float(s[:-1]) * 1e3
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            return None

    try:
        _finviz_delay()
        resp = requests.get(
            FINVIZ_QUOTE_URL,
            params={"t": ticker},
            headers=_FINVIZ_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()

        # Detect bot-block / CAPTCHA pages
        text_lower = resp.text.lower()
        if "captcha" in text_lower or ("robot" in text_lower and len(resp.text) < 5000):
            logger.warning("%s: Finviz returned a bot-block page — skipping", ticker)
            return empty

        soup = BeautifulSoup(resp.text, "html.parser")

        # Snapshot table: pairs of label/value <td> cells
        metrics: Dict[str, str] = {}
        for table in soup.find_all("table", class_="snapshot-table2"):
            cells = table.find_all("td")
            for i in range(0, len(cells) - 1, 2):
                key = cells[i].text.strip()
                val = cells[i + 1].text.strip()
                if key:
                    metrics[key] = val

        if not metrics:
            logger.debug("%s: Finviz snapshot table empty or not found", ticker)
            return empty

        # Trailing P/E preferred; fall back to Forward P/E
        pe = _parse_fv(metrics.get("P/E")) or _parse_fv(metrics.get("Forward P/E"))
        peg = _parse_fv(metrics.get("PEG"))
        eps_5y = _parse_fv(metrics.get("EPS next 5Y"))

        logger.debug(
            "%s Finviz: P/E=%s  PEG=%s  EPS5Y=%s",
            ticker,
            metrics.get("P/E", "-"),
            metrics.get("PEG", "-"),
            metrics.get("EPS next 5Y", "-"),
        )
        return {
            "pe_ratio": pe,
            "peg_ratio": peg,
            "eps_growth_next_5y": eps_5y,
            "analyst_count": None,
        }

    except Exception as exc:
        logger.debug("%s: Finviz quote fetch failed: %s", ticker, exc)
        return empty


def get_stock_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch all financial metrics needed by Edgeware filters/scorer.

    Returns a dict on success or None if data is unusable.
    Errors are logged; exceptions are caught so the pipeline continues.
    """
    try:
        # ── yfinance (structural / revenue / gross-margin data) ───────────
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Sanity-check: bail early on empty yfinance responses
        if not info or info.get("quoteType") is None:
            logger.warning("%s: yfinance returned no usable info", ticker)
            return None

        # ── Finviz (primary source for valuation multiples) ───────────────
        fv = get_finviz_data(ticker)

        # P/E: Finviz primary; yfinance fallback (trailing preferred, then forward)
        pe_yf = _safe_float(info.get("trailingPE")) or _safe_float(info.get("forwardPE"))
        pe_ratio = fv["pe_ratio"] if fv["pe_ratio"] is not None else pe_yf

        # PEG: Finviz primary; yfinance fallback
        peg_yf = _safe_float(info.get("pegRatio"))
        peg_ratio = fv["peg_ratio"] if fv["peg_ratio"] is not None else peg_yf

        # Analyst count: Finviz free snapshot doesn't expose it; always yfinance
        analyst_count = info.get("numberOfAnalystOpinions")

        gm_current, gm_prior = _get_annual_gross_margin(stock, info)

        data: Dict[str, Any] = {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "market_cap": _safe_float(info.get("marketCap")),
            "pe_ratio": pe_ratio,
            "peg_ratio": peg_ratio,
            "analyst_count": analyst_count,
            "eps_growth_next_5y": fv["eps_growth_next_5y"],
            "gross_margin_current": gm_current,
            "gross_margin_prior": gm_prior,
            "revenue_quarters": _get_quarterly_revenue(stock),
            # Fields populated later in the pipeline
            "net_insider_buying": 0.0,
            "insider_transactions": [],
            "sector_median_pe": None,
            "revenue_acceleration": None,
            "gm_expansion": None,
            "pe_ratio_vs_sector": None,
            "watch_flags": [],
            "score": 0.0,
            "score_components": {},
        }
        return data

    except Exception as exc:
        logger.error("%s: unexpected error in get_stock_data: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# SEC EDGAR Form 4 (insider transactions)
# ---------------------------------------------------------------------------

def get_insider_transactions(
    ticker: str,
    days: int = 90,
) -> Dict[str, Any]:
    """
    Scrape SEC EDGAR Form 4 filings for *ticker* over the last *days* days.

    Returns:
        {
            "net_buying": float,            # net dollar value (buy − sell)
            "total_bought": float,          # gross $ purchased
            "total_sold": float,            # gross $ sold
            "transactions": List[dict],     # individual transactions
        }
    """
    empty = {"net_buying": 0.0, "total_bought": 0.0, "total_sold": 0.0, "transactions": []}

    cik_map = _load_cik_map()
    cik = cik_map.get(ticker.upper())
    if cik is None:
        logger.debug("%s: CIK not found in EDGAR map", ticker)
        return empty

    # ------------------------------------------------------------------
    # Step 1 — fetch submissions JSON for this company
    # ------------------------------------------------------------------
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    resp = _edgar_get(submissions_url)
    if resp is None:
        return empty

    try:
        data = resp.json()
    except Exception as exc:
        logger.debug("%s: failed to parse submissions JSON: %s", ticker, exc)
        return empty

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    if not forms:
        return empty

    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Step 2 — find Form 4 filings within the date window
    # ------------------------------------------------------------------
    form4_filings: List[Tuple[int, str, str]] = []
    for form, filing_date, accession, primary_doc in zip(
        forms, dates, accessions, primary_docs
    ):
        if form != "4":
            continue
        if filing_date < cutoff_str:
            continue
        form4_filings.append((cik, accession, primary_doc))

    if not form4_filings:
        logger.debug("%s: no Form 4 filings in last %d days", ticker, days)
        return empty

    # ------------------------------------------------------------------
    # Step 3 — fetch and parse each Form 4 XML
    # ------------------------------------------------------------------
    all_transactions: List[Dict[str, Any]] = []
    total_bought = 0.0
    total_sold = 0.0

    for cik_num, accession, primary_doc in form4_filings:
        accession_clean = accession.replace("-", "")
        xml_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_num}/{accession_clean}/{primary_doc}"
        )
        xml_resp = _edgar_get(xml_url)
        if xml_resp is None:
            continue

        try:
            # BeautifulSoup + lxml-xml handles both well-formed XML and some quirks
            soup = BeautifulSoup(xml_resp.content, "lxml-xml")
        except Exception:
            try:
                soup = BeautifulSoup(xml_resp.content, "xml")
            except Exception as exc:
                logger.debug("%s: XML parse error for %s: %s", ticker, accession, exc)
                continue

        txns = _parse_form4_xml(soup)
        for txn in txns:
            if txn["code"] == "A":
                total_bought += txn["value"]
            elif txn["code"] == "D":
                total_sold += txn["value"]
            all_transactions.append(txn)

    net_buying = total_bought - total_sold
    return {
        "net_buying": net_buying,
        "total_bought": total_bought,
        "total_sold": total_sold,
        "transactions": all_transactions,
    }


def _parse_form4_xml(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extract non-derivative stock transactions from a parsed Form 4 XML.

    Only counts A (acquired / open-market buy) and D (disposed / sale) codes.
    Option grants and other derivative transactions are excluded.
    """
    results: List[Dict[str, Any]] = []

    def _val(tag, parent) -> Optional[str]:
        """Find a <value> child inside *tag* which is inside *parent*."""
        el = parent.find(tag)
        if el is None:
            return None
        v = el.find("value")
        return v.text.strip() if v else None

    for txn in soup.find_all("nonDerivativeTransaction"):
        try:
            code = _val("transactionAcquiredDisposedCode", txn)
            if code not in ("A", "D"):
                continue

            shares_str = _val("transactionShares", txn)
            price_str = _val("transactionPricePerShare", txn)
            date_str = _val("transactionDate", txn)

            shares = _safe_float(shares_str) or 0.0
            price = _safe_float(price_str) or 0.0
            value = shares * price

            results.append(
                {
                    "code": code,
                    "shares": shares,
                    "price": price,
                    "value": value,
                    "date": date_str,
                }
            )
        except Exception as exc:
            logger.debug("Form 4 txn parse error: %s", exc)
            continue

    return results
