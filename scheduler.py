"""
scheduler.py — Runs the Edgeware pipeline every Monday at 07:00 local time.

Usage:
    python scheduler.py           # runs indefinitely
    python scheduler.py --once    # run immediately once, then exit (useful for testing)
    python scheduler.py --at HH:MM  # override the scheduled time

The process logs all pipeline output to edgeware.log (set up in screener.py).
Keep it alive with systemd, launchd, Docker, or a screen/tmux session.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import schedule
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


def _run_pipeline_job() -> None:
    """Wrapper that the scheduler calls every Monday."""
    logger.info(
        "Scheduler trigger: starting Edgeware pipeline at %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    try:
        # Import here so screener logging is initialised first
        from screener import run_pipeline

        result = run_pipeline()
        if result is None:
            logger.error("Pipeline returned None — check edgeware.log for details")
        else:
            logger.info(
                "Pipeline finished successfully: %d stock(s) in top 10", len(result)
            )
    except Exception as exc:
        # Catch-all: scheduler must not die on pipeline errors
        logger.exception("Uncaught exception in pipeline job: %s", exc)


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------


def start_scheduler(run_time: str = "07:00") -> None:
    """
    Block forever, running the pipeline every Monday at *run_time*.

    Args:
        run_time: HH:MM string in local time.
    """
    logger.info(
        "Edgeware scheduler started — pipeline will run every Monday at %s",
        run_time,
    )

    schedule.every().monday.at(run_time).do(_run_pipeline_job)

    # Show next scheduled run
    next_run = schedule.next_run()
    logger.info("Next scheduled run: %s", next_run)

    while True:
        schedule.run_pending()
        time.sleep(30)  # check every 30 seconds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edgeware Stock Screener — weekly scheduler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the pipeline immediately once and exit (skips the scheduler).",
    )
    parser.add_argument(
        "--at",
        default="07:00",
        metavar="HH:MM",
        help="Time of day to run the pipeline each Monday (24-hour format).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Logging must be initialised before importing screener
    import logging
    import os

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

    args = _parse_args()

    if args.once:
        logger.info("--once flag set: running pipeline immediately")
        _run_pipeline_job()
        sys.exit(0)

    try:
        start_scheduler(run_time=args.at)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (KeyboardInterrupt)")
        sys.exit(0)
