#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime, timedelta

from consumer.db import query_range
from monitoring.drift_detector import load_baseline, load_current_from_db, generate_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WINDOW_HOURS = 24


def main():
    parser = argparse.ArgumentParser(description="Short-term Evidently drift detection")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", default="/data/reports")
    parser.add_argument("--window-hours", type=int, default=WINDOW_HOURS)
    args = parser.parse_args()

    now = datetime.utcnow()
    start = (now - timedelta(hours=args.window_hours)).isoformat()
    end = now.isoformat()

    logger.info("Short-term window: %s to %s", start, end)
    df = query_range(start, end)
    if df.empty:
        logger.warning("No inference records in the last %d hours", args.window_hours)
        return

    reference = load_baseline(args.baseline)
    current = load_current_from_db(df)
    generate_reports(reference, current, args.output_dir, tag="short")


if __name__ == "__main__":
    main()
