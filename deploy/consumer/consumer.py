#!/usr/bin/env python3
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime

from kafka import KafkaConsumer

from consumer.db import init_db, insert_batch

BATCH_SIZE = int(os.getenv("CONSUMER_BATCH_SIZE", "100"))
FLUSH_INTERVAL = int(os.getenv("CONSUMER_FLUSH_INTERVAL", "5"))
TOPIC = os.getenv("KAFKA_TOPIC", "inference-events")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")

logger = logging.getLogger(__name__)
_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Received signal %d, shutting down...", signum)
    _running = False


def _flush(buffer: deque):
    if not buffer:
        return
    records = list(buffer)
    insert_batch(records)
    logger.info("Flushed %d records to DuckDB", len(records))
    buffer.clear()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    init_db()
    logger.info("DuckDB initialized at %s", os.getenv("DUCKDB_PATH", "/data/duckdb/inference.duckdb"))

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode()),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        session_timeout_ms=30000,
        heartbeat_interval_ms=10000,
    )

    logger.info("Kafka consumer subscribed to %s (broker: %s)", TOPIC, BOOTSTRAP_SERVERS)

    buffer = deque()
    last_flush = time.monotonic()

    while _running:
        msg_set = consumer.poll(timeout_ms=1000)
        now = time.monotonic()

        for _tp, messages in msg_set.items():
            for msg in messages:
                val = msg.value
                ts = datetime.fromtimestamp(msg.timestamp / 1000.0)
                buffer.append((
                    ts,
                    json.dumps(val.get("input", {})),
                    val.get("prediction"),
                    val.get("probability_readmitted"),
                    val.get("probability_not_readmitted"),
                ))

        if len(buffer) >= BATCH_SIZE or (buffer and (now - last_flush) >= FLUSH_INTERVAL):
            _flush(buffer)
            last_flush = now

    _flush(buffer)
    consumer.close()
    logger.info("Consumer shut down gracefully")


if __name__ == "__main__":
    main()
