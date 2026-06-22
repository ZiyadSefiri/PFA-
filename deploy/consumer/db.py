import json
import os
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(os.getenv("DUCKDB_PATH", "/data/duckdb/inference.duckdb"))


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def init_db():
    con = _connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id BIGINT PRIMARY KEY,
            ts TIMESTAMP,
            input_json JSON,
            prediction INTEGER,
            probability_readmitted DOUBLE,
            probability_not_readmitted DOUBLE
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_id START 1")
    con.commit()
    con.close()


def insert_batch(records: list[tuple]):
    con = _connect()
    con.executemany(
        "INSERT INTO inference_logs VALUES (nextval('seq_id'), ?, ?, ?, ?, ?)",
        records,
    )
    con.commit()
    con.close()


def query_range(start_ts: str, end_ts: str) -> pd.DataFrame:
    con = _connect()
    df = con.execute(
        "SELECT * FROM inference_logs WHERE ts >= ? AND ts < ? ORDER BY ts",
        (start_ts, end_ts),
    ).fetchdf()
    con.close()
    return df


def query_latest(n: int) -> pd.DataFrame:
    con = _connect()
    df = con.execute(
        "SELECT * FROM inference_logs ORDER BY ts DESC LIMIT ?", (n,)
    ).fetchdf()
    con.close()
    return df
