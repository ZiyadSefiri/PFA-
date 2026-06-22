#!/usr/bin/env python3
import os
from pathlib import Path

import streamlit as st
import pandas as pd

from consumer.db import query_latest, query_range

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/data/reports"))
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/data/duckdb/inference.duckdb")

st.set_page_config(page_title="Drift Monitor", layout="wide")
st.title("Diabetic Readmission — Drift & Monitoring Dashboard")

col1, col2, col3 = st.columns(3)
try:
    latest = query_latest(1)
    total = len(query_range("1970-01-01", "2100-01-01")) if False else 0
except Exception:
    latest = pd.DataFrame()
    total = 0

# Quick stats
stats_placeholder = st.empty()

try:
    total_df = query_range("1970-01-01", "2100-01-01")
    total_records = len(total_df)
    latest_ts = total_df["ts"].max() if not total_df.empty else "—"
    col1.metric("Total Inferences", total_records)
    col2.metric("Latest Inference", str(latest_ts)[:19] if latest_ts != "—" else "—")
    col3.metric("DB Size", f"{Path(str(DUCKDB_PATH)).stat().st_size / 1024:.0f} KB" if Path(str(DUCKDB_PATH)).exists() else "N/A")
except Exception as e:
    st.warning(f"Cannot query DuckDB: {e}")

st.divider()

# Report browser
st.subheader("Evidently Reports")
reports = sorted(REPORTS_DIR.glob("*.html")) if REPORTS_DIR.exists() else []

if reports:
    selected = st.selectbox("Select a report", [r.name for r in reports], index=len(reports) - 1)
    selected_path = REPORTS_DIR / selected
    with open(selected_path) as f:
        st.components.v1.html(f.read(), height=800, scrolling=True)
else:
    st.info("No Evidently reports generated yet. Run a drift detection job first.")

st.divider()
st.caption("Architecture: FastAPI → Redpanda → Consumer → DuckDB → Evidently (short/long) → Reports")
