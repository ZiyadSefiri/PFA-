import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASELINE_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
]

TARGET = "readmitted"
PREDICTION = "prediction"


def _stringify_diag(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["diag_1", "diag_2", "diag_3", "race", "max_glu_serum", "A1Cresult"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", pd.NA).fillna("missing")
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).replace("nan", pd.NA)
    return df


def load_baseline(path: str) -> pd.DataFrame:
    logger.info("Loading baseline data from %s", path)
    df = pd.read_csv(path)
    df = df.replace("?", pd.NA)
    df = _stringify_diag(df)

    df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"], errors="ignore")
    df = df.dropna(subset=["race", "diag_1", "diag_2", "diag_3", "gender"])
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df.reset_index(drop=True)

    for col in BASELINE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[BASELINE_COLS + [TARGET]]

    df[TARGET] = df[TARGET].apply(lambda x: 1 if x == "<30" else 0)

    logger.info("Baseline shape: %s rows x %s cols", df.shape[0], df.shape[1])
    return df


def load_current_from_logs(log_dir: str) -> pd.DataFrame:
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Inference log directory not found: {log_dir}")

    records = []
    for fpath in sorted(log_path.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    for inp, out in zip(entry["input"], entry["output"]):
                        inp[PREDICTION] = out["prediction"]
                        inp[TARGET] = None
                        records.append(inp)

    if not records:
        raise ValueError("No inference records found in log directory")

    df = pd.DataFrame(records)
    df = df.replace("?", pd.NA)
    df = _stringify_diag(df)

    for col in BASELINE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[BASELINE_COLS + [TARGET, PREDICTION]]
    logger.info("Current data shape: %s rows x %s cols", df.shape[0], df.shape[1])
    return df


def build_column_mapping() -> ColumnMapping:
    numerical_features = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
    ]
    categorical_features = [c for c in BASELINE_COLS if c not in numerical_features]

    return ColumnMapping(
        target=TARGET,
        prediction=PREDICTION,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )


def generate_reports(reference: pd.DataFrame, current: pd.DataFrame, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    column_mapping = build_column_mapping()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info("Generating Data Drift report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    drift_path = output_path / f"data_drift_report_{timestamp}.html"
    data_drift_report.save_html(str(drift_path))
    logger.info("Data Drift report saved to %s", drift_path)

    summary = data_drift_report.as_dict()
    drift_metrics = summary.get("metrics", [])
    logger.info("Drift summary: %s", json.dumps(drift_metrics, indent=2, default=str))

    current_with_target = current.dropna(subset=[TARGET])
    if len(current_with_target) > 0:
        logger.info("Generating Model Quality report (%s labeled records)...", len(current_with_target))
        quality_report = Report(metrics=[ClassificationPreset()])
        quality_report.run(reference_data=reference, current_data=current_with_target, column_mapping=column_mapping)
        quality_path = output_path / f"model_quality_report_{timestamp}.html"
        quality_report.save_html(str(quality_path))
        logger.info("Model Quality report saved to %s", quality_path)
    else:
        logger.info("Skipping Model Quality report: no labeled data in current window")


def main():
    parser = argparse.ArgumentParser(description="Evidently AI Drift & Quality Monitor")
    parser.add_argument("--baseline", required=True, help="Path to baseline training CSV")
    parser.add_argument("--log-dir", required=True, help="Directory with inference JSONL logs")
    parser.add_argument("--output-dir", default="./reports", help="Directory for output reports")
    args = parser.parse_args()

    reference = load_baseline(args.baseline)
    current = load_current_from_logs(args.log_dir)
    generate_reports(reference, current, args.output_dir)


if __name__ == "__main__":
    main()
