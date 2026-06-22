import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.kafka_producer import init_producer, close_producer, send_inference
from app.model_loader import get_model, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetic Readmission Predictor",
    description="ML model inference for diabetic readmission risk",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("Loading model on startup...")
    try:
        get_model()
        logger.info("Model loaded successfully. Info: %s", get_model_info())
    except Exception as e:
        logger.error("Failed to load model: %s", e)

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
    try:
        await init_producer(bootstrap)
    except Exception as e:
        logger.warning("Kafka unavailable at %s — inference events will be dropped: %s", bootstrap, e)


@app.on_event("shutdown")
async def shutdown():
    await close_producer()


@app.get("/health")
async def health():
    info = get_model_info()
    return {"status": "ok", "model_uri": info.get("uri", "unknown")}


DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
DROP_COLS = ["encounter_id", "patient_nbr", "readmitted", "weight", "payer_code", "medical_specialty"]


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace("?", pd.NA)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    for col in DIAG_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", pd.NA).fillna("missing")

    for col in ["race", "max_glu_serum", "A1Cresult"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", pd.NA).fillna("missing")

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).replace("nan", pd.NA)
        df = df[df["gender"] != "Unknown/Invalid"]

    return df


@app.post("/predict")
async def predict(payload: dict):
    instances = payload.get("instances")
    if instances is None:
        raise HTTPException(status_code=400, detail="Missing 'instances' field")

    if not isinstance(instances, list) or len(instances) == 0:
        raise HTTPException(status_code=400, detail="'instances' must be a non-empty list")

    try:
        df = pd.DataFrame(instances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    raw_records = df.to_dict(orient="records")
    df = _preprocess(df)
    if df.empty:
        raise HTTPException(status_code=400, detail="All instances were filtered out during preprocessing")

    model = get_model()

    try:
        preds = model.predict(df)
        probs = model.predict_proba(df)

        results = []
        for i in range(len(df)):
            results.append({
                "prediction": int(preds[i]),
                "probability_readmitted": float(probs[i][1]),
                "probability_not_readmitted": float(probs[i][0]),
            })

        await send_inference(raw_records, results)

        return {"predictions": results}

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
