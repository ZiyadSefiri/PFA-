# Diabetic Readmission Prediction — MLOps Pipeline

End-to-end machine learning pipeline for predicting hospital readmission within 30 days (diabetic patients). Features MLflow experiment tracking, containerized FastAPI inference, Redpanda/Kafka async streaming, DuckDB analytics storage, Evidently AI drift monitoring, and a Streamlit dashboard — all orchestrated on Minikube (rootless Podman).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Model Training](#model-training)
4. [Inference API](#inference-api)
5. [Async Data Pipeline](#async-data-pipeline)
6. [Drift Monitoring](#drift-monitoring)
7. [Dashboard](#dashboard)
8. [Containerization](#containerization)
9. [Kubernetes Deployment](#kubernetes-deployment)
10. [API Reference](#api-reference)
11. [Local Development](#local-development)
12. [Troubleshooting](#troubleshooting)

---

## Architecture

```
                          ┌──────────────────┐
                          │   MLflow Tracking │
                          │  (mlflow.db +     │
                          │   mlartifacts/)   │
                          └────────┬─────────┘
                                   │ train/log
                                   v
┌──────────┐   HTTP    ┌──────────────────────┐   async    ┌──────────────┐
│  Client  │ ────────> │  FastAPI Inference   │ ──────────> │   Redpanda   │
│  curl/   │ <──────── │  (:8000 /predict)    │  (produce)  │  (Kafka API) │
│  App     │   JSON    └──────────────────────┘             │  :9092       │
                                                           └──────┬───────┘
                                                                  │ poll
                                                                  v
                                                     ┌────────────────────┐
                                                     │  Kafka Consumer    │
                                                     │  (batch flush      │
                                                     │   every 5s/100rec)│
                                                     └────────┬───────────┘
                                                              │ batch insert
                                                              v
                                                     ┌────────────────────┐
                                                     │  DuckDB            │
                                                     │  (analytics DB,    │
                                                     │   columnar, file)  │
                                                     └──┬───────────┬─────┘
                                                        │           │
                                        ┌───────────────┘           └──────────────┐
                                        v                                           v
                          ┌──────────────────────┐              ┌──────────────────────┐
                          │  Short-Term Drift     │              │  Long-Term Drift     │
                          │  (CronJob every 6h)   │              │  (CronJob weekly)    │
                          │  window=24h           │              │  window=7d           │
                          └──────────┬───────────┘              └──────────┬───────────┘
                                     │                                     │
                                     v                                     v
                          ┌──────────────────────┐              ┌──────────────────────┐
                          │  HTML Drift Reports   │              │  HTML Drift Reports   │
                          │  (Evidently AI)       │              │  (Evidently AI)       │
                          └──────────┬───────────┘              └──────────┬───────────┘
                                     │                                     │
                                     └────────────────┬────────────────────┘
                                                      v
                                          ┌──────────────────────┐
                                          │  Streamlit Dashboard │
                                          │  (:8501)             │
                                          │  (report browser +   │
                                          │   inference stats)   │
                                          └──────────────────────┘
```

### Data Flow Summary

| Step | Component | Protocol | Description |
|------|-----------|----------|-------------|
| 1 | Client → FastAPI | HTTP POST | Send raw patient features as JSON |
| 2 | FastAPI → Redpanda | Async Kafka | Publish inference input + prediction to `inference-events` topic |
| 3 | Consumer → DuckDB | Batch insert | Poll topic, buffer records, flush every 5s or 100 records |
| 4 | CronJob → DuckDB | SQL query (range) | Short-term (24h window) and long-term (7d window) queries |
| 5 | CronJob → Evidently | In-memory | Compare current vs baseline data, generate HTML reports |
| 6 | Dashboard → Reports | Read HTML | Streamlit loads and displays latest Evidently reports |

---

## Repository Layout

```
.
├── README.md                          # This file
├── Training/                          # Model training (Jupyter)
│   ├── data/diabetic_data.csv         # UCI Diabetes dataset (19MB)
│   ├── utility/main.py                # CSV loader helper
│   ├── train.ipynb                    # Logistic Regression training
│   ├── train_rf.ipynb                 # Random Forest + SMOTE training
│   └── apply_fixes.py                 # Notebook modifier for RF pipeline
├── mlflow.db                          # MLflow tracking DB (SQLite, 804KB)
├── mlartifacts/                       # MLflow model artifacts
│   └── 1/models/
│       ├── m-3bf4...904fd/            # Logistic Regression (40KB)
│       ├── m-3e69...1fc9/             # Large RF (196MB)
│       ├── m-ea0e...5c29/             # RF (21MB)
│       └── m-f2cd...33a1/             # RF + SMOTE (26MB) ← active
└── deploy/                            # Production deployment
    ├── Containerfile                  # Multi-purpose container image
    ├── requirements.txt               # Pinned Python dependencies
    ├── .dockerignore
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                    # FastAPI (v2 — async Kafka producer)
    │   ├── model_loader.py            # MLflow model loader (lazy)
    │   └── kafka_producer.py          # Async Kafka producer (aiokafka)
    ├── consumer/
    │   ├── __init__.py
    │   ├── db.py                      # DuckDB schema + batch operations
    │   └── consumer.py                # Kafka consumer (long-running)
    ├── monitoring/
    │   ├── __init__.py
    │   ├── drift_detector.py          # Shared Evidently report logic
    │   ├── short_term.py              # Short-term drift entry point
    │   ├── long_term.py               # Long-term drift entry point
    │   └── dashboard.py               # Streamlit dashboard
    └── k8s/
        ├── redpanda.yaml              # Redpanda StatefulSet + Service
        ├── consumer.yaml              # Kafka consumer Deployment + PVCs
        ├── deployment.yaml            # FastAPI inference Deployment
        ├── service.yaml               # Inference NodePort (30800)
        ├── cronjobs.yaml              # Short/long-term Evidently CronJobs
        └── dashboard.yaml             # Streamlit Dashboard + NodePort (30801)
```

---

## Model Training

### Pipeline

The active model (`m-f2cdec93641243389b1b075d919833a1`, 26 MB) is a scikit-learn pipeline trained via Jupyter notebooks:

```
Raw CSV → CustomMapper → ColumnTransformer → SMOTE → RandomForestClassifier
```

### CustomMapper

| Transformation | Columns | Logic |
|---|---|---|
| Age bin → int | `age` | `[0-10)`→0 ... `[90-100)`→9 |
| Binary mapping | `change`, `diabetesMed`, `gender` | No/Male→0, Yes/Ch/Female→1 |
| Medication flag | 17 medication cols | `!= 'No'` → 1, else 0 |
| Drop constants | citoglipton, metformin-rosiglitazone, examide, acetohexamide, troglitazone, glimepiride-pioglitazone | Removed |

### ColumnTransformer

| Group | Transformer | Columns |
|---|---|---|
| `num` | `SimpleImputer(strategy='mean')` | time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses |
| `cat` | `SimpleImputer('missing') → OneHotEncoder(handle_unknown='ignore')` | race, max_glu_serum, A1Cresult, 17 medication cols |
| `diag_target_enc` | `TargetEncoder(smooth='auto')` | diag_1, diag_2, diag_3 |
| `remainder` | passthrough | age, gender, change, diabetesMed |

### Classifier

- `RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight='balanced')`
- SMOTE oversampling before classifier
- Threshold tuning via precision-recall curve

### Training Output (from best run)

| Metric | Value |
|---|---|
| Accuracy | ~0.70 |
| F1 (binary) | ~0.55 |
| ROC-AUC | ~0.72 |
| Input features | 44 raw → 98 encoded |
| Model size | 26 MB |

---

## Inference API

### Endpoints

#### `GET /health`

Returns model load status and artifact URI.

```json
{"status": "ok", "model_uri": "mlartifacts/1/models/m-f2cd...33a1/artifacts"}
```

#### `POST /predict`

Accepts raw patient records. The model pipeline handles all preprocessing internally.

**Request:**
```json
{
  "instances": [
    {
      "race": "Caucasian",
      "gender": "Female",
      "age": "[0-10)",
      "admission_type_id": 6,
      "discharge_disposition_id": 25,
      "admission_source_id": 1,
      "time_in_hospital": 1,
      "num_lab_procedures": 41,
      "num_procedures": 0,
      "num_medications": 1,
      "number_outpatient": 0,
      "number_emergency": 0,
      "number_inpatient": 0,
      "diag_1": "250.83",
      "diag_2": "?",
      "diag_3": "?",
      "number_diagnoses": 1,
      "max_glu_serum": "None",
      "A1Cresult": "None",
      "metformin": "No",
      "...": "..."
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 0,
      "probability_readmitted": 0.298,
      "probability_not_readmitted": 0.702
    }
  ]
}
```

### Input Columns (44 features)

All columns from the raw dataset except `encounter_id`, `patient_nbr`, `readmitted`, `weight`, `payer_code`, `medical_specialty`.

Diagnosis columns (`diag_1`, `diag_2`, `diag_3`) are automatically cast to string and nulls filled with `"missing"`. Unknown/Invalid gender is rejected.

### Lifecycle

| Event | Action |
|---|---|
| Startup | Load model from baked-in `mlartifacts/` (or `MODEL_URI` env), connect Kafka producer |
| Predict | Run model, return prediction, publish `{input, prediction, probabilities}` to Redpanda `inference-events` topic |
| Shutdown | Gracefully close Kafka producer |

---

## Async Data Pipeline

### Redpanda (Kafka-compatible broker)

Single-node StatefulSet. No Zookeeper dependency. Topic `inference-events` is auto-created on first produce.

| Config | Value |
|---|---|
| Image | `redpandadata/redpanda:v24.3.1` |
| Ports | 9092 (Kafka API), 9644 (Admin API) |
| Resources | 250m CPU / 512Mi request, 1CPU / 1Gi limit |
| Storage | 2Gi PVC (ReadWriteOnce) |

### Kafka Consumer

Long-running Python process that:

1. Polls `inference-events` every 1s
2. Buffers records in a deque
3. Flushes to DuckDB when:
   - Buffer reaches 100 records, OR
   - 5 seconds since last flush
4. Handles SIGTERM/SIGINT for graceful shutdown

### DuckDB Schema

```sql
CREATE TABLE inference_logs (
    id                      BIGINT PRIMARY KEY,
    ts                      TIMESTAMP,       -- Kafka message timestamp
    input_json              JSON,            -- Raw input features
    prediction              INTEGER,         -- 0 or 1
    probability_readmitted  DOUBLE,
    probability_not_readmitted DOUBLE
);

CREATE SEQUENCE seq_id START 1;
```

Design rationale:
- **Columnar storage** — analytical queries (group by, time range) are orders of magnitude faster than row-based SQLite
- **Embedded** — no server process, single file on PVC, accessible from any pod
- **JSON column** — stores full raw input for schema-flexible drift analysis
- **Auto-increment ID** — sequential across consumer restarts

---

## Drift Monitoring

### Evidently AI Integration

Both short-term and long-term jobs use the same core logic in `drift_detector.py`:

1. **Load baseline** — Read `diabetic_data.csv`, apply the same cleaning as training (drop ID cols, filter Unknown/Invalid, stringify diags, binarize target)
2. **Load current** — Query DuckDB for time window, parse `input_json`, align columns to baseline schema
3. **Column mapping** — 8 numerical features, 36 categorical features
4. **Generate reports**:
   - `DataDriftPreset` — Feature-by-feature distribution comparison (Kolmogorov-Smirnov for numerical, Jensen-Shannon for categorical)
   - `ClassificationPreset` — Model quality metrics (only if labeled data is available in the window)
5. **Save HTML** — Self-contained reports to shared `evidently-reports` PVC

### Short-Term CronJob

| Field | Value |
|---|---|
| Schedule | `0 */6 * * *` (every 6 hours) |
| Window | Last 24 hours |
| Tag | `short` |
| Resources | 200m CPU / 256Mi request |

### Long-Term CronJob

| Field | Value |
|---|---|
| Schedule | `0 2 * * 0` (every Sunday at 2 AM) |
| Window | Last 7 days |
| Tag | `long` |
| Resources | 200m CPU / 256Mi request |

### Output Files

```
/data/reports/
├── data_drift_report_short_20260620_120000.html
├── data_drift_report_long_20260620_020000.html
├── model_quality_report_short_20260620_120000.html
└── model_quality_report_long_20260620_020000.html
```

---

## Dashboard

### Streamlit UI (`drift-dashboard:8501`)

Displays:
- **Total inferences** — row count from DuckDB
- **Latest inference timestamp**
- **DuckDB file size**
- **Report browser** — dropdown selector to view any generated Evidently HTML report inline

Access via `minikube service drift-dashboard --url` or `kubectl port-forward svc/drift-dashboard 8501:8501`.

---

## Containerization

### Image Details

| Property | Value |
|---|---|
| Base image | `python:3.13-slim` |
| Size | ~1.33 GB (includes mlflow, sklearn, evidently, streamlit, aiokafka) |
| Build method | Single-stage `podman build` or `minikube image build` |
| Model | Baked in (`mlartifacts/1/models/m-f2cd...33a1/`, 26 MB) |
| Entry points | `uvicorn` (default), `python -m consumer.consumer`, `python -m monitoring.short_term`, `python -m monitoring.long_term`, `streamlit run` |

### Containerfile Structure

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY deploy/app/ app/
COPY deploy/consumer/ consumer/
COPY deploy/monitoring/ monitoring/
COPY mlartifacts/.../ mlartifacts/.../
ENV MODEL_URI=mlartifacts/1/models/m-f2cd...33a1/artifacts
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build Commands

```bash
# Podman (local)
podman build -t diabetic-readmission:latest -f deploy/Containerfile .

# Minikube (builds inside cluster VM, avoids push/pull)
minikube image build -t diabetic-readmission:latest -f deploy/Containerfile .
```

---

## Kubernetes Deployment

### Prerequisites

- Minikube v1.35+ with `--driver=podman`
- 4 CPU cores, 4 GB RAM minimum
- Rootless Podman configured

### Full Deploy Sequence

```bash
# 1. Build & load image
podman build -t diabetic-readmission:latest -f deploy/Containerfile .
podman save diabetic-readmission:latest -o /tmp/diabetic-image.tar
minikube image load /tmp/diabetic-image.tar
rm /tmp/diabetic-image.tar

# 2. Deploy Redpanda (Kafka) first
kubectl apply -f deploy/k8s/redpanda.yaml
kubectl rollout status statefulset/redpanda

# 3. Deploy infrastructure (PVCs, consumer)
kubectl apply -f deploy/k8s/consumer.yaml

# 4. Deploy inference API
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

# 5. Deploy monitoring (CronJobs + Dashboard)
kubectl apply -f deploy/k8s/cronjobs.yaml
kubectl apply -f deploy/k8s/dashboard.yaml

# 6. Wait for all pods
kubectl get pods -w
```

### All Resources

| Kind | Name | Replicas | Exposed Port |
|---|---|---|---|
| StatefulSet | `redpanda` | 1 | 9092 (cluster-internal) |
| Deployment | `diabetic-readmission-api` | 2 | NodePort 30800 |
| Deployment | `inference-consumer` | 1 | — |
| Deployment | `drift-dashboard` | 1 | NodePort 30801 |
| CronJob | `drift-short-term` | — | — |
| CronJob | `drift-long-term` | — | — |
| PVC | `duckdb-data` | 1 Gi | — |
| PVC | `evidently-reports` | 1 Gi | — |

### Testing

```bash
# Inference API
kubectl port-forward svc/diabetic-readmission-service 9999:8000
curl http://localhost:9999/health
curl -X POST http://localhost:9999/predict -H "Content-Type: application/json" -d '{"instances":[...]}'

# Dashboard
kubectl port-forward svc/drift-dashboard 8501:8501
# Open http://localhost:8501

# Minikube tunnel (alternative)
minikube service diabetic-readmission-service --url
minikube service drift-dashboard --url
```

---

## Resource Sizing

| Pod | Request CPU | Request Mem | Limit CPU | Limit Mem |
|---|---|---|---|---|
| `redpanda` | 250m | 512 Mi | 1 | 1 Gi |
| `diabetic-readmission-api` | 500m | 512 Mi | 2 | 2 Gi |
| `inference-consumer` | 100m | 128 Mi | 500m | 512 Mi |
| `drift-dashboard` | 100m | 128 Mi | 500m | 512 Mi |
| `drift-short-term` (job) | 200m | 256 Mi | 1 | 1 Gi |
| `drift-long-term` (job) | 200m | 256 Mi | 1 | 1 Gi |

---

## Local Development (No Kubernetes)

```bash
# 1. Create and activate venv
python3 -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -r deploy/requirements.txt

# 3. Start Redpanda (or Kafka) locally
podman run -d --name redpanda --network host \
  docker.redpanda.com/redpandadata/redpanda:v24.3.1 \
  redpanda start --smp 1 --memory 1G \
  --node-id 0 --check=false

# 4. Start consumer (in background)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
DUCKDB_PATH=/tmp/inference.duckdb \
python -m consumer.consumer &

# 5. Start FastAPI
MODEL_URI=mlartifacts/1/models/m-f2cd...33a1/artifacts \
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir deploy

# 6. Run drift detection manually
python -m monitoring.short_term \
  --baseline Training/data/diabetic_data.csv \
  --output-dir /tmp/reports

# 7. Start dashboard
streamlit run monitoring/dashboard.py \
  --server.port 8501 \
  --server.headless true
```

---

## Dependencies

```
fastapi==0.115.12        # HTTP API framework
uvicorn[standard]         # ASGI server
mlflow==3.14.0            # Model registry & loading
cloudpickle==3.1.2        # Model serialization
numpy==2.4.6             # Numerical computing
pandas==2.3.3            # DataFrame operations
scikit-learn==1.9.0      # ML pipeline
scipy==1.17.1            # Scientific computing
psutil==7.2.2            # System metrics (MLflow req)
imbalanced-learn==0.13.0 # SMOTE sampler
evidently==0.4.36        # Data drift & quality monitoring
pydantic==2.11.1         # Validation (FastAPI req)
aiokafka==0.12.0         # Async Kafka producer
kafka-python==2.0.3      # Sync Kafka consumer
duckdb==1.2.1            # Embedded analytics database
streamlit==1.44.1        # Monitoring dashboard
```

---

## Troubleshooting

### "Connection reset by peer"

**Cause:** Rootless Podman's pasta network driver.  
**Fix:** Use `--network host` for local tests, or use `kubectl port-forward` in Minikube.

### "Permission denied" on volume mounts

**Cause:** SELinux blocking container access to host files.  
**Fix:** Add `:Z` flag to volume mounts (e.g. `-v $(pwd)/mlartifacts:/app/mlartifacts:ro,Z`).

### "ImagePullBackOff" / "ErrImagePull"

**Cause:** Minikube (containerd) cannot see the Podman image.  
**Fix:** Use `minikube image load /path/to/image.tar` after `podman save`, or build directly with `minikube image build`.

### "Failed to load model" in container

**Cause:** App user can't read model files (permissions mismatch between host UID and container user).  
**Fix:** Remove explicit `USER appuser` and let the container run as root (which maps to the host user in rootless Podman).

### "No inference records" in drift jobs

**Cause:** No client has called `/predict` yet, or Kafka/consumer is down.  
**Check:** `kubectl logs deployment/inference-consumer` and verify Redpanda is running.

### DuckDB file not found

**Cause:** Consumer pod hasn't started yet, or PVC not mounted.  
**Check:** `kubectl get pvc` to verify Bound status, `kubectl describe pod inference-consumer` for volume mount errors.

---

## Notes

- The image is 1.33 GB due to the breadth of dependencies (mlflow, evidently, streamlit). For production, split into separate images for each component.
- The model is baked into the image (26 MB). To update, rebuild the image with new artifacts.
- Redpanda auto-creates topics on first produce — no manual topic setup needed.
- DuckDB supports concurrent readers + single writer. Short/long-term CronJobs mount the DB as read-only to avoid conflicts with the consumer.
- All Evidently reports are self-contained HTML files (embedded JS/CSS) — no server needed to view them.
