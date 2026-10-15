# End-to-End Fraud Detection System

A real-time fraud scoring application for PaySim mobile money transactions. The API serves the saved preprocessing and XGBoost pipeline with the decision threshold selected during the existing ML workflow. The React interface sends transaction fields to FastAPI and displays the probability and threshold decision.

## Inference contract

The production artifacts are kept in `models/production/`:

- `fraud_detection_pipeline.joblib` — fitted preprocessing and classifier pipeline.
- `decision_threshold.json` — selected validation threshold (`decision_threshold`, currently `0.995`).

The saved pipeline expects `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`. It includes median imputation and scaling for numeric columns and one hot encoding for `type`. Accepted values are `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, and `TRANSFER`. No target label is accepted by the inference API.

## Run locally

Use Python 3.13 and Node.js 22 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
$env:FRONTEND_ORIGIN = "http://localhost:5173"
.\backend\start.ps1
```

The launcher works from the repository root or the `backend` directory (`.\start.ps1`). If port 8000 is already in use, start on another port with `.\backend\start.ps1 -Port 8001`.

In another terminal, start the frontend:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

The six-page UI is at `http://localhost:5173`, the API health endpoint at `http://localhost:8001/health`, and the interactive API docs at `http://localhost:8001/docs`. Override artifact locations with `FRAUD_MODEL_PATH` and `FRAUD_THRESHOLD_PATH` if needed. Set `VITE_API_BASE_URL` to change the browser API URL; its local default is `http://localhost:8001/api/v1`. For a same-origin deployment, use `/api/v1`; Vite and the Docker frontend proxy the API and `/health` requests.

## Docker Compose

From the repository root, run:

```sh
docker compose up --build
```

The frontend is served at `http://localhost:8080` and proxies `/api/` and `/health` to the backend container. The backend image includes only the API source and two production artifacts; it does not copy data, notebooks, MLflow history, or training code. `FRONTEND_ORIGIN` can be set in a local `.env` file to restrict browser access to the required frontend origin.

## API

The React console includes Overview, Risk assessment, Batch scoring, Monitoring, Model registry, and Transactions pages. The assessment view uses a native Three.js animated risk orb with a reduced-motion and WebGL fallback. The UI is personalized for Ritesh Girhe.

| Route | Purpose |
| --- | --- |
| `GET /health` | API and production-model readiness |
| `POST /api/v1/predict` | Score one transaction |
| `POST /api/v1/predict/batch` | Score 1–500 transactions |
| `GET /api/v1/predictions` | Paginated persisted history; supports `is_fraud` and `labeled` filters |
| `PATCH /api/v1/predictions/{transaction_id}/label` | Attach a known 0/1 outcome |
| `GET /api/v1/model/info` | Production model version, threshold, and evaluation metrics |
| `GET /api/v1/monitoring/metrics` | Process-local request, latency, and prediction counters |
| `GET /api/v1/monitoring/summary` | Stored prediction counts and labeled-outcome metrics |
| `GET /api/v1/monitoring/drift` | PSI and KS drift against the production reference sample |
| `GET /api/v1/reports/eda` | EDA report JSON |
| `GET /api/v1/reports/comparison` | Model comparison report JSON |

Predictions are appended to `data/predictions/predictions.jsonl` (override with `FRAUD_PREDICTIONS_PATH`). This file may contain transaction data and is intentionally excluded from version control. Process request counters and latency percentiles reset when the API process restarts; persisted prediction history does not. Invalid transaction fields are rejected with HTTP 422.

## Tests

From the repository root:

```sh
python -m pytest -q
cd frontend && npm test && npm run build
```

The suite includes mocked prediction, batch, history, labeling, monitoring, model/report route tests plus a separate integration check that scores one small transaction with the saved production artifacts. API tests use a temporary JSONL store and do not pollute local prediction history. Tests do not load the PaySim dataset or retrain the model.

## Delivery phases 19–23

- **Phase 19 – FastAPI backend:** single and batch inference, paginated prediction history, label feedback, model metadata, reports, health checks, and monitoring routes.
- **Phase 20 – React + Three.js frontend:** six responsive views, animated risk visualization, CSV batch scoring, model details, monitoring, and transaction feedback.
- **Phase 21 – Monitoring:** request/latency counters, stored prediction summaries, labeled precision/recall/PR-AUC, and feature PSI/KS drift indicators.
- **Phase 22 – Testing:** backend route and model integration tests, frontend API tests, and a Vite production build.
- **Phase 23 – Run and deployment:** local PowerShell run steps above and Docker Compose deployment above.

## Existing ML work

Training, notebooks, evaluation output, production artifacts, and the existing `mlflow.db` experiment history are maintained separately from the inference application. Starting the API and running its tests do not invoke training or write to MLflow.

## Training workflow (phases 11–18)

The training workflow is implemented in `src/models/workflow.py`. Run it from the repository root with the project environment activated:

```powershell
python -m src.models.workflow
```

It creates the legitimate-only and weighted-logistic baselines, compares logistic regression, random forest, and XGBoost with training-only imbalance strategies, runs three-fold stratified PR-AUC cross-validation, tunes the top two model families using the time-based validation split, selects a decision threshold by validation F2, and evaluates the selected pipeline once on the held-out test split.

The training sample keeps every training fraud row and samples up to 30,000 legitimate rows to fit within workstation memory. Validation and test evaluation use their complete saved splits. The sampling choice, data hash, parameters, metrics, and serialized pipeline for each run are stored under `experiments/`; reports are written under `reports/`. Threshold and final test output are written to `reports/threshold_curve.csv` and `reports/final_evaluation.json`.

The selected artifact is written to `models/registry/vN/` and promoted to `models/production/`. Production includes `model.joblib`, the API-compatible `fraud_detection_pipeline.joblib`, `decision_threshold.json`, `metadata.json`, and `reference_sample.csv`. The saved pipeline takes the same seven raw transaction fields accepted by the API. Feature engineering can derive `is_merchant_dest` when the destination ID is present, but this model excludes it because the API does not accept `nameDest`.

Promote or roll back a registry version with:

```powershell
python -m src.models.registry promote v1
python -m src.models.registry rollback v1
```
