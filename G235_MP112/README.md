# G235 · MP112 — EV Charging Session Energy (Bhopal)

| | |
|---|---|
| **Group id** | G235 |
| **Problem id** | MP112 |
| **Course** | SCSE3040 MLOps — mini project |
| **Faculty / Batch** | Upasna Joshi / 38 |

| Enrolment no. | GitHub |
|---|---|
| S24CSEU1655 | mokshasharma |
| S24CSEU1143 | AnantaUpadhyay |
| S24CSEU1972 | YANUGULAREDDYHARSHITH |
| S24CSEU1657 | AditiDhulia |
| S24CSEU1665 | RaghavVats |

**Client:** Bhopal Green Charge Network (a made-up organisation).
**Task:** when a vehicle plugs in, predict `energy_kwh`, the energy the session
will deliver. This is a **regression** task, scored with **MAE** (kWh).

---

## 1. Results at a glance

| Item | Value |
|---|---|
| v1 model | Ridge on engineered + pairwise-interaction features (`ridge_interact`) |
| v1 5-fold CV MAE (cleaned train) | **1.66 kWh** |
| Most drifted feature (PSI) | **`session_minutes`** (PSI 3.30) |
| Worst-served group (v1 on `later.csv`) | **Residential** (MAE 16.57 vs ~2.1 elsewhere) |
| v1 MAE on last 30% of `later.csv` | 4.968 |
| v2 MAE on the same rows | 2.047 (**58.8% lower**) |
| Promotion rule (≥ 8% lower) | **promote**, so v2 is `@champion` (registry version 2) |
| Holdout | `predictions/holdout_predictions.csv`, 250 rows, `record_id,prediction` |

All numbers are written by the code into [reports/findings.yaml](reports/findings.yaml).

---

## 2. The seven data checks (`src/ev_energy/data_checks.py`)

Counts are on `train.csv` **as given** (1,032 rows).

| Kind | Present | Details | How the code finds it |
|---|---|---|---|
| `leaking_column` | **no** | — | `train.csv` has no column that is missing from `holdout_features.csv` except the target. No feature has Spearman \|r\| ≥ 0.98 with the target. |
| `unit_mixup` | **yes** | `vehicle_age_yr`, `Residential`, factor **12** | Each group's mean is compared with the other groups' mean. Residential is ×11.4 on the mean and ×12.0 on the max (96 vs 8), so it is months, not years. The closest exact conversion factor is 12. |
| `sentinel_value` | **yes** | `queue_before` = **-999**, **50** rows | `value_counts()` of every numeric column, checked against the typical sentinel numbers. |
| `duplicate_rows` | **yes** | **32** extra copies | `df.duplicated().sum()` |
| `stuck_sensor` | **no** | — | Sorted by date, the longest run of one repeated value in any numeric column covers only 2 days. |
| `target_typos` | **yes** | **11** rows | The sorted target jumps from 56.75 to 349.1 (×6.2 gap). The 11 values above the gap are 10× too big. |
| `spelling_variants` | **yes** | `vehicle`, **3** true levels | 20 raw spellings (`" car"`, `CAR`, `car.`, `thr.`, `two wheeler`, …). Each is normalised (case, spaces, dots) and prefix-matched to the clean levels in `later.csv`. |

Cleaning (`src/ev_energy/cleaning.py`) runs in this order:

1. drop the duplicate rows
2. turn `-999` into NaN (filled with the median inside the model pipeline)
3. divide Residential `vehicle_age_yr` by 12
4. divide the 11 typo targets by 10
5. map every spelling to `two_wheeler` / `three_wheeler` / `car`

Run it yourself:

```bash
cd src && python -m ev_energy.data_checks
```

---

## 3. Repository layout

```
README.md                 this file
requirements.txt          pinned libraries
params.yaml               every path and setting (read with yaml.safe_load)
pytest.ini                puts src/ and . on the path for tests
src/ev_energy/
  config.py               params, paths, SHA-256 of train.csv
  data_checks.py          the seven checks
  cleaning.py             raw -> clean train table
  features.py             feature engineering + sklearn Pipeline factory
  train.py                v1: 5 candidate runs in MLflow, register best, @champion
  monitoring.py           PSI, MAE per station_type, v2 retrain, promotion rule
  predict.py              holdout predictions with @champion, export models/model.joblib
  findings.py             fills reports/findings.yaml
  logging_utils.py        JSON log formatter (stdout)
  pipeline.py             runs every stage in order
app/main.py               FastAPI: POST /predict, GET /health
tests/                    15 pytest tests (TestClient included; no CSVs needed)
models/model.joblib       champion pipeline loaded by the API
mlflow.db                 MLflow tracking + model registry (SQLite)
reports/findings.yaml     the filled form
reports/run_state.json    run ids / versions handed between stages
predictions/holdout_predictions.csv
Dockerfile, .dockerignore
.github/workflows/ci.yml  pytest on every push
k8s/deployment.yaml, k8s/service.yaml
data/raw/*.csv(.dvc)      data under DVC
```

---

## 4. How to run

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# whole pipeline from scratch (delete mlflow.db, mlruns/ and reports/run_state.json first
# so that v1 is registry version 1 and v2 is version 2)
cd src
python -m ev_energy.pipeline

# or stage by stage
python -m ev_energy.data_checks     # 7 checks -> printed as JSON
python -m ev_energy.train           # v1 runs + registry + @champion
python -m ev_energy.monitoring      # PSI, per-group MAE, v2, promotion
python -m ev_energy.predict         # holdout CSV + models/model.joblib
python -m ev_energy.findings        # reports/findings.yaml
cd ..

pytest -v                           # tests
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs
```

### Data version control (DVC)

```bash
pip install dvc
git init                     # if not already a repo
dvc init
dvc add data/raw/train.csv data/raw/later.csv
git add data/raw/train.csv.dvc data/raw/later.csv.dvc data/raw/.gitignore .dvc/config data/raw/holdout_features.csv
git commit -m "Track raw data with DVC"
```

`data_sha256` logged in every MLflow run =
`d1625c55d78fabce1d0acc6875ebd07c10b322ed076a51dd9578a5d706b089cb` (the original `train.csv`).
**Never open `train.csv` in Excel and save it**, because that changes the hash.

---

## 5. Modelling

**Features.** These are the 8 request fields plus two engineered features known at plug-in time:

- `max_energy_kwh = charger_kw × session_minutes / 60`, the most the charger could deliver
- `battery_room_pct = 100 − battery_start_pct`

The numeric columns are median-imputed (for the `-999` rows) and scaled. The categories are one-hot encoded.
All of this sits inside one sklearn `Pipeline`, so the API passes raw fields straight to the model.

**v1 candidates** (`params.yaml → training.candidates`, one MLflow run each, all logging `data_sha256`, `cv_mae`, `train_mae`, `mae_later`):

| Run | 5-fold CV MAE |
|---|---|
| ridge | 1.816 |
| **ridge_interact** (Ridge + all pairwise interactions) | **1.662** ← v1 |
| gbr_deep | 2.053 |
| gbr_shallow | 2.118 |
| rf_300 | 2.477 |

Energy is close to linear in session time and charger power, so a linear model beats the trees.
The interaction terms let the slope differ by `station_type` and `vehicle`.

---

## 6. Monitoring and retraining (`src/ev_energy/monitoring.py`)

**PSI** uses 10 bins at the deciles of cleaned `train.csv`, with open outer bins and
`PSI = Σ (a − e) · ln(a / e)`, where `e` and `a` are the bin shares (floored at 1e-4).

| Column | PSI |
|---|---|
| battery_start_pct | 0.027 |
| **session_minutes** | **3.299** (train max 240 min; later mean 214, max 333) |
| charger_kw | 0.034 |
| temp_c | 0.064 |
| vehicle_age_yr | 0.028 |
| queue_before | 0.056 |

**v1 MAE per `station_type` on `later.csv`:** Highway 2.11, Mall 2.14, **Residential 16.57**.
v1's Residential errors grow with session length (r ≈ 0.88), so the energy delivered per
minute at Residential stations has gone up. This is concept drift, not just input drift.

**v2.** `later.csv` is sorted by date. v2 is trained on cleaned train plus the first 70% of `later.csv` (210 rows)
and compared on the last 30% (90 rows). The new rows reflect the current regime, so each `later.csv` row may count
more than an old row. That weight is chosen from {1, 3, 5, 10, 20} with a time-ordered inner split of
the first 70% only, so the check rows are never used for tuning. The chosen weight is 20.

| | MAE on last 30% |
|---|---|
| v1 | 4.968 |
| v2 | 2.047 |

The improvement is 58.8%, which is at least 8%, so the decision is **promote**. v2 (registry version 2) gets the alias `@champion`.
`predict.py` loads `models:/mp112-ev-energy-model@champion`, writes the holdout CSV and saves the
same pipeline to `models/model.joblib` for the API.

---

## 7. Prediction service (`app/main.py`)

```bash
# local
PYTHONPATH=src:. uvicorn app.main:app --port 8000         # Windows: set PYTHONPATH=src;.
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "battery_start_pct": 16, "session_minutes": 210, "charger_kw": 9.4, "temp_c": 25.8,
  "vehicle_age_yr": 1.1, "queue_before": 2, "vehicle": "two_wheeler", "station_type": "Residential"}'
# -> {"energy_kwh": 61.821, "model_version": "champion"}
```

The request schema accepts **exactly** the 8 fields (`extra="forbid"`). The client's rules are pydantic `Field` limits, so a bad request gets **422** and never reaches the model:

- `session_minutes: Field(..., gt=0, le=400)`, so anything above 400 is rejected
- `charger_kw: Field(..., ge=1.5)`, so anything below 1.5 is rejected
- `vehicle` and `station_type` are `Literal`s with the clean levels; other fields have sensible ranges

`GET /health` returns `{"status": "ok", "model_loaded": true}`.

**Logging.** Every prediction writes one JSON line to stdout, with a timestamp, request id, model
version, inputs, prediction and latency. The request contains no personal data.

---

## 8. Container, CI, Kubernetes

```bash
docker build -t mp112-api:1.0 .
docker run -p 8000:8000 mp112-api:1.0

kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

- **Dockerfile:** `python:3.12-slim`; `requirements.txt` is installed *before* the code is copied (layer cache); `EXPOSE 8000`; uvicorn on `0.0.0.0:8000`.
- **CI:** `.github/workflows/ci.yml` runs on every push (and pull requests), installs `requirements.txt` and runs `pytest`. The tests build their own small data and model, so CI needs neither the CSVs nor MLflow artifacts.
- **Kubernetes:** Deployment `mp112-api` (3 replicas, image `mp112-api:1.0`, container port 8000) and Service `mp112-svc` (port 80 → 8000). The selector `app: mp112-api` matches the pod labels. Readiness probe on `/health` after 5 s; liveness probe on `/health` after 20 s.

---

## 9. Tests (`pytest -v`, 15 tests)

- each data check on synthetic data: duplicates, sentinel, unit factor 12, target typos, no stuck sensor, spelling mapping
- PSI is ~0 for identical data and large for shifted data
- end-to-end `clean_frame`
- API via `TestClient`: health, a valid prediction, both client rules at their boundaries (400/401, 1.5/1.4), missing or extra fields, an unknown vehicle, and the JSON log line
