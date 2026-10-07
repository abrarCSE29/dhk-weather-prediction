# Dhaka Weather Forecast Service

Self-training next-day max-temperature model for Dhaka. Data comes from the free Open-Meteo API,
models are versioned in MLflow, and a FastAPI server exposes a versioned, rate-limited API plus a one-page UI.

## Architecture

```
Open-Meteo API --> train.py --> MLflow (runs + registry, alias "champion")
                                      |
Browser (index.html) --> FastAPI /api/v1/predict --> loads champion --> prediction
                          ^ rate limiting, version headers
APScheduler: weekly retrain  |  POST /api/v1/train: manual retrain
```

## Structure

| File | Purpose |
|---|---|
| `app/weather.py` | Fetch data, aggregate to daily, `build_features()` (shared by train and serve) |
| `app/train.py` | Train, log to MLflow, register model, promote to `champion` |
| `app/main.py` | FastAPI app, scheduler, rate limiting, versioned routes |
| `app/static/index.html` | Minimal web UI |
| `Dockerfile`, `docker-compose.yml` | Deployment (API on :8000, MLflow UI on :5000) |

## Run

Copy `.env.example` to `.env.local`. Leave the Supabase fields blank to use the local Docker volume, or fill them with your Supabase database and Storage S3 settings to persist MLflow metadata and artifacts in Supabase. `.env.local` is ignored by Git.

```bash
docker compose up --build
```
1. First start: no model exists, so training begins in the background (about 1-3 minutes; downloads 2015-today).
   `/health` shows `model_loaded: true` when ready.
2. Open **http://localhost:8000** (UI), **/docs** (Swagger), **http://localhost:5000** (MLflow).

Without Docker: `uv pip install --python weather_env/bin/python -r requirements.txt` and `uv run --python weather_env/bin/python uvicorn app.main:app --reload`. With Supabase fields configured, the app builds the SQLAlchemy tracking URI from the database fields and maps the Storage credentials to MLflow's S3 settings. With those fields blank, MLflow uses its local file store.

## How prediction by date works

The model is **one-day-ahead**: for date D it fetches the 15 days before D, builds features from day D-1,
and predicts D's max temperature. Therefore valid dates run from 2015-02-01 to tomorrow. Dates further out are
rejected (422) because no honest forecast exists for them from this model.

## API (v1)

| Method | Path | Notes | Limit |
|---|---|---|---|
| GET | `/api/v1/predict?date=YYYY-MM-DD` | Predicted max temp (°C) | 20/min per IP |
| GET | `/api/v1/model` | Champion version, training status | 30/min |
| POST | `/api/v1/train` | Header `X-Admin-Key`; returns 202 | 2/hour |
| GET | `/health` | Liveness + model loaded | exempt |

Example:
```bash
curl "localhost:8000/api/v1/predict?date=2026-10-07"
# {"date":"2026-10-07","predicted_temp_max_c":33.1,"model_version":"3","api_version":"1"}
```
Status codes: 422 bad date, 429 rate limited, 502 weather provider down, 503 model/data not ready.

## Versioning

- **API:** version lives in the URL (`/api/v1/...`). Breaking changes go into a new router (`/api/v2`) mounted
  beside v1; v1 stays until deprecated. Every response carries `X-API-Version`.
- **Model:** MLflow Model Registry. Each training run registers a new version; the alias `champion` points to the
  one in production. Responses carry `X-Model-Version` and the body includes `model_version`.
- **Rollback:** in the MLflow UI (or `MlflowClient().set_registered_model_alias(...)`) repoint `champion` to an
  older version, then `POST /api/v1/train` or restart the API to reload.

## Automatic training

- On startup with no champion, and every Monday 02:00 (`RETRAIN_DOW`), the service retrains.
- Each run logs params, `mae`, `baseline_mae` (tomorrow = today), and the data end date.
- **Promotion gate:** a new version becomes champion only if its holdout MAE (last 365 days) beats the
  persistence baseline; the very first model is always promoted.

## Rate limiting

`slowapi`, per client IP, in memory. Limits are env-configurable (`RATE_DEFAULT`, `RATE_PREDICT`). Responses include
`X-RateLimit-*` headers; excess requests get 429. Behind a reverse proxy, forward the real client IP
(`uvicorn --proxy-headers`). For several workers or replicas use a shared store:
`Limiter(storage_uri="redis://redis:6379")`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ADMIN_KEY` | `change-me` | Key for `/api/v1/train` (**change it**) |
| `RATE_PREDICT` | `20/minute` | Predict limit |
| `RATE_DEFAULT` | `60/minute` | Default limit |
| `RETRAIN_DOW` | `mon` | Weekly retrain day |
| `MLFLOW_TRACKING_URI` | detected from database fields or MLflow default | Override the tracking store URI directly |
| `DATABASE_URI` | blank | Optional full SQLAlchemy URI; takes precedence over the split database fields |
| `user`, `password`, `host`, `port`, `database` | blank | Supabase Postgres connection fields; use the Session pooler values for local IPv4 networks |
| `S3_ENDPOINT`, `S3_REGION`, `BUCKET_NAME`, `ACCESS_KEY`, `SECRET_ACCESS_KEY` | blank | Supabase Storage S3 settings for model artifacts |

## Known limitations

- Trained on ERA5 reanalysis but served with Open-Meteo's recent observations for the last week, so there is a small source mismatch.
- A one-day persistence baseline is hard to beat; expect modest MAE gains. Check `mae` vs `baseline_mae` in MLflow.
- Run a single Uvicorn worker (scheduler and limiter state are in-process). Local Compose uses SQLite when Supabase fields are blank; Supabase mode uses Postgres and Storage.
- Add authentication (not just an admin key) and HTTPS before exposing publicly.
