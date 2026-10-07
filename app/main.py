import os
import threading
from contextlib import asynccontextmanager
from datetime import date as Date
from datetime import timedelta
from logging import getLogger

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from mlflow.tracking import MlflowClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .train import NAME, train
from .weather import FEATURES, build_features, fetch_daily

API_VERSION = "1"
MIN_PREDICTION_DATE = Date(2015, 2, 1)
logger = getLogger(__name__)

state = {"model": None, "version": None, "training": False}
limiter = Limiter(
    key_func=get_remote_address,
    headers_enabled=True,
    default_limits=[os.getenv("RATE_DEFAULT", "10/minute")],
)


def load_champion():
    mv = MlflowClient().get_model_version_by_alias(NAME, "champion")
    state["model"] = mlflow.sklearn.load_model(f"models:/{NAME}@champion")
    state["version"] = mv.version


def retrain():
    """Train a new model, then load it if MLflow marks it champion."""
    if state["training"]:
        return
    state["training"] = True
    try:
        logger.info("Training result: %s", train())
        load_champion()
    except Exception:
        logger.exception("Training or champion loading failed")
    finally:
        state["training"] = False


@asynccontextmanager
async def lifespan(app):
    try:
        load_champion()
    except Exception:  # noqa: BLE001
        logger.info("No champion model could be loaded; starting initial training")
        threading.Thread(target=retrain, daemon=True).start()
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        retrain,
        "cron",
        day_of_week=os.getenv("RETRAIN_DOW", "mon"),
        hour=2,
    )
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="Dhaka Weather Forecast API", version=API_VERSION, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],)

@app.middleware("http")
async def version_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-API-Version"] = API_VERSION
    if state["version"]:
        response.headers["X-Model-Version"] = str(state["version"])
    return response


v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@v1.get("/predict")
@limiter.limit(os.getenv("RATE_PREDICT", "20/minute"))
def predict(
    request: Request,
    response: Response,
    date: Date = Query(..., description="YYYY-MM-DD"),  # noqa: B008
):
    if state["model"] is None:
        raise HTTPException(503, "Model not ready yet (initial training in progress)")
    if not MIN_PREDICTION_DATE <= date <= Date.today() + timedelta(days=1):  # noqa: DTZ011
        raise HTTPException(422, "date must be between 2015-02-01 and tomorrow")
    prediction_date = pd.Timestamp(date)
    previous_day = pd.Timedelta(days=1)
    try:
        daily = fetch_daily(
            prediction_date - 15 * previous_day,
            prediction_date - previous_day,
        )
    except requests.RequestException:
        raise HTTPException(502, "Weather provider unavailable")
    features = build_features(daily).dropna().tail(1)
    if features.empty or features["time"].iloc[0] != prediction_date - previous_day:
        raise HTTPException(503, "Weather data for the previous day is not available yet")
    temperature_change = state["model"].predict(features[FEATURES])[0]
    predicted_temperature = float(features["temp_max"].iloc[0] + temperature_change)
    return {
        "date": str(date),
        "predicted_temp_max_c": round(predicted_temperature, 1),
        "model_version": state["version"],
        "api_version": API_VERSION,
    }


@v1.get("/model")
@limiter.limit("30/minute")
def model_info(request: Request, response: Response):
    return {"name": NAME, "champion_version": state["version"], "training": state["training"]}


@v1.post("/train", status_code=202)
@limiter.limit("2/hour")
def trigger_training(request: Request, response: Response, x_admin_key: str = Header(None)):
    if x_admin_key != os.getenv("ADMIN_KEY", "change-me"):
        raise HTTPException(401, "Invalid admin key")
    threading.Thread(target=retrain, daemon=True).start()
    return {"status": "training started"}


app.include_router(v1)


@app.get("/health", include_in_schema=False)
@limiter.exempt
def health(request: Request):
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/", include_in_schema=False)
@limiter.exempt
def index(request: Request):
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
