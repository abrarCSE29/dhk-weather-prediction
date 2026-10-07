import hashlib
import json
import logging
import os
from io import StringIO

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")  # Load .env file if present

# Uvicorn routes this logger to its configured console handler in local and Render runs.
logger = logging.getLogger("uvicorn.error")

_redis = None
_redis_checked = False


def _get_redis():
    """Return the optional Upstash Redis client, initializing it once."""
    global _redis, _redis_checked
    if _redis_checked:
        return _redis

    _redis_checked = True
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not url and not token:
        logger.info("Upstash weather cache disabled: credentials are not configured")
        return None
    if not url or not token:
        logger.warning("Upstash cache disabled: both URL and token must be configured")
        return None

    try:
        from upstash_redis import Redis

        _redis = Redis(url=url, token=token)
        logger.info("Upstash weather cache client initialized")
    except Exception:
        logger.exception("Could not initialize Upstash cache; continuing without cache")
    return _redis


def _cache_key(url, params):
    """Build a stable key without exposing request details in Redis keys."""
    payload = json.dumps([url, params], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"dhk-weather-pred:weather:open-meteo:v1:{digest}"

ARCHIVE = "https://archive-api.open-meteo.com/v1/era5"
FORECAST = "https://api.open-meteo.com/v1/forecast"
LAT, LON = 23.81, 90.41  # Dhaka
HOURLY = "temperature_2m,relative_humidity_2m,precipitation"

FEATURES = [
    "temp_max",
    "temp_min",
    "temp_mean",
    "humidity_mean",
    "rain_sum",
    "lag1",
    "lag2",
    "lag7",
    "roll3",
    "roll7",
    "rain_roll3",
    "diff1",
    "anom7",
    "temp_range",
    "hum_change",
    "doy_sin",
    "doy_cos",
    "month",
    "is_monsoon",
]


def fetch_daily(start, end):
    """Fetch Open-Meteo hourly data and aggregate it to daily weather values."""
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    old = pd.Timestamp(end) <= pd.Timestamp.today().normalize() - pd.Timedelta(days=8)
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": HOURLY,
        "timezone": "Asia/Dhaka",
    }
    if old:
        params["start_date"] = start_date.strftime("%Y-%m-%d")
        params["end_date"] = end_date.strftime("%Y-%m-%d")
    else:
        # Open-Meteo forbids combining past_days with explicit start/end dates.
        params["past_days"] = 16
    url = ARCHIVE if old else FORECAST
    cache = _get_redis()
    cache_params = {
        **params,
        "requested_start_date": start_date.strftime("%Y-%m-%d"),
        "requested_end_date": end_date.strftime("%Y-%m-%d"),
    }
    key = _cache_key(url, cache_params)
    if cache:
        try:
            cached = cache.get(key)
            if cached:
                daily = pd.read_json(StringIO(cached), orient="records")
                daily["time"] = pd.to_datetime(daily["time"])
                logger.info(
                    "Weather cache hit for %s through %s",
                    start_date.date(),
                    end_date.date(),
                )
                return daily
            logger.info(
                "Weather cache miss for %s through %s",
                start_date.date(),
                end_date.date(),
            )
        except Exception:
            logger.warning("Upstash cache read failed; fetching weather from provider", exc_info=True)

    response = requests.get(url, timeout=60, params=params)
    response.raise_for_status()
    hourly = pd.DataFrame(response.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly = hourly[
        (hourly["time"] >= start_date)
        & (hourly["time"] < end_date + pd.Timedelta(days=1))
    ]
    daily = (
        hourly.set_index("time")
        .resample("D")
        .agg(
            temp_max=("temperature_2m", "max"),
            temp_min=("temperature_2m", "min"),
            temp_mean=("temperature_2m", "mean"),
            humidity_mean=("relative_humidity_2m", "mean"),
            rain_sum=("precipitation", "sum"),
        )
        .reset_index()
    )
    if cache:
        ttl = 30 * 24 * 60 * 60 if old else 15 * 60
        try:
            cache.set(key, daily.to_json(orient="records", date_format="iso"), ex=ttl)
            logger.info(
                "Weather data cached for %s through %s (TTL %s seconds)",
                start_date.date(),
                end_date.date(),
                ttl,
            )
        except Exception:
            logger.warning("Upstash cache write failed; returning provider response", exc_info=True)
    return daily


def build_features(daily):
    """Shared by training AND serving (avoids training-serving skew)."""
    df = daily.sort_values("time").reset_index(drop=True).copy()
    t = df["temp_max"]
    df["lag1"] = t.shift(1)
    df["lag2"] = t.shift(2)
    df["lag7"] = t.shift(7)
    df["roll3"] = t.rolling(3).mean()
    df["roll7"] = t.rolling(7).mean()
    df["rain_roll3"] = df["rain_sum"].rolling(3).sum()
    df["diff1"] = t - df["lag1"]
    df["anom7"] = t - df["roll7"]
    df["temp_range"] = t - df["temp_min"]
    df["hum_change"] = df["humidity_mean"].diff()
    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["time"].dt.month
    df["is_monsoon"] = df["month"].between(6, 9).astype(int)
    return df
