import numpy as np
import pandas as pd
import requests

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
    old = pd.Timestamp(end) <= pd.Timestamp.today().normalize() - pd.Timedelta(days=8)
    response = requests.get(
        ARCHIVE if old else FORECAST,
        timeout=60,
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": str(start)[:10],
            "end_date": str(end)[:10],
            "hourly": HOURLY,
            "timezone": "Asia/Dhaka",
        },
    )
    response.raise_for_status()
    hourly = pd.DataFrame(response.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    return (
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
