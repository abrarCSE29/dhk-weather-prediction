import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from .weather import FEATURES, build_features, fetch_daily

NAME = "dhaka-temp"
EXPERIMENT_NAME = NAME
MODEL_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "random_state": 42,
    "verbose": -1,
}
SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
]


def train(start_year=2015):
    """Train, evaluate, log, and promote a Dhaka next-day temperature model."""
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=8)
    daily_frames = [
        fetch_daily(f"{year}-01-01", min(pd.Timestamp(f"{year}-12-31"), end))
        for year in range(start_year, end.year + 1)
    ]
    data = build_features(pd.concat(daily_frames, ignore_index=True))
    data["target"] = data["temp_max"].shift(-1) - data["temp_max"]
    data = data.dropna().reset_index(drop=True)

    holdout_start = data["time"].max() - pd.Timedelta(days=365)
    training_data = data[data["time"] <= holdout_start]
    holdout_data = data[data["time"] > holdout_start]
    validation_model = LGBMRegressor(**MODEL_PARAMS).fit(
        training_data[FEATURES], training_data["target"]
    )
    mae = mean_absolute_error(
        holdout_data["target"],
        validation_model.predict(holdout_data[FEATURES]),
    )
    baseline_mae = mean_absolute_error(
        holdout_data["target"], np.zeros(len(holdout_data))
    )
    final_model = LGBMRegressor(**MODEL_PARAMS).fit(data[FEATURES], data["target"])

    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts"),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                **MODEL_PARAMS,
                "rows": len(data),
                "data_end": str(data["time"].max().date()),
            }
        )
        mlflow.log_metrics({"mae": mae, "baseline_mae": baseline_mae})
        mlflow.sklearn.log_model(
            final_model,
            name="model",
            registered_model_name=NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    version = client.search_model_versions(f"run_id='{run.info.run_id}'")[0].version

    try:
        client.get_model_version_by_alias(NAME, "champion")
        has_champion = True
    except Exception:  # noqa: BLE001
        has_champion = False

    promoted = not has_champion or mae < baseline_mae
    if promoted:
        client.set_registered_model_alias(NAME, "champion", version)
    return {
        "version": version,
        "mae": mae,
        "baseline_mae": baseline_mae,
        "promoted": promoted,
    }
