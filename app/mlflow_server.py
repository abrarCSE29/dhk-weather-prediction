"""Start the MLflow UI with the same store configuration as the API."""

import os
import subprocess
import sys

from . import mlflow_config  # Configure MLflow environment before starting the server.


def main():
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        os.getenv("PORT", os.getenv("MLFLOW_UI_PORT", "5000")),
        "--backend-store-uri",
        os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        "--default-artifact-root",
        os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts"),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
