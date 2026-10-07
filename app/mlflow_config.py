"""Load MLflow's database and artifact-store settings from the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configure_mlflow_environment():
    """Map local Supabase settings to the standard MLflow and S3 variables."""
    load_dotenv(PROJECT_ROOT / ".env.local")

    database_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("DATABASE_URI")
    if not database_uri:
        database_fields = {
            "username": os.getenv("user") or os.getenv("SUPABASE_DB_USER"),
            "password": os.getenv("password") or os.getenv("SUPABASE_DB_PASSWORD"),
            "host": os.getenv("host") or os.getenv("SUPABASE_DB_HOST"),
            "port": os.getenv("port") or os.getenv("SUPABASE_DB_PORT"),
            "database": os.getenv("database") or os.getenv("SUPABASE_DB_NAME"),
        }
        if any(database_fields[name] for name in ("username", "password", "host")):
            missing = [name for name, value in database_fields.items() if not value]
            if missing:
                raise RuntimeError(
                    "Incomplete Supabase database configuration; missing: "
                    + ", ".join(missing)
                )
            database_uri = URL.create(
                "postgresql+psycopg2",
                username=database_fields["username"],
                password=database_fields["password"],
                host=database_fields["host"],
                port=int(database_fields["port"]),
                database=database_fields["database"],
            ).render_as_string(hide_password=False)

    if database_uri:
        os.environ["MLFLOW_TRACKING_URI"] = database_uri
    elif not os.getenv("MLFLOW_TRACKING_URI") and Path("/data").is_dir():
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////data/mlflow.db"

    storage = {
        "endpoint": os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT"),
        "access_key": os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("ACCESS_KEY"),
        "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("SECRET_ACCESS_KEY"),
        "region": os.getenv("AWS_DEFAULT_REGION") or os.getenv("S3_REGION"),
        "bucket": os.getenv("BUCKET_NAME"),
    }
    if any(storage.values()):
        missing = [name for name, value in storage.items() if not value]
        if missing:
            raise RuntimeError(
                "Incomplete Supabase Storage configuration; missing: "
                + ", ".join(missing)
            )

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = storage["endpoint"].rstrip("/")
        os.environ["AWS_ACCESS_KEY_ID"] = storage["access_key"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = storage["secret_key"]
        os.environ["AWS_DEFAULT_REGION"] = storage["region"]

        artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "")
        if not artifact_root.startswith("s3://"):
            os.environ["MLFLOW_ARTIFACT_ROOT"] = f"s3://{storage['bucket']}/mlflow"
    elif not os.getenv("MLFLOW_ARTIFACT_ROOT") and Path("/data").is_dir():
        os.environ["MLFLOW_ARTIFACT_ROOT"] = "/data/artifacts"


configure_mlflow_environment()
