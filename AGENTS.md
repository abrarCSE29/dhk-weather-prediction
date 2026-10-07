# Repository Guidelines

## Project Structure

Application code is in `app/`: `weather.py` fetches and prepares Open-Meteo data, `train.py` trains and registers the model in MLflow, and `main.py` serves the FastAPI API and schedules retraining. The small browser interface is `app/static/index.html`. `Dockerfile` and `docker-compose.yml` define the API and MLflow services. There is no dedicated test or asset directory currently.

## Build, Run, and Development

- `docker compose up --build` builds and starts the API on port 8000 and MLflow on port 5000; persistent MLflow data is stored in the `mldata` volume.
- `pip install -r requirements.txt` installs Python dependencies for local development.
- `uvicorn app.main:app --reload` runs the API locally with reload enabled. The service expects an MLflow tracking store; without Docker it defaults to local MLflow storage.
- Visit `/docs` for the interactive API documentation and `/health` for readiness status.

## Coding Style and Naming

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, and uppercase names for module constants (for example, `FEATURES`). Keep feature creation in `app/weather.py` shared between training and prediction to avoid training-serving differences. Use focused functions and preserve the API's `/api/v1` versioning pattern. No formatter or linter is configured in the repository.

## Testing

No test suite or coverage requirement is present. When adding behavior, add focused tests (for example, under `tests/` with `test_*.py`) for feature generation, date validation, and model-promotion rules, and document how to run them. Keep tests independent of live weather API calls and MLflow services where practical by using fixtures or mocks.

## Commits and Pull Requests

The available checkout does not expose Git commit history, so no established commit convention can be verified. Use short imperative commit subjects, such as `Add date validation test`. Pull requests should explain the behavior change, list relevant configuration or API effects, link related issues, and include UI screenshots when the web interface changes. Note any required environment variables or deployment steps.

## Configuration and Operations

Set `ADMIN_KEY` to a non-default secret before exposing the training endpoint. Keep the deployment at one Uvicorn worker: the scheduler and in-memory rate limiter are process-local. Do not commit credentials, generated model data, or local MLflow databases.
