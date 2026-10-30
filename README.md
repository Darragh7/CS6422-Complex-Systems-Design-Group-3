# Scoreline

## First-time setup

- Windows: run `./setup.ps1` from PowerShell at the repository root.
- Linux/macOS VM: run `sh setup.sh` from the repository root.

Both scripts prepare a Python virtual environment, install `backend/requirements.txt`, and create `.env` from `.env.example` when needed. Set your local database/API values before running the API.

## Run with Docker Compose

Install Docker Engine/Desktop with the Compose plugin. Copy `.env.example` to `.env`, set `DATABASE_URL` to the PostgreSQL connection string from Supabase, and set a strong `SECRET_KEY`. Then run:

```sh
docker compose up --build
```

Compose builds and starts the FastAPI backend, which connects to your Supabase database; it does not run a local PostgreSQL container. Check `http://localhost:8000/ready` to verify both the API and database connection, and open `http://localhost:8000/docs` to inspect the API. Stop it with Ctrl+C (or `docker compose down`). See [the deployment guide](docs/development-deployment.md) for the Kubernetes workflow.

## Run in Kubernetes

For a local Minikube setup, follow [the deployment guide](docs/development-deployment.md). It covers building/loading the container, creating local-only secrets, applying manifests, and checking the API.

## Project docs

- [Development process](docs/DEV_PROCESS.pdf)
- [Security documentation](docs/security/)
- [Licensing review and prediction disclaimer](docs/security/licensing-and-disclaimers.md)
