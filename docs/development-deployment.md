## 1. Prepare Supabase and local tools

Create a Supabase project and copy its PostgreSQL connection string from **Project Settings → Database → Connection string**. Use the session pooler string if your network does not support Supabase's direct database connection (for example, some IPv4-only environments). Keep the password URL-encoded if it contains special characters. This application connects to PostgreSQL using `DATABASE_URL`; do not use the Supabase REST URL or anon key.

If this is a new database, open the Supabase SQL Editor and run [`database/schema.sql`](../database/schema.sql). The schema enables row-level security; configure policies separately if clients access these tables through Supabase APIs. The backend connects directly to PostgreSQL.

Install Python 3.11+, Docker Engine/Desktop with the Compose plugin, and (for Kubernetes) `kubectl` and Minikube. Start Docker Desktop/Engine and ensure the Docker daemon is available. From the repository root:

```sh
sh setup.sh
```

On Windows, use `.\setup.ps1` in PowerShell instead. The setup script prepares the Python environment and copies `.env.example` to `.env` if needed; it does not install system packages or configure Supabase.

Edit `.env` and set:

- `DATABASE_URL` to the Supabase PostgreSQL connection string.
- `SECRET_KEY` to a unique random value. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- `FOOTBALL_API_KEY` if you want live football-data.org API results; otherwise leave it empty.

Keep `.env` private and out of Git. Never paste connection strings or credentials into issues, pull requests, chat, logs, or screenshots.

## 2. Start and test with Docker Compose (Step 3)

Docker Compose builds and runs the API container; PostgreSQL is hosted by Supabase, not in Docker. In the repository root, run:

```sh
docker compose up --build
```

In another terminal, check the database-aware readiness endpoint and API docs:

```sh
curl --fail http://localhost:8000/ready
```

On PowerShell, use `Invoke-RestMethod http://localhost:8000/ready`. A successful response is `{"status":"ready","database":"connected"}`; a database connection failure returns HTTP 503. `/health` is a liveness check and does not test the database. You can also inspect the API at `http://localhost:8000/docs`. Stop Compose with Ctrl+C or `docker compose down`; this does not delete data in Supabase.

## 3. Launch the same app in local Kubernetes (Step 10)

Ensure Docker Desktop/Engine is running. Start Minikube and build the backend image into the cluster:

```sh
minikube start
minikube image build -t scoreline-backend:local -f backend/Dockerfile backend
```

Create the namespace, then create/update a Kubernetes Secret from the same root `.env` values. This command sends all entries in `.env` to the backend container; do not use it if that file contains unrelated secrets:

```sh
kubectl apply -f kubernetes/namespace.yaml
kubectl create secret generic scoreline-secrets --namespace scoreline --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -
```

Alternatively, use `kubernetes/secrets.yaml` as a reference when creating a secret through your preferred secret manager. It is only a placeholder template; never apply it with placeholder values or commit populated credentials. Kubernetes Secret objects are not encrypted merely because Kubernetes stores them as Secrets.

Apply the configuration, backend, and service. There is no in-cluster PostgreSQL deployment: the backend connects directly to Supabase.

```sh
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/backend-deployment.yaml
kubectl apply -f kubernetes/backend-service.yaml
kubectl -n scoreline rollout status deployment/scoreline-backend
```

Forward the service to localhost:

```sh
kubectl -n scoreline port-forward service/scoreline-backend 8000:8000
```

In another terminal, verify the API-to-Supabase connection:

```sh
curl --fail http://localhost:8000/ready
```

On PowerShell, use `Invoke-RestMethod http://localhost:8000/ready`. A successful response is `{"status":"ready","database":"connected"}`. Inspect pods and logs with `kubectl -n scoreline get pods` and `kubectl -n scoreline logs deployment/scoreline-backend`.

To remove the local development deployment:

```sh
kubectl delete namespace scoreline
minikube stop
```

This does not remove or modify your Supabase project or its database contents. Live football API requests also need a valid provider key.

## Continuous integration

GitHub Actions runs the repository secret scan and builds the backend Docker image for pushes and pull requests. This checks that the Docker build context and Dockerfile remain usable; deploying to a remote registry is intentionally not part of CI.
