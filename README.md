# Scoreline

## First-time setup

1. Install and start Docker Desktop (with Docker Compose).
2. From the project root, create your local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Edit `.env` and set `DATABASE_URL` to your Supabase PostgreSQL connection string and `SECRET_KEY` to a strong secret. Set `FOOTBALL_API_KEY` if you want to use the external football data service. Keep `.env` private; it is for local configuration.

## Run the development environment

From the project root, build and start both services:

```powershell
docker compose up --build
```

- Django website: <http://localhost:8000/webplay/>
- FastAPI documentation: <http://localhost:8002/docs>
- FastAPI/database readiness: <http://localhost:8002/ready>

Django serves the website, while FastAPI serves the API in a separate container. The API connects to Supabase using `DATABASE_URL`. To stop the services, press **Ctrl+C**. To start them in the background, use `docker compose up --build -d`, and stop them with `docker compose down`.


## Project docs

- [Kubernetes deployment guide](docs/development-deployment.md)
- [Development process](docs/DEV_PROCESS.pdf)
- [Security documentation](docs/security/)
- [Licensing review and prediction disclaimer](docs/security/licensing-and-disclaimers.md)
