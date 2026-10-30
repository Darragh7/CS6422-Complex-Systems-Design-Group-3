#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install Python 3.11 or newer and rerun this script." >&2
  exit 1
fi

if [ ! -d backend ] || [ ! -f backend/requirements.txt ]; then
  echo "Run setup.sh from the project checkout; backend/requirements.txt is missing." >&2
  exit 1
fi

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit it with local settings before starting the app."
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt

cat <<'EOF'

Setup complete.
Run the API with:
  .venv/bin/python -m uvicorn app.main:app --app-dir backend --reload

Run the Docker development stack with:
  docker compose up --build
EOF
