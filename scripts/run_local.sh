#!/usr/bin/env bash
# Run database, backend (reload) and Vite dev server. Ctrl+C stops backend and frontend.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run scripts/setup_local.sh first"; exit 1; }

docker compose up -d db
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8001 &
backend=$!
trap 'kill $backend 2>/dev/null || true' EXIT INT TERM

echo "Backend:  http://127.0.0.1:8001/docs"
echo "Frontend: http://127.0.0.1:5173"
npm --prefix frontend run dev
