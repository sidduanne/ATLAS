#!/usr/bin/env bash
# One-time local setup: .env, database container, Python venv, schema, frontend deps.
# Safe to re-run; it never deletes data.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }
set -a; . ./.env; set +a

echo "==> Starting PostgreSQL + pgvector (127.0.0.1:5433)"
docker compose up -d db
for _ in $(seq 1 60); do
  [ "$(docker inspect -f '{{.State.Health.Status}}' multimodal-retrieval-db 2>/dev/null)" = healthy ] && break
  sleep 1
done

echo "==> Python environment (.venv)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" "torch>=2.2,<3"
.venv/bin/python -m pip install -r backend/requirements-dev.txt

echo "==> Database schema and indexes"
PYTHONPATH=backend .venv/bin/python backend/init_db.py --indexes

echo "==> Frontend dependencies"
npm --prefix frontend ci

if ! command -v tesseract >/dev/null; then
  echo "NOTE: Tesseract not found. OCR for scanned PDFs/images is disabled until you run:"
  echo "      sudo apt install tesseract-ocr tesseract-ocr-eng"
fi
echo "Setup complete. Optional: PYTHONPATH=backend .venv/bin/python backend/download_models.py"
echo "Start the app with: scripts/run_local.sh"
