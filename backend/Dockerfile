FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch by default (avoids multi-GB CUDA wheels). For an NVIDIA host,
# build with --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "$TORCH_INDEX_URL" "torch>=2.2,<3"
COPY backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY backend ./backend
COPY experiments ./experiments
COPY scripts ./scripts
ENV PYTHONPATH=/app/backend DATA_DIR=/app/data HF_HOME=/app/data/models
EXPOSE 8001
# One worker on purpose: experiment-job tracking and latency stats are process-local.
CMD ["sh", "-c", "python backend/init_db.py --indexes && exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --proxy-headers --forwarded-allow-ips='*'"]
