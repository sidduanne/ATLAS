FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models DISABLE_SAFETENSORS_CONVERSION=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch by default (avoids multi-GB CUDA wheels). For an NVIDIA host,
# build with TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "$TORCH_INDEX_URL" "torch>=2.2,<3"
COPY backend/requirements.txt ./backend/
RUN pip install -r backend/requirements.txt

# Bake both frozen encoders (~700 MB) into the image so the first search is fast
# and the server needs no model downloads at runtime. PRELOAD_MODELS=0 skips this.
ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import CLIPModel, CLIPProcessor; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"; \
    else mkdir -p /app/data && ln -s /app/data/models /opt/models; fi
# Baked models: never contact the Hugging Face Hub at runtime (works on offline servers).
ENV HF_HUB_OFFLINE=${PRELOAD_MODELS}

COPY backend ./backend
COPY experiments ./experiments
COPY scripts ./scripts
ENV PYTHONPATH=/app/backend DATA_DIR=/app/data
EXPOSE 8001
# One worker on purpose: experiment-job tracking and latency stats are process-local.
CMD ["sh", "-c", "python -c 'from app.models import init_db; init_db()' && exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --proxy-headers --forwarded-allow-ips='*'"]
