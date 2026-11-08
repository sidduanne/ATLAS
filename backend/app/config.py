import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
RESULTS_DIR = ROOT / "experiments" / "results"
CORS_ORIGINS = [origin for origin in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin]

LABELS = ("relevant", "duplicate", "malformed", "out_of_domain", "irrelevant")

MB = 1024 * 1024
MAX_PDF_BYTES = 500 * MB
MAX_BATCH_BYTES = 2048 * MB
MAX_BATCH_FILES = 50
MAX_IMAGE_BYTES = 20 * MB
MAX_IMAGE_PIXELS = 20_000_000
EMBED_BATCH_SIZE = 32


def data_path(relative: str) -> Path:
    path = (DATA_DIR / relative).resolve()
    if not path.is_relative_to(DATA_DIR):
        raise ValueError("Invalid asset path")
    return path
