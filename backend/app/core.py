"""Settings, database session, the user-facing error type and the data-directory guard."""
import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set (Docker Compose sets it for the backend container).")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class UserError(Exception):
    """An expected failure caused by the request (bad file, empty query, ...).

    Services raise it without knowing about HTTP; main.py renders it as {"detail": message}.
    """

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def data_path(relative: str) -> Path:
    """Resolve a stored relative path, refusing anything outside DATA_DIR."""
    path = (DATA_DIR / relative).resolve()
    if not path.is_relative_to(DATA_DIR):
        raise ValueError("Invalid asset path")
    return path
