"""Integration tests use isolated PostgreSQL schemas, never the user's corpus."""
import hashlib
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.core
import app.main
import app.models  # noqa: F401  (registers every table on Base)
import app.services.encoders
from app.core import Base, engine, get_db


def vector(value, dimension=384):
    rng = np.random.default_rng(int(hashlib.sha256(value.encode()).hexdigest()[:8], 16))
    values = rng.normal(size=dimension)
    return (values/np.linalg.norm(values)).tolist()


@pytest.fixture
def db(tmp_path, monkeypatch):
    schema = "atlas_test_" + uuid4().hex
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    test_engine = engine.execution_options(schema_translate_map={None: schema})
    Base.metadata.create_all(test_engine)
    monkeypatch.setattr(app.core, "DATA_DIR", tmp_path)
    try:
        with Session(test_engine) as session:
            yield session
    finally:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


@pytest.fixture
def client(db, monkeypatch):
    # Every caller goes through app.services.encoders, so patching it replaces all models.
    encoders = app.services.encoders
    monkeypatch.setattr(encoders, "embed_texts", lambda texts: [vector(t) for t in texts])
    monkeypatch.setattr(encoders, "embed_query", vector)
    monkeypatch.setattr(encoders, "embed_images", lambda images: [vector(str(i), 512) for i, _ in enumerate(images)])
    monkeypatch.setattr(encoders, "embed_image_query", lambda q: vector("0", 512))
    def session_override():
        try:
            yield db
        except Exception:
            db.rollback()
            raise
    app.main.app.dependency_overrides[get_db] = session_override
    with TestClient(app.main.app) as test_client:
        yield test_client
    app.main.app.dependency_overrides.clear()
