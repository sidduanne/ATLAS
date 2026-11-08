"""Add missing tables without deleting or rewriting existing documents/passages."""
import argparse
from sqlalchemy import text

from app.database import Base, engine
import app.models  # Registers every additive table.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexes", action="store_true", help="Also create cosine HNSW and keyword GIN indexes")
    args = parser.parse_args()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    if args.indexes:
        with engine.begin() as connection:
            for statement in (
                "CREATE INDEX IF NOT EXISTS passages_embedding_hnsw ON passages USING hnsw (embedding vector_cosine_ops)",
                "CREATE INDEX IF NOT EXISTS figures_embedding_hnsw ON figures USING hnsw (embedding vector_cosine_ops)",
                "CREATE INDEX IF NOT EXISTS passages_keyword_gin ON passages USING gin (to_tsvector('english', content))",
            ):
                connection.execute(text(statement))
    print("Schema ready; existing documents and passages preserved.")


if __name__ == "__main__":
    main()
