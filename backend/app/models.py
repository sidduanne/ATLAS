"""Database tables. The original documents/passages tables are kept; the rest were added alongside them."""
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core import Base, engine


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(30), default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    passages: Mapped[list["Passage"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Passage(Base):
    __tablename__ = "passages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))  # Sentence-BERT MiniLM
    document: Mapped[Document] = relationship(back_populates="passages")


class SourceAsset(Base):
    """The retained original file of a document."""
    __tablename__ = "source_assets"
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    storage_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    media_type: Mapped[str] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(200))


class Figure(Base):
    __tablename__ = "figures"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int | None] = mapped_column(Integer)
    caption: Mapped[str] = mapped_column(Text, default="")
    storage_path: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(512))  # CLIP


class ImageText(Base):
    """Cached OCR outcome for a figure."""
    __tablename__ = "image_text"
    figure_id: Mapped[int] = mapped_column(ForeignKey("figures.id", ondelete="CASCADE"), primary_key=True)
    text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30))
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewLabel(Base):
    """Current human label of a passage or figure (one per item)."""
    __tablename__ = "review_labels"
    __table_args__ = (UniqueConstraint("item_type", "item_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_type: Mapped[str] = mapped_column(String(20))
    item_id: Mapped[int] = mapped_column(Integer, index=True)
    label: Mapped[str] = mapped_column(String(30))
    note: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ReviewEvent(Base):
    """Append-only audit trail of every label submission."""
    __tablename__ = "review_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_type: Mapped[str] = mapped_column(String(20))
    item_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(30))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvaluationQuery(Base):
    __tablename__ = "evaluation_queries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(String(500))
    modality: Mapped[str] = mapped_column(String(10))
    verified: Mapped[bool] = mapped_column(default=False)


class EvaluationCandidate(Base):
    """A pooled search result awaiting (or holding) a human relevance judgment."""
    __tablename__ = "evaluation_candidates"
    __table_args__ = (UniqueConstraint("query_id", "item_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("evaluation_queries.id", ondelete="CASCADE"), index=True)
    item_id: Mapped[int] = mapped_column(Integer)
    relevant: Mapped[bool | None] = mapped_column(nullable=True)


def init_db():
    """Create missing tables and search indexes. Never drops or rewrites existing data."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        for statement in (
            "CREATE INDEX IF NOT EXISTS passages_embedding_hnsw ON passages USING hnsw (embedding vector_cosine_ops)",
            "CREATE INDEX IF NOT EXISTS figures_embedding_hnsw ON figures USING hnsw (embedding vector_cosine_ops)",
            "CREATE INDEX IF NOT EXISTS passages_keyword_gin ON passages USING gin (to_tsvector('english', content))",
        ):
            connection.execute(text(statement))
