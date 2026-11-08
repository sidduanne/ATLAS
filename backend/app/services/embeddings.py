from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    model = SentenceTransformer(MODEL_NAME)
    model.eval()
    model.requires_grad_(False)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert text passages into normalized 384-dimensional vectors."""
    if not texts:
        return []

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 10,
    )

    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Convert one search query into a vector."""
    return embed_texts([query])[0]
