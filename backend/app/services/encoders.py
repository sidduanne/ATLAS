"""Frozen encoders: Sentence-BERT (MiniLM, 384-d) for text and CLIP (512-d) for images and image queries.

Models load lazily on first use and are never trained. All callers go through these
functions (as `encoders.embed_texts(...)`), so tests can replace them in one place.
"""
import os
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from app.core import DATA_DIR

# Inference only needs published weights; never start a hub conversion job during a request.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def text_model() -> SentenceTransformer:
    model = SentenceTransformer(TEXT_MODEL)
    model.eval()
    model.requires_grad_(False)
    return model


@lru_cache(maxsize=1)
def clip_model():
    try:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
        model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True)
    except OSError:  # Not baked into the image: download into the data volume.
        cache = str(DATA_DIR / "models")
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL, cache_dir=cache)
        model = CLIPModel.from_pretrained(CLIP_MODEL, cache_dir=cache)
    model.eval()
    model.requires_grad_(False)
    return processor, model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Normalized 384-d vectors for passages."""
    if not texts:
        return []
    vectors = text_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) > 10)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def _clip_encode(images=None, text=None) -> list[list[float]]:
    processor, model = clip_model()
    with torch.inference_mode():
        if images is not None:
            vectors = model.get_image_features(**processor(images=images, return_tensors="pt"))
        else:  # CLIP's text encoder accepts at most 77 tokens.
            inputs = processor(text=text, padding=True, truncation=True, max_length=77, return_tensors="pt")
            vectors = model.get_text_features(**inputs)
        if not isinstance(vectors, torch.Tensor):  # Newer transformers return a model output.
            vectors = vectors.pooler_output
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    result = vectors.cpu().numpy()
    if result.ndim != 2 or result.shape[1] != 512 or not np.isfinite(result).all():
        raise ValueError("CLIP returned invalid embeddings")
    return result.tolist()


def embed_images(images) -> list[list[float]]:
    """Normalized 512-d CLIP vectors for PIL images."""
    return _clip_encode(images=images) if images else []


def embed_image_query(query: str) -> list[float]:
    """Normalized 512-d CLIP text vector, comparable with image vectors."""
    return _clip_encode(text=[query])[0]
