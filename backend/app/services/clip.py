"""Frozen CLIP encoders. Model weights load lazily; no inference at app startup."""
import os
from functools import lru_cache

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from app.config import DATA_DIR

# Avoid the hub's optional background conversion service; inference only needs
# the published weights and must not start a conversion job during a request.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
MODEL = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def get_clip():
    try:
        processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True)
        model = CLIPModel.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        cache = str(DATA_DIR / "models")
        processor = CLIPProcessor.from_pretrained(MODEL, cache_dir=cache)
        model = CLIPModel.from_pretrained(MODEL, cache_dir=cache)
    model.eval()
    model.requires_grad_(False)
    return processor, model


def _encode(images=None, text=None):
    processor, model = get_clip()
    with torch.inference_mode():
        if images is not None:
            inputs = processor(images=images, return_tensors="pt")
            vectors = model.get_image_features(**inputs)
        else:
            inputs = processor(text=text, padding=True, truncation=True, max_length=77, return_tensors="pt")
            vectors = model.get_text_features(**inputs)
        # Recent transformers versions may return a model output.
        if not isinstance(vectors, torch.Tensor):
            vectors = vectors.pooler_output
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    result = vectors.cpu().numpy()
    if result.ndim != 2 or result.shape[1] != 512 or not np.isfinite(result).all():
        raise ValueError("CLIP returned invalid embeddings")
    return result.tolist()


def embed_images(images):
    return _encode(images=images) if images else []


def embed_image_query(query):
    return _encode(text=[query])[0]
