"""Download and cache both frozen encoders ahead of time, so first searches are fast."""
from app.services.clip import get_clip
from app.services.embeddings import get_embedding_model


def main():
    get_embedding_model()
    print("Sentence-BERT ready")
    get_clip()
    print("CLIP ready")


if __name__ == "__main__":
    main()
