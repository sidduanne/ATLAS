from sqlalchemy import select

from app.database import SessionLocal
from app.models.document import Passage
from app.services.embeddings import embed_texts


BATCH_SIZE = 32


def main():
    total_updated = 0

    with SessionLocal() as db:
        while True:
            passages = db.scalars(
                select(Passage)
                .where(Passage.embedding.is_(None))
                .order_by(Passage.id)
                .limit(BATCH_SIZE)
            ).all()

            if not passages:
                break

            vectors = embed_texts([p.content for p in passages])

            for passage, vector in zip(passages, vectors):
                passage.embedding = vector

            db.commit()
            total_updated += len(passages)

            print(f"Embedded {total_updated} passages so far.")

    print(f"Finished. Total passages embedded: {total_updated}")


if __name__ == "__main__":
    main()
