from collections import defaultdict
from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal
from app.models.document import Document


SEARCH_ROOTS = [
    Path.home() / "Downloads",
    Path.home() / "Documents",
    Path.home() / "Desktop",
]


def main():
    print("Scanning local PDF folders...", flush=True)

    files_by_name = defaultdict(list)

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                files_by_name[path.name].append(path.resolve())

    with SessionLocal() as db:
        documents = db.scalars(
            select(Document).order_by(Document.id)
        ).all()

        records = [
            {
                "id": doc.id,
                "filename": doc.filename,
                "title": doc.title,
            }
            for doc in documents
        ]

    matched = 0
    missing = 0
    ambiguous = 0

    output_path = Path("pdf_match_report.tsv").resolve()

    with output_path.open("w", encoding="utf-8") as report:
        report.write("document_id\tfilename\tstatus\tlocal_paths\n")

        for doc in records:
            paths = files_by_name.get(doc["filename"], [])

            if len(paths) == 1:
                status = "MATCHED"
                matched += 1
            elif len(paths) == 0:
                status = "MISSING"
                missing += 1
            else:
                status = "AMBIGUOUS"
                ambiguous += 1

            path_text = " | ".join(str(p) for p in paths)

            report.write(
                f'{doc["id"]}\t{doc["filename"]}\t'
                f"{status}\t{path_text}\n"
            )

            print(
                f'[{status}] ID={doc["id"]} '
                f'{doc["filename"]} ({len(paths)} local matches)'
            )

    print("\nMatching summary")
    print(f"Database records: {len(records)}")
    print(f"Unique local filenames: {len(files_by_name)}")
    print(f"Exactly one local match: {matched}")
    print(f"No local match: {missing}")
    print(f"Multiple local matches: {ambiguous}")
    print(f"Report: {output_path}")
    print("Database was not modified.")


if __name__ == "__main__":
    main()
