"""Ingest a curated local manifest with explicit provenance and reuse licenses."""
import argparse
import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.core import SessionLocal
from app.models import SourceAsset
from app.services.ingestion import ingest_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--apply', action='store_true', help='Without this flag, validate and preview only')
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())['documents']
    prepared = []
    for entry in entries:
        path = (args.manifest.resolve().parent / entry['path']).resolve()
        if not path.is_file() or path.suffix.lower() != '.pdf':
            parser.error(f'Expected a local PDF: {path}')
        if not entry.get('source_url') or not entry.get('license'):
            parser.error(f'Provide source_url and an actual reuse license for {path.name}')
        prepared.append((entry, path, hashlib.sha256(path.read_bytes()).hexdigest()))
    with SessionLocal() as db:
        for entry, path, checksum in prepared:
            previous = db.scalar(select(SourceAsset).where(SourceAsset.sha256 == checksum))
            if previous:
                print(f'SKIP {path.name}: already stored as document {previous.document_id}')
                continue
            if not args.apply:
                print(f'WOULD IMPORT {path.name}: {entry["license"]} / {entry["source_url"]}')
                continue
            result = ingest_pdf(path, path.name, db)
            source = db.get(SourceAsset, result['document_id'])
            source.source_url = entry['source_url']; source.license = entry['license']
            db.commit()
            print(json.dumps(result))


if __name__ == '__main__':
    main()
