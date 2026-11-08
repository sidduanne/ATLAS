"""Restore legacy sources from the existing match report after page-text checks.

Only MATCHED rows with one readable PDF and matching stored passage text qualify.
No broad filesystem scan. Ambiguous files are left for human selection.
"""
import argparse
import csv
import json
from pathlib import Path

import pymupdf
from sqlalchemy import select

from app.database import SessionLocal
from app.models import Document, Passage, SourceAsset
from app.services.assets import attach_pdf


def verify_source(path, document, db):
    samples = db.scalars(select(Passage).where(Passage.document_id == document.id).order_by(Passage.id).limit(3)).all()
    usable = [p for p in samples if p.page_number and len(p.content.split()) >= 8]
    if not usable:
        return False
    with pymupdf.open(path) as pdf:
        for passage in usable:
            if passage.page_number > len(pdf):
                return False
            stored = ' '.join(passage.content.lower().split())[:160]
            extracted = ' '.join(pdf[passage.page_number-1].get_text().lower().split())
            if stored not in extracted:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('pdf_match_report.tsv'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/source_recovery.json'))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--limit', type=int, default=10, help='Maximum sources to attach in one run')
    args = parser.parse_args()
    results = []
    applied = 0
    with SessionLocal() as db, args.report.open() as report:
        for row in csv.DictReader(report, delimiter='\t'):
            record = {'document_id': int(row['document_id']), 'filename': row['filename']}
            document = db.get(Document, record['document_id'])
            if document is None or document.filename != row['filename']:
                record['status'] = 'stale_record'
            elif db.get(SourceAsset, document.id):
                record['status'] = 'already_attached'
            elif row['status'] != 'MATCHED' or ' | ' in row['local_paths']:
                record['status'] = 'needs_manual_selection'
            else:
                path = Path(row['local_paths'])
                try:
                    if not path.is_file() or path.suffix.lower() != '.pdf' or not verify_source(path, document, db):
                        record['status'] = 'missing_or_text_mismatch'
                    elif args.apply and applied < args.limit:
                        record.update(attach_pdf(path, document, db)); record['status'] = 'restored'; applied += 1
                        print(f"Restored #{document.id}: {document.filename} ({record.get('figures_created', 0)} figures)", flush=True)
                    else:
                        record['status'] = 'verified_candidate'
                except Exception as exc:
                    db.rollback(); record.update(status='failed', error=str(getattr(exc, 'orig', exc))[:500])
            results.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'applied': args.apply, 'results': results}, indent=2))
    from collections import Counter
    print(json.dumps(dict(Counter(r['status'] for r in results)), indent=2))


if __name__ == '__main__':
    main()
