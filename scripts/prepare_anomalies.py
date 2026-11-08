"""Prepare controlled annotation candidates; never treat suggested labels as human labels.

Writes a standalone JSON dataset for manual review. Import the candidate texts through
Library > Add documents > Text, then label through Review. No application writes.
"""
import argparse
import json
from pathlib import Path

from app.database import SessionLocal
from app.models import Passage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passage-id', type=int, required=True)
    parser.add_argument('--output', type=Path, default=Path('experiments/datasets/annotation_candidates.json'))
    args = parser.parse_args()
    with SessionLocal() as db:
        passage = db.get(Passage, args.passage_id)
        if passage is None:
            parser.error('Passage does not exist')
        text = passage.content
        source = passage.document_id
    candidates = [
        ('original', text, 'relevant'),
        ('duplicate', text, 'duplicate'),
        ('malformed', ' '.join(text.split()[::4]) + ' [CORRUPTED: missing words, broken extraction]', 'malformed'),
        ('out_of_domain', 'Recipe: cream butter and sugar, add flour, then bake the cookies until golden.', 'out_of_domain'),
        ('irrelevant', 'The venue opens at nine; refreshments are served near the registration desk.', 'irrelevant'),
        ('ambiguous', 'The model improves performance in some cases, but the measurements are not specified.', None),
    ]
    report = {'source_document_id': source, 'source_passage_id': args.passage_id,
              'notes': 'Controlled examples for manual review. Suggested labels are construction hints, not verified ground truth. Judge relevance against the declared technical domain. Exact duplicates must stay in the same experimental split.',
              'items': [{'id': name, 'content': content, 'suggested_label': label, 'human_label': None, 'verified': False} for name, content, label in candidates]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f'Prepared {len(candidates)} unverified candidates: {args.output}')


if __name__ == '__main__':
    main()
