# What you need to do next

The application and experiment tooling are implemented. Human judgment is the remaining prerequisite for the research results.

Open http://127.0.0.1:5173 and refresh the page if it was already open.

## 1. Label items in Review

Start with **Text passages**. Read each item in the context of your technical corpus and choose:

| Label | Meaning |
| --- | --- |
| relevant | Readable, useful technical content within the intended domain |
| duplicate | A redundant copy of an item already represented in the corpus; check context/source |
| malformed | Broken extraction, illegible text, or an unusable figure crop |
| out_of_domain | Content outside the project's declared technical domain |
| irrelevant | Within the broad domain but not useful content, such as navigation or boilerplate |

Use notes to record your reasoning. If an item is ambiguous, choose **Skip uncertain item for now** rather than guessing. Reset skipped items when ready to revisit them. Skipping creates no label. Correct labels using Recent labels.

The live classifier starts after **4 labels spanning at least 2 classes** for that modality. The experiment requires **at least 12 labels**, at least 2 classes, and sufficient independent source documents for a held-out split. Twelve is only an execution minimum, not a credible study size. Aim for **50–100+ honestly judged text items**, with several independent documents per class. Label figures separately if you also want an image-classifier experiment.

Do not assign incorrect labels just to balance classes. If nearly all real items are relevant, add and manually judge controlled anomaly examples; the existing `scripts/prepare_anomalies.py` prepares examples without inventing human labels. Duplicate and malformed items can overlap; use a consistent annotation rule and document it. Duplicate status may require information beyond the cached embedding.

## 2. Judge retrieval relevance in Experiments

Expand **Human relevance judgments**. Six draft queries have been prepared: three text and three image queries. Each has an unjudged pool of retrieved candidates.

1. Select a draft query.
2. Read each candidate and inspect its source or figure.
3. Click **Relevant** or **Not relevant** relative to that specific question.
4. After judging every candidate and identifying at least one relevant result, click **Verify completed judgments**.
5. Add more independent evaluation queries for a useful held-out study.

These binary query judgments are separate from the five anomaly labels in Review. No suggested relevance or anomaly label has been treated as human ground truth.

Pooling takes the top 5 results from each applicable retrieval method and removes duplicate IDs. Recall is measured relative to the judged pool, which may miss relevant items outside it. Expand the query set and judge thoroughly before making conclusions.

## 3. Run the experiments

The readiness panel explains missing prerequisites. Once labels/judgments are sufficient:

- **Evaluate judged queries** runs text/CLIP evaluation against an immutable snapshot of the verified query set.
- **Compare active learning** uses the selected passage or figure labels, fixed held-out source/duplicate groups, equal budgets, and multiple seeds.
- **Benchmark index** runs exact versus HNSW on increasing actual corpus sizes.

The backend runs one experiment at a time in the background. Keep it running; run-status tracking resets on restart. Status and diagnostic output appear in the panel. After completion, click **Refresh results** to reload dashboard reports. The CLI runners remain available.

The system will not fabricate an active-learning improvement if uncertainty sampling performs worse or makes no difference.

## 4. Complete source metadata

Automatic recovery restored **66 source PDFs** and indexed **2,034 figure crops**. The original 91 document records and 6,409 passages were preserved.

The remaining **25 sources** need attention: **23 ambiguous matches**, plus **2 missing/text-mismatched files**. In Library, attach the correct original file. The source-recovery report in `experiments/results/source_recovery.json` identifies the records. Cropping is heuristic: inspect figures and label unusable crops as malformed.

Use **Edit source metadata** in Library to record the original source URL and actual reuse license. Recovery does not infer a license from a filename or hosting site. All 66 restored sources currently need their license recorded.

## Verification

The updated implementation passed **17 automated tests** and a React/TypeScript production build. Tests include judgment verification/invalidation, blocked experiments without human ground truth, bounded run submission, skipping without labels, real model inference, and sanitization of NUL characters found in real PDF captions.
