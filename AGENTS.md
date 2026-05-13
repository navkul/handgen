# AGENTS.md

## Long-Term Goal

This folder is the staging area for a future standalone product repo.

The long-term product is a deployed subscription site where users can turn source material into their own handwriting. Source material may include:

- typed text,
- screenshots,
- GenAI chat transcripts,
- copied notes or proofs,
- mixed English and math.

The product should let a user upload handwriting references and a symbol/math worksheet, then generate downloadable handwritten documents that preserve the source content while matching the user's writing style.

The product must not be optimized for signatures, checks, IDs, contracts, legal documents, financial documents, or other deceptive identity-use cases. Focus on ordinary notes, letters, study material, prose, and math-based proofs.

## Current MVP

The current MVP is local-first and has no UI.

Its goal is to prove the core rendering pipeline before building the deployed app:

1. Store one user's natural handwriting style reference locally.
2. Later ingest a user-written digit/symbol worksheet.
3. Accept a text file containing English and optional math spans.
4. Generate English prose with DiffBrush in short chunks.
5. Generate English variables inside math spans with DiffBrush context prompts.
6. Generate digits, Greek, and non-language math symbols from the worksheet glyph bank with slight randomized variation and ink normalization.
7. Compose the final handwritten document as SVG and PNG.
8. Save machine-readable manifests for debugging and later evaluation.

Math spans are marked with `[[...]]`. Text outside those delimiters is prose. Text inside those delimiters is symbol/math content.

## Current Stack

- Python package and CLI under `handgen/`
- SQLite metadata database at `data/handgen.sqlite3`
- Filesystem storage for input assets, generated crops, SVGs, PNGs, logs, and manifests
- DiffBrush checkpoint and vendor code referenced from `../04_cpu_neural_scout/vendor/DiffBrush`
- SVG as canonical output
- PNG rendered from SVG with `rsvg-convert`

Do not add a frontend, web backend, subscription system, auth, cloud storage, or queue until the local MVP pipeline is reliable.

## Operating Rules

- Treat this `repo/` folder as the only writable product repo area.
- Do not modify the older research agent folders unless explicitly asked.
- Prefer clean product modules over copying entire research scripts.
- Keep large model weights out of this repo.
- Keep generated images and manifests inspectable.
- Store metadata in SQLite; store large files on disk.
- Preserve exact source text in sidecars even when DiffBrush prose is not visually exact.
- Do not present raw DiffBrush math as correct; English variables may route through DiffBrush context-component generation, while digits, Greek, and relation/operator symbols should route through worksheet glyph-bank composition.

## Useful Commands

```bash
python3 -m handgen init
python3 -m handgen writer create writer_001
python3 -m handgen style add writer_001 data/style_refs/writer_001/natural_prose_primary.png
python3 -m handgen document render writer_001 examples/prose_only.txt --out outputs/prose_only
```

After a new worksheet exists:

```bash
python3 -m handgen worksheet ingest writer_001 data/worksheets/writer_001/symbol_worksheet_01.png
python3 -m handgen document render writer_001 examples/mixed_proof.txt --out outputs/mixed_proof
```

## Near-Term Priorities

1. Make worksheet ingestion robust for the new worksheet format.
2. Improve DiffBrush prose chunking, cropping, and reranking.
3. Improve page layout for longer documents.
4. Add lightweight verification for source preservation and missing glyph coverage.
5. Keep the CLI workflow stable before moving this folder into a standalone repository.
