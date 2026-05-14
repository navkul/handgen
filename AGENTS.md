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
5. Generate English prose and slot-carrier lines with DiffBrush in short chunks.
6. Generate inline ASCII variables, Greek, and non-language math symbols from the worksheet glyph bank with slight randomized variation and ink normalization. Digits currently use isolated DiffBrush connected components.
7. Compose the final handwritten document as SVG and PNG.
8. Save machine-readable manifests for debugging and later evaluation.

Math spans are marked with `[[...]]`. Text outside those delimiters is prose. Text inside those delimiters is symbol/math content.

## Current Stack

- Python package and CLI under `handgen/`
- SQLite metadata database at `data/handgen.sqlite3`
- Filesystem storage for input assets, generated crops, SVGs, PNGs, logs, and manifests
- DiffBrush vendor code under `models/diffbrush/third_party_repo/`
- DiffBrush checkpoint stored locally at `models/diffbrush/third_party_repo/model_zoo/DiffBrush-ckpt.pt` and ignored by Git
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
- Do not present raw DiffBrush math as correct; ASCII variables, Greek, and relation/operator symbols should route through worksheet glyph-bank composition. Digits are still DiffBrush components for now and must remain documented as a known MVP limit until moved to worksheet rendering.

## Documentation Rules

The docs in `docs/` are part of the product state, not optional notes. For every commit, check whether the code, data contracts, pipeline behavior, assumptions, tradeoffs, or known findings changed. If they did, update the relevant docs in the same commit.

This does not mean every commit needs a new doc entry. It means the committed docs must describe the current codebase after the commit lands.

Current docs:

- `docs/architecture.md` describes system boundaries, core components, persistence, rendering routes, external dependencies, and scope constraints.
- `docs/pipeline.md` describes the end-to-end CLI, ingestion, rendering, output, and verification flow.
- `docs/findings.md` records current empirical findings, limitations, open questions, and resolved understanding.
- `docs/decisions.md` records durable decisions and tradeoffs. It is not a changelog.

Examples:

- If a commit changes worksheet row detection, update `docs/pipeline.md` if the ingestion flow changed and `docs/findings.md` if the supported worksheet assumptions changed.
- If a commit changes where digits, Greek, operators, or variables are rendered from, update `docs/architecture.md`, `docs/pipeline.md`, and the relevant decision in `docs/decisions.md`.
- If a commit changes manifest fields, source-contract behavior, or OCR gating, update `docs/pipeline.md` and `docs/decisions.md` if the source-preservation contract changed.
- If a commit only renames a private helper or reformats code without changing behavior, no docs edit is required.
- Do not add `CHANGELOG.md` until the project has versioned releases, external users, or a standalone package boundary.

## Useful Commands

```bash
python -m handgen init
python -m handgen writer create writer_001
python -m handgen style add writer_001 data/style_refs/writer_001/natural_prose_primary.png
python -m handgen document render writer_001 examples/prose_only.txt --out outputs/prose_only
```

After a new worksheet exists:

```bash
python -m handgen worksheet ingest writer_001 data/worksheets/writer_001/writer_001_worksheet_01.png
python -m handgen document render writer_001 examples/mixed_proof.txt --out outputs/mixed_proof
```

## Near-Term Priorities

1. Make worksheet ingestion robust for the new worksheet format.
2. Improve DiffBrush prose chunking, cropping, and reranking.
3. Improve page layout for longer documents.
4. Add lightweight verification for source preservation and missing glyph coverage.
5. Keep the CLI workflow stable before moving this folder into a standalone repository.
