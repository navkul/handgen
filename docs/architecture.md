# Architecture

This document describes the current Handgen MVP architecture. Keep it aligned with the codebase whenever a commit changes system boundaries, persistence, rendering routes, model integration, or artifact contracts.

## Product Boundary

Handgen is currently a local-first Python CLI for proving the core handwriting rendering pipeline before any deployed product work. The MVP has no frontend, backend service, subscription system, auth, cloud storage, queue, or multi-user runtime.

The product direction is a subscription site where users can turn ordinary source material into handwritten notes, letters, study material, prose, and math-based proofs in their own writing style. The system must not be optimized for signatures, checks, IDs, contracts, legal documents, financial documents, or deceptive identity-use cases.

## Core Components

- `handgen/cli.py` exposes the local CLI commands and connects the database, filesystem assets, worksheet ingestion, and document rendering.
- `handgen/db.py` owns the SQLite schema and metadata helpers.
- `handgen/parser.py` parses source text and `[[...]]` math spans into a render plan that preserves source text.
- `handgen/ingest/worksheet.py` rasterizes worksheet inputs and extracts labeled glyph-bank crops.
- `handgen/diffbrush/runner.py` loads the DiffBrush model in-process and reuses it across every prose chunk and variable in a render, plus candidate generation, OCR reranking, and isolated component extraction.
- `handgen/render/document.py` orchestrates document rendering, source-preservation checks, SVG composition, PNG export, and output manifests.
- `handgen/render/math.py` composes math spans from worksheet glyphs and selected DiffBrush components.
- `handgen/render/ink.py` normalizes ink alpha, crop trimming, and handwritten asset appearance.
- `handgen/eval/ocr.py` and `handgen/eval/source_contract.py` provide lightweight verification around generated text and manifest source reconstruction.

## Persistence Model

SQLite stores metadata only. Large and inspectable artifacts stay on disk.

- Database: `data/handgen.sqlite3`
- Style references: `data/style_refs/<writer_id>/`
- Worksheets and glyph banks: `data/worksheets/<writer_id>/`
- Render outputs: `outputs/<job_name>/`

Generated manifests are part of the debugging and evaluation surface. They should preserve source text, source hashes, render routes, selected artifacts, warnings, and source-contract results.

## Rendering Model

SVG is the canonical document output because it keeps layout, embedded images, and composition metadata inspectable. PNG is a derivative rendered from SVG with `rsvg-convert`.

The current rendering split is:

- Prose outside `[[...]]` routes through source-locked DiffBrush generation.
- Math spans inside `[[...]]` route through math-aware rendering.
- Inline ASCII variables route through worksheet glyph-bank composition.
- Greek, relation symbols, operators, and punctuation route through worksheet glyph-bank composition.
- Digits currently use isolated DiffBrush connected components and are called out as a known MVP limit.
- Raw DiffBrush math should not be treated as reliable source-preserving math output.

## External Dependencies

DiffBrush is vendored into this repo by default:

- `models/diffbrush/third_party_repo/`
- `models/diffbrush/third_party_repo/model_zoo/DiffBrush-ckpt.pt`
- `handgen/diffbrush/run_single.py`

Runtime overrides are supported with:

- `DIFFBRUSH_ROOT`
- `DIFFBRUSH_CHECKPOINT`
- `DIFFBRUSH_DEVICE` (default `auto`; falls back from `mps` to `cpu` on failure)
- `DIFFBRUSH_STEPS` (default `20` DDIM sampling steps)
- `HANDGEN_DIFFBRUSH_PROSE_CANDIDATES` (default `4`; more candidates improve selection opportunities but increase render time)

DiffBrush is invoked in-process. The model is built once on the first generate call in a render and held in memory for the rest of that render, so each subsequent chunk avoids the model-load cost. This replaces the earlier subprocess-per-chunk approach and is roughly 3.75× faster end-to-end on a 3-chunk test.

Image processing depends on Pillow and NumPy. PNG export depends on `rsvg-convert` being available at runtime.

## Safety and Scope Constraints

Do not add deployed-product infrastructure until the local CLI pipeline is reliable. Do not commit model weights, generated output directories, local SQLite databases, virtual environments, or unrelated research workspace changes. Reusable writer assets and worksheet-derived crops may be committed only when they are safe to share.

## Documentation Update Example

If a commit adds a web API, this file must change because the product boundary and component model changed. Add the service to Core Components, update Product Boundary, and describe any new persistence or runtime dependencies.

If a commit only fixes spacing math in `handgen/render/math.py` without changing the route, artifact shape, or system boundary, this file probably does not need an edit. Update `docs/pipeline.md` or `docs/findings.md` only if the behavior or finding changed.
