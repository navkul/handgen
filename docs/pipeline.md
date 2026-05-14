# Pipeline

This document is the operational source of truth for how Handgen turns a text file into handwritten document artifacts. Keep it aligned with the code whenever a commit changes parse behavior, routing, generation, composition, verification, outputs, or manifests.

## CLI Entry Points

Initialize local state:

```bash
python3 -m handgen init
```

Create a writer:

```bash
python3 -m handgen writer create writer_001
```

Register a natural handwriting style reference:

```bash
python3 -m handgen style add writer_001 data/style_refs/writer_001/natural_prose_primary.png
```

Ingest a worksheet:

```bash
python -m handgen worksheet ingest writer_001 data/worksheets/writer_001/writer_001_worksheet_01.png
```

Render a document:

```bash
python -m handgen document render writer_001 examples/mixed_proof.txt --out outputs/mixed_proof
```

## Source Parsing

Input source is a UTF-8 text file. Math spans are marked with `[[...]]`.

- Text outside delimiters is prose.
- Text inside delimiters is math/symbol content.
- The parser strips the markup delimiters from the render source while preserving the intended source text.
- The parsed source text is recorded in the render manifest and protected by the source contract.

Example:

```text
Pick [[0 < a < b]]
Done
```

renders as source text:

```text
Pick 0 < a < b
Done
```

## Worksheet Ingestion

Worksheet ingestion currently assumes the worksheet follows the expected MVP row order. The ingest step:

1. Rasterizes the source worksheet if needed.
2. Builds a dark-pixel mask.
3. Groups connected components into rows.
4. Assigns rows to expected label specs.
5. Splits rows into glyph groups.
6. Saves row images, glyph-bank crops, and `manifest.json`.
7. Registers worksheet and glyph metadata in SQLite.

The worksheet manifest is the handoff between ingestion and rendering. It records row specs, extracted glyphs, crop paths, image sizes, hashes, warnings, and assumptions.

## Document Rendering

`handgen/render/document.py` orchestrates rendering.

1. Read source text from disk.
2. Parse source text into a render plan.
3. Look up the latest writer style reference in SQLite.
4. Detect whether math or carrier rendering is needed.
5. Look up the latest worksheet manifest when math glyphs are required.
6. Build a math style profile from the style reference and worksheet glyph measurements.
7. Generate or prepare visual assets for each line and span.
8. Compose the final SVG.
9. Render PNG from SVG.
10. Write `manifest.json`.
11. Run the source contract and fail with `source_contract_failed.json` if source reconstruction does not match.
12. Register output artifacts in SQLite.

## Prose Route

Prose is generated with DiffBrush candidates and selected only after text verification. The current policy favors source preservation over visual novelty.

Candidate search is controlled by environment variables:

- `HANDGEN_DIFFBRUSH_PROSE_CANDIDATES`: number of candidates per prompt variant, default `4`.
- `HANDGEN_DIFFBRUSH_LAST_RESORT_RETRIES`: extra attempts when no candidate clears the best-effort floor, default `2`.
- `HANDGEN_DIFFBRUSH_ALLOW_BEST_EFFORT`: keep rendering a selected DiffBrush candidate when OCR cannot certify exactness, default enabled.
- `HANDGEN_DIFFBRUSH_MIN_BEST_EFFORT_SCORE`: minimum OCR/token match score for best-effort selection before last-resort fallback, default `0.12`.

The speed tradeoff is linear in generated candidates after the one-time model load is paid. On `examples/fragmented_math_full.txt` with `DIFFBRUSH_STEPS=20`, no last-resort retries, and the repo-local checkpoint, one candidate took `17.78s` wall time and four candidates took `42.09s` wall time. Four candidates was about `2.37x` slower for that example and can improve selection, but it does not guarantee OCR-perfect prose.

The prose route records evidence such as source text hashes, candidate metadata, OCR or estimated word boxes, selected artifact paths, and exactness information in the manifest.

Prose chunking is configurable via `handgen/chunking.py`. Four strategies are available — `balanced_dp` (minimise squared deviation from a ~42-char target; **production default**), `current` (legacy greedy word-count, reachable only when `--max-chars` is set as an escape hatch), `punctuation_first` (split on `.`, `;`, `:`, `,` then rebalance), and `variable_length_sampling` (per-chunk targets sampled from `[35, 50]`). All strategies preserve whole words and target IAM-line-shaped chunks because DiffBrush is trained on that distribution (see `docs/findings.md`).

By default, every `handgen document render` of prose-only content routes through `balanced_dp`. The `--max-words` flag is now a no-op for prose; `--max-chars` switches the prose pipeline to the legacy char-greedy chunker as an explicit escape hatch.

`python -m handgen.eval.chunk_compare WRITER_ID --strategies current balanced_dp punctuation_first variable_length_sampling --out outputs/chunking_compare/latest` renders the same benchmark prose across all four strategies and emits a comparison grid PNG plus `summary.json` with per-cell chunk lengths, OCR repeat counts, and match scores.

## Math Route

Math spans use a mixed route:

- Worksheet glyphs are used for ASCII variables, Greek, relation symbols, operators, and punctuation where available.
- Digits currently use isolated DiffBrush connected components generated from natural handwriting context.
- Symbols are normalized with ink and sizing profiles so worksheet crops blend with the prose style.
- Each rendered math token records source character metadata for source reconstruction.

Missing worksheet glyphs should be treated as rendering blockers or explicit warnings, not silent substitutions.

## Output Artifacts

A render output directory may include:

- `document.svg`: canonical composed output.
- `document.png`: PNG derivative.
- `manifest.json`: machine-readable render manifest.
- `source_contract_failed.json`: failure details when source preservation fails.
- `diffbrush_runs/`: per-span DiffBrush inputs, outputs (`sample.png`), and `result.json`. One subdirectory per chunk or variable; all of them share the same in-memory model within a single render.
- `prose/`, `prose_candidates/`, `variables/`, `math/`, `glyph_rgba/`, `glyph_variants/`: intermediate visual assets.
- `logs/`: runtime logs and diagnostic output.

## Verification

The source contract reconstructs rendered source text from manifest spans and line records. A render should fail if the reconstructed text differs from the parsed source text.

OCR checks are used for DiffBrush prose candidates where possible. OCR is a gate for generated prose quality, not the only source-preservation mechanism.

## Documentation Update Example

If a commit changes digit rendering from isolated DiffBrush components to worksheet glyph-bank crops, update Math Route and any affected Output Artifacts or Verification notes.

If a commit changes the render manifest schema, update Output Artifacts and Verification so future readers know how to inspect the new manifest.

If a commit only renames an internal helper without changing behavior, this file probably does not need an edit.
