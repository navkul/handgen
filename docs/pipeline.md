# Pipeline

## Enroll a writer

Run `handgen init` and `handgen writer create <writer_id>`. Set `HANDGEN_DATA_DIR` before invoking the CLI to use a separate local catalog. The bundled commands in the README create two example writers. [Quick start](../README.md#try-it-with-the-bundled-samples)

```bash
handgen style add writer_001 data/style_refs/writer_001/natural_prose_primary.png \
  --transcript examples/transcripts/writer_001_primary.txt
handgen worksheet ingest writer_001 data/worksheets/writer_001/writer_001_worksheet_01.png
```

Natural reference images may contain a single line or a single column of handwriting. `style add` flattens the paper, estimates skew, removes page rules, locates rows, and prepares short conditioning windows at their natural aspect ratio. Use `--crop LEFT TOP RIGHT BOTTOM` in original, orientation-corrected image pixels to select a column or omit printed headers. Multiple registrations interleave windows from different references. Re-registering a source rebuilds its prepared crops.

An optional UTF-8 transcript supplies one nonempty line per handwritten row. The row count must match. Only plausible boundaries with actual ink gaps receive word labels. A row with ambiguous word boundaries is still available for model conditioning. The CLI prints which rows yielded word samples. Their manifest records the source box, baseline, body height, adjacent gap, exact label and digest. The renderer uses case-sensitive whole-word matches and can add sampled punctuation around a known word.

## Ingest the worksheet

Use a page with clearly separated handwritten marks in known order. The built-in layout covers digits, punctuation and operators, Greek, lowercase letters and uppercase letters. Ingestion accepts an image or page one of a PDF (`--dpi` defaults to 220). It estimates skew, compensates for paper brightness, suppresses long grid lines, groups connected components into symbols and reconciles detected rows against the ordered sections. A section can wrap onto another physical row.

For different label orders or repeated variants, pass `--layout path/to/layout.json`. A small custom sheet could use the following layout.

```json
{"rows": [
  {"id": "digits", "labels": ["0", "1", "2"]},
  {"id": "symbols", "labels": ["x", "x", "+", "=", "theta"]}
]}
```

The manifest shows the source SHA-256, processed page, deskew angle, detected row assignments, individual crops and label coverage. Labels depend on writing order. Inconsistent counts fail ingestion. The default layout recognizes the supplied worksheet’s missing gamma and records it explicitly. The read path verifies registered crop hashes and treats a damaged crop as unavailable.

## Render

Write a UTF-8 file, for example `Pick [[0 < a < b]].` Explicit math spans must open and close on the same line. Plain text is prose by default. `--infer-math` enables the older heuristic for unmarked expressions. Explicit spans take precedence on their lines.

```bash
handgen document render writer_001 examples/mixed_proof.txt --out outputs/proof --require-verified
```

1. Copy the input bytes into a new run and record the delimiter-free text, line endings and source offsets. Preflight all math and, in samples mode, all prose. Failures list the missing characters and source lines in `missing_glyphs.json`.
2. Split prose using the balanced dynamic-programming chunker. A `--max-chars` override selects the greedy splitter and supports `--max-words`. Long uninterrupted tokens can span placements without acquiring spaces.
3. In automatic mode, try a whole chunk of known word samples first. Otherwise generate short DiffBrush candidates and read the actual ink. The runner varies reference lines and seeds, sizes the canvas to the prompt, and can generate extra context for a short prefix, cut it out and read it again. A failed line may still contain usable runs. Recover those only at clear ink gaps with a matching independent OCR read. Fill remaining words from transcribed ink, worksheet letters or a local VATr++ checkpoint. For unresolved generated alternatives, compare character edits. Explicit `--prose samples|diffbrush|vatr` selects one prose route.
4. Draw math at its own width and baseline. Prefer a direct labeled glyph. A supported composite such as `≤` uses the writer’s `<` and `-`. Unicode superscript and subscript digits use the writer’s sampled digit at the corresponding height. Without a worksheet digit, relaxed generation can use an isolated DiffBrush component labeled `unverified_model_digit`. Strict and samples-only runs require coverage.
5. `--carriers` optionally generates a short line with a temporary word in place of each expression. Retain the surrounding prose only when the full line, word boundaries, ink gaps and independently re-read prose crops agree. Other lines use direct span composition. The carrier report records each outcome.
6. The layout buffers one physical row, aligns its ink, wraps to its measured width, and starts a new page when required. Each page is a 1500 × 1900 SVG with embedded ink. `rsvg-convert` writes the corresponding PNG. A seed controls sample choice and model candidates.

Each successful run includes `source.txt`, `parse_plan.json`, the per-route crops, `warnings.json`, `manifest.json`, `document.svg` and `document.png`. Later pages are `document-002.svg`, `document-002.png`, and so on. Open the paths under `outputs.pages` in the manifest to retrieve every page. A failed render job records the error and retains the previous successful render. Relative asset paths in worksheet and style packs support moving the catalog.

## Fit and evaluate

```bash
handgen style fit writer_001 --out outputs/fit --steps 100 --seed 732 \
  --test-reference data/style_refs/writer_001/natural_prose_alt03.png examples/transcripts/writer_001_holdout.txt
handgen eval render writer_001 --source examples/prose_only.txt \
  --prose vatr --adapter outputs/fit/writer.safetensors --out outputs/fit-eval
```

Fitting needs at least four case-insensitive word families from aligned transcripts. Only ASCII alphabetic words up to 32 letters enter the fit. The split groups capitalization variants. Training samples provide the reference memory. Optimize a shared 512-channel offset and score the original memory plus checkpoints on held-out word types. Step zero can win. The optional test image is processed after model selection. It must be separate from the fitting images, and only unseen alphabetic words are scored. `fit.json` includes word lists, hashes, seed, device, validation history and the selected step. A comparison image shows the real crop, initial conditioning and selected conditioning.

Passing `--adapter` to automatic rendering allows the fitted model to repair novel words from a failed line. Actual transcribed words retain priority. If a fitted repair fails OCR and worksheet samples cover the text, use the worksheet. Each adapter-backed run saves a copy of its selected memory and its digest in `prose_adapter` so later fits can reuse the same output folder.

Each placement records its evidence as `transcript_label`, `worksheet_labels`, `ocr_match`, `whitespace`, `best_effort` or `unverified_model_digit`. Only `ocr_match` sets the visual OCR certification flag. `--require-verified` accepts labels and matching OCR, and rejects the two unresolved generation methods. An exact source reconstruction is required before page publication.

`eval render --source FILE` uses the same routes and defaults as a document render, with options for seed, prose mode, adapter, inferred math, carriers and strict verification. With no source argument it exercises the built-in mixed-line fixtures and carrier inference. It groups neighboring prose placements on physical baselines, reads each exported region, counts character edits, and checks every placed image for visible ink. OCR normalizes whitespace and spacing before common punctuation while preserving letter case and the punctuation itself. When prose is present, the evaluation’s overall pass additionally requires zero page-readback edits. The report includes the observations, denominators, contact sheet and render manifest. Inspect worksheet token labels and crops separately when evaluating math.
