# Architecture

Handgen is a local Python CLI for generating ordinary handwritten notes, letters, study material and math proofs. A writer supplies natural prose and, for math, a labeled worksheet. Model weights, references, SQLite records and generated pages live on the local machine.

## Components

| Component | Responsibility |
| --- | --- |
| `handgen/cli.py`, `db.py`, `paths.py` | Writer enrollment, asset registration, job state and artifact digests. |
| `handgen/ingest/paper.py`, `style_lines.py`, `style.py` | Flatten paper illumination, remove rules, separate lines, produce model windows, and align optional transcripts to observed word gaps. |
| `handgen/ingest/style_metrics.py` | Estimate space width and word advances in units of the writer’s letter-body height. Enable character-class width correction only when it improves held-out error. |
| `handgen/ingest/worksheet.py` | Deskew a page, identify rows and groups of marks, assign ordered labels, and publish a hash-checked glyph bank. |
| `handgen/parser.py`, `chunking.py`, `render/content.py` | Preserve line endings, map source spans to character offsets, and plan short prompts and formula runs. |
| `handgen/diffbrush/` | Load the vendored line generator and VAE on demand. Draw candidates from multiple prepared references, normalize the ink and rank blind OCR reads. |
| `handgen/vatr/` | Run the VATr++ word generator with local weights. Fit a writer-specific conditioning vector on labeled samples and evaluate it on separate word types. |
| `handgen/render/assets.py`, `repair.py`, `carriers.py`, `math.py` | Choose a rendering route, extract readable generated substrings, reuse actual word images, and compose worksheet glyphs into prose or notation. |
| `handgen/render/document.py`, `layout.py`, `ink.py` | Preflight coverage, align ink on a common baseline, wrap and paginate, publish the outputs, and register their hashes. |
| `handgen/eval/` | Reconstruct source text from placements, run local OCR, and measure text and ink on the rasterized pages. |

## Data and reproducibility

`HANDGEN_DATA_DIR` can override the default `data/` catalog. SQLite stores writers, raw style reference metadata, prepared style manifest pointers, the current worksheet and glyphs per writer, documents, render jobs, and output hashes. Large files live on disk.

- `style_refs/<writer>/` holds original inputs and the `prepared/` packs. Each pack is keyed by image digest, preparation version, optional crop and transcript. Crop records include source coordinates and hashes. Older single-line registrations can still resolve their original image.
- `worksheets/<writer>/<stem>/` holds the processed page, row images, glyph bank and manifests. Each registered recipe has a snapshot manifest with paths relative to its directory. The loader also understands earlier manifests written with absolute paths.
- `<render-output>/runs/<job>-<unique>/` holds a complete render. Each SVG embeds the pixels it uses. The output directory also has convenience copies of the first page and the latest successful manifest. The manifest names all pages in its run.
- Writer adapters live in the requested fit directory. Their metadata binds the saved conditioning memory to a writer ID and the SHA-256 of the base VATr++ checkpoint.

## Rendering and evidence

Text inside `[[...]]` always enters the math compositor. Outside those delimiters, the default renderer can use transcribed word crops, OCR-checked DiffBrush lines, newly recognized subregions, worksheet lettering or optional VATr++ words. Math uses the writer’s labeled digits, variables, Greek, operators and punctuation. The compositor records which sampled strokes it used for derived operators and superscript or subscript digits. A legacy fallback can isolate a model-generated digit when its worksheet crop is missing and relaxed rendering is permitted.

Each run stores both the original source bytes and the plain text with markup removed. Piece IDs, character ranges, original line endings, labels, and generation targets permit reconstruction. OCR agreement, a user’s sample label, and unverified generated ink have distinct evidence fields. The evaluation harness additionally measures what an OCR engine sees in the exported PNG, where layout and resampling can change legibility.

## Runtime

The CPU-only samples route needs Pillow, NumPy and librsvg. PDF worksheet input adds Poppler. Optional generation uses local PyTorch on CUDA, Apple MPS or CPU and a local reader. Model instances and constant diffusion conditioning are reused during a render. The two model families can share the bundled Unifont archetypes. The model guide lists installation paths and checkpoint digests. The research credits identify each upstream component. [Model setup](../models/README.md) · [Research credits](references.md)
