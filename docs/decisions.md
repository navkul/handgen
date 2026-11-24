# Decisions

## Run experiments through a local CLI

The same local command can enroll handwriting, render a note and measure the resulting page. Keeping references, fitted weights and evaluation artifacts on one machine lets a researcher inspect the ink alongside each recorded decision.

## Store metadata in SQLite and ink on disk

SQLite links writer assets, the active worksheet, documents, job state and output digests. Source images, prepared crops, manifests, adapters and pages remain individually inspectable files. Content-based names protect different scans, annotations and worksheet layouts from collisions. Per-run output directories allow repeated renders and distinguish artifacts from different seeds or models.

## Publish SVG pages with two source records

SVG is canonical because it retains individual placements with embedded assets. PNGs are derivatives rendered by librsvg. Preserve the original UTF-8 bytes, including markup and line endings, alongside the delimiter-free text and a SHA-256 for each. Reconstruct the latter from ordered source ranges before publishing. Pagination changes page geometry while preserving this contract.

## Choose the source of each piece of handwriting

A labeled word sample preserves the writer’s letter shapes, joins and stroke widths. DiffBrush supplies context-sensitive prose, and VATr++ supplies individual novel words. Worksheet ink is the source of truth for inline variables, digits, Greek and operators. A direct sample has priority over a construction from existing marks. Digit isolation from DiffBrush is an explicitly unverified fallback for incomplete legacy worksheets and is disallowed by strict and samples-only rendering.

Use balanced prose chunking by default and preserve an explicit character-limit override for comparisons. Slot carriers are opt-in. They require detectable cuts and independently readable prose on the carrier line. Direct composition preserves math at its natural width when those observations are unavailable.

## Preserve distinctions in the evidence

A source hash proves which text was supplied. A worksheet label or transcribed word establishes which sample the renderer placed. OCR agreement provides reader evidence about the pixels. Keep all three available in manifests. Source preservation is mandatory. Strict mode additionally requires every visible placement to have a label or a matching OCR read. Ordinary rendering may retain a generated approximation and must record the intended and observed text.

Measure exported pages separately. Region OCR may make different errors from crop recognition, and an OCR reader can agree with an incorrect shape. Expose actual observations, character counts and ink bounds so someone can inspect disagreements.

## Calibrate a writer only when the available evidence supports it

Estimate natural word and space advance from transcript-aligned ink. Apply a personalized width correction only when held-out word families improve on a simple writer-specific baseline. Keep the base VATr++ generator frozen and fit a small conditioning offset from labeled words. Group case variants across splits, include the original conditioning in selection, and save the metric history. An optional independent reference measures new words only after selection. Use the fitted memory explicitly so a writer can compare it with the base generator.

## Load neural models on demand

The sample-only path works on CPU without neural weights. A neural render loads a model once and reuses its conditioning where the inputs are constant. Models, decoder assets and optional recognition weights can be staged locally. Git stores the inference integration, published weight references, checksums and example inputs. Generated jobs and model binaries are excluded.

## Maintain current documentation

Update the architecture, pipeline, findings and relevant decisions when behavior or evidence changes. Keep a release changelog for the future standalone, versioned product.
