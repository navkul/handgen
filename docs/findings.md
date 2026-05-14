# Findings

This document captures current engineering observations, empirical behavior, limitations, and open problems. It is not a changelog. Update it when a commit changes what we know about the system, proves or disproves an approach, introduces a notable limitation, or resolves a documented finding.

## Current Findings

### Worksheet ingestion is the most important near-term reliability risk

The MVP depends on worksheet rows matching the expected format closely enough for automatic row grouping and label assignment. The code records assumptions and warnings in the worksheet manifest, but robust handling of layout drift is still a priority.

Implication: changes to `handgen/ingest/worksheet.py` should usually include either tests around row/label behavior or an update here if they change known ingestion limits.

### Source preservation is a product invariant, not just a rendering detail

The system stores parsed source text and source hashes in manifests and verifies rendered-line reconstruction through `handgen/eval/source_contract.py`. This is necessary because handwriting generation can be visually approximate while the source sidecar must remain exact.

Implication: any commit that changes parsing, span routing, manifest line records, or token metadata must consider source-contract behavior.

### DiffBrush prose needs verification and reranking

DiffBrush can produce useful handwriting-like prose, but generated text can drift, repeat, omit, or add content. The MVP therefore treats DiffBrush output as candidates and uses OCR/text checks before composition where possible.

The current default generates four candidates per prompt variant. In a fresh benchmark of `examples/fragmented_math_full.txt` with `DIFFBRUSH_STEPS=20` and no last-resort retries, one candidate took `17.78s` wall time while four candidates took `42.09s`. That is about `2.37x` slower for this document. Candidate search can improve the selected line, but OCR still sees some low-quality lines, so higher candidate counts are a quality/runtime knob rather than a correctness guarantee.

The current OCR gate (tesseract token comparison) is a fairly weak content check. A stronger inference-time verification — running HTR/OCR on every candidate, scoring against the target text, and regenerating failed chunks — is the most direct next step to suppress repeated and drifted words. The DiffBrush paper's discriminators only help at training time; inference has no hard content check today.

Implication: visual quality improvements should not bypass source-locking checks unless the replacement verification strategy is documented. Candidate-count changes should be benchmarked because they directly trade runtime for selection quality.

### Prose chunk size matters because DiffBrush is trained on IAM-line-shaped prompts

DiffBrush's training distribution is IAM handwriting lines, which cluster around ~30–50 characters. Empirically, asking it to render 20- or 30-char chunks produces *more* repeated words than 40-char chunks, not fewer — the model is most reliable when the prompt looks like what it was trained on. Shorter is not safer.

`handgen/chunking.py` provides four prose chunking strategies (`current`, `balanced_dp`, `punctuation_first`, `variable_length_sampling`) and `handgen/eval/chunk_compare.py` renders side-by-side comparisons. In a 5-sentence benchmark on writer_002, the DP and punctuation-first splitters both eliminated repeats *and* lifted average OCR match score by ~33% versus the greedy splitter — mostly by killing the small orphan tail chunks (7–18 chars) that the greedy splitter emits at sentence ends.

`balanced_dp` is the production default for prose-only renders as of 2026-05-14 (see `docs/decisions.md`). The greedy chunker remains reachable as an escape hatch via `--max-chars`. Carrier/slot lines still use their own `_chunk_carrier` splitter; this finding only governs the pure-prose path.

Implication: chunking should target ~42 chars per chunk, never split words, and avoid tiny tails. Future improvements should pursue *semantically* coherent chunks (clause/punctuation boundaries, overlap/context windows) rather than just smaller ones.

### Chunking algorithm runtime is negligible — optimize for correctness

The four chunking strategies range from O(n) greedy to O(n²) DP. On benchmark sentences the DP is ~2.2× slower than the variable-length sampler — but in absolute terms that's 27 µs vs 12 µs per sentence. Total render time is dominated by diffusion sampling (seconds per chunk), so chunker complexity is rounding error.

Implication: pick chunking strategies on output quality alone. Do not optimize chunking for speed at the cost of structure.

### Diffusion sampling dominates runtime; persistent GPU is the production path

DiffBrush is a diffusion model — generation is dozens of denoising steps (a UNet forward pass per step) that gradually turn random noise into a coherent handwritten line, not a single forward pass. This is why every chunk costs ~1–3 seconds on M-series MPS even after the model is loaded.

In a recent run of `handgen/eval/chunk_compare.py` (5 sentences × 4 strategies, writer_002), the harness took ~30 minutes wall-clock: 73 prose chunks × 5.7 average DiffBrush candidates per chunk = 416 diffusion sampling passes. None of the 73 chunks passed OCR exactness on the first candidate for this writer's style, so the runner walked the full candidate budget every time.

A single mid-tier cloud GPU (L4, A10G) typically runs Stable-Diffusion-class UNets 3–5× faster than M-series MPS at batch size 1. An A100 is ~10–15×, an H100 ~20×. The same 30-minute local run would be roughly 1–3 minutes on an L4/A10G, and under a minute on an A100 with batched candidate generation. Batching the 4 per-chunk candidates into a single forward pass (currently sequential in the runner) is an additional ~4× available on CUDA but not on MPS.

Implication: when moving toward production, host DiffBrush as a persistent GPU service that keeps the model in memory and accepts text+style requests, rather than per-job model loads (see `docs/decisions.md`). Tesseract OCR is CPU-only and will become a meaningful share of remaining time after GPU speedup — worth parallelizing per-candidate.

### Raw DiffBrush math is not trusted as exact math

Math symbols need exact source preservation. The MVP routes ASCII letters, Greek, operators, relation symbols, and punctuation through worksheet glyph-bank composition. Digits still use isolated DiffBrush components for now and are recorded as a known limit in render manifests.

Implication: math rendering changes should preserve token-level source metadata and avoid relying on unconstrained generated math text.

### DiffBrush is now loaded once per render, not once per chunk

Earlier the renderer spawned a fresh Python subprocess for every prose chunk and variable, which paid a ~10s model-load cost on every call. The runner is now in-process: the model is built on the first call inside a `document render` and reused for every subsequent chunk and variable in that same render. On a 3-chunk prose test this drops the total from ~37s to ~10s (~3.75× faster end-to-end), with warm chunks at ~2s and the first chunk paying a one-time ~5–6s load.

Implication: the speedup only applies within a single `document render` call. Running the CLI repeatedly (one call per document) still pays the load cost each time. If multi-document batching is needed, expose a callable that holds one `DiffBrushRunner` across documents.

### SVG is useful for debugging because composition remains inspectable

The SVG output keeps embedded visual assets, positions, sizes, and document structure inspectable before PNG conversion. This fits the local-first MVP better than treating the final PNG as the only meaningful artifact.

Implication: rendering changes should preserve enough manifest and SVG detail to debug layout and source reconstruction.

## Open Questions

- How tolerant should worksheet ingestion be to row spacing, skew, extra marks, and missing glyphs before it requires manual intervention?
- Should missing glyph coverage fail before rendering or produce a partial render with explicit manifest errors?
- What is the right threshold for accepting OCR-verified prose candidates versus falling back to estimated layout?
- How should longer documents paginate while keeping manifests easy to inspect?
- When the repo becomes standalone, which generated sample assets should remain committed as fixtures?
- Can semantically coherent chunking (clause boundaries, conjunction breaks, phrase units) reduce repetition further than purely length-based splitting?
- Does overlap/context chunking — generating each chunk with a few neighboring words and keeping only the intended center or suffix — help DiffBrush avoid mid-sentence resets?
- Would an HTR-based content gate (replacing or augmenting tesseract) raise the OCR-exactness rate enough to make the candidate budget elastic instead of always-exhausted?

## Documentation Update Example

If a commit adds skew correction and tests show worksheet ingestion now handles rotated scans, update the worksheet finding with the new supported range and any remaining limits.

If a commit removes an old limitation, do not add a historical note just to preserve the past. Replace or delete the stale finding so this file describes the current codebase.
