# Decisions

This document records durable technical and product decisions. It is intentionally not a changelog. Add or update entries only when a commit changes a meaningful decision, reverses a decision, or adds a tradeoff future contributors need to understand.

## Decision Format

Use this shape for new entries:

```md
## YYYY-MM-DD: Short Decision Title

Decision: State the current decision in one or two sentences.

Reason: Explain why this choice fits the MVP.

Tradeoff: Name the cost, risk, or thing deferred.

Update rule: Explain what kind of future commit should revisit this entry.
```

If a later commit changes a decision, edit the existing entry so it reflects the current state. Add a short "Updated" note only when the reason for the reversal matters.

## 2026-05-14: Keep The MVP Local-First

Decision: Handgen remains a local-first Python CLI until the rendering pipeline is reliable.

Reason: The current risk is generation correctness, source preservation, worksheet ingestion, and inspectable artifacts. Product infrastructure would add complexity before the core output is trustworthy.

Tradeoff: There is no deployed user workflow, account model, queue, cloud storage, or subscription layer yet.

Update rule: Revisit this entry when a commit introduces any frontend, backend service, hosted runtime, or multi-user product boundary.

## 2026-05-14: Store Metadata In SQLite And Large Artifacts On Disk

Decision: SQLite stores writer, worksheet, glyph, document, job, and artifact metadata. Images, SVGs, PNGs, logs, model outputs, and manifests live on disk.

Reason: The MVP needs local inspectability and simple reproducibility without hiding generated artifacts inside the database.

Tradeoff: Filesystem paths are part of the working contract, so moving assets requires care.

Update rule: Revisit this entry if artifact storage changes, paths become virtualized, or the database starts storing large binary blobs.

## 2026-05-14: Use SVG As The Canonical Document Output

Decision: The final composed document is written as SVG first, then rendered to PNG as a derivative.

Reason: SVG keeps layout and embedded assets inspectable, which is important while the renderer is still evolving.

Tradeoff: PNG appearance depends on the SVG renderer, currently `rsvg-convert`.

Update rule: Revisit this entry if PDF or PNG becomes the canonical output or if SVG no longer contains enough diagnostic information.

## 2026-05-14: Preserve Source Text In Manifests And Contracts

Decision: Render manifests must preserve parsed source text, source hashes, span routing, and enough token metadata for source reconstruction.

Reason: Visual handwriting generation can be approximate, but the machine-readable source sidecar must remain exact for debugging and evaluation.

Tradeoff: Render code and manifests carry more metadata than a pure image-generation pipeline would need.

Update rule: Revisit this entry when parser output, manifest schema, source-contract behavior, or OCR gating changes.

## 2026-05-14: Split Prose Generation From Math Symbol Rendering

Decision: Prose routes through source-locked DiffBrush generation. Inline ASCII variables, Greek, operators, relation symbols, and punctuation route through worksheet glyph-bank composition where possible. Digits still use isolated DiffBrush components for the current MVP iteration.

Reason: DiffBrush is useful for natural handwriting, but raw generated math is not trusted as exact source-preserving notation.

Tradeoff: Mixed rendering requires ink normalization, sizing rules, and route-specific verification. Digit rendering is still less exact than worksheet glyph rendering and remains a known limit.

Update rule: Revisit this entry when digit, Greek, operator, punctuation, or variable routing changes.

## 2026-05-13: Run DiffBrush In-Process And Reuse Models Across Spans

Decision: `handgen/diffbrush/runner.py` builds the DiffBrush model in-process on the first call inside a `document render` and reuses it for every prose chunk and variable in that render. No subprocess per chunk, no persistent daemon.

Reason: The earlier subprocess-per-chunk path paid a ~10s model-load cost on every call. On a 3-chunk test, switching to in-process reuse brings the total from ~37s to ~10s (~3.75×) with warm chunks at ~2s. The pattern matches how the Autoscribe benchmark drives DiffBrush.

Tradeoff: The host Python process now holds the model in memory (CPU or MPS) for the duration of a render, so it should not also import other large models concurrently. Speedup only applies within a single `document render` call — repeated CLI invocations still pay one model-load each.

Update rule: Revisit this entry if rendering moves to a long-running server, batches multiple documents in one process, or if DiffBrush is replaced by a different prose model.

## 2026-05-14: Do Not Maintain A Changelog Yet

Decision: The repo does not need a changelog while it is a local MVP staging repo without versioned releases or external users.

Reason: Git history, `docs/findings.md`, and this decision log are better suited to current development.

Tradeoff: There is no release-oriented narrative for non-developer users yet.

Update rule: Add `CHANGELOG.md` when the project has versioned releases, external consumers, or a standalone package boundary.
