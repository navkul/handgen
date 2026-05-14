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

Implication: visual quality improvements should not bypass source-locking checks unless the replacement verification strategy is documented.

### Raw DiffBrush math is not trusted as exact math

Math symbols need exact source preservation. The MVP routes symbols through worksheet glyph-bank composition and uses DiffBrush only for prose-like or isolated variable components where appropriate.

Implication: math rendering changes should preserve token-level source metadata and avoid relying on unconstrained generated math text.

### SVG is useful for debugging because composition remains inspectable

The SVG output keeps embedded visual assets, positions, sizes, and document structure inspectable before PNG conversion. This fits the local-first MVP better than treating the final PNG as the only meaningful artifact.

Implication: rendering changes should preserve enough manifest and SVG detail to debug layout and source reconstruction.

## Open Questions

- How tolerant should worksheet ingestion be to row spacing, skew, extra marks, and missing glyphs before it requires manual intervention?
- Should missing glyph coverage fail before rendering or produce a partial render with explicit manifest errors?
- What is the right threshold for accepting OCR-verified prose candidates versus falling back to estimated layout?
- How should longer documents paginate while keeping manifests easy to inspect?
- When the repo becomes standalone, which generated sample assets should remain committed as fixtures?

## Documentation Update Example

If a commit adds skew correction and tests show worksheet ingestion now handles rotated scans, update the worksheet finding with the new supported range and any remaining limits.

If a commit removes an old limitation, do not add a historical note just to preserve the past. Replace or delete the stale finding so this file describes the current codebase.
