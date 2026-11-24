from __future__ import annotations

from typing import Any


def verify_source_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the plain source from placements. OCR is a separate check."""
    spans = manifest.get("spans", [])
    spans_by_id = {span["id"]: span for span in spans}
    errors: list[dict[str, Any]] = []
    if len(spans_by_id) != len(spans):
        errors.append({"code": "duplicate_span_id"})
    used: set[str] = set()
    reconstructed: list[str] = []
    cursor = 0
    for i, line in enumerate(manifest.get("lines", [])):
        pieces: list[str] = []
        for span_id in line.get("span_ids", []):
            span = spans_by_id.get(span_id)
            if span is None:
                errors.append({"code": "missing_span_record", "line_index": i, "span_id": span_id})
                continue
            if span_id in used:
                errors.append({"code": "span_used_twice", "span_id": span_id})
            used.add(span_id)
            text = span["text"]
            if "source_start" in span and (span["source_start"], span["source_end"]) != (cursor, cursor + len(text)):
                errors.append({"code": "source_offset_mismatch", "span_id": span_id})
            cursor += len(text)
            pieces.append(text)
            # Both worksheet words and math use the same token compositor.
            block = span.get("math") or span.get("glyph_run")
            if block is not None:
                visible = "".join(t.get("source_character", t.get("text", "")) for t in block["tokens"])
                if visible != text:
                    errors.append({"code": "math_token_reconstruction_mismatch", "span_id": span_id,
                                   "expected": text, "actual": visible})
            exemplar = span.get("word_sample")
            if exemplar is not None and exemplar["text"] != text:
                errors.append({"code": "sample_label_mismatch", "span_id": span_id})
            if "vatr" in span and span["vatr"]["target"] != text:
                errors.append({"code": "generation_target_mismatch", "span_id": span_id})
            if "recovered_crop" in span and span["recovered_crop"]["text"].split() != text.split():
                errors.append({"code": "crop_target_mismatch", "span_id": span_id})
            if "carrier_crop" in span and span["carrier_crop"].get("text", "").split() != text.split():
                errors.append({"code": "carrier_crop_target_mismatch", "span_id": span_id})
            run = span.get("diffbrush")
            if run and "display_text" in run and not span.get("carrier_crop") and run["display_text"].split() != text.split():
                errors.append({"code": "generation_target_mismatch", "span_id": span_id})
        actual = "".join(pieces)
        if actual != line["text"]:
            errors.append({"code": "line_reconstruction_mismatch", "line_index": i, "expected": line["text"], "actual": actual})
        ending = line.get("ending", "\n" if i + 1 < len(manifest["lines"]) else "")
        reconstructed.extend((actual, ending))
        cursor += len(ending)
    for unused in sorted(set(spans_by_id) - used):
        errors.append({"code": "unplaced_span", "span_id": unused})
    rendered = "".join(reconstructed)
    expected = manifest.get("source_text", "")
    if rendered != expected:
        errors.append({"code": "source_reconstruction_mismatch", "expected": expected, "actual": rendered})
    return {
        "schema_version": "handgen_source_contract_v2",
        "passed": not errors,
        "checks": ["each_span_placed_once_in_source_order", "rendered_lines_and_endings_reconstruct_source",
                   "worksheet_tokens_and_sample_labels_match_source", "generation_targets_match_source"],
        "rendered_source_text": rendered,
        "errors": errors,
    }
