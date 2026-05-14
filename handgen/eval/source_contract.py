from __future__ import annotations

from typing import Any


def verify_source_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    spans_by_id = {span["id"]: span for span in manifest.get("spans", [])}
    errors: list[dict[str, Any]] = []
    rendered_lines: list[str] = []

    for line in manifest.get("lines", []):
        pieces: list[str] = []
        for span_id in line.get("span_ids", []):
            span = spans_by_id.get(span_id)
            if span is None:
                errors.append({"code": "missing_span_record", "line_index": line.get("index"), "span_id": span_id})
                continue
            pieces.append(span.get("text", ""))
            if span.get("type") == "math":
                math_text = _math_tokens_to_text(span.get("math", {}).get("tokens", []))
                if math_text != span.get("text", ""):
                    errors.append(
                        {
                            "code": "math_token_reconstruction_mismatch",
                            "span_id": span_id,
                            "expected": span.get("text", ""),
                            "actual": math_text,
                        }
                    )
        rendered_line = "".join(pieces)
        rendered_lines.append(rendered_line)
        if rendered_line != line.get("text", ""):
            errors.append(
                {
                    "code": "line_reconstruction_mismatch",
                    "line_index": line.get("index"),
                    "expected": line.get("text", ""),
                    "actual": rendered_line,
                }
            )

    rendered_source = "\n".join(rendered_lines)
    expected_source = manifest.get("source_text", "")
    if rendered_source != expected_source:
        errors.append({"code": "source_reconstruction_mismatch", "expected": expected_source, "actual": rendered_source})

    return {
        "schema_version": "handgen_source_contract_v1",
        "passed": not errors,
        "checks": [
            "rendered_span_text_reconstructs_each_source_line",
            "rendered_lines_reconstruct_source_text",
            "prose_artifacts_reconstruct_source_text",
            "math_token_sequence_reconstructs_math_span_text",
        ],
        "rendered_source_text": rendered_source,
        "errors": errors,
    }


def _math_tokens_to_text(tokens: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for token in tokens:
        if token.get("label") == "space":
            pieces.append(token.get("source_character", token.get("text", " ")))
        else:
            pieces.append(token.get("source_character", ""))
    return "".join(pieces)
