"""Short prompts and worksheet slot detection for mixed handwriting lines."""

from __future__ import annotations

import os
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..utils import png_size, read_json, require_pillow_numpy

MATH_OPERATOR_CHARS = set("<>+=*/|-")
CARRIER_TARGET_CHARS = 42
CARRIER_MIN_CHARS = 36
PADDING_WORDS = ("and", "then", "the", "line", "continues", "with", "simple", "words")
CLAUSE_STARTERS = {"and", "but", "then"}
EXPRESSION_RE = re.compile(r"\b[A-Za-z0-9]\b(?:\s*[+\-=*/<>]\s*\b[A-Za-z0-9]\b)+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
LIKELY_VARIABLE_NAMES = set("xyzXYZ")
VARIABLE_CONTEXT_PREV = {"if", "then", "let", "suppose", "assume", "where", "when", "given", "for", "than"}
VARIABLE_CONTEXT_NEXT = {"is", "are", "must", "equals", "true", "false", "even", "odd", "positive", "negative", "prime"}
ARTICLE_FOLLOWERS = {"number", "prime", "line", "point", "value", "set", "term", "case", "thing", "proof"}
PLACEHOLDER_VOCABULARY = ("blank", "term", "value", "number", "thing", "letter", "amount")
PLACEHOLDER_WORDS = set(PLACEHOLDER_VOCABULARY)
DEFAULT_SLOT_PLACEHOLDERS = (
    (10_000, "blank"),
)


def _carrier_for_line(source_line: str, known_variables: set[str] | None = None) -> dict[str, Any]:
    known_variables = known_variables if known_variables is not None else set()
    fragments = _math_fragments_for_line(source_line, known_variables)
    carrier_text = _carrier_text_without_fragments(source_line, fragments)
    slot_carrier = _slot_carrier_plan(source_line, fragments)
    source_words = [{"text": match.group(0), "start": match.start(), "end": match.end()} for match in WORD_RE.finditer(source_line)]
    kept_source_words = [
        word for word in source_words if not any(word["start"] >= fragment["source_start"] and word["end"] <= fragment["source_end"] for fragment in fragments)
    ]
    carrier_words = [
        {"text": match.group(0), "start": match.start(), "end": match.end()}
        for match in WORD_RE.finditer(carrier_text)
    ]
    slot_words = [
        {"text": match.group(0), "start": match.start(), "end": match.end()}
        for match in WORD_RE.finditer(slot_carrier["text"])
    ]
    slot_word_indices = _slot_word_indices(slot_words, slot_carrier["placeholders"])
    insertions: list[dict[str, Any]] = []
    for insertion_idx, fragment in enumerate(fragments):
        prev_idx = next((idx for idx in range(len(kept_source_words) - 1, -1, -1) if kept_source_words[idx]["end"] <= fragment["source_start"]), None)
        next_idx = next((idx for idx, word in enumerate(kept_source_words) if word["start"] >= fragment["source_end"]), None)
        insertions.append(
            {
                "text": fragment["text"],
                "source_start": fragment["source_start"],
                "source_end": fragment["source_end"],
                "prev_word_index": prev_idx,
                "next_word_index": next_idx,
                "slot_word_index": slot_word_indices[insertion_idx] if insertion_idx < len(slot_word_indices) else None,
                "slot_placeholder": slot_carrier["placeholders"][insertion_idx] if insertion_idx < len(slot_carrier["placeholders"]) else None,
                "anchor": "between_words" if prev_idx is not None and next_idx is not None else "line_edge",
            }
        )
    return {
        "source_text": source_line,
        "carrier_text": carrier_text,
        "slot_carrier_text": slot_carrier["text"],
        "carrier_words": carrier_words,
        "slot_carrier_words": slot_words,
        "insertions": insertions,
        "known_variables_after_line": sorted(known_variables),
    }


def _math_fragments_for_line(source_line: str, known_variables: set[str]) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for match in EXPRESSION_RE.finditer(source_line):
        fragments.append({"text": match.group(0), "source_start": match.start(), "source_end": match.end(), "kind": "expression"})
        known_variables.update(ch for ch in match.group(0) if ch.isalpha())
    source_words = [{"text": match.group(0), "start": match.start(), "end": match.end()} for match in WORD_RE.finditer(source_line)]
    for idx, word in enumerate(source_words):
        text = word["text"]
        if len(text) != 1 or not text.isalpha():
            continue
        if any(word["start"] >= fragment["source_start"] and word["end"] <= fragment["source_end"] for fragment in fragments):
            continue
        prev_text = source_words[idx - 1]["text"].casefold() if idx > 0 else ""
        next_text = source_words[idx + 1]["text"].casefold() if idx + 1 < len(source_words) else ""
        is_article = text == "a" and next_text in ARTICLE_FOLLOWERS
        is_variable = (
            text in known_variables
            or text in LIKELY_VARIABLE_NAMES
            or prev_text in VARIABLE_CONTEXT_PREV
            or next_text in VARIABLE_CONTEXT_NEXT
        )
        if is_variable and not is_article:
            fragments.append({"text": text, "source_start": word["start"], "source_end": word["end"], "kind": "single_variable"})
            known_variables.add(text)
    fragments.sort(key=lambda item: (item["source_start"], item["source_end"]))
    deduped: list[dict[str, Any]] = []
    last_end = -1
    for fragment in fragments:
        if fragment["source_start"] < last_end:
            continue
        deduped.append(fragment)
        last_end = fragment["source_end"]
    return deduped


def _carrier_text_without_fragments(source_line: str, fragments: list[dict[str, Any]]) -> str:
    chars = list(source_line)
    for fragment in fragments:
        for idx in range(fragment["source_start"], fragment["source_end"]):
            chars[idx] = " "
    text = "".join(chars)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slot_carrier_plan(source_line: str, fragments: list[dict[str, Any]]) -> dict[str, Any]:
    pieces: list[str] = []
    placeholders: list[dict[str, Any]] = []
    cursor = 0
    for slot_index, fragment in enumerate(fragments):
        pieces.append(source_line[cursor : fragment["source_start"]])
        context = _fragment_placeholder_context(source_line, fragment)
        token = _placeholder_for_fragment(fragment["text"], context=context)
        placeholders.append({"slot_index": slot_index, "text": token, "source_text": fragment["text"], "context": context})
        pieces.append(token)
        cursor = fragment["source_end"]
    pieces.append(source_line[cursor:])
    return {"text": _normalize_generated_line_text("".join(pieces)), "placeholders": placeholders}


def _fragment_placeholder_context(source_line: str, fragment: dict[str, Any]) -> str:
    clause_start = max(source_line.rfind(".", 0, fragment["source_start"]), source_line.rfind(",", 0, fragment["source_start"]), source_line.rfind(";", 0, fragment["source_start"]))
    clause_text = source_line[clause_start + 1 : fragment["source_start"]]
    words = [match.group(0).casefold() for match in WORD_RE.finditer(clause_text)]
    first = words[0] if words else ""
    if first == "if":
        return "if_short_slot"
    if first == "let":
        return "let_short_slot"
    if first == "then":
        return "then_slot_clause"
    if first == "and":
        return "and_slot_clause"
    if first == "suppose":
        return "suppose_slot_clause"
    return "generic_slot_clause"


def _placeholder_for_fragment(fragment_text: str, *, context: str | None = None) -> str:
    visible_len = len(re.sub(r"\s+", "", fragment_text))
    for max_len, placeholder in _slot_placeholders(context=context):
        if visible_len <= max_len:
            return placeholder
    return _slot_placeholders(context=context)[-1][1]


def _slot_placeholders(*, context: str | None = None) -> tuple[tuple[int, str], ...]:
    policy_path = os.environ.get("HANDGEN_PLACEHOLDER_POLICY")
    if not policy_path:
        return DEFAULT_SLOT_PLACEHOLDERS
    try:
        policy = read_json(Path(policy_path))
        entries = []
        if context:
            entries = policy.get("slot_placeholders_by_context", {}).get(context, [])
        if not entries:
            entries = policy.get("slot_placeholders", [])
        parsed = tuple((int(item["max_visible_len"]), str(item["placeholder"])) for item in entries)
    except (OSError, KeyError, TypeError, ValueError):
        return DEFAULT_SLOT_PLACEHOLDERS
    if not parsed or any(placeholder not in PLACEHOLDER_VOCABULARY for _, placeholder in parsed):
        return DEFAULT_SLOT_PLACEHOLDERS
    return parsed


def _slot_word_indices(slot_words: list[dict[str, Any]], placeholders: list[dict[str, Any]]) -> list[int | None]:
    indices: list[int | None] = []
    search_from = 0
    for placeholder in placeholders:
        token = placeholder["text"].casefold()
        found = None
        for idx in range(search_from, len(slot_words)):
            if slot_words[idx]["text"].casefold() == token:
                found = idx
                search_from = idx + 1
                break
        indices.append(found)
    return indices


def _normalize_generated_line_text(text: str) -> str:
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_carrier(carrier_text: str, target_chars: int = CARRIER_TARGET_CHARS) -> list[dict[str, Any]]:
    words = list(re.finditer(r"\S+", carrier_text))
    carrier_words = [{"text": match.group(0), "start": match.start(), "end": match.end()} for match in WORD_RE.finditer(carrier_text)]
    chunks: list[dict[str, Any]] = []
    start_word = 0
    while start_word < len(words):
        end_word = start_word
        end_pos = words[start_word].end()
        while end_word + 1 < len(words):
            next_end = words[end_word + 1].end()
            if next_end - words[start_word].start() > target_chars and end_word >= start_word:
                break
            end_word += 1
            end_pos = words[end_word].end()
        start_pos = words[start_word].start()
        text = carrier_text[start_pos:end_pos].strip()
        chunks.append(
            {
                "text": text,
                "carrier_start": start_pos,
                "carrier_end": end_pos,
                "word_start": _word_index_at_or_after(carrier_words, start_pos),
                "word_end": _word_index_after(carrier_words, end_pos),
            }
        )
        start_word = end_word + 1
    chunks = _split_carrier_chunks_on_natural_breaks(carrier_text, carrier_words, chunks)
    for chunk in chunks:
        chunk["padding_text"] = ""
        chunk["prompt_text"] = _iam_like_prompt_text(chunk["text"])
        chunk["prompt_variants"] = _iam_like_prompt_variants(chunk["text"])
    return chunks


def _iam_like_prompt_text(text: str) -> str:
    return _iam_like_prompt_variants(text)[0]


def _iam_like_prompt_variants(text: str) -> list[str]:
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if "blank" in {match.group(0).casefold() for match in WORD_RE.finditer(normalized_text)}:
        if len(normalized_text) < CARRIER_MIN_CHARS:
            padding = _padding_for(normalized_text)
            return [f"{normalized_text} {padding}".strip()]
        return [normalized_text or text]
    prompt = re.sub(r"[,;:]", "", text)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    words = [match.group(0) for match in WORD_RE.finditer(prompt)]
    lowered = [word.casefold() for word in words]
    placeholder = next((word for word in words if word.casefold() in PLACEHOLDER_WORDS), None)
    policy_variants = _policy_prompt_variants(words, lowered, placeholder, prompt)
    if policy_variants:
        return policy_variants
    variants: list[str] = []
    if placeholder and len(words) <= 2:
        first = lowered[0] if lowered else ""
        if first == "if":
            variants.extend(
                [
                    f"If {placeholder} is positive",
                    f"If {placeholder} is true",
                    f"If {placeholder} is useful",
                    f"If {placeholder} is small",
                ]
            )
        if first == "let":
            variants.extend([f"Let {placeholder} be useful", f"Let {placeholder} be fixed"])
        if first in {"then", "and"}:
            variants.extend([f"{words[0].capitalize()} {placeholder} is useful", f"{words[0].capitalize()} {placeholder} is positive"])
    if lowered and lowered[0] in CLAUSE_STARTERS:
        prompt = prompt[:1].upper() + prompt[1:]
    variants.append(prompt or text)
    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _policy_prompt_variants(words: list[str], lowered: list[str], placeholder: str | None, prompt: str) -> list[str]:
    policy_path = os.environ.get("HANDGEN_CARRIER_PROMPT_POLICY")
    if not policy_path:
        return []
    context = _carrier_prompt_context(words, lowered, placeholder)
    if not context:
        return []
    try:
        policy = read_json(Path(policy_path))
        context_policy = policy.get("contexts", {}).get(context, {})
        templates = []
        if placeholder:
            templates = context_policy.get("prompt_templates_by_placeholder", {}).get(placeholder.casefold(), [])
        if not templates:
            templates = context_policy.get("prompt_templates", [])
    except (OSError, AttributeError):
        return []
    if not templates:
        return []
    variants: list[str] = []
    first = words[0] if words else ""
    for template in templates:
        try:
            variant = str(template).format(placeholder=placeholder or "", prompt=prompt, first=first)
        except (KeyError, ValueError):
            continue
        variant = re.sub(r"\s+", " ", variant).strip()
        if variant and variant not in variants:
            variants.append(variant)
    return variants


def _carrier_prompt_context(words: list[str], lowered: list[str], placeholder: str | None) -> str | None:
    if not words:
        return None
    first = lowered[0]
    if placeholder and len(words) <= 2 and first == "if":
        return "if_short_slot"
    if placeholder and len(words) <= 2 and first == "let":
        return "let_short_slot"
    if placeholder and first == "then":
        return "then_slot_clause"
    if placeholder and first == "and":
        return "and_slot_clause"
    if placeholder and first == "suppose":
        return "suppose_slot_clause"
    if placeholder:
        return "generic_slot_clause"
    return None


def _split_carrier_chunks_on_natural_breaks(
    carrier_text: str,
    carrier_words: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    split_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        start = int(chunk["carrier_start"])
        if int(chunk["carrier_end"]) - start <= CARRIER_TARGET_CHARS:
            kept = dict(chunk)
            kept["split_reason"] = "target_char_window"
            split_chunks.append(kept)
            continue
        boundaries = [start]
        for match in re.finditer(r"[.;!?]\s+", carrier_text[start : int(chunk["carrier_end"])]):
            boundary = start + match.end()
            if boundary - boundaries[-1] >= 8:
                boundaries.append(boundary)
        for word in carrier_words:
            if not (start < word["start"] < int(chunk["carrier_end"])):
                continue
            if word["text"].casefold().strip(".,;:!?") in CLAUSE_STARTERS and word["start"] - boundaries[-1] >= 8:
                boundaries.append(word["start"])
        boundaries.append(int(chunk["carrier_end"]))
        boundaries = sorted(set(boundaries))
        for idx, left in enumerate(boundaries[:-1]):
            right = boundaries[idx + 1]
            text = carrier_text[left:right].strip()
            if not text:
                continue
            split_chunks.append(
                {
                    "text": text,
                    "carrier_start": left,
                    "carrier_end": right,
                    "word_start": _word_index_at_or_after(carrier_words, left),
                    "word_end": _word_index_after(carrier_words, right),
                    "split_reason": "natural_sentence_or_clause_boundary" if len(boundaries) > 2 else "target_char_window",
                }
            )
    return _merge_short_carrier_chunks(carrier_text, carrier_words, split_chunks)


def _merge_short_carrier_chunks(
    carrier_text: str,
    carrier_words: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    idx = 0
    while idx < len(chunks):
        chunk = chunks[idx]
        word_count = int(chunk["word_end"]) - int(chunk["word_start"])
        is_tiny = len(str(chunk["text"]).strip()) < 12 or word_count < 2
        if is_tiny and _chunk_has_placeholder(carrier_words, chunk):
            kept = dict(chunk)
            kept["split_reason"] = "kept_short_slot_chunk_for_iam_prompt_padding"
            merged.append(kept)
        elif is_tiny and merged:
            previous = merged.pop()
            combined = _carrier_chunk_from_bounds(carrier_text, carrier_words, int(previous["carrier_start"]), int(chunk["carrier_end"]))
            combined["split_reason"] = "merged_short_orphan_chunk"
            merged.append(combined)
        elif is_tiny and idx + 1 < len(chunks):
            next_chunk = chunks[idx + 1]
            combined = _carrier_chunk_from_bounds(carrier_text, carrier_words, int(chunk["carrier_start"]), int(next_chunk["carrier_end"]))
            combined["split_reason"] = "merged_short_leading_chunk"
            merged.append(combined)
            idx += 1
        else:
            merged.append(chunk)
        idx += 1
    return merged


def _chunk_has_placeholder(carrier_words: list[dict[str, Any]], chunk: dict[str, Any]) -> bool:
    return any(
        word["text"].casefold() in PLACEHOLDER_WORDS
        for word in carrier_words[int(chunk["word_start"]) : int(chunk["word_end"])]
    )


def _carrier_chunk_from_bounds(
    carrier_text: str,
    carrier_words: list[dict[str, Any]],
    left: int,
    right: int,
) -> dict[str, Any]:
    return {
        "text": carrier_text[left:right].strip(),
        "carrier_start": left,
        "carrier_end": right,
        "word_start": _word_index_at_or_after(carrier_words, left),
        "word_end": _word_index_after(carrier_words, right),
    }


def _word_index_at_or_after(words: list[dict[str, Any]], char_pos: int) -> int:
    for idx, word in enumerate(words):
        if word["end"] > char_pos:
            return idx
    return len(words)


def _word_index_after(words: list[dict[str, Any]], char_pos: int) -> int:
    for idx, word in enumerate(words):
        if word["start"] >= char_pos:
            return idx
    return len(words)


def _padding_for(text: str) -> str:
    words: list[str] = []
    idx = 0
    while len(f"{text} {' '.join(words)}".strip()) < CARRIER_TARGET_CHARS:
        words.append(PADDING_WORDS[idx % len(PADDING_WORDS)])
        idx += 1
    return " ".join(words)


def _slot_box_for_display(
    crop: Path,
    slot_box: dict[str, Any],
    *,
    rendered_width: float,
    rendered_height: float,
    source: str,
) -> tuple[dict[str, Any], str]:
    display_box = _display_space_word_box(crop, slot_box, rendered_width=rendered_width, rendered_height=rendered_height, source=source)
    refined = _refine_slot_box_from_alpha_ink(crop, display_box, rendered_width=rendered_width, rendered_height=rendered_height)
    if refined is None:
        return display_box, source
    return refined, f"visual_alpha_refined_from_{source}"


def _display_space_word_box(
    crop: Path,
    slot_box: dict[str, Any],
    *,
    rendered_width: float,
    rendered_height: float,
    source: str,
) -> dict[str, Any]:
    if source != "ocr":
        return dict(slot_box)
    image_w, image_h = png_size(crop)
    sx = rendered_width / max(1.0, float(image_w))
    sy = rendered_height / max(1.0, float(image_h))
    return dict(slot_box) | {
        "left": float(slot_box["left"]) * sx,
        "right": float(slot_box["right"]) * sx,
        "top": float(slot_box["top"]) * sy,
        "bottom": float(slot_box["bottom"]) * sy,
        "width": float(slot_box["width"]) * sx,
        "height": float(slot_box["height"]) * sy,
    }


def _refine_slot_box_from_alpha_ink(
    crop: Path,
    display_box: dict[str, Any],
    *,
    rendered_width: float,
    rendered_height: float,
) -> dict[str, Any] | None:
    Image, np = require_pillow_numpy()
    image = Image.open(crop).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    sx = image.width / max(1.0, rendered_width)
    sy = image.height / max(1.0, rendered_height)
    left = float(display_box["left"]) * sx
    right = float(display_box["right"]) * sx
    top = float(display_box.get("top", 0.0)) * sy
    bottom = float(display_box.get("bottom", rendered_height)) * sy
    est_w = max(1.0, right - left)
    est_h = max(1.0, bottom - top)
    x0 = max(0, int(left - est_w * 0.18))
    x1 = min(image.width, int(right + est_w * 0.18) + 1)
    y0 = max(0, int(top - est_h * 0.40))
    y1 = min(image.height, int(bottom + est_h * 0.40) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = alpha[y0:y1, x0:x1] > 18
    if not bool(roi.any()):
        return None
    runs = _merged_ink_column_runs(roi.any(axis=0), max_gap=max(3, int(est_w * 0.05)))
    selected_run = _select_placeholder_ink_run(runs, expected_left=left - x0, expected_right=right - x0)
    if selected_run is None:
        return None
    run_left, run_right = selected_run
    selected_roi = roi[:, run_left:run_right]
    if not bool(selected_roi.any()):
        return None
    ys, xs = np.where(selected_roi)
    ink_left = x0 + run_left + int(xs.min())
    ink_right = x0 + run_left + int(xs.max()) + 1
    ink_top = y0 + int(ys.min())
    ink_bottom = y0 + int(ys.max()) + 1
    min_width = est_w * 0.25
    if ink_right - ink_left < min_width:
        return None
    pad_x = max(3, int(est_w * 0.08))
    pad_y = max(2, int(est_h * 0.08))
    display_left = max(0.0, (ink_left - pad_x) / sx)
    display_right = min(rendered_width, (ink_right + pad_x) / sx)
    display_top = max(0.0, (ink_top - pad_y) / sy)
    display_bottom = min(rendered_height, (ink_bottom + pad_y) / sy)
    return dict(display_box) | {
        "left": display_left,
        "right": display_right,
        "top": display_top,
        "bottom": display_bottom,
        "width": display_right - display_left,
        "height": display_bottom - display_top,
        "visual_detection": {
            "method": "alpha_column_run_overlap",
            "search_box": {
                "left": round(x0 / sx, 3),
                "right": round(x1 / sx, 3),
                "top": round(y0 / sy, 3),
                "bottom": round(y1 / sy, 3),
            },
            "selected_run": {
                "left": round((x0 + run_left) / sx, 3),
                "right": round((x0 + run_right) / sx, 3),
            },
            "run_count": len(runs),
        },
    }


def _merged_ink_column_runs(columns: Any, *, max_gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_seen: int | None = None
    for idx, has_ink in enumerate(columns.tolist()):
        if has_ink:
            if start is None:
                start = idx
            last_seen = idx
            continue
        if start is not None and last_seen is not None and idx - last_seen > max_gap:
            runs.append((start, last_seen + 1))
            start = None
            last_seen = None
    if start is not None and last_seen is not None:
        runs.append((start, last_seen + 1))
    return runs


def _select_placeholder_ink_run(
    runs: list[tuple[int, int]],
    *,
    expected_left: float,
    expected_right: float,
) -> tuple[int, int] | None:
    if not runs:
        return None
    expected_center = (expected_left + expected_right) / 2.0

    def score(run: tuple[int, int]) -> tuple[float, float, int]:
        left, right = run
        overlap = max(0.0, min(float(right), expected_right) - max(float(left), expected_left))
        center_distance = abs(((left + right) / 2.0) - expected_center)
        return (overlap, -center_distance, right - left)

    return max(runs, key=score)


def carrier_prose_crops(line, assets):
    """Extract prose groups only when a whole carrier and its word boundaries agree.

    The removed placeholder has its own measurable gap on both sides. The
    caller lays the math on the same baseline at its natural width.
    """
    import numpy as np
    from PIL import Image

    from ..diffbrush import DiffBrushUnavailable
    from ..diffbrush.runner import ProseVerificationError
    from ..eval.ocr import OCRUnavailable, ocr_image, verify_ocr_text

    spans = line.spans
    fragments = []
    cursor = 0
    for span in spans:
        if span.kind == "math":
            fragments.append({"text": span.text, "source_start": cursor, "source_end": cursor + len(span.text)})
        cursor += len(span.text)
    if not fragments or not any(s.kind == "prose" and s.text.strip() for s in spans):
        return {}, None
    plan = _slot_carrier_plan(line.text, fragments)
    tokens, groups = [], {}
    slot_idx = 0
    for span in spans:
        left = len(tokens)
        if span.kind == "math":
            tokens.append(plan["placeholders"][slot_idx]["text"])
            slot_idx += 1
        else:
            tokens.extend(span.text.split())
            if len(tokens) > left:
                groups[span.id] = (left, len(tokens))
    # Adjacent punctuation gets its own token here so that cropping a slot
    # can never erase a source comma or full stop.
    prompt = " ".join(tokens)
    report = {"source_text": line.text, "slot_carrier_text": prompt, "insertions": fragments,
              "used": False}
    if len(prompt) > 52:
        return {}, report | {"reason": "carrier_exceeds_line_budget"}
    try:
        run = assets.generate(prompt, prompt_text=_iam_like_prompt_text(prompt), carrier=True)
    except ProseVerificationError:
        return {}, report | {"reason": "ocr_unverified"}
    except (DiffBrushUnavailable, OCRUnavailable):
        if assets.mode != "auto":
            raise
        return {}, report | {"reason": "generator_unavailable"}
    report["diffbrush"] = run
    boxes = run.get("ocr_word_boxes", {}).get("words", [])
    if not run["exactness_certified"] or [b["text"] for b in boxes] != tokens:
        return {}, report | {"reason": "unverified_word_boundaries"}
    with Image.open(run["styled_crop"]["path"]) as source:
        rgba = source.convert("RGBA")
    active = np.asarray(rgba.getchannel("A")).max(axis=0) > 12
    cuts = [0]
    for prev, nxt in pairwise(boxes):
        left, right = int(prev["right"]), int(nxt["left"])
        if not (cuts[-1] < left < right <= rgba.width):
            return {}, report | {"reason": "overlapping_word_boxes"}
        blanks = np.flatnonzero(~active[left:right]) + left
        if not len(blanks):
            return {}, report | {"reason": "touching_word_ink"}
        cuts.append(int(blanks[np.argmin(abs(blanks - (left + right) / 2))]))
    cuts.append(rgba.width)
    cropped = {}
    for span_id, (start, end) in groups.items():
        box = (cuts[start], 0, cuts[end], rgba.height)
        cropped_image = rgba.crop(box)
        ink = cropped_image.getchannel("A").getbbox()
        if not ink:
            return {}, report | {"reason": "empty_prose_region"}
        path = assets.out_dir / "carrier_prose" / f"{span_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cropped_image.crop(ink).save(path)
        text = " ".join(tokens[start:end])
        reading = ocr_image(path, psm=7)
        verification = verify_ocr_text(text, reading["text"])
        if not verification["passed"]:
            return {}, report | {"reason": "cropped_prose_ocr_mismatch"}
        cropped[span_id] = (path, {"text": text, "ocr_verification": verification, "engine": reading["engine"],
                                  "word_start": start, "word_end": end, "source_box": list(box),
                                  "trim_box": list(ink), "carrier_id": run["id"]})
    return cropped, report | {"used": True}
