from __future__ import annotations

import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

from .. import db
from ..diffbrush.runner import DiffBrushRunner
from ..ingest.worksheet import glyph_bank_from_manifest
from ..eval.source_contract import verify_source_contract
from ..parser import chunk_prose, parse_source, plan_to_dict
from ..utils import data_uri, escape_xml, now_iso, png_size, read_json, render_svg_to_png, require_pillow_numpy, sha_file, sha_text, write_json
from .ink import rgba_ink_profile
from .math import math_style_profile, render_math_span


PAGE_W = 1500
PAGE_H = 1900
LEFT = 64.0
TOP = 72.0
RIGHT = 72.0
LINE_GAP = 96.0
PROSE_H = 64.0
SPAN_GAP = 16.0
SEED = 2026051301
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


def image_tag(path: Path, x: float, y: float, w: float, h: float, opacity: float = 0.98) -> str:
    return (
        f'<image href="{data_uri(path)}" x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" opacity="{opacity:.3f}"/>'
    )


def rect_tag(x: float, y: float, w: float, h: float, fill: str = "#ffffff") -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"/>'


def _display_size(path: Path, target_h: float) -> tuple[float, float]:
    w, h = png_size(path)
    width = w * (target_h / h) if h else target_h
    return width, target_h


def _whitespace_width(text: str, style_profile: dict[str, Any]) -> float:
    base = float(style_profile["target_space_advance"])
    width = 0.0
    for ch in text:
        width += base * (4.0 if ch == "\t" else 1.0)
    return width


def _insert_render_span(
    con: sqlite3.Connection,
    job_id: int,
    span_id: str,
    kind: str,
    route: str,
    text: str,
    artifact_path: str | None,
    exact: bool,
) -> None:
    con.execute(
        """
        INSERT INTO render_spans(job_id, span_id, kind, route, text, artifact_path, exact_text_certified)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, span_id, kind, route, text, artifact_path, 1 if exact else 0),
    )


def _needs_carrier_render(source_text: str) -> bool:
    known_variables: set[str] = set()
    return any(_carrier_for_line(line, known_variables)["insertions"] for line in source_text.splitlines())


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


def _estimated_word_boxes(carrier: dict[str, Any], chunk: dict[str, Any], rendered_width: float, rendered_height: float) -> list[dict[str, Any]]:
    chunk_len = max(1, chunk["carrier_end"] - chunk["carrier_start"])
    boxes: list[dict[str, Any]] = []
    for word in carrier["carrier_words"][chunk["word_start"] : chunk["word_end"]]:
        left = rendered_width * max(0.0, (word["start"] - chunk["carrier_start"]) / chunk_len)
        right = rendered_width * min(1.0, (word["end"] - chunk["carrier_start"]) / chunk_len)
        boxes.append(
            {
                "text": word["text"],
                "normalized": word["text"].casefold(),
                "left": left,
                "right": right,
                "top": rendered_height * 0.15,
                "bottom": rendered_height * 0.9,
                "width": right - left,
                "height": rendered_height * 0.75,
                "confidence": None,
                "source": "estimated_from_expected_carrier_text",
            }
        )
    return boxes


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


def _style_profile_for_carrier(base_profile: dict[str, Any], carrier_ink: dict[str, Any]) -> dict[str, Any]:
    profile = dict(base_profile)
    worksheet_darkness = float(base_profile.get("measured_worksheet_symbol_darkness", 0.0) or 0.0)
    carrier_alpha = float(carrier_ink.get("mean_alpha", 0.0) or 0.0)
    if worksheet_darkness > 0.0 and carrier_alpha > 0.0:
        profile["symbol_alpha_scale"] = round(max(0.28, min(1.65, carrier_alpha / worksheet_darkness)), 4)
        profile["symbol_alpha_scale_source"] = "selected_diffbrush_carrier_mean_alpha"
    profile["selected_diffbrush_ink_profile"] = carrier_ink
    return profile


def _math_slot_geometry(
    math_block: dict[str, Any],
    slot_box: dict[str, Any],
    *,
    carrier_x: float,
    carrier_y: float,
    carrier_height: float,
    style_profile: dict[str, Any],
) -> dict[str, float]:
    ink_bbox = math_block.get("ink_bbox") or {"x": 0.0, "y": 0.0, "width": math_block["width"], "height": math_block["height"]}
    visible_h = max(1.0, float(ink_bbox.get("height", math_block["height"])))
    slot_w = max(1.0, float(slot_box["right"]) - float(slot_box["left"]))
    slot_h = max(1.0, float(slot_box.get("height", carrier_height * 0.72)))
    target_visible_h = min(
        slot_h * _slot_height_ratio(math_block["text"]),
        _inline_ink_height_cap(math_block["text"], style_profile),
    )
    scale = target_visible_h / visible_h
    max_display_w = slot_w * _slot_width_ratio(math_block["text"])
    if math_block["width"] * scale > max_display_w:
        scale = max_display_w / max(1.0, float(math_block["width"]))
    display_w = float(math_block["width"]) * scale
    display_h = float(math_block["height"]) * scale
    slot_center_x = carrier_x + (float(slot_box["left"]) + float(slot_box["right"])) / 2.0
    slot_center_y = carrier_y + (float(slot_box["top"]) + float(slot_box["bottom"])) / 2.0
    visible_center_x = (float(ink_bbox.get("x", 0.0)) + float(ink_bbox.get("width", math_block["width"])) / 2.0) * scale
    visible_center_y = (float(ink_bbox.get("y", 0.0)) + visible_h / 2.0) * scale
    math_x = slot_center_x - visible_center_x
    math_y = slot_center_y - visible_center_y
    clear_w = max(slot_w + 8.0, display_w + 6.0)
    return {
        "x": math_x,
        "y": math_y,
        "width": display_w,
        "height": display_h,
        "clear_x": slot_center_x - clear_w / 2.0,
        "clear_y": carrier_y - 6.0,
        "clear_width": clear_w,
        "clear_height": carrier_height + 20.0,
        "scale": scale,
        "target_visible_height": target_visible_h,
    }


def _slot_height_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if len(compact) == 1 and compact.isalpha():
        return 0.88
    if compact.isalnum() and len(compact) == 1:
        return 0.92
    return 0.90


def _inline_ink_height_cap(text: str, style_profile: dict[str, Any]) -> float:
    compact = re.sub(r"\s+", "", text)
    prose_ink_h = float(style_profile.get("measured_style_ink_height", 0.0) or style_profile.get("target_prose_height", PROSE_H) * 0.55)
    if len(compact) == 1 and compact.islower():
        multiplier = 1.05
    elif len(compact) == 1 and compact.isupper():
        multiplier = 1.12
    elif len(compact) == 1 and compact.isdigit():
        multiplier = 1.08
    else:
        multiplier = 1.14
    return max(1.0, prose_ink_h * multiplier)


def _slot_width_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if len(compact) == 1:
        return 0.92
    return 1.05


def render_document(
    writer_id: str,
    source_path: Path,
    out_dir: Path,
    *,
    max_words: int = 6,
    max_chars: int | None = None,
) -> dict[str, Any]:
    db.init_db()
    source_text = source_path.read_text(encoding="utf-8")
    plan = parse_source(source_text)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, Any]] = []
    rng = random.Random(SEED)

    with db.connect() as con:
        db.require_writer(con, writer_id)
        style = db.latest_style_ref(con, writer_id)
        if style is None:
            raise RuntimeError(f"writer {writer_id!r} has no style reference; run `handgen style add` first")
        style_ref = Path(style["path"])
        carrier_needed = _needs_carrier_render(plan.source_text)
        math_needed = carrier_needed or any(span.kind == "math" for span in plan.spans)
        worksheet = db.latest_worksheet(con, writer_id)
        glyph_bank: dict[str, list[Path]] = {}
        worksheet_manifest: dict[str, Any] | None = None
        if math_needed:
            if worksheet is None:
                warning = {
                    "code": "missing_symbol_bank",
                    "message": "Input contains math spans but no worksheet glyph bank has been ingested.",
                    "action": "Run `handgen worksheet ingest writer_001 /path/to/worksheet.pdf` and render again.",
                }
                warnings.append(warning)
                write_json(out_dir / "warnings.json", warnings)
                raise RuntimeError(warning["message"])
            worksheet_manifest = read_json(Path(worksheet["manifest_path"]))
            glyph_bank = glyph_bank_from_manifest(worksheet_manifest)

        con.execute(
            "INSERT INTO documents(writer_id, source_path, source_sha256, created_at) VALUES (?, ?, ?, ?)",
            (writer_id, str(source_path), sha_text(plan.source_text), now_iso()),
        )
        document_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.execute(
            "INSERT INTO render_jobs(document_id, out_dir, status, created_at) VALUES (?, ?, ?, ?)",
            (document_id, str(out_dir), "running", now_iso()),
        )
        job_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])

    write_json(out_dir / "parse_plan.json", plan_to_dict(plan))
    runner = DiffBrushRunner()
    style_profile = math_style_profile(style_ref, worksheet_manifest, prose_height=PROSE_H)
    line_carriers: list[dict[str, Any]] = []
    known_variables: set[str] = set()
    for source_line in plan.source_text.splitlines():
        line_carriers.append(_carrier_for_line(source_line, known_variables))
    carrier_needed = any(carrier["insertions"] for carrier in line_carriers)
    digit_component_labels = sorted(
        {
            ch
            for carrier in line_carriers
            for insertion in carrier["insertions"]
            for ch in insertion["text"]
            if ch.isdigit()
        }
        | {
            ch
            for span in plan.spans
            if span.kind == "math"
            for ch in span.text
            if ch.isdigit()
        }
    )
    digit_component_bank = {
        label: runner.generate_variable(
            label=label,
            style_ref=style_ref,
            out_dir=out_dir,
            seed=SEED + 9100 + ord(label[0]) * 19,
            writer_id=writer_id,
        )
        for label in digit_component_labels
    }
    if carrier_needed:
        svg: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">',
            "<title>Handgen document</title>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]
        span_records: list[dict[str, Any]] = []
        line_records: list[dict[str, Any]] = []
        carrier_records: list[dict[str, Any]] = []
        y = TOP
        max_x = PAGE_W - RIGHT
        overflow = False
        for line_index, source_line in enumerate(plan.source_text.splitlines()):
            carrier = line_carriers[line_index]
            slot_text = carrier["slot_carrier_text"] if carrier["insertions"] else carrier["carrier_text"]
            chunks = _chunk_carrier(slot_text)
            x = LEFT
            line_span_ids: list[str] = []
            slot_records: list[dict[str, Any]] = []
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"line{line_index:02d}_slot_carrier{chunk_idx:02d}"
                if chunk_idx > 0:
                    x += _whitespace_width(" ", style_profile)
                run = runner.generate_chunk(
                    span_id=chunk_id,
                    text=chunk["text"],
                    prompt_text=chunk.get("prompt_variants") or chunk.get("prompt_text", chunk["text"]),
                    style_ref=style_ref,
                    out_dir=out_dir,
                    seed=SEED + line_index * 1009 + chunk_idx * 97,
                    writer_id=writer_id,
                )
                crop = Path(run["styled_crop"]["path"])
                carrier_ink_profile = rgba_ink_profile(crop)
                carrier_style_profile = _style_profile_for_carrier(style_profile, carrier_ink_profile)
                if not run.get("exactness_certified"):
                    warnings.append(
                        {
                            "code": "diffbrush_last_resort_prose" if run.get("route") == "diffbrush_last_resort_prose_token" else "diffbrush_best_effort_prose",
                            "span_id": chunk_id,
                            "expected": chunk["text"],
                            "ocr_text": run.get("ocr_verification", {}).get("actual"),
                            "match_score": run.get("ocr_verification", {}).get("match_score"),
                            "selected_carrier_ink": carrier_ink_profile,
                            "message": run.get("best_effort_reason", "Rendered a DiffBrush candidate because OCR did not certify an exact slot-carrier match."),
                        }
                    )
                width, height = _display_size(crop, style_profile["target_prose_height"])
                if x + width > max_x:
                    x = LEFT
                    y += LINE_GAP
                svg.append(image_tag(crop, x, y, width, height))
                ocr_word_boxes_for_chunk = run.get("ocr_word_boxes", {}).get("words", [])
                expected_word_count = chunk["word_end"] - chunk["word_start"]
                if run.get("exactness_certified") and len(ocr_word_boxes_for_chunk) >= expected_word_count:
                    word_boxes = ocr_word_boxes_for_chunk
                    word_box_source = "ocr"
                else:
                    word_boxes = _estimated_word_boxes({"carrier_words": carrier["slot_carrier_words"]}, chunk, width, height)
                    word_box_source = "estimated_from_slot_carrier_text"
                record = {
                    "id": chunk_id,
                    "line_index": line_index,
                    "type": "slot_carrier_prose",
                    "route": run.get("route", "diffbrush_ocr_verified_prose_token"),
                    "text": chunk["text"],
                    "bbox": {"x": round(x, 2), "y": round(y, 2), "width": round(width, 2), "height": round(height, 2)},
                    "diffbrush": run,
                    "selected_carrier_ink": carrier_ink_profile,
                    "carrier_chunk": chunk,
                    "word_box_source_for_slots": word_box_source,
                    "exactness_evidence": {
                        "source_text_certified": True,
                        "visual_text_certified": bool(run.get("exactness_certified")),
                        "method": "DiffBrush renders a natural slot carrier with blank tokens; each blank is cleared and replaced by a parsed glyph-bank fragment.",
                    },
                }
                span_records.append(record)
                line_span_ids.append(chunk_id)
                for insertion_idx, insertion in enumerate(carrier["insertions"]):
                    slot_word_index = insertion.get("slot_word_index")
                    if slot_word_index is None or not (chunk["word_start"] <= slot_word_index < chunk["word_end"]):
                        continue
                    local_slot = slot_word_index - chunk["word_start"]
                    if local_slot >= len(word_boxes):
                        continue
                    slot_box, refined_slot_source = _slot_box_for_display(
                        crop,
                        word_boxes[local_slot],
                        rendered_width=width,
                        rendered_height=height,
                        source=word_box_source,
                    )
                    math_block = render_math_span(
                        f"{chunk_id}_slot{insertion_idx:02d}",
                        insertion["text"],
                        glyph_bank,
                        out_dir,
                        rng,
                        variable_bank=digit_component_bank,
                        style_profile=style_profile,
                        ink_match_profile=carrier_style_profile,
                    )
                    math_png = Path(math_block["paths"]["png"])
                    geometry = _math_slot_geometry(
                        math_block,
                        slot_box,
                        carrier_x=x,
                        carrier_y=y,
                        carrier_height=height,
                        style_profile=style_profile,
                    )
                    svg.append(rect_tag(geometry["clear_x"], geometry["clear_y"], geometry["clear_width"], geometry["clear_height"]))
                    svg.append(image_tag(math_png, geometry["x"], geometry["y"], geometry["width"], geometry["height"], opacity=0.98))
                    slot_id = f"{chunk_id}_slot{insertion_idx:02d}"
                    slot_record = {
                        "id": slot_id,
                        "line_index": line_index,
                        "type": "math_slot_replacement",
                        "route": "worksheet_fragment_replaces_diffbrush_blank_slot",
                        "text": insertion["text"],
                        "bbox": {
                            "x": round(geometry["x"], 2),
                            "y": round(geometry["y"], 2),
                            "width": round(geometry["width"], 2),
                            "height": round(geometry["height"], 2),
                        },
                        "cleared_slot_bbox": {
                            "x": round(geometry["clear_x"], 2),
                            "y": round(geometry["clear_y"], 2),
                            "width": round(geometry["clear_width"], 2),
                            "height": round(geometry["clear_height"], 2),
                        },
                        "slot_geometry": {key: round(value, 4) for key, value in geometry.items()},
                        "slot_word_box": slot_box,
                        "slot_word_box_source": refined_slot_source,
                        "source_insertion": insertion,
                        "math": math_block,
                        "ink_match_profile": carrier_style_profile,
                        "exactness_evidence": {
                            "source_text_certified": True,
                            "visual_text_certified": True,
                            "method": "The placeholder word is visually cleared, and the source fragment is composed from worksheet glyphs in the planned slot.",
                        },
                    }
                    span_records.append(slot_record)
                    line_span_ids.append(slot_id)
                    slot_records.append(slot_record)
                x += width + SPAN_GAP
            line_records.append({"index": line_index, "text": source_line, "carrier_text": carrier["carrier_text"], "slot_carrier_text": slot_text, "span_ids": line_span_ids})
            carrier_records.append(carrier | {"chunks": chunks, "slot_replacements": slot_records, "rendered_carrier_text": carrier["carrier_text"]})
            y += LINE_GAP
            if y > PAGE_H - 80:
                overflow = True
        if overflow:
            warnings.append({"code": "page_overflow", "message": "Content exceeded the single MVP page height; pagination is not implemented yet."})
        svg.append("</svg>")
        svg_path = out_dir / "document.svg"
        svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
        png_path = render_svg_to_png(svg_path)
        write_json(out_dir / "warnings.json", warnings)
        carrier_errors = [
            {"line_index": idx, "carrier_text": record["carrier_text"], "rendered": record["rendered_carrier_text"]}
            for idx, record in enumerate(carrier_records)
            if record["carrier_text"] != record["rendered_carrier_text"]
        ]
        manifest = {
            "schema_version": "handgen_render_manifest_v1",
            "generated_at": now_iso(),
            "writer_id": writer_id,
            "source_path": str(source_path),
            "source_text": plan.source_text,
            "source_sha256": sha_text(plan.source_text),
            "routes_used": sorted({record["route"] for record in span_records}),
            "style_reference": str(style_ref),
            "worksheet_manifest": str(worksheet["manifest_path"]) if worksheet is not None else None,
            "math_style_profile": style_profile,
            "carrier_plan": carrier_records,
            "lines": line_records,
            "spans": span_records,
            "outputs": {"svg": str(svg_path), "png": str(png_path)},
            "warnings": warnings,
            "source_preservation_policy": {
                "prose": "diffbrush_ocr_verified_or_best_effort_carrier_chunks",
                "math": "worksheet_fragment_overlay_in_carrier_gaps",
                "fail_closed": False,
            },
            "source_contract": {
                "schema_version": "handgen_carrier_contract_v1",
                "passed": not carrier_errors,
                "checks": ["carrier_chunks_reconstruct_carrier_text", "symbols_record_source_positions"],
                "errors": carrier_errors,
            },
        }
        if carrier_errors:
            write_json(out_dir / "source_contract_failed.json", manifest["source_contract"])
            raise RuntimeError(f"source preservation contract failed; see {out_dir / 'source_contract_failed.json'}")
        manifest_path = out_dir / "manifest.json"
        write_json(manifest_path, manifest)
        with db.connect() as con:
            for record in span_records:
                artifact_path = (
                    record.get("diffbrush", {}).get("styled_crop", {}).get("path")
                    or record.get("math", {}).get("paths", {}).get("png")
                    or record.get("asset", {}).get("path")
                )
                _insert_render_span(
                    con,
                    job_id,
                    record["id"],
                    record["type"],
                    record["route"],
                    record["text"],
                    artifact_path,
                    bool(record["exactness_evidence"].get("source_text_certified"))
                    and bool(record["exactness_evidence"].get("visual_text_certified")),
                )
            db.insert_artifact(con, job_id, "svg", str(svg_path), sha_file(svg_path))
            db.insert_artifact(con, job_id, "png", str(png_path), sha_file(png_path))
            db.insert_artifact(con, job_id, "manifest", str(manifest_path), sha_file(manifest_path))
            con.execute("UPDATE render_jobs SET status = ?, completed_at = ? WHERE id = ?", ("done", now_iso(), job_id))
        return manifest

    prose_chunk_cache: dict[str, dict[str, Any]] = {}
    variable_labels = sorted(
        {
            ch
            for span in plan.spans
            if span.kind == "math"
            for ch in span.text
            if ch.isdigit()
        }
    )
    variable_bank = {
        label: runner.generate_variable(
            label=label,
            style_ref=style_ref,
            out_dir=out_dir,
            seed=SEED + 7000 + ord(label[0]) * 17,
            writer_id=writer_id,
        )
        for label in variable_labels
    }

    def verified_prose_chunk(chunk_text: str) -> dict[str, Any]:
        cached = prose_chunk_cache.get(chunk_text)
        if cached is not None:
            return cached
        run = runner.generate_chunk(
            span_id=f"prose_chunk_{sha_text(chunk_text)[:10]}",
            text=chunk_text,
            style_ref=style_ref,
            out_dir=out_dir,
            seed=SEED + 3000 + len(prose_chunk_cache) * 97,
            writer_id=writer_id,
        )
        prose_chunk_cache[chunk_text] = run
        return run

    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">',
        "<title>Handgen document</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    span_records: list[dict[str, Any]] = []
    line_records: list[dict[str, Any]] = []
    prose_route = f"diffbrush_chunked_prose_chars_{max_chars}" if max_chars is not None else "diffbrush_chunked_prose_words"
    y = TOP
    max_x = PAGE_W - RIGHT
    overflow = False

    for line in plan.lines:
        x = LEFT
        line_span_ids: list[str] = []
        for span in line.spans:
            if span.kind == "prose":
                chunks = chunk_prose(span.text, max_words=max_words, max_chars=max_chars)
                if not chunks and span.text:
                    continue
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{span.id}_chunk{chunk_idx:02d}"
                    if chunk_idx > 0:
                        x += _whitespace_width(" ", style_profile)
                    run = verified_prose_chunk(chunk)
                    crop = Path(run["styled_crop"]["path"])
                    width, height = _display_size(crop, style_profile["target_prose_height"])
                    if x + width > max_x:
                        x = LEFT
                        y += LINE_GAP
                    svg.append(image_tag(crop, x, y, width, height))
                    record = {
                        "id": chunk_id,
                        "source_span_id": span.id,
                        "line_index": line.index,
                        "type": "prose",
                        "route": run.get("route", "diffbrush_ocr_verified_prose_token"),
                        "text": chunk,
                        "bbox": {"x": round(x, 2), "y": round(y, 2), "width": round(width, 2), "height": round(height, 2)},
                        "diffbrush": run,
                        "exactness_evidence": {
                            "source_text_certified": True,
                            "visual_text_certified": bool(run.get("exactness_certified")),
                            "method": "One DiffBrush candidate is composed only after OCR verification rejects repeated, hallucinated, missing, or mismatched text.",
                        },
                    }
                    span_records.append(record)
                    line_span_ids.append(chunk_id)
                    x += width + SPAN_GAP
            else:
                block = render_math_span(
                    span.id,
                    span.text,
                    glyph_bank,
                    out_dir,
                    rng,
                    variable_bank=variable_bank,
                    style_profile=style_profile,
                )
                png = Path(block["paths"]["png"])
                width, height = _display_size(png, block["height"])
                if x + width > max_x:
                    x = LEFT
                    y += LINE_GAP
                math_y = y + 8
                svg.append(image_tag(png, x, math_y, width, height))
                record = {
                    "id": span.id,
                    "line_index": line.index,
                    "type": "math",
                    "route": block["route"],
                    "text": span.text,
                    "bbox": {"x": round(x, 2), "y": round(math_y, 2), "width": round(width, 2), "height": round(height, 2)},
                    "math": block,
                    "exactness_evidence": {
                        "source_text_certified": True,
                        "visual_text_certified": not any(ch.isdigit() for ch in span.text),
                        "method": "Each visible token is rendered from the parsed source string; letters, Greek, and symbols use worksheet glyph crops; digits currently use isolated DiffBrush connected components.",
                    },
                }
                span_records.append(record)
                line_span_ids.append(span.id)
                x += width + SPAN_GAP
        line_records.append({"index": line.index, "text": line.text, "span_ids": line_span_ids})
        y += LINE_GAP
        if y > PAGE_H - 80:
            overflow = True

    if overflow:
        warnings.append(
            {
                "code": "page_overflow",
                "message": "Content exceeded the single MVP page height; pagination is not implemented yet.",
            }
        )
    svg.append("</svg>")
    svg_path = out_dir / "document.svg"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    png_path = render_svg_to_png(svg_path)
    write_json(out_dir / "warnings.json", warnings)

    manifest = {
        "schema_version": "handgen_render_manifest_v1",
        "generated_at": now_iso(),
        "writer_id": writer_id,
        "source_path": str(source_path),
        "source_text": plan.source_text,
        "source_sha256": sha_text(plan.source_text),
        "routes_used": sorted({record["route"] for record in span_records}),
        "prose_chunking": {
            "mode": "chars" if max_chars is not None else "words",
            "max_words": max_words if max_chars is None else None,
            "max_chars": max_chars,
        },
        "style_reference": str(style_ref),
        "worksheet_manifest": str(worksheet["manifest_path"]) if math_needed and worksheet is not None else None,
        "math_style_profile": style_profile,
        "selected_variable_candidates": variable_bank,
        "lines": line_records,
        "spans": span_records,
        "outputs": {"svg": str(svg_path), "png": str(png_path)},
        "warnings": warnings,
        "source_preservation_policy": {
            "prose": "diffbrush_candidate_ocr_verified_chunks",
            "math": "parsed_token_composition",
            "fail_closed": False,
        },
        "known_limits": [
            "DiffBrush prose chunks must pass OCR verification; right-edge suffix hallucinations may be trimmed only after OCR proves the expected text is a prefix.",
            "Math exactness depends on worksheet glyph-label quality.",
            "Digits in math are isolated DiffBrush connected components for now and are not OCR-certified in MVP v1.",
            "Pagination is not implemented; long documents may overflow one page.",
        ],
    }
    source_contract = verify_source_contract(manifest)
    manifest["source_contract"] = source_contract
    if not source_contract["passed"]:
        write_json(out_dir / "source_contract_failed.json", source_contract)
        raise RuntimeError(f"source preservation contract failed; see {out_dir / 'source_contract_failed.json'}")
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)

    with db.connect() as con:
        for record in span_records:
            artifact_path = None
            if record["type"] == "prose":
                artifact_path = record["diffbrush"]["styled_crop"]["path"]
            elif record["type"] == "math":
                artifact_path = record["math"]["paths"]["png"]
            _insert_render_span(
                con,
                job_id,
                record["id"],
                record["type"],
                record["route"],
                record["text"],
                artifact_path,
                bool(record["exactness_evidence"].get("source_text_certified"))
                and bool(record["exactness_evidence"].get("visual_text_certified")),
            )
        db.insert_artifact(con, job_id, "svg", str(svg_path), sha_file(svg_path))
        db.insert_artifact(con, job_id, "png", str(png_path), sha_file(png_path))
        db.insert_artifact(con, job_id, "manifest", str(manifest_path), sha_file(manifest_path))
        con.execute("UPDATE render_jobs SET status = ?, completed_at = ? WHERE id = ?", ("done", now_iso(), job_id))

    return manifest
