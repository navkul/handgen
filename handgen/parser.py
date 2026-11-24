from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    id: str
    line_index: int
    index_in_line: int
    kind: str
    text: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class ParsedLine:
    index: int
    text: str
    spans: tuple[Span, ...]
    ending: str = ""


@dataclass(frozen=True)
class ParsePlan:
    source_text: str
    lines: tuple[ParsedLine, ...]
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class ProseToken:
    kind: str
    text: str


def _math_end(markup: str, start: int) -> int:
    # Consume closing brackets belonging to indices such as A[B[i]].
    # Brackets in half-open intervals may have no matching square bracket.
    depth = 0
    fallback = -1
    for match in re.compile(r"\[+|\]+").finditer(markup, start):
        length = match.end() - match.start()
        if markup[match.start()] == "[":
            if length >= 2 and fallback >= 0:
                break
            depth += length
        else:
            if length >= depth + 2:
                return match.start() + depth
            if length >= 2:
                fallback = match.end() - 2
            depth = max(0, depth - length)
    return fallback


def parse_source(source_text: str) -> ParsePlan:
    lines: list[ParsedLine] = []
    spans: list[Span] = []
    cursor = 0
    for line_index, raw_line in enumerate(source_text.splitlines(keepends=True)):
        markup = raw_line.splitlines()[0]
        ending = raw_line[len(markup):]
        plain = ""
        parts: list[Span] = []
        pos = 0
        part_index = 0
        while pos < len(markup):
            if markup.startswith("[[", pos):
                end = _math_end(markup, pos + 2)
                if end == -1:
                    raise ValueError(f"unclosed math delimiter on line {line_index + 1}")
                text = markup[pos + 2 : end]
                kind = "math"
                pos = end + 2
            else:
                next_math = markup.find("[[", pos)
                if next_math == -1:
                    text = markup[pos:]
                    pos = len(markup)
                else:
                    text = markup[pos:next_math]
                    pos = next_math
                kind = "prose"
            if text == "":
                continue
            start = cursor + len(plain)
            plain += text
            span = Span(
                id=f"line{line_index:02d}_{kind}{part_index:02d}",
                line_index=line_index,
                index_in_line=part_index,
                kind=kind,
                text=text,
                source_start=start,
                source_end=cursor + len(plain),
            )
            parts.append(span)
            spans.append(span)
            part_index += 1
        lines.append(ParsedLine(index=line_index, text=plain, spans=tuple(parts), ending=ending))
        cursor += len(plain) + len(ending)
    return ParsePlan(source_text="".join(line.text + line.ending for line in lines), lines=tuple(lines), spans=tuple(spans))


def _split_long_token(token: str, max_chars: int) -> list[str]:
    return [token[i : i + max_chars] for i in range(0, len(token), max_chars)]


def chunk_prose(text: str, max_words: int = 6, max_chars: int | None = None) -> list[str]:
    if max_chars is not None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            for piece in _split_long_token(word, max_chars):
                if not current:
                    current = [piece]
                    current_len = len(piece)
                    continue
                proposed_len = current_len + 1 + len(piece)
                if proposed_len > max_chars:
                    chunks.append(" ".join(current))
                    current = [piece]
                    current_len = len(piece)
                    continue
                current.append(piece)
                current_len = proposed_len
        if current:
            chunks.append(" ".join(current))
        return chunks
    if not text.strip():
        return []
    chunks = []
    current = ""
    word_count = 0
    for match in re.finditer(r"\S+\s*", text):
        token = match.group(0)
        if word_count >= max_words and current:
            chunks.append(current.strip())
            current = ""
            word_count = 0
        current += token
        word_count += 1
    if current:
        chunk = current.strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def tokenize_prose(text: str) -> list[ProseToken]:
    tokens: list[ProseToken] = []
    pos = 0
    while pos < len(text):
        start = pos
        is_space = text[pos].isspace()
        while pos < len(text) and text[pos].isspace() == is_space:
            pos += 1
        tokens.append(ProseToken(kind="space" if is_space else "text", text=text[start:pos]))
    return tokens


def plan_to_dict(plan: ParsePlan) -> dict:
    return {
        "schema_version": "handgen_parse_plan_v1",
        "source_text": plan.source_text,
        "delimiter_policy": "Text inside [[...]] is math. Other text is prose.",
        "lines": [
            {
                "index": line.index,
                "text": line.text,
                "ending": line.ending,
                "spans": [
                    {
                        "id": span.id,
                        "kind": span.kind,
                        "text": span.text,
                        "source_start": span.source_start,
                        "source_end": span.source_end,
                    }
                    for span in line.spans
                ],
            }
            for line in plan.lines
        ],
    }
