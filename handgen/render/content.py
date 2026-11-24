"""Preserve source ranges while planning model prompts and worksheet runs."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import replace

from ..parser import ParsedLine, Span
from .carriers import _math_fragments_for_line


def infer_math_spans(line: ParsedLine, known: set[str]) -> ParsedLine:
    if not line.spans or any(span.kind == "math" for span in line.spans):
        return line
    fragments = _math_fragments_for_line(line.text, known)
    if not fragments:
        return line
    parts: list[Span] = []
    start = line.spans[0].source_start

    def add(left: int, right: int, kind: str) -> None:
        if left < right:
            idx = len(parts)
            parts.append(Span(f"line{line.index:02d}_{kind}{idx:02d}", line.index, idx,
                              kind, line.text[left:right], start + left, start + right))

    left = 0
    for fragment in fragments:
        add(left, fragment["source_start"], "prose")
        add(fragment["source_start"], fragment["source_end"], "math")
        left = fragment["source_end"]
    add(left, len(line.text), "prose")
    return replace(line, spans=tuple(parts))


def prose_slices(source: str, split: Callable[[str], list[str]]) -> Iterator[str]:
    """Map normalized prompts back to contiguous slices with their original gaps."""
    cursor = 0
    for chunk in split(source):
        words = chunk.split()
        if not words:
            continue
        gap = re.match(r"\s*", source[cursor:]).group()
        if gap:
            yield gap
            cursor += len(gap)
        pattern = r"\s+".join(map(re.escape, words))
        match = re.compile(pattern).match(source, cursor)
        if not match:
            raise ValueError(f"chunker changed the source near offset {cursor}: {chunk!r}")
        if any(len(word) > 52 for word in words):
            # A URL or uninterrupted identifier can exceed both a model's
            # content window and a page. Keep its characters contiguous.
            for token in re.findall(r"\s+|\S+", match.group()):
                if token.isspace():
                    yield token
                else:
                    yield from (token[i:i + 48] for i in range(0, len(token), 48))
        else:
            yield match.group()
        cursor = match.end()
    tail = source[cursor:]
    if tail.strip():
        raise ValueError(f"chunker skipped source text near offset {cursor}")
    if tail:
        yield tail


def math_slices(source: str, max_chars: int = 36) -> Iterator[str]:
    """Keep ordinary formulas together, wrap long expressions near operators."""
    while len(source) > max_chars:
        breaks = list(re.finditer(r"\s+|(?=[+\-−=<>≤≥])", source[:max_chars + 1]))
        end = next((m.start() for m in reversed(breaks) if m.start() >= max_chars // 2), max_chars)
        yield source[:end]
        source = source[end:]
    if source:
        yield source
