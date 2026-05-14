"""Prose chunking strategies for DiffBrush prompt construction.

Every strategy takes a single ``text`` string and returns ``list[str]`` chunks
that, when concatenated, reproduce the original word sequence with original
whitespace preserved. Chunks are emitted as slices of the original ``text``
rather than as ``" ".join(words)`` so the source-reconstruction contract in
:mod:`handgen.eval.source_contract` holds for inputs with non-standard
whitespace (double spaces, tabs, etc.). No strategy ever splits a word.

Strategies are pure functions so they can be unit-tested in isolation and
dispatched by name from :mod:`handgen.eval.chunk_compare`.
"""

from __future__ import annotations

import random
import re
from typing import Callable

from .parser import chunk_prose

# IAM training lines cluster around ~30-50 chars; aim for ~42 with a hard
# 52-char ceiling so the model rarely sees prompts outside its distribution.
DEFAULT_TARGET_CHARS = 42
DEFAULT_HARD_MAX = 52
DEFAULT_MIN_CHARS = 28

# Punctuation priorities for the punctuation-first splitter. Earlier marks
# are stronger sentence boundaries and get tried first.
_PUNCT_PRIORITY: tuple[tuple[str, str], ...] = (
    ("period", r"(?<=[.!?])\s+"),
    ("semicolon", r"(?<=;)\s+"),
    ("colon", r"(?<=:)\s+"),
    ("comma", r"(?<=,)\s+"),
)


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) byte spans of every whitespace-separated word
    in ``text``, in left-to-right order. The whitespace between words stays
    implicit in the gaps between spans, which is what lets callers slice the
    original text and preserve unusual spacing exactly."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _slice_chunk(text: str, spans: list[tuple[int, int]], i: int, j: int) -> str:
    """Slice ``text`` from word ``i`` through word ``j`` inclusive, preserving
    whatever whitespace originally sat between those words."""
    return text[spans[i][0] : spans[j][1]]


# ---------------------------------------------------------------------------
# Current strategy (legacy baseline, retained for benchmarking)
# ---------------------------------------------------------------------------


def current(text: str) -> list[str]:
    """The legacy greedy chunker (``max_words=6``). Retained for benchmarking
    new strategies against the historical default via the chunk_compare harness."""
    return chunk_prose(text, max_words=6, max_chars=None)


# ---------------------------------------------------------------------------
# A. Balanced DP split  (production default)
# ---------------------------------------------------------------------------


def balanced_dp(
    text: str,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    hard_max_chars: int = DEFAULT_HARD_MAX,
) -> list[str]:
    """Dynamic-programming split that minimises Σ(target − chunk_len)².

    Never splits a word. Subject to ``chunk_len <= hard_max_chars`` unless a
    single word already exceeds that ceiling (in which case the word stands
    alone — handgen tokens are short enough that this is rare).

    Output chunks are slices of the original ``text`` so non-standard
    whitespace between words is preserved exactly for source reconstruction.
    """
    spans = _word_spans(text)
    if not spans:
        return []

    n = len(spans)

    def chunk_len(i: int, j: int) -> int:
        # Length of the actual emitted slice, including any original whitespace.
        return spans[j][1] - spans[i][0]

    INF = float("inf")
    cost = [INF] * (n + 1)
    split_at = [-1] * (n + 1)
    cost[n] = 0.0

    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            length = chunk_len(i, j)
            if length > hard_max_chars and j > i:
                break
            penalty = (target_chars - length) ** 2
            total = penalty + cost[j + 1]
            if total < cost[i]:
                cost[i] = total
                split_at[i] = j

    chunks: list[str] = []
    i = 0
    while i < n:
        j = split_at[i]
        if j < 0:  # defensive: single oversized word
            j = i
        chunks.append(_slice_chunk(text, spans, i, j))
        i = j + 1
    return chunks


# ---------------------------------------------------------------------------
# B. Punctuation-first split
# ---------------------------------------------------------------------------


def punctuation_first(
    text: str,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    hard_max_chars: int = DEFAULT_HARD_MAX,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    """Split on punctuation in priority order, then balance the fragments.

    Pass 1 cuts at the strongest punctuation that yields at least one
    fragment within ``min_chars..hard_max_chars``. Pass 2 merges short
    fragments forward and recursively rebalances oversized fragments via
    :func:`balanced_dp` so every output chunk stays IAM-shaped.
    """
    text = text.strip()
    if not text:
        return []

    fragments = _split_on_strongest_punctuation(text, hard_max_chars)
    fragments = _merge_short_fragments(fragments, min_chars, hard_max_chars)

    chunks: list[str] = []
    for fragment in fragments:
        if len(fragment) <= hard_max_chars:
            chunks.append(fragment)
            continue
        # Oversized fragment (e.g. a long unpunctuated clause): fall back to
        # the DP splitter so we still respect the hard ceiling.
        chunks.extend(balanced_dp(fragment, target_chars=target_chars, hard_max_chars=hard_max_chars))
    return chunks


def _split_on_strongest_punctuation(text: str, hard_max_chars: int) -> list[str]:
    candidates: list[list[str]] = []
    for _name, pattern in _PUNCT_PRIORITY:
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        if len(parts) <= 1:
            continue
        candidates.append(parts)
        # If this split already keeps every fragment under the ceiling, stop
        # at the strongest mark that fits — no need to over-fragment.
        if all(len(part) <= hard_max_chars for part in parts):
            return parts
    # No punctuation produced an under-ceiling split; return the finest cut
    # we did find (likely commas), or the whole text if nothing matched.
    return candidates[-1] if candidates else [text]


def _merge_short_fragments(
    fragments: list[str], min_chars: int, hard_max_chars: int
) -> list[str]:
    """Greedy forward-merge: combine short fragments into the next one when
    doing so keeps the result under the ceiling."""
    merged: list[str] = []
    buffer = ""
    for fragment in fragments:
        if not buffer:
            buffer = fragment
            continue
        combined = f"{buffer} {fragment}".strip()
        if len(buffer) < min_chars and len(combined) <= hard_max_chars:
            buffer = combined
        else:
            merged.append(buffer)
            buffer = fragment
    if buffer:
        # Trailing tiny tail: pull it back into the previous chunk if it fits.
        if merged and len(buffer) < min_chars:
            tail = f"{merged[-1]} {buffer}".strip()
            if len(tail) <= hard_max_chars:
                merged[-1] = tail
                return merged
        merged.append(buffer)
    return merged


# ---------------------------------------------------------------------------
# D. Variable-length target sampling
# ---------------------------------------------------------------------------


def variable_length_sampling(
    text: str,
    *,
    target_min: int = 35,
    target_max: int = 50,
    hard_max_chars: int = 52,
    seed: int = 0xC4_05_4A,
) -> list[str]:
    """Greedy split with a per-chunk target drawn from ``[target_min, target_max]``.

    The varying target avoids the failure mode where a fixed target lands
    repeatedly on the same awkward word boundary. The RNG is seeded for
    deterministic reproducibility across runs.

    Output chunks are slices of the original ``text`` so non-standard
    whitespace between words is preserved exactly.
    """
    spans = _word_spans(text)
    if not spans:
        return []

    rng = random.Random(seed)
    n = len(spans)

    def slice_len(i: int, j: int) -> int:
        return spans[j][1] - spans[i][0]

    chunk_ranges: list[tuple[int, int]] = []
    start_idx = 0
    j = 0
    target = rng.randint(target_min, target_max)

    while j < n:
        proposed = slice_len(start_idx, j)
        if proposed <= target:
            j += 1
            continue
        # Past the soft target: admit one more word only if it fits the hard
        # ceiling AND the existing chunk is still uncomfortably short.
        current_len = slice_len(start_idx, j - 1) if j > start_idx else 0
        if proposed <= hard_max_chars and current_len < target_min and j > start_idx:
            j += 1
            continue
        emit_end = max(start_idx, j - 1)
        chunk_ranges.append((start_idx, emit_end))
        start_idx = emit_end + 1
        j = start_idx
        target = rng.randint(target_min, target_max)

    if start_idx < n:
        chunk_ranges.append((start_idx, n - 1))

    # Tail merge: if the last chunk is too short and combining with the
    # previous one still fits under the ceiling, fuse them by re-slicing.
    if len(chunk_ranges) >= 2:
        last_start, last_end = chunk_ranges[-1]
        prev_start, _prev_end = chunk_ranges[-2]
        last_len = slice_len(last_start, last_end)
        combined_len = slice_len(prev_start, last_end)
        if last_len < target_min and combined_len <= hard_max_chars:
            chunk_ranges[-2] = (prev_start, last_end)
            chunk_ranges.pop()

    return [_slice_chunk(text, spans, i, j) for i, j in chunk_ranges]


# ---------------------------------------------------------------------------
# Registry — dispatched by name from the CLI / harness.
# ---------------------------------------------------------------------------

Strategy = Callable[[str], list[str]]

STRATEGIES: dict[str, Strategy] = {
    "current": current,
    "balanced_dp": balanced_dp,
    "punctuation_first": punctuation_first,
    "variable_length_sampling": variable_length_sampling,
}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"unknown chunking strategy {name!r}; known: {known}") from exc
