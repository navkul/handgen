"""Estimate prose word and space widths in units of the writer's body height."""

from __future__ import annotations

from typing import Any

import numpy as np

_CLASSES = ("narrow", "wide", "capital", "regular", "punctuation")
_PRIOR = np.array([0.62, 1.4, 1.18, 1.0, 0.6])


def _letter_counts(word: str) -> np.ndarray:
    counts = np.zeros(len(_CLASSES))
    for c in word:
        if c in "ilItfj":
            group = 0
        elif c in "mwMW":
            group = 1
        elif c.isupper():
            group = 2
        elif c.isalnum():
            group = 3
        else:
            group = 4
        counts[group] += 1
    return counts


def _prose_word(word: str) -> bool:
    return word.isascii() and not any(c.isspace() for c in word) and any(c.isalpha() for c in word)


def _fit(fractions: np.ndarray, advance: np.ndarray) -> tuple[float, np.ndarray]:
    # Each observation is the ink width per character, pooled over repeated
    # copies of its word. Ridge regularization stabilizes rare letter classes.
    typical = float(np.median(advance))
    identity = np.eye(len(_CLASSES))
    weights = np.linalg.solve(fractions.T @ fractions + identity, fractions.T @ advance + typical * _PRIOR)
    return typical, weights.clip(0.3 * typical, 2 * typical)


def writing_metrics(exemplars: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    gaps = []
    tokens = []
    observed = []
    for word, records in sorted(exemplars.items()):
        widths = []
        for sample in records:
            body = float(sample["body_height"])
            if body <= 0:
                continue
            widths.append(float(sample["width"]) / body)
            gap = sample.get("gap_before")
            if gap is not None and gap > 0:
                gaps.append(gap / body)
        if widths and _prose_word(word):
            tokens.append(word)
            observed.append(float(np.median(widths)))
    result: dict[str, Any] = {
        "unit": "line_body_height", "space_count": len(gaps),
        "median_space": round(float(np.median(gaps)), 4) if gaps else None,
        "word_count": len(tokens), "width_calibrated": False,
    }
    families = sorted({w.casefold() for w in tokens})
    if len(families) < 6:
        return result
    # Hold out every occurrence of a word, including capitalization variants.
    # For a larger vocabulary eight disjoint folds keep this linear in samples.
    folds = len(families) if len(families) <= 64 else 8
    assignment = {word: i % folds for i, word in enumerate(families)}
    fold_ids = np.array([assignment[w.casefold()] for w in tokens])
    counts = np.stack([_letter_counts(w) for w in tokens])
    sizes = counts.sum(axis=1)
    fractions = counts / sizes[:, None]
    measured = np.asarray(observed)
    advance = measured / sizes
    fixed_errors = []
    fitted_errors = []
    for i in range(folds):
        test = fold_ids == i
        typical, weights = _fit(fractions[~test], advance[~test])
        fixed_errors.extend(np.abs(sizes[test] * typical / measured[test] - 1))
        fitted_errors.extend(np.abs(counts[test] @ weights / measured[test] - 1))
    fixed = float(np.mean(fixed_errors))
    fitted = float(np.mean(fitted_errors))
    # A percentage point and a 5% relative gain avoid enabling a correction
    # on a tie or a negligible apparent advantage on these few samples.
    calibrated = fitted + max(0.01, 0.05 * fixed) < fixed
    result.update({
        "width_calibrated": calibrated, "heldout_word_families": len(families), "heldout_folds": folds,
        "heldout_fixed_mape": round(fixed, 5), "heldout_fitted_mape": round(fitted, 5),
    })
    if calibrated:
        _, weights = _fit(fractions, advance)
        result["advance_by_class"] = dict(zip(_CLASSES, weights.round(5).tolist()))
    return result


def estimated_word_width(word: str, metrics: dict[str, Any], *, body_height: float) -> float | None:
    if not metrics["width_calibrated"] or not _prose_word(word):
        return None
    weights = [metrics["advance_by_class"][name] for name in _CLASSES]
    return float(_letter_counts(word) @ weights) * body_height


def estimated_space_width(metrics: dict[str, Any], *, body_height: float, fallback: float) -> float:
    count = metrics["space_count"]
    if not count:
        return fallback
    # Four prior observations at the existing page spacing damp small samples.
    measured = metrics["median_space"] * body_height
    return (measured * count + fallback * 4) / (count + 4)
