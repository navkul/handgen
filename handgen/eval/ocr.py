from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def text_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def verify_ocr_text(expected: str, actual: str) -> dict[str, Any]:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    expected_tokens = expected_norm.split() if expected_norm else []
    actual_tokens = actual_norm.split() if actual_norm else []
    repeated_tokens = _repeated_tokens(actual_tokens)
    hallucinated_tokens = [token for token in actual_tokens if token not in expected_tokens]
    missing_tokens = [token for token in expected_tokens if token not in actual_tokens]
    prefix_matches = len(actual_tokens) > len(expected_tokens) and actual_tokens[: len(expected_tokens)] == expected_tokens
    trimmable_suffix_tokens = actual_tokens[len(expected_tokens) :] if prefix_matches else []
    passed = actual_norm == expected_norm
    score = _match_score(expected_tokens, actual_tokens)
    return {
        "schema_version": "handgen_ocr_text_verification_v1",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "expected_normalized": expected_norm,
        "actual_normalized": actual_norm,
        "expected_tokens": expected_tokens,
        "actual_tokens": actual_tokens,
        "repeated_tokens": repeated_tokens,
        "hallucinated_tokens": hallucinated_tokens,
        "missing_tokens": missing_tokens,
        "prefix_matches_expected": prefix_matches,
        "trimmable_suffix_tokens": trimmable_suffix_tokens,
        "match_score": score,
        "reject_reasons": [] if passed else _reject_reasons(expected_norm, actual_norm, repeated_tokens, hallucinated_tokens, missing_tokens),
    }


def ocr_image(path: Path, *, psm: int = 8, allow_missing: bool = False) -> dict[str, Any]:
    binary = os.environ.get("HANDGEN_TESSERACT", "tesseract")
    resolved = shutil.which(binary)
    if resolved is None:
        if allow_missing:
            return {"engine": "tesseract", "available": False, "text": "", "stderr": f"{binary!r} not found"}
        raise RuntimeError(f"OCR verifier requires tesseract; set HANDGEN_TESSERACT or install tesseract")
    proc = subprocess.run(
        [resolved, str(path), "stdout", "--psm", str(psm)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "engine": "tesseract",
        "available": True,
        "binary": resolved,
        "psm": psm,
        "text": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def ocr_word_boxes(path: Path, *, psm: int = 7) -> dict[str, Any]:
    binary = os.environ.get("HANDGEN_TESSERACT", "tesseract")
    resolved = shutil.which(binary)
    if resolved is None:
        raise RuntimeError(f"OCR verifier requires tesseract; set HANDGEN_TESSERACT or install tesseract")
    proc = subprocess.run(
        [resolved, str(path), "stdout", "--psm", str(psm), "tsv"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    words: list[dict[str, Any]] = []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"engine": "tesseract", "available": True, "binary": resolved, "psm": psm, "words": words, "stderr": proc.stderr.strip()}
    header = lines[0].split("\t")
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) != len(header):
            continue
        row = dict(zip(header, cols))
        text = row.get("text", "").strip()
        norm = normalize_text(text)
        if not norm:
            continue
        try:
            conf = float(row.get("conf", "-1"))
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
        except (KeyError, ValueError):
            continue
        if conf < 0:
            continue
        words.append(
            {
                "text": text,
                "normalized": norm,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "right": left + width,
                "bottom": top + height,
                "confidence": conf,
            }
        )
    return {
        "engine": "tesseract",
        "available": True,
        "binary": resolved,
        "psm": psm,
        "words": words,
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def ocr_legibility_score(path: Path, *, psm: int = 7, allow_missing: bool = True) -> dict[str, Any]:
    try:
        boxes = ocr_word_boxes(path, psm=psm)
    except RuntimeError as exc:
        if allow_missing:
            return {"available": False, "score": 0.0, "error": str(exc)}
        raise
    words = boxes.get("words", [])
    confidences = [max(0.0, min(100.0, float(word.get("confidence", 0.0)))) for word in words]
    if not confidences:
        return {"available": True, "score": 0.0, "word_count": 0, "mean_confidence": 0.0, "psm": psm}
    mean_confidence = sum(confidences) / len(confidences)
    return {
        "available": True,
        "score": round(mean_confidence / 100.0, 4),
        "word_count": len(confidences),
        "mean_confidence": round(mean_confidence, 3),
        "psm": psm,
    }


def _repeated_tokens(tokens: list[str]) -> list[str]:
    repeated: list[str] = []
    for idx, token in enumerate(tokens[1:], start=1):
        if token == tokens[idx - 1] and token not in repeated:
            repeated.append(token)
    return repeated


def _match_score(expected_tokens: list[str], actual_tokens: list[str]) -> float:
    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0
    common = _lcs_len(expected_tokens, actual_tokens)
    recall = common / len(expected_tokens)
    precision = common / len(actual_tokens)
    prefix = 1.0 if actual_tokens[: len(expected_tokens)] == expected_tokens else 0.0
    return round(max(0.0, min(1.0, 0.62 * recall + 0.30 * precision + 0.08 * prefix)), 4)


def _lcs_len(left: list[str], right: list[str]) -> int:
    prev = [0] * (len(right) + 1)
    for token in left:
        curr = [0]
        for idx, other in enumerate(right, start=1):
            curr.append(prev[idx - 1] + 1 if token == other else max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def _reject_reasons(
    expected_norm: str,
    actual_norm: str,
    repeated_tokens: list[str],
    hallucinated_tokens: list[str],
    missing_tokens: list[str],
) -> list[str]:
    reasons: list[str] = []
    if not actual_norm:
        reasons.append("ocr_empty")
    if repeated_tokens:
        reasons.append("repeated_tokens")
    if hallucinated_tokens:
        reasons.append("hallucinated_tokens")
    if missing_tokens:
        reasons.append("missing_expected_tokens")
    if actual_norm and actual_norm != expected_norm and not reasons:
        reasons.append("text_mismatch")
    return reasons
