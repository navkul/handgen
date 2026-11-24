"""Recover only independently readable parts of an uncertified handwriting line."""

from __future__ import annotations

import itertools
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from ..eval.ocr import ocr_image, ocr_word_boxes, verify_ocr_text
from ..utils import sha_file, sha_text


def _word_cuts(ink: np.ndarray, boxes: list[dict]) -> list[int | None]:
    """Give each OCR boundary a cut only if enough consecutive columns are clear."""
    height, width = ink.shape
    active = ink.max(axis=0) > 5
    edges = np.flatnonzero(np.diff(~active, prepend=False, append=False)).reshape(-1, 2)
    gaps = [(int(a), int(b)) for a, b in edges if b - a >= max(2, round(height * 0.065))]
    cuts: list[int | None] = [0]
    for prev, nxt in itertools.pairwise(boxes):
        # The distance between reported boxes can be smaller than the true
        # gap. Locate the whole ink-free strip that intersects that distance.
        left, right = prev["right"], nxt["left"]
        middle = (left + right) / 2
        allowed = [(a, b) for a, b in gaps if left <= right and a <= right and b >= left]
        if allowed:
            a, b = min(allowed, key=lambda ab: abs((ab[0] + ab[1]) / 2 - middle))
            cuts.append(int((a + b) // 2))
        else:
            cuts.append(None)
    cuts.append(width)

    # An OCR engine can omit leading/trailing ink entirely. Keep such ink
    # away from the proposed run if a true gap isolates the first/last box.
    occupied = np.flatnonzero(active)
    if len(occupied):
        margin = max(2, height * 0.22)
        left, right = boxes[0]["left"], boxes[-1]["right"]
        if left - occupied[0] > margin:
            possible = [b for a, b in gaps if occupied[0] < a < left and b <= left + margin]
            cuts[0] = max(possible, default=None)
        if occupied[-1] - right > margin:
            possible = [a for a, b in gaps if right - margin <= a and right < b < occupied[-1]]
            cuts[-1] = min(possible, default=None)
    return cuts


def recover_prose_runs(text: str, run: dict[str, Any], out_dir: Path) -> dict[int, dict]:
    """Map source word indices to nonoverlapping, OCR-confirmed runs of original ink.

    The caller keeps the exact whitespace around each run. Within a recovered
    run, text and character offsets are slices of the supplied source.
    """
    expected = list(re.finditer(r"\S+", text))
    asset = run.get("styled_crop", {})
    if run.get("exactness_certified") or not expected or not asset.get("path"):
        return {}
    source = Path(asset["path"])
    if not source.is_file():
        return {}
    digest = sha_file(source)
    if asset.get("sha256") and asset["sha256"] != digest:
        return {}
    try:
        observations = run.get("ocr_word_boxes") or ocr_word_boxes(source)
    except (RuntimeError, OSError):
        return {}
    boxes = observations.get("words", [])
    if not boxes:
        return {}
    with Image.open(source) as original:
        image = original.convert("RGBA")
    paper = Image.new("RGBA", image.size, "white")
    paper.alpha_composite(image)
    ink = 255 - np.asarray(paper.convert("L"))
    if not all(0 <= b.get("left", -1) < b.get("right", -1) <= image.width for b in boxes):
        return {}
    if any(a["left"] > b["left"] for a, b in itertools.pairwise(boxes)):
        return {}

    # Match whitespace-delimited words literally, including case and punctuation.
    # A second recognition reads the isolated raster without access to the source.
    matcher = SequenceMatcher(None, [m.group() for m in expected], [b["text"] for b in boxes], autojunk=False)
    cuts = _word_cuts(ink, boxes)
    recovered = {}
    directory = out_dir / "recovered_prose"
    directory.mkdir(parents=True, exist_ok=True)
    key = sha_text(digest + text)[:18]
    with TemporaryDirectory(dir=directory) as scratch:

        def recover(start: int, stop: int, first_box: int) -> None:
            last_box = first_box + stop - start
            # Exclude an edge word that touches rejected ink. Its remaining
            # neighbors can still have clear cuts on both sides.
            while start < stop and cuts[first_box] is None:
                start += 1
                first_box += 1
            while start < stop and cuts[last_box] is None:
                stop -= 1
                last_box -= 1
            if start >= stop:
                return
            left, right = cuts[first_box], cuts[last_box]
            if right <= left:
                return
            ys, xs = np.where(ink[:, left:right] > 5)
            if not len(xs):
                return
            box = (
                max(left, left + int(xs.min()) - 2),
                max(0, int(ys.min()) - 2),
                min(right, left + int(xs.max()) + 3),
                min(image.height, int(ys.max()) + 3),
            )
            source_start, source_end = expected[start].start(), expected[stop - 1].end()
            phrase = text[source_start:source_end]
            name = f"{key}_words{start:02d}-{stop:02d}.png"
            candidate = Path(scratch) / name
            image.crop(box).save(candidate)
            try:
                reading = ocr_image(candidate, psm=7)
            except (RuntimeError, OSError):
                return
            verification = verify_ocr_text(phrase, reading["text"])
            if verification["passed"] and phrase.split() == reading["text"].split():
                path = directory / name
                os.replace(candidate, path)
                recovered[start] = {
                    "end_word": stop,
                    "text": phrase,
                    "source_start": source_start,
                    "source_end": source_end,
                    "path": str(path),
                    "sha256": sha_file(path),
                    "crop_box": list(box),
                    "source_crop": str(source),
                    "source_sha256": digest,
                    "ocr_box_range": [first_box, last_box],
                    "ocr_verification": {
                        **verification,
                        "engine": reading.get("engine"),
                        "separate_crop_read": True,
                    },
                }
                return
            # A word-box match can disappear when the surrounding language
            # context is removed. Check shorter neighboring runs at real gaps.
            inner = [i for i in range(first_box + 1, last_box) if cuts[i] is not None]
            if inner:
                split = min(inner, key=lambda i: abs(i - (first_box + last_box) / 2))
                next_word = start + split - first_box
                recover(start, next_word, first_box)
                recover(next_word, stop, split)

        for block in matcher.get_matching_blocks():
            if block.size:
                recover(block.a, block.a + block.size, block.b)
    return recovered
