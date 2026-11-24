"""Recover complete handwritten lines from a scan for local style conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .paper import components, otsu_threshold, paper_luminance, remove_rules


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(active, prepend=False, append=False))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _ink_threshold(dark: np.ndarray) -> int:
    return max(18, round(otsu_threshold(dark) * 0.75))


def _page_angle(paper: Image.Image) -> float:
    if paper.height < 0.28 * paper.width:
        return 0.0
    small = paper.copy()
    small.thumbnail((900, 900), Image.Resampling.BILINEAR)
    dark = 255 - np.asarray(small)
    y, x = np.nonzero(dark > _ink_threshold(dark))
    if len(x) < 80:
        return 0.0
    step = max(1, len(x) // 60_000)
    x, y = x[::step] - small.width / 2, y[::step] - small.height / 2
    offset = math.ceil(math.hypot(*small.size))
    angles = np.arange(-6, 6.01, 0.25)
    scores = []
    for angle in np.deg2rad(angles):
        rows = np.rint(y * np.cos(angle) - x * np.sin(angle)).astype(int) + offset
        projection = np.bincount(rows, minlength=2 * offset + 1).astype(float)
        smooth = np.convolve(projection, [0.25, 0.5, 0.25], mode="same")
        scores.append(float(smooth @ smooth))
    best = int(np.argmax(scores))
    if scores[best] < scores[len(angles) // 2] * 1.08:
        return 0.0
    return float(angles[best])


def _prepare_page(source: Path, crop: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original)
        source_size = image.size
        if crop is not None:
            left, top, right, bottom = crop
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError(f"Crop {crop} must be inside the oriented source image ({image.width} x {image.height}).")
            image = image.crop(crop)
        image.thumbnail((3200, 3200), Image.Resampling.LANCZOS)
        paper = paper_luminance(image)
    angle = _page_angle(paper)
    deskewed = paper.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=255) if angle else paper
    dark = remove_rules(255 - np.asarray(deskewed))
    threshold = _ink_threshold(dark)
    mask = dark >= threshold
    # An isolated pixel cannot define a reference line or span its word spaces.
    p = np.pad(mask, 1)
    h, w = mask.shape
    neighbours = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            neighbours += p[dy:dy + h, dx:dx + w]
    mask &= neighbours >= 3
    if np.count_nonzero(mask) < 16:
        raise ValueError("No handwriting found. Use a line or a single-column page of dark writing on light paper.")
    if mask.mean() > 0.72:
        raise ValueError("Too little visible paper around the writing. Use dark handwriting on a light background.")
    floor = max(5, round(threshold * 0.18))
    high = max(floor + 80, float(np.percentile(dark[mask], 95)))
    ink = np.clip((dark.astype(np.float32) - floor) * (255 / (high - floor)), 0, 255).round().astype(np.uint8)
    # Keep antialiasing and pencil edges attached to real strokes.
    p = np.pad(mask, 1)
    support = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            support |= p[dy:dy + h, dx:dx + w]
    ink[~support] = 0
    return ink, mask, {
        "source_size": list(source_size),
        "crop": list(crop) if crop else None,
        "analysis_size": list(paper.size),
        "deskewed_size": list(deskewed.size),
        "deskew_degrees": angle,
        "bounds_coordinates": "exclusive pixel edges in the EXIF-oriented source",
    }


def source_box(box: tuple[int, int, int, int], geometry: dict[str, Any]) -> list[int]:
    w, h = geometry["analysis_size"]
    sw, sh = geometry["source_size"]
    sx, sy, right, bottom = geometry["crop"] or [0, 0, sw, sh]
    cw, ch = right - sx, bottom - sy
    dw, dh = geometry["deskewed_size"]
    angle = math.radians(geometry["deskew_degrees"])
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=float)
    corners -= [dw / 2, dh / 2]
    c, s = math.cos(angle), math.sin(angle)
    original = (corners @ [[c, s], [-s, c]] + [w / 2, h / 2]) * [cw / w, ch / h] + [sx, sy]
    low = np.maximum([sx, sy], np.floor(original.min(axis=0))).astype(int)
    high = np.minimum([right, bottom], np.ceil(original.max(axis=0))).astype(int)
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


@dataclass
class HandwritingLine:
    box: tuple[int, int, int, int]
    ink: np.ndarray
    mask: np.ndarray


def _line_samples(ink: np.ndarray, mask: np.ndarray) -> list[HandwritingLine]:
    # Voting with stroke centers keeps ascenders/descenders from merging
    # neighboring lines. Assign each complete connected stroke to a peak.
    labels = np.zeros(mask.shape, dtype=np.int32)
    boxes = components(mask, labels=labels)
    if not boxes:
        return []
    heights = np.array([c["y1"] - c["y0"] for c in boxes])
    order = np.argsort(heights)
    areas = np.array([c["area"] for c in boxes])
    cumulative = np.cumsum(areas[order])
    size = max(3, int(heights[order[np.searchsorted(cumulative, cumulative[-1] / 2)]]))
    mass_y = np.bincount(labels[mask], weights=np.nonzero(mask)[0])
    centers = np.array([mass_y[c["component_id"]] / c["area"] for c in boxes])
    usable = np.array([h <= 4 * size and not (c["x1"] - c["x0"] > 16 * size and h < 0.22 * size)
                       for h, c in zip(heights, boxes)])
    anchors = usable & (heights >= 0.42 * size) & (areas >= max(5, size))
    if not anchors.any():
        return []
    votes = np.bincount(centers[anchors].round().astype(int), weights=areas[anchors].clip(max=3 * size ** 2),
                        minlength=mask.shape[0])
    sigma = max(1.2, 0.30 * size)
    offsets = np.arange(-math.ceil(3 * sigma), math.ceil(3 * sigma) + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    # Full then sliced also handles a tiny image shorter than the kernel.
    density = np.convolve(votes, kernel / kernel.sum(), "full")[len(offsets) // 2:][:len(votes)]
    maxima = np.flatnonzero((density >= np.r_[0, density[:-1]]) & (density > np.r_[density[1:], 0])
                            & (density >= max(0.8, 0.10 * size)))
    def same_line(peak: int, candidate: int) -> bool:
        a, b = sorted((peak, candidate))
        if b - a < size:
            return True
        return b - a < 2.1 * size and density[a:b + 1].min() > 0.43 * min(density[a], density[b])

    peaks: list[int] = []
    for candidate in maxima[np.argsort(density[maxima])[::-1]]:
        if not any(same_line(peak, int(candidate)) for peak in peaks):
            peaks.append(int(candidate))
    if not peaks:
        return []
    peaks.sort()
    peak_rows = np.array(peaks)
    # Retain small dots and punctuation in the neighborhood of actual strokes.
    context = Image.new("1", (mask.shape[1], mask.shape[0]))
    draw = ImageDraw.Draw(context)
    for c, anchor in zip(boxes, anchors):
        if anchor:
            draw.rectangle((c["x0"] - 2 * size, c["y0"] - size, c["x1"] + 2 * size, c["y1"] + size), fill=1)
    near_writing = np.asarray(context)
    assignments = np.full(len(mass_y), -1, dtype=np.int32)
    grouped: list[list[dict[str, int]]] = [[] for _ in peaks]
    for c, center, keep in zip(boxes, centers, usable):
        if not keep or not near_writing[c["y0"]:c["y1"], c["x0"]:c["x1"]].any():
            continue
        group = int(np.argmin(abs(peak_rows - center)))
        if abs(peaks[group] - center) <= 1.7 * size:
            assignments[c["component_id"]] = group
            grouped[group].append(c)
    samples = []
    for i, group in enumerate(grouped):
        if not group or sum(c["area"] for c in group) < 16:
            continue
        x = max(0, min(c["x0"] for c in group) - 1)
        y = max(0, min(c["y0"] for c in group) - 1)
        right = min(mask.shape[1], max(c["x1"] for c in group) + 1)
        bottom = min(mask.shape[0], max(c["y1"] for c in group) + 1)
        region = np.s_[y:bottom, x:right]
        line_mask = assignments[labels[region]] == i
        support = np.asarray(Image.fromarray(line_mask.astype(np.uint8)).filter(ImageFilter.MaxFilter(3))).astype(bool)
        samples.append(HandwritingLine((x, y, right, bottom), np.where(support, ink[region], 0), line_mask))
    return samples


def handwriting_lines(
    source: Path, crop: tuple[int, int, int, int] | None = None,
) -> tuple[list[HandwritingLine], dict[str, Any]]:
    ink, mask, geometry = _prepare_page(source, crop)
    lines = _line_samples(ink, mask)
    if not lines:
        raise ValueError("No usable handwriting lines found in the style reference.")
    return lines, geometry


def body_metrics(mask: np.ndarray) -> tuple[int, int]:
    counts = mask.sum(axis=1)
    rows = np.flatnonzero(counts >= max(1, counts.max() * 0.35))
    top, baseline = int(rows[0]), int(rows[-1] + 1)
    return baseline, max(1, baseline - top)


def ink_gaps(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(a, b) for a, b in _runs(~mask.any(axis=0)) if a > 0 and b < mask.shape[1]]


def model_windows(mask: np.ndarray, body_height: int) -> list[tuple[int, int]]:
    height, width = mask.shape
    max_width = (1024 - 8) * height / 56
    gaps = [(a, b) for a, b in ink_gaps(mask) if b - a >= max(2, body_height * 0.42)]
    spans = []
    left = 0
    while width - left > max_width:
        choices = [(a, b) for a, b in gaps if left < a <= left + max_width]
        if not choices:
            # A single, unusually long word still fits by uniform downscaling.
            choices = [(a, b) for a, b in gaps if left < a]
            if not choices:
                break
        target = left + (width - left) / math.ceil((width - left) / max_width)
        def cut_score(gap: tuple[int, int], target: float = target) -> float:
            space = min((gap[1] - gap[0]) / body_height, 2)
            return space - 2 * abs(gap[0] - target) / max_width
        a, b = max(choices, key=cut_score)
        spans.append((left, min(a + 1, b)))
        left = max(a, b - 1)
    spans.append((left, width))
    return spans


def condition_image(ink: np.ndarray) -> Image.Image:
    h, w = ink.shape
    scale = min(56 / h, 1016 / w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    sample = Image.fromarray(255 - ink).resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (max(512, math.ceil((sample.width + 8) / 16) * 16), 64), 255)
    canvas.paste(sample, (4, (64 - sample.height) // 2))
    return canvas


def primary_condition_image(source: Path) -> Image.Image:
    lines, _ = handwriting_lines(source)
    line = max(lines, key=lambda item: item.ink.shape[1])
    _, body_height = body_metrics(line.mask)
    left, end = max(model_windows(line.mask, body_height), key=lambda span: span[1] - span[0])
    return condition_image(line.ink[:, left:end])


def style_word_crops(refs: list[Path]) -> list[Image.Image]:
    """Extract unlabeled word images for a word-conditioned synthesis model."""
    crops = []
    for path in refs:
        lines, _ = handwriting_lines(path)
        for line in lines:
            line_mask = line.mask
            _, body_height = body_metrics(line_mask)
            cuts = [(a + b) // 2 for a, b in ink_gaps(line_mask) if b - a >= max(3, 0.8 * body_height)]
            edges = [0, *cuts, line.ink.shape[1]]
            for a, b in pairwise(edges):
                part = Image.fromarray(line.ink[:, a:b])
                bbox = part.getbbox()
                if bbox and bbox[3] - bbox[1] >= body_height * 0.5:
                    word = ImageOps.invert(part.crop(bbox))
                    crops.append(ImageOps.expand(word, border=1, fill=255))
    return crops
