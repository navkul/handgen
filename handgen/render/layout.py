"""Baseline-aligned flow of handwriting crops onto fixed-size SVG pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..utils import data_uri, escape_xml

PAGE_W = 1500
PAGE_H = 1900
LEFT = 72
RIGHT = 72
TOP = 72
BOTTOM = 72
PROSE_H = 40
LINE_GAP = 68


def ink_geometry(path: Path) -> tuple[float, float]:
    """Estimate the baseline and x-height of a transparent line crop, in pixels.

    A quantile over occupied columns gives more weight to letter bodies than
    to sparse ascenders, descenders and dots.
    """
    with Image.open(path) as im:
        alpha = np.asarray(im.convert("RGBA").getchannel("A"))
    mask = alpha > max(12, int(alpha.max()) * 0.24)
    columns = mask[:, mask.any(axis=0)]
    if not columns.size:
        raise ValueError(f"empty handwriting crop: {path}")
    bottom = columns.shape[0] - columns[::-1].argmax(axis=0)
    top = columns.argmax(axis=0)
    baseline = float(np.quantile(bottom, 0.67))
    body = max(1.0, baseline - float(np.quantile(top, 0.42)))
    return baseline, body


class PageLayout:
    """Buffer a row until its tallest ascender and descender are known."""

    def __init__(self) -> None:
        self.pages: list[list[str]] = [[]]
        self.top = float(TOP)
        self.x = float(LEFT)
        self.pending_gap = 0.0
        self.row: list[tuple[Path, float, float, float, float, dict]] = []

    def newline(self) -> None:
        ascent = max((rise for _, _, _, _, rise, _ in self.row), default=PROSE_H * .8)
        descent = max((h - rise for _, _, _, h, rise, _ in self.row), default=PROSE_H * .2)
        if self.top + ascent + descent > PAGE_H - BOTTOM:
            self.pages.append([])
            self.top = float(TOP)
        baseline = self.top + ascent
        for path, x, w, h, rise, record in self.row:
            y = baseline - rise
            self.pages[-1].append(
                f'<image id="{escape_xml(record["id"])}" href="{data_uri(path)}" '
                f'x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" preserveAspectRatio="none"/>'
            )
            record.update(page=len(self.pages), baseline=round(baseline, 3),
                          bbox={"x": round(x, 2), "y": round(y, 2), "width": round(w, 2), "height": round(h, 2)})
        self.top += max(LINE_GAP, ascent + descent + .30 * PROSE_H)
        self.row.clear()
        self.x = float(LEFT)
        self.pending_gap = 0.0

    def space(self, advance: float) -> None:
        self.pending_gap += advance

    def place(self, path: Path, width: float, height: float, asset_baseline: float, record: dict) -> None:
        # Word crops keep their intrinsic ascender/descender proportions. An
        # unusually tall stroke increases row spacing for all ink on this row.
        fit = min(1.0, (PAGE_W - LEFT - RIGHT) / width, (PAGE_H - TOP - BOTTOM) / height)
        w, h, rise = width * fit, height * fit, asset_baseline * fit
        if self.row and self.x + self.pending_gap + w > PAGE_W - RIGHT:
            self.newline()
        x = min(self.x + self.pending_gap, PAGE_W - RIGHT - w)
        self.row.append((path, x, w, h, rise, record))
        self.x = x + w
        self.pending_gap = 0.0

    def svg(self, page: int) -> str:
        return "\n".join([
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">',
            f'<title>Handgen document, page {page}</title>',
            '<rect width="100%" height="100%" fill="white"/>',
            *self.pages[page - 1], "</svg>", "",
        ])
