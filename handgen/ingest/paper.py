from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def paper_luminance(image: Image.Image) -> Image.Image:
    """Flatten paper illumination while preserving grayscale stroke edges."""
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("L", image.size, 255)
        white.paste(rgba.convert("L"), mask=rgba.getchannel("A"))
        gray = np.asarray(white)
    else:
        gray = np.asarray(image.convert("L"))

    # The brightest part of each tile measures the local paper. Interpolate
    # between tile centers to avoid putting block boundaries into the ink.
    tile = max(16, min(96, max(image.size) // 24))
    height, width = gray.shape
    rows = (height + tile - 1) // tile
    cols = (width + tile - 1) // tile
    padded = np.pad(gray, ((0, rows * tile - height), (0, cols * tile - width)), mode="edge")
    tiles = padded.reshape(rows, tile, cols, tile)
    bright = np.quantile(tiles, 0.88, axis=(1, 3), method="nearest").astype(np.uint8)
    paper = Image.fromarray(bright).resize((cols * tile, rows * tile), Image.Resampling.BILINEAR)
    background = np.asarray(paper, dtype=np.float32)[:height, :width]
    corrected = np.minimum(gray * np.float32(255) / np.maximum(background, 1), 255)
    return Image.fromarray(corrected.round().astype(np.uint8))


def otsu_threshold(intensity: np.ndarray) -> int:
    """Find the split maximizing between-class intensity variance."""
    histogram = np.bincount(intensity.ravel(), minlength=256)
    count = np.cumsum(histogram, dtype=float)
    weight = np.cumsum(histogram * np.arange(256), dtype=float)
    denominator = count * (count[-1] - count)
    variance = np.divide((weight[-1] * count - weight * count[-1]) ** 2, denominator,
                         out=np.zeros_like(count), where=denominator > 0)
    return int(np.argmax(variance))


def remove_rules(dark: np.ndarray) -> np.ndarray:
    """Subtract long paper rules and carry ink across saturated crossings."""
    # Across a ruled row most pixels measure the rule's own darkness. Across
    # handwriting most pixels still measure paper. Work on the long axis first.
    axes = [dark, dark.T] if dark.shape[0] >= 0.28 * dark.shape[1] else [dark]
    for rows in axes:
        levels = np.quantile(rows, 0.20, axis=1, method="nearest")
        edges = np.flatnonzero(np.diff(levels >= 8, prepend=False, append=False))
        for top, bottom in edges.reshape(-1, 2):
            if bottom - top > max(4, 0.015 * rows.shape[0]):
                continue
            values = levels[top:bottom, None].astype(np.float32)
            corrected = np.maximum(rows[top:bottom] - values, 0) * (255 / np.maximum(255 - values, 1))
            if top > 0 and bottom < len(rows):
                # Where dark ink crosses a dark line the two saturate together.
                bridge = np.minimum(rows[top - 1], rows[bottom])
                corrected = np.maximum(corrected, np.minimum(rows[top:bottom], bridge))
            rows[top:bottom] = corrected.round().astype(np.uint8)
    return dark


def components(mask: np.ndarray, *, labels: np.ndarray | None = None) -> list[dict[str, int]]:
    """Find 8-connected ink runs. Run overlap keeps large scans inexpensive."""
    parents: list[int] = []
    runs: list[tuple[int, int, int, int]] = []

    def root(i: int) -> int:
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    previous: list[tuple[int, int, int]] = []
    for y, scanline in enumerate(mask):
        edges = np.flatnonzero(np.diff(scanline, prepend=False, append=False)).tolist()
        current = []
        j = 0
        for x0, x1 in zip(edges[::2], edges[1::2]):
            while j < len(previous) and previous[j][1] < x0:
                j += 1
            k = j
            if k < len(previous) and previous[k][0] <= x1:
                label = root(previous[k][2])
                k += 1
                while k < len(previous) and previous[k][0] <= x1:
                    parents[root(previous[k][2])] = label
                    k += 1
            else:
                label = len(parents)
                parents.append(label)
            current.append((x0, x1, label))
            runs.append((y, x0, x1, label))
        previous = current
    boxes: dict[int, dict[str, int]] = {}
    for y, x0, x1, label in runs:
        key = root(label)
        if labels is not None:
            labels[y, x0:x1] = key + 1
        box = boxes.setdefault(key, {"component_id": key + 1, "x0": x0, "y0": y, "x1": x1, "y1": y + 1, "area": 0})
        box["x0"] = min(x0, box["x0"])
        box["x1"] = max(x1, box["x1"])
        box["y1"] = y + 1
        box["area"] += x1 - x0
    return list(boxes.values())
