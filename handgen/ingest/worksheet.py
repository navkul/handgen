from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..utils import now_iso, read_json, sha_file, write_json
from .paper import components, otsu_threshold, paper_luminance, remove_rules

ROW_SPECS = [
    {"id": "digits", "kind": "glyphs", "labels": list("0123456789")},
    {
        "id": "punctuation_operators",
        "kind": "glyphs",
        "labels": [
            ".",
            ",",
            ";",
            ":",
            "!",
            "?",
            "'",
            '"',
            "-",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "<",
            ">",
            "+",
            "=",
            "*",
            "/",
            "_",
            "|",
        ],
    },
    {
        "id": "greek_symbols",
        "kind": "glyphs",
        "labels": ["alpha", "beta", "gamma", "theta", "lambda", "mu", "pi", "sigma"],
    },
    {"id": "lowercase_letters", "kind": "glyphs", "labels": list("abcdefghijklmnopqrstuvwxyz")},
    {"id": "uppercase_letters", "kind": "glyphs", "labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
]


def slug(label: str) -> str:
    names = {
        ".": "period",
        ",": "comma",
        ";": "semicolon",
        ":": "colon",
        "!": "bang",
        "?": "question",
        "'": "apostrophe",
        '"': "quote",
        "-": "hyphen",
        "(": "lparen",
        ")": "rparen",
        "[": "lbracket",
        "]": "rbracket",
        "{": "lbrace",
        "}": "rbrace",
        "<": "lt",
        ">": "gt",
        "+": "plus",
        "=": "equals",
        "*": "asterisk",
        "/": "slash",
        "_": "underscore",
        "|": "bar",
    }
    return names.get(label, re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "glyph")


def _paper_image(source: Path, work: Path, dpi: int) -> tuple[Image.Image, tuple[int, int]]:
    page = source
    if source.suffix.lower() == ".pdf":
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-singlefile", str(source), str(work / "page")],
            check=True,
        )
        page = work / "page.png"
    with Image.open(page) as original:
        upright = ImageOps.exif_transpose(original)
        original_size = upright.size
        upright.thumbnail((3600, 3600), Image.Resampling.LANCZOS)
        return paper_luminance(upright), original_size


def dark_mask(image: Image.Image) -> np.ndarray:
    """Separate ink from paper using between-class variance in luminance."""
    arr = np.asarray(image.convert("L"))
    return arr <= min(242, otsu_threshold(arr))


def _glyph_height(comps: list[dict[str, int]]) -> float:
    if not comps:
        raise ValueError("No handwriting detected on the worksheet")
    areas = np.sort([c["area"] for c in comps])
    # Use an ink-area weighted median to avoid measuring the size of page speckles.
    typical_area = areas[np.searchsorted(np.cumsum(areas), areas.sum() * 0.5)]
    return max(3.0, float(np.median([c["y1"] - c["y0"] for c in comps if c["area"] >= 0.22 * typical_area])))


def _rotation(comps: list[dict[str, int]], height: float) -> float:
    anchors = [c for c in comps if height * 0.5 <= c["y1"] - c["y0"] <= height * 2.1]
    if len(anchors) < 5:
        return 0.0
    x = np.array([(c["x0"] + c["x1"]) / 2 for c in anchors])
    y = np.array([(c["y0"] + c["y1"]) / 2 for c in anchors])
    if np.ptp(x) < 4 * height:
        return 0.0
    angles = np.arange(-9.0, 9.01, 0.25)
    scores = []
    # The correct angle clusters glyph centers into narrow horizontal bands.
    # Linear binning avoids a score jump when a glyph crosses a bin boundary.
    for angle in angles:
        projected = (y - x * np.tan(np.deg2rad(angle))) / max(1.0, height * 0.35)
        projected -= projected.min()
        bins = projected.astype(int)
        fractions = projected - bins
        hist = np.bincount(bins, 1 - fractions, minlength=bins.max() + 2)
        hist += np.bincount(bins + 1, fractions, minlength=hist.size)
        smooth = np.convolve(hist, [0.25, 0.5, 0.25])
        scores.append(float(smooth @ smooth))
    best = int(np.argmax(scores))
    if scores[best] < 1.045 * scores[len(angles) // 2]:
        return 0.0
    return float(angles[best])


def _rule_rotation(mask: np.ndarray) -> float:
    # Long ruled/grid lines can join many letters into a single component.
    # In that case measure the lines themselves, refining to less than one
    # pixel of tilt over the width of a typical scan.
    y, x = np.nonzero(mask)
    stride = max(1, len(x) // 60_000)
    y = y[::stride] - mask.shape[0] / 2
    x = x[::stride] - mask.shape[1] / 2

    def score(angle: float) -> float:
        slope = np.tan(np.deg2rad(angle))
        result = 0.0
        for projected in (y - slope * x, x + slope * y):
            bins = np.rint(projected).astype(int)
            histogram = np.bincount(bins - bins.min()).astype(float)
            result += float(histogram @ histogram)
        return result

    coarse = max(np.arange(-9.0, 9.01, 0.25), key=score)
    return round(float(max(np.arange(coarse - 0.15, coarse + 0.151, 0.025), key=score)), 3)


def _clear_margin_shadows(gray: Image.Image, mask: np.ndarray, comps: list[dict[str, int]]) -> Image.Image:
    # An illumination jump at the edge of a photographed sheet can dominate
    # both the threshold and glyph-height estimate. Identify only broad, dense
    # margin marks far larger than the other ink, preserving nearby letters.
    if len(comps) < 8:
        return gray
    span = min(gray.size)
    min_area = max(0.002 * mask.size, 25 * np.percentile([c["area"] for c in comps], 75))
    oversized = [
        c
        for c in comps
        if c["area"] > min_area
        and min(c["x0"], c["y0"], gray.width - c["x1"], gray.height - c["y1"]) < 0.055 * span
    ]
    if not oversized:
        return gray
    labels = np.zeros(mask.shape, np.int32)
    components(mask, labels=labels)
    remove = np.zeros_like(mask)
    for c in oversized:
        region = np.s_[c["y0"] : c["y1"], c["x0"] : c["x1"]]
        component = labels[region] == c["component_id"]
        # Dense shadow patches fill paper-sized tiles. Narrow lines of a grid
        # still have abundant paper inside the same tiles.
        small = Image.fromarray(component).convert("L")
        tile = max(4.0, span * 0.013)
        small = small.resize(tuple(max(1, round(n / tile)) for n in small.size), Image.Resampling.BOX)
        if small.getextrema()[1] > 245:
            remove[region] |= component
    if not remove.any():
        return gray
    cleaned = gray.copy()
    cleaned.paste(255, mask=Image.fromarray(remove).convert("L").filter(ImageFilter.MaxFilter(3)))
    return cleaned


def _foreground(gray: Image.Image) -> tuple[Image.Image, list[dict[str, int]], float, float]:
    mask = dark_mask(gray)
    comps = components(mask)
    original = gray
    gray = _clear_margin_shadows(gray, mask, comps)
    if gray is not original:
        mask = dark_mask(gray)
        comps = components(mask)

    def spans_page(c: dict[str, int]) -> bool:
        w, h = c["x1"] - c["x0"], c["y1"] - c["y0"]
        if w > 0.55 * gray.width and h > 0.65 * gray.height:
            return c["area"] < 0.2 * w * h  # a connected grid
        return (w > 0.55 * gray.width and w > 4.5 * h) or (h > 0.65 * gray.height and h > 4.5 * w)

    has_rule = len(comps) > 2 and any(spans_page(c) for c in comps)
    angle = _rule_rotation(mask) if has_rule else _rotation(comps, _glyph_height(comps))
    if angle:
        gray = gray.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    if has_rule:
        dark = 255 - np.asarray(gray)
        # Exclude page margins introduced by the rotations when measuring how
        # much of each row is ruled. Retain coordinates in the deskewed page.
        bounds = Image.fromarray(dark_mask(gray)).getbbox()
        if bounds:
            x0, y0, x1, y1 = bounds
            region = dark[y0:y1, x0:x1]
            remove_rules(region)
            gray = Image.fromarray(255 - dark)
    if angle or has_rule:
        mask = dark_mask(gray)
        comps = components(mask)
    height = _glyph_height(comps)
    min_area = max(2, height * height * 0.006)
    comps = [c for c in comps if c["area"] >= min_area]
    if not comps:
        raise ValueError("No usable ink components detected on the worksheet")
    keep = np.zeros(mask.shape, dtype=bool)
    for c in comps:
        box = np.s_[c["y0"] : c["y1"], c["x0"] : c["x1"]]
        keep[box] |= mask[box]
    # Recover soft stroke edges around ink, while discarding the page's dot grid.
    support = Image.fromarray(keep).convert("L").filter(ImageFilter.MaxFilter(3))
    arr = np.asarray(gray)
    darkness = 255.0 - arr.astype("float32")
    ink_white = max(1.0, float(np.percentile(darkness[keep], 85)))
    cleaned = np.where(np.asarray(support) > 0, 255 - np.minimum(255, darkness * (255 / ink_white)), 255)
    return Image.fromarray(cleaned.astype("uint8")), comps, height, angle


def group_rows(comps: list[dict[str, int]], height: float | None = None) -> list[dict[str, Any]]:
    height = _glyph_height(comps) if height is None else height
    rows: list[dict[str, Any]] = []
    for c in sorted(comps, key=lambda c: (c["y0"] + c["y1"], c["x0"])):
        center = (c["y0"] + c["y1"]) / 2
        if not rows or center - rows[-1]["cy"] > height * 1.35:
            rows.append({"cy": center, "components": []})
        row = rows[-1]
        row["components"].append(c)
        row["cy"] += (center - row["cy"]) / len(row["components"])
    for row in rows:
        parts = row["components"]
        # Join dots, crossbars and double quotes at the writer's scale.
        boxes: list[list[int]] = []
        for c in sorted(parts, key=lambda c: c["x0"]):
            if boxes and c["x0"] - boxes[-1][2] <= 0.34 * height:
                prior = boxes[-1]
                prior[1] = min(prior[1], c["y0"])
                prior[2] = max(prior[2], c["x1"])
                prior[3] = max(prior[3], c["y1"])
            else:
                boxes.append([c["x0"], c["y0"], c["x1"], c["y1"]])
        row["boxes"] = boxes
    return rows


def _labels_for_groups(spec: dict[str, Any], group_count: int) -> tuple[list[str], list[str]]:
    labels = list(spec["labels"])
    if spec == ROW_SPECS[2] and group_count == 7:
        return [label for label in labels if label != "gamma"], ["gamma"]
    return labels, []


def _read_specs(layout: Path | None) -> list[dict[str, Any]]:
    if layout is None:
        return ROW_SPECS
    template = read_json(layout)
    specs = template.get("rows") if isinstance(template, dict) else None
    if not isinstance(specs, list) or not specs:
        raise ValueError("Worksheet layout must contain a nonempty rows array")
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("id"), str) or not spec["id"]:
            raise ValueError("Each worksheet layout row needs an id")
        labels = spec.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(s, str) or not s.strip() for s in labels)
        ):
            raise ValueError("Each worksheet layout row needs a nonempty list of labels in writing order")
    return specs


def _assign_rows(
    rows: list[dict[str, Any]], specs: list[dict[str, Any]], *, default_layout: bool
) -> list[tuple[int, int]]:
    counts = [len(row["boxes"]) for row in rows]

    @lru_cache(None)
    def match(section: int, start: int) -> tuple[tuple[int, int], ...] | None:
        if start == len(rows) and (default_layout or section == len(specs)):
            return ()
        if section == len(specs):
            return None
        expected = len(specs[section]["labels"])
        allowed = {expected}
        if default_layout and specs[section]["id"] == "greek_symbols":
            allowed.add(expected - 1)
        count = 0
        for stop in range(start + 1, len(rows) + 1):
            count += counts[stop - 1]
            if count > expected:
                break
            if count in allowed:
                tail = match(section + 1, stop)
                if tail is not None:
                    return ((start, stop),) + tail
        return None

    assignment = match(0, 0)
    if not assignment:
        expected = ", ".join(f"{s['id']} ({len(s['labels'])})" for s in specs)
        raise ValueError(
            f"Cannot align worksheet. Detected glyph counts by physical row {counts}. "
            f"Expected sections in order: {expected}. Separate touching marks or supply "
            "--layout with the labels actually written."
        )
    return list(assignment)


def _save_crop(
    image: Image.Image, box: tuple[int, ...] | list[int], path: Path, base: Path, pad: int
) -> dict[str, Any]:
    x0, y0, x1, y1 = box
    padded = (max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad))
    # Pad with paper so a neighboring symbol never leaks into the crop.
    crop = Image.new("L", (padded[2] - padded[0], padded[3] - padded[1]), 255)
    crop.paste(image.crop(tuple(box)), (x0 - padded[0], y0 - padded[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)
    return {
        "path": str(path.relative_to(base)),
        "bbox": list(padded),
        "ink_bbox": list(box),
        "width": crop.width,
        "height": crop.height,
        "sha256": sha_file(path),
    }


def ingest_worksheet(
    writer_id: str, source: Path, out_dir: Path, dpi: int = 220, *, layout: Path | None = None
) -> dict[str, Any]:
    if dpi <= 0:
        raise ValueError("Worksheet DPI must be positive")
    specs = _read_specs(layout)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha_file(source)
    recipe = json.dumps([2, source_sha, dpi, specs, layout is None], sort_keys=True).encode()
    asset_dir = Path("assets") / hashlib.sha256(recipe).hexdigest()[:16]
    snapshot_name = f"manifest-{asset_dir.name}.json"
    # Build a complete bank before publishing it. A failed ingest leaves the
    # preceding manifest and all the crops that it references intact.
    with tempfile.TemporaryDirectory(dir=out_dir) as work_dir:
        stage = Path(work_dir)
        gray, source_size = _paper_image(source, stage, dpi)
        image, comps, height, angle = _foreground(gray)
        rows = group_rows(comps, height)
        assignments = _assign_rows(rows, specs, default_layout=layout is None)
        assets = stage / asset_dir
        assets.mkdir(parents=True, exist_ok=True)
        page = assets / "worksheet_page.png"
        image.save(page)
        glyphs: dict[str, list[dict[str, Any]]] = {}
        row_records = []
        for index, (start, stop) in enumerate(assignments):
            spec = specs[index]
            section = rows[start:stop]
            boxes = [box for row in section for box in row["boxes"]]
            labels, missing = _labels_for_groups(spec, len(boxes)) if layout is None else (spec["labels"], [])
            section_box = (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
            key = f"{index:02d}_{slug(spec['id'])}"
            row_crop = _save_crop(image, section_box, assets / "rows" / f"{key}.png", stage, pad=8)
            record = {
                "index": index,
                "id": spec["id"],
                "kind": "glyphs",
                "crop": row_crop,
                "physical_row_count": len(section),
                "source_row_indices": list(range(start, stop)),
                "detected_x_group_count": len(boxes),
                "label_count": len(spec["labels"]),
                "detected_label_sequence": labels,
            }
            if missing:
                record["inferred_missing_labels"] = missing
            row_records.append(record)
            for j, (label, box) in enumerate(zip(labels, boxes, strict=True)):
                name = slug(label)
                path = assets / "glyph_bank" / name.lower() / f"{key}_{j:02d}_{name}.png"
                # Include the mask's antialiased fringe before adding blank padding.
                fringe = (
                    max(0, box[0] - 1),
                    max(0, box[1] - 1),
                    min(image.width, box[2] + 1),
                    min(image.height, box[3] + 1),
                )
                glyph = _save_crop(image, fringe, path, stage, pad=3)
                glyph.update(label=label, row_id=spec["id"], row_index=index, label_index=j)
                glyphs.setdefault(label, []).append(glyph)
            for label in missing:
                glyphs.setdefault(label, []).append(
                    {"label": label, "row_id": spec["id"], "missing_crop": True}
                )
        available = {label for label, crops in glyphs.items() if any(c.get("path") for c in crops)}
        required = {label for spec in specs for label in spec["labels"]}
        manifest = {
            "schema_version": "handgen_worksheet_manifest_v2",
            "writer_id": writer_id,
            "generated_at": now_iso(),
            "source_path": str(source.resolve()),
            "source_sha256": source_sha,
            "snapshot_name": snapshot_name,
            "local_source": str(source.resolve()),
            "page_png": str(page.relative_to(stage)),
            "dpi": dpi,
            "source_page_size": list(source_size),
            "page_size": list(image.size),
            "deskew_degrees": angle,
            "estimated_glyph_height": round(height, 2),
            "detected_component_count": len(comps),
            "detected_row_count": len(rows),
            "used_row_count": len(row_records),
            "rows": row_records,
            "glyphs": glyphs,
            "coverage": {
                "available_labels": sorted(available),
                "missing_labels": sorted(required - available),
            },
            "layout": str(layout) if layout else "default",
            "known_limits": ["Sections and separated handwriting must follow the specified label order."],
        }
        # Use atomic file replacement on retry as well, allowing repair of a
        # damaged crop without modifying assets of a different page or layout.
        for path in assets.rglob("*.png"):
            destination = out_dir / path.relative_to(stage)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination)
        write_json(stage / "manifest.json", manifest)
        # Readers registered against this recipe remain consistent if another
        # ingest publishes a different layout before the database transaction.
        os.link(stage / "manifest.json", stage / snapshot_name)
        os.replace(stage / snapshot_name, out_dir / snapshot_name)
        os.replace(stage / "manifest.json", out_dir / "manifest.json")
    return load_worksheet_manifest(out_dir / snapshot_name)


def load_worksheet_manifest(path: Path) -> dict[str, Any]:
    """Resolve portable v2 paths and pre-move absolute v1 paths at read time."""
    manifest = read_json(path)
    base = path.resolve().parent

    def local_glyph(candidate: Path) -> Path:
        # v1 on case-insensitive macOS could record "glyph_bank/A/..." while
        # Git actually stored both letter cases in "glyph_bank/a/...".
        parent = candidate.parent
        if not candidate.is_file() and parent.parent.name == "glyph_bank" and parent.parent.is_dir():
            for sibling in parent.parent.iterdir():
                if sibling.name.casefold() == parent.name.casefold() and (sibling / candidate.name).is_file():
                    return sibling / candidate.name
        return candidate

    def resolve(value: str, folder: str = "") -> str:
        asset = Path(value)
        if not asset.is_absolute():
            return str(local_glyph(base / asset))
        if folder and folder in asset.parts:
            # The manifest's local bank takes precedence over a stale checkout.
            old_directory = next(p for p in asset.parents if p.name == folder)
            local = local_glyph(base / folder / asset.relative_to(old_directory))
            if local.exists() or not asset.exists():
                return str(local)
        return str(asset)

    for crops in manifest.get("glyphs", {}).values():
        for crop in crops:
            if crop.get("path"):
                crop["path"] = resolve(crop["path"], "glyph_bank")
    for row in manifest.get("rows", []):
        crop = row.get("crop", {})
        if crop.get("path"):
            crop["path"] = resolve(crop["path"], "rows")
    if manifest.get("page_png"):
        page = Path(manifest["page_png"])
        manifest["page_png"] = (
            str(base / page.name)
            if page.is_absolute() and (base / page.name).exists()
            else resolve(str(page))
        )
    return manifest


def glyph_bank_from_manifest(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    bank = {}
    for label, crops in manifest.get("glyphs", {}).items():
        paths = []
        for crop in crops:
            if crop.get("missing_crop") or not crop.get("path"):
                continue
            path = Path(crop["path"])
            if path.is_file() and (not crop.get("sha256") or sha_file(path) == crop["sha256"]):
                paths.append(path)
        if paths:
            bank[label] = paths
    return bank
