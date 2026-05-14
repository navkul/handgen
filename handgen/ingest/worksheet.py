from __future__ import annotations

import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

from ..utils import now_iso, png_size, require_pillow_numpy, sha_file, write_json


ROW_SPECS = [
    {"id": "digits", "kind": "glyphs", "labels": list("0123456789")},
    {
        "id": "punctuation_operators",
        "kind": "glyphs",
        "labels": [".", ",", ";", ":", "!", "?", "'", '"', "-", "(", ")", "[", "]", "{", "}", "<", ">", "+", "=", "*", "/", "_", "|"],
    },
    {"id": "greek_symbols", "kind": "glyphs", "labels": ["alpha", "beta", "gamma", "theta", "lambda", "mu", "pi", "sigma"]},
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


def rasterize_source(source: Path, out_png: Path, dpi: int) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".pdf":
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-singlefile", str(source), str(out_png.with_suffix(""))],
            check=True,
        )
    else:
        Image, _ = require_pillow_numpy()
        Image.open(source).convert("RGB").save(out_png)


def dark_mask(image: Any) -> Any:
    _, np = require_pillow_numpy()
    arr = np.asarray(image.convert("L"))
    return arr < 105


def components(mask: Any) -> list[dict[str, int]]:
    _, np = require_pillow_numpy()
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[dict[str, int]] = []
    for y in range(h):
        xs = np.where(mask[y] & ~seen[y])[0]
        for x0 in xs.tolist():
            if seen[y, x0] or not mask[y, x0]:
                continue
            q: deque[tuple[int, int]] = deque([(x0, y)])
            seen[y, x0] = True
            min_x = max_x = x0
            min_y = max_y = y
            area = 0
            while q:
                x, yy = q.popleft()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, yy)
                max_y = max(max_y, yy)
                for nx, ny in ((x + 1, yy), (x - 1, yy), (x, yy + 1), (x, yy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if area >= 18 and (max_x - min_x) >= 2 and (max_y - min_y) >= 2:
                out.append({"x0": min_x, "y0": min_y, "x1": max_x + 1, "y1": max_y + 1, "area": area})
    return out


def group_rows(comps: list[dict[str, int]]) -> list[dict[str, Any]]:
    comps = sorted(comps, key=lambda c: ((c["y0"] + c["y1"]) / 2, c["x0"]))
    rows: list[dict[str, Any]] = []
    for comp in comps:
        cy = (comp["y0"] + comp["y1"]) / 2
        for row in rows:
            if abs(cy - row["cy"]) < 58:
                row["components"].append(comp)
                n = len(row["components"])
                row["cy"] = ((row["cy"] * (n - 1)) + cy) / n
                break
        else:
            rows.append({"cy": cy, "components": [comp]})
    normalized = []
    for row in sorted(rows, key=lambda r: r["cy"]):
        comps_row = sorted(row["components"], key=lambda c: c["x0"])
        if sum(c["area"] for c in comps_row) < 60:
            continue
        normalized.append(
            {
                "cy": row["cy"],
                "x0": min(c["x0"] for c in comps_row),
                "y0": min(c["y0"] for c in comps_row),
                "x1": max(c["x1"] for c in comps_row),
                "y1": max(c["y1"] for c in comps_row),
                "components": comps_row,
            }
        )
    return normalized


def x_groups(mask: Any, row_box: tuple[int, int, int, int], gap: int = 28) -> list[tuple[int, int]]:
    _, np = require_pillow_numpy()
    x0, y0, x1, y1 = row_box
    sub = mask[y0:y1, x0:x1]
    active_cols = np.where(sub.sum(axis=0) > 0)[0]
    if len(active_cols) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = int(active_cols[0])
    prev = int(active_cols[0])
    for x in active_cols[1:]:
        x = int(x)
        if x - prev > gap:
            groups.append((x0 + start, x0 + prev + 1))
            start = x
        prev = x
    groups.append((x0 + start, x0 + prev + 1))
    return groups


def crop_save(image: Any, box: tuple[int, int, int, int], path: Path, pad: int = 14) -> dict[str, Any]:
    w, h = image.size
    x0, y0, x1, y1 = box
    padded = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(padded).convert("L")
    crop.save(path)
    return {"path": str(path), "bbox": list(padded), "width": crop.width, "height": crop.height, "sha256": sha_file(path)}


def _row_box(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (row["x0"], row["y0"], row["x1"], row["y1"])


def _consume_spec_rows(mask: Any, rows: list[dict[str, Any]], start_idx: int, spec_idx: int) -> tuple[list[dict[str, Any]], int]:
    spec = ROW_SPECS[spec_idx]
    labels = spec["labels"]
    remaining_specs = len(ROW_SPECS) - spec_idx - 1
    max_idx = len(rows) - remaining_specs
    consumed: list[dict[str, Any]] = []
    detected = 0
    row_idx = start_idx
    while row_idx < max_idx:
        row = rows[row_idx]
        row_box = _row_box(row)
        groups = x_groups(mask, row_box)
        consumed.append({"source_row_index": row_idx, "row": row, "row_box": row_box, "groups": groups})
        detected += len(groups)
        row_idx += 1
        if spec["id"] == "greek_symbols" and detected == len(labels) - 1:
            break
        if detected >= len(labels):
            break
    return consumed, row_idx


def _labels_for_groups(spec: dict[str, Any], group_count: int) -> tuple[list[str], list[str]]:
    labels = list(spec["labels"])
    if (
        spec["id"] == "greek_symbols"
        and group_count == 7
        and labels == ["alpha", "beta", "gamma", "theta", "lambda", "mu", "pi", "sigma"]
    ):
        return ["alpha", "beta", "theta", "lambda", "mu", "pi", "sigma"], ["gamma"]
    return labels, []


def ingest_worksheet(writer_id: str, source: Path, out_dir: Path, dpi: int = 220) -> dict[str, Any]:
    Image, _ = require_pillow_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    local_source = source
    if source.suffix.lower() == ".pdf":
        page_png = out_dir / "worksheet_page.png"
        rasterize_source(source, page_png, dpi)
    else:
        page_png = source
    image = Image.open(page_png).convert("RGB")
    mask = dark_mask(image)
    comps = components(mask)
    rows = group_rows(comps)
    glyphs: dict[str, list[dict[str, Any]]] = {}
    row_records: list[dict[str, Any]] = []
    row_idx = 0
    for idx, spec in enumerate(ROW_SPECS):
        consumed_rows, row_idx = _consume_spec_rows(mask, rows, row_idx, idx)
        if not consumed_rows:
            continue
        x0 = min(item["row_box"][0] for item in consumed_rows)
        y0 = min(item["row_box"][1] for item in consumed_rows)
        x1 = max(item["row_box"][2] for item in consumed_rows)
        y1 = max(item["row_box"][3] for item in consumed_rows)
        row_crop = crop_save(image, (x0, y0, x1, y1), out_dir / "rows" / f"{idx:02d}_{spec['id']}.png", pad=24)
        record: dict[str, Any] = {
            "index": idx,
            "id": spec["id"],
            "kind": spec["kind"],
            "crop": row_crop,
            "physical_row_count": len(consumed_rows),
            "source_row_indices": [item["source_row_index"] for item in consumed_rows],
        }
        if spec["kind"] == "glyphs":
            groups = [(item["row"], group) for item in consumed_rows for group in item["groups"]]
            labels = list(spec["labels"])
            group_labels, inferred_missing_labels = _labels_for_groups(spec, len(groups))
            record["detected_x_group_count"] = len(groups)
            record["label_count"] = len(labels)
            if group_labels != labels:
                record["detected_label_sequence"] = group_labels
                record["inferred_missing_labels"] = inferred_missing_labels
            for label_idx, label in enumerate(group_labels):
                if label_idx < len(groups):
                    row, (gx0, gx1) = groups[label_idx]
                    glyph = crop_save(
                        image,
                        (gx0, row["y0"], gx1, row["y1"]),
                        out_dir / "glyph_bank" / slug(label) / f"{spec['id']}_{label_idx:02d}_{slug(label)}.png",
                        pad=18,
                    )
                    glyph.update({"label": label, "row_id": spec["id"], "row_index": idx, "label_index": label_idx})
                    glyphs.setdefault(label, []).append(glyph)
                else:
                    glyphs.setdefault(label, []).append({"label": label, "row_id": spec["id"], "missing_crop": True})
            for label in inferred_missing_labels:
                glyphs.setdefault(label, []).append({"label": label, "row_id": spec["id"], "missing_crop": True})
        row_records.append(record)
    available = {label for label, crops in glyphs.items() if any(not c.get("missing_crop") for c in crops)}
    manifest = {
        "schema_version": "handgen_worksheet_manifest_v1",
        "writer_id": writer_id,
        "generated_at": now_iso(),
        "source_path": str(source),
        "local_source": str(local_source),
        "page_png": str(page_png),
        "dpi": dpi,
        "page_size": list(png_size(page_png)),
        "detected_component_count": len(comps),
        "detected_row_count": len(rows),
        "used_row_count": len(row_records),
        "rows": row_records,
        "glyphs": glyphs,
        "coverage": {"available_labels": sorted(available)},
        "known_limits": [
            "Automatic row-to-template assignment assumes the worksheet follows the expected MVP row order.",
            "Punctuation/operator crops should be visually reviewed before relying on dense math output.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def glyph_bank_from_manifest(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    bank: dict[str, list[Path]] = {}
    for label, crops in manifest.get("glyphs", {}).items():
        paths = []
        for crop in crops:
            if not crop.get("missing_crop") and crop.get("path"):
                paths.append(Path(crop["path"]))
        if paths:
            bank[label] = paths
    return bank
