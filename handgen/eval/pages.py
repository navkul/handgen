"""Read actual exported pages back with local OCR."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image

from .ocr import ocr_image, verify_ocr_text


def _read_region(paper: Image.Image, spans: list[dict], source: str, path: Path) -> dict:
    boxes = [s["bbox"] for s in spans]
    bounds = (max(0, int(min(b["x"] for b in boxes)) - 2),
              max(0, int(min(b["y"] for b in boxes)) - 2),
              min(paper.width, round(max(b["x"] + b["width"] for b in boxes)) + 3),
              min(paper.height, round(max(b["y"] + b["height"] for b in boxes)) + 3))
    paper.crop(bounds).save(path)
    start, end = spans[0]["source_start"], spans[-1]["source_end"]
    ocr = ocr_image(path, psm=7)
    return {"page": spans[0]["page"], "bbox": bounds, "path": str(path), "source_start": start, "source_end": end,
            "span_ids": [s["id"] for s in spans], "engine": ocr["engine"],
            "verification": verify_ocr_text(source[start:end], ocr["text"])}


def assess_pages(manifest: dict, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    by_page = defaultdict(list)
    for span in manifest["spans"]:
        if "page" in span:
            by_page[span["page"]].append(span)

    regions, empty = [], []
    for page in manifest["outputs"]["pages"]:
        with Image.open(page["png"]) as image:
            paper = image.convert("RGB")
        group = []
        number = page["number"]
        for span in by_page[number]:
            b = span["bbox"]
            tile = paper.crop((int(b["x"]), int(b["y"]), round(b["x"] + b["width"]), round(b["y"] + b["height"])))
            if not tile.convert("L").point(lambda gray: 255 if gray < 238 else 0).getbbox():
                empty.append(span["id"])
            if group and (span["type"] != "prose" or span["baseline"] != group[-1]["baseline"]
                          or span["line_index"] != group[-1]["line_index"]):
                path = work / f"page{number:03d}_prose{len(regions):04d}.png"
                regions.append(_read_region(paper, group, manifest["source_text"], path))
                group = []
            if span["type"] == "prose":
                group.append(span)
        if group:
            path = work / f"page{number:03d}_prose{len(regions):04d}.png"
            regions.append(_read_region(paper, group, manifest["source_text"], path))
    edits = sum(r["verification"]["character_edit_distance"] for r in regions)
    characters = sum(len(r["verification"]["expected_visible"]) for r in regions)
    return {"pages": len(manifest["outputs"]["pages"]), "empty_ink_spans": empty,
            "prose_regions": regions, "exact_regions": sum(r["verification"]["passed"] for r in regions),
            "character_edits": edits, "target_characters": characters,
            "character_error_rate": edits / characters if characters else None}
