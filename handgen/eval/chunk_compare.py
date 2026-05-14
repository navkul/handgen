from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

from .. import db
from ..render import document as document_mod
from ..render import math as math_mod


GRID_MODES = (
    ("original", None),
    ("chars40", 40),
    ("chars30", 30),
    ("chars20", 20),
)

CELL_SCALE = 0.45
CELL_PADDING = 20
HEADER_H = 42
LABEL_W = 220
ROW_BG = (248, 248, 248, 255)
GRID_BG = (255, 255, 255, 255)
TEXT_COLOR = (24, 24, 24, 255)
RULE_COLOR = (214, 214, 214, 255)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    kind: str
    text: str


def _placeholder_png(svg_path: Path) -> Path:
    png_path = svg_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    width = max(1, int(round(float(root.attrib.get("width", "8")))))
    height = max(1, int(round(float(root.attrib.get("height", "8")))))
    Image.new("RGBA", (width, height), (255, 255, 255, 0)).save(png_path)
    return png_path


def _load_cases(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase(**item) for item in raw]


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in ("Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _multiline(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> None:
    x0, y0, x1, y1 = box
    words = text.split()
    lines: list[str] = []
    current = ""
    max_width = x1 - x0
    for word in words:
        trial = word if not current else f"{current} {word}"
        width = draw.textbbox((0, 0), trial, font=font)[2]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    y = y0
    for line in lines[:4]:
        draw.text((x0, y), line, fill=TEXT_COLOR, font=font)
        y += 16
        if y > y1:
            break


def _render_preview_from_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    width = 1500
    height = 1900
    canvas = Image.new("RGBA", (width, height), GRID_BG)
    for span in manifest.get("spans", []):
        bbox = span["bbox"]
        x = int(round(bbox["x"]))
        y = int(round(bbox["y"]))
        w = max(1, int(round(bbox["width"])))
        h = max(1, int(round(bbox["height"])))
        if span["type"] == "prose":
            asset_path = Path(span["diffbrush"]["styled_crop"]["path"])
        else:
            asset_path = Path(span["math"]["paths"]["png"])
        tile = Image.open(asset_path).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
        canvas.alpha_composite(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _crop_preview(preview_path: Path, manifest: dict[str, Any], output_path: Path) -> Path:
    image = Image.open(preview_path).convert("RGBA")
    xs: list[int] = []
    ys: list[int] = []
    xe: list[int] = []
    ye: list[int] = []
    for span in manifest.get("spans", []):
        bbox = span["bbox"]
        xs.append(int(bbox["x"]))
        ys.append(int(bbox["y"]))
        xe.append(int(bbox["x"] + bbox["width"]))
        ye.append(int(bbox["y"] + bbox["height"]))
    if not xs:
        crop = image
    else:
        pad = 24
        box = (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(image.width, max(xe) + pad),
            min(image.height, max(ye) + pad),
        )
        crop = image.crop(box)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return output_path


def _build_grid(cases: list[BenchmarkCase], output_dir: Path, grid_path: Path) -> Path:
    sample = Image.open(output_dir / cases[0].id / "original" / "preview_crop.png").convert("RGBA")
    cell_w = int(sample.width * CELL_SCALE)
    cell_h = int(sample.height * CELL_SCALE)
    header_font = _load_font(18)
    label_font = _load_font(15)
    text_font = _load_font(13)
    width = LABEL_W + len(GRID_MODES) * (cell_w + CELL_PADDING) + CELL_PADDING
    row_h = max(cell_h, 94) + CELL_PADDING
    height = HEADER_H + len(cases) * row_h + CELL_PADDING
    canvas = Image.new("RGBA", (width, height), GRID_BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, width, HEADER_H), fill=ROW_BG)
    for idx, (label, _) in enumerate(GRID_MODES):
        x = LABEL_W + CELL_PADDING + idx * (cell_w + CELL_PADDING)
        draw.text((x, 12), label, fill=TEXT_COLOR, font=header_font)
    draw.line((0, HEADER_H, width, HEADER_H), fill=RULE_COLOR, width=1)

    for row_idx, case in enumerate(cases):
        y = HEADER_H + row_idx * row_h
        if row_idx % 2 == 0:
            draw.rectangle((0, y, width, y + row_h), fill=ROW_BG)
        draw.text((18, y + 12), f"{case.id} [{case.kind}]", fill=TEXT_COLOR, font=label_font)
        _multiline(draw, (18, y + 34, LABEL_W - 16, y + row_h - 12), case.text, text_font)
        for col_idx, (mode_label, _) in enumerate(GRID_MODES):
            x = LABEL_W + CELL_PADDING + col_idx * (cell_w + CELL_PADDING)
            crop_path = output_dir / case.id / mode_label / "preview_crop.png"
            cell = Image.open(crop_path).convert("RGBA")
            scaled = cell.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.alpha_composite(scaled, (x, y + 12))
            draw.rectangle((x, y + 12, x + cell_w, y + 12 + cell_h), outline=RULE_COLOR, width=1)
        draw.line((0, y + row_h, width, y + row_h), fill=RULE_COLOR, width=1)

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path)
    return grid_path


def run_comparison(writer_id: str, cases_path: Path, output_dir: Path) -> dict[str, str]:
    cases = _load_cases(cases_path)
    db.init_db()
    output_dir.mkdir(parents=True, exist_ok=True)
    document_mod.render_svg_to_png = _placeholder_png
    math_mod.render_svg_to_png = _placeholder_png

    for case in cases:
        case_dir = output_dir / case.id
        source_path = case_dir / "source.txt"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(case.text + "\n", encoding="utf-8")
        for mode_label, max_chars in GRID_MODES:
            render_dir = case_dir / mode_label
            manifest = document_mod.render_document(
                writer_id,
                source_path.resolve(),
                render_dir.resolve(),
                max_chars=max_chars,
            )
            manifest_path = render_dir / "manifest.json"
            preview_path = _render_preview_from_manifest(manifest, render_dir / "preview_full.png")
            _crop_preview(preview_path, manifest, render_dir / "preview_crop.png")
            if not manifest_path.exists():
                raise RuntimeError(f"Expected manifest was not written: {manifest_path}")

    grid_path = _build_grid(cases, output_dir, output_dir / "chunking_grid.png")
    return {
        "grid": str(grid_path),
        "cases": str(cases_path),
        "root": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a chunking comparison grid for handgen.")
    parser.add_argument("writer_id")
    parser.add_argument("--cases", default="examples/chunking_benchmark.json")
    parser.add_argument("--out", default="outputs/chunking_compare")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_comparison(args.writer_id, Path(args.cases).resolve(), Path(args.out).resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
