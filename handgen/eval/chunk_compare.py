"""Render the same prose under several chunking strategies and compare.

The harness produces:

* one rendered preview PNG per (sentence × strategy) cell,
* one ``chunking_grid.png`` with rows=sentences and columns=strategies,
* a ``summary.json`` with per-cell chunk lists, lengths, and OCR signals,
* a ``README.txt`` pointing at all of the above.

Run with::

    python -m handgen.eval.chunk_compare WRITER_ID \
        --cases examples/chunking_benchmark.json \
        --strategies current balanced_dp punctuation_first variable_length_sampling \
        --out outputs/chunking_compare/latest

Defaults match the four strategies described in :mod:`handgen.chunking`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

from .. import db
from ..chunking import STRATEGIES, get_strategy
from ..render import document as document_mod
from ..render import math as math_mod


DEFAULT_STRATEGIES = ("current", "balanced_dp", "punctuation_first", "variable_length_sampling")

CELL_SCALE = 0.45
CELL_PADDING = 20
HEADER_H = 56
LABEL_W = 280
CHUNK_TEXT_H = 64
ROW_BG = (248, 248, 248, 255)
GRID_BG = (255, 255, 255, 255)
TEXT_COLOR = (24, 24, 24, 255)
MUTED_COLOR = (110, 110, 110, 255)
RULE_COLOR = (214, 214, 214, 255)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    kind: str
    text: str


def _placeholder_png(svg_path: Path) -> Path:
    """SVG→PNG stub so the harness works without librsvg installed.

    The chunk comparison only consumes the styled prose crops on disk; the
    full-document PNG produced by the renderer is irrelevant here, so we
    can short-circuit ``render_svg_to_png`` with a blank canvas.
    """
    png_path = svg_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    width = max(1, int(round(float(root.attrib.get("width", "8")))))
    height = max(1, int(round(float(root.attrib.get("height", "8")))))
    Image.new("RGBA", (width, height), (255, 255, 255, 0)).save(png_path)
    return png_path


def _load_cases(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for item in raw:
        cases.append(BenchmarkCase(id=item["id"], kind=item.get("kind", "prose"), text=item["text"]))
    return cases


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in ("Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _multiline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int, int] = TEXT_COLOR,
    max_lines: int = 4,
    line_height: int = 16,
) -> int:
    """Word-wrap ``text`` inside ``box``. Returns the y-coordinate after the
    last drawn line so callers can stack additional content below."""
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
    for line in lines[:max_lines]:
        draw.text((x0, y), line, fill=fill, font=font)
        y += line_height
        if y > y1:
            break
    return y


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


def _build_grid(
    cases: list[BenchmarkCase],
    strategies: list[str],
    output_dir: Path,
    grid_path: Path,
    summary: dict[str, dict[str, dict[str, Any]]],
) -> Path:
    """Compose the rows-as-sentences × columns-as-strategies grid PNG."""
    sample_path = output_dir / cases[0].id / strategies[0] / "preview_crop.png"
    sample = Image.open(sample_path).convert("RGBA")
    cell_w = int(sample.width * CELL_SCALE)
    cell_h = int(sample.height * CELL_SCALE)
    header_font = _load_font(20)
    label_font = _load_font(15)
    text_font = _load_font(13)
    chunk_font = _load_font(11)

    width = LABEL_W + len(strategies) * (cell_w + CELL_PADDING) + CELL_PADDING
    row_content_h = cell_h + CHUNK_TEXT_H
    row_h = max(row_content_h, 110) + CELL_PADDING
    height = HEADER_H + len(cases) * row_h + CELL_PADDING
    canvas = Image.new("RGBA", (width, height), GRID_BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, width, HEADER_H), fill=ROW_BG)
    for idx, label in enumerate(strategies):
        x = LABEL_W + CELL_PADDING + idx * (cell_w + CELL_PADDING)
        draw.text((x, 14), label, fill=TEXT_COLOR, font=header_font)
    draw.line((0, HEADER_H, width, HEADER_H), fill=RULE_COLOR, width=1)

    for row_idx, case in enumerate(cases):
        y = HEADER_H + row_idx * row_h
        if row_idx % 2 == 0:
            draw.rectangle((0, y, width, y + row_h), fill=ROW_BG)
        draw.text((18, y + 12), f"{case.id} [{case.kind}]", fill=TEXT_COLOR, font=label_font)
        _multiline(draw, (18, y + 34, LABEL_W - 16, y + row_h - 12), case.text, text_font)
        for col_idx, strategy in enumerate(strategies):
            x = LABEL_W + CELL_PADDING + col_idx * (cell_w + CELL_PADDING)
            crop_path = output_dir / case.id / strategy / "preview_crop.png"
            cell = Image.open(crop_path).convert("RGBA")
            scaled = cell.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.alpha_composite(scaled, (x, y + 12))
            draw.rectangle((x, y + 12, x + cell_w, y + 12 + cell_h), outline=RULE_COLOR, width=1)
            entry = summary[case.id][strategy]
            lens = entry["chunk_lengths"]
            repeated = entry.get("repeated_tokens_total", 0)
            chunks_caption = (
                f"chunks={len(lens)}  lens={lens}\n"
                f"repeats={repeated}  avg_match={entry.get('avg_match_score', 0.0):.2f}"
            )
            _multiline(
                draw,
                (x, y + 12 + cell_h + 4, x + cell_w, y + 12 + cell_h + CHUNK_TEXT_H),
                chunks_caption,
                chunk_font,
                fill=MUTED_COLOR,
                max_lines=3,
                line_height=14,
            )
        draw.line((0, y + row_h, width, y + row_h), fill=RULE_COLOR, width=1)

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path)
    return grid_path


def _summarise_manifest(case_text: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract per-cell metrics from a render manifest."""
    chunks: list[str] = []
    repeated_total = 0
    match_scores: list[float] = []
    for span in manifest.get("spans", []):
        if span.get("type") != "prose":
            continue
        # The manifest text carries a trailing space between chunks for the
        # source-reconstruction contract; report the prompt content only.
        chunks.append(span.get("text", "").rstrip(" "))
        verification = (
            span.get("diffbrush", {})
            .get("ocr_verification", {})
        )
        repeated_total += len(verification.get("repeated_tokens", []) or [])
        score = verification.get("match_score")
        if isinstance(score, (int, float)):
            match_scores.append(float(score))
    avg_match = sum(match_scores) / len(match_scores) if match_scores else 0.0
    return {
        "source_text": case_text,
        "chunks": chunks,
        "chunk_lengths": [len(c) for c in chunks],
        "num_chunks": len(chunks),
        "repeated_tokens_total": repeated_total,
        "avg_match_score": round(avg_match, 4),
    }


def run_comparison(
    writer_id: str,
    cases_path: Path,
    output_dir: Path,
    strategies: list[str],
) -> dict[str, Any]:
    cases = _load_cases(cases_path)
    if not cases:
        raise RuntimeError(f"benchmark file {cases_path} contains no cases")
    db.init_db()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Skip the librsvg full-page render; we only need the styled crops.
    document_mod.render_svg_to_png = _placeholder_png
    math_mod.render_svg_to_png = _placeholder_png

    summary: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_summary: dict[str, Any] = {}
        case_dir = output_dir / case.id
        source_path = case_dir / "source.txt"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(case.text + "\n", encoding="utf-8")
        for strategy_name in strategies:
            strategy = get_strategy(strategy_name)
            render_dir = case_dir / strategy_name
            manifest = document_mod.render_document(
                writer_id,
                source_path.resolve(),
                render_dir.resolve(),
                chunking_strategy=strategy,
                chunking_strategy_name=strategy_name,
            )
            manifest_path = render_dir / "manifest.json"
            preview_path = _render_preview_from_manifest(manifest, render_dir / "preview_full.png")
            crop_path = _crop_preview(preview_path, manifest, render_dir / "preview_crop.png")
            if not manifest_path.exists():
                raise RuntimeError(f"Expected manifest was not written: {manifest_path}")
            entry = _summarise_manifest(case.text, manifest)
            entry["output_image"] = str(crop_path)
            entry["manifest_path"] = str(manifest_path)
            case_summary[strategy_name] = entry
        summary[case.id] = case_summary

    grid_path = _build_grid(cases, strategies, output_dir, output_dir / "chunking_grid.png", summary)

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "writer_id": writer_id,
        "cases_path": str(cases_path),
        "strategies": strategies,
        "grid_path": str(grid_path),
        "results": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                f"Chunking comparison for writer={writer_id}",
                f"Benchmark cases: {cases_path}",
                f"Strategies:      {', '.join(strategies)}",
                "",
                "Outputs:",
                f"  comparison grid : {grid_path}",
                f"  summary JSON    : {summary_path}",
                "  per-cell crops  : <case_id>/<strategy>/preview_crop.png",
                "  full manifests  : <case_id>/<strategy>/manifest.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "grid": str(grid_path),
        "summary": str(summary_path),
        "readme": str(readme_path),
        "cases": str(cases_path),
        "root": str(output_dir),
        "strategies": strategies,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a chunking comparison grid for handgen.")
    parser.add_argument("writer_id", help="Registered writer id (e.g. writer_001, writer_002)")
    parser.add_argument("--cases", default="examples/chunking_benchmark.json")
    parser.add_argument("--out", default="outputs/chunking_compare/latest")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=sorted(STRATEGIES),
        help="Chunking strategies to compare (one column per strategy).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_comparison(
        args.writer_id,
        Path(args.cases).resolve(),
        Path(args.out).resolve(),
        list(args.strategies),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
