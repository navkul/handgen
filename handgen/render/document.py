from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any

from .. import db
from ..diffbrush.runner import DiffBrushRunner
from ..ingest.worksheet import glyph_bank_from_manifest
from ..parser import Span, chunk_prose, parse_source, plan_to_dict
from ..utils import data_uri, escape_xml, now_iso, png_size, read_json, render_svg_to_png, sha_file, sha_text, write_json
from .math import math_style_profile, render_math_span


PAGE_W = 1500
PAGE_H = 1900
LEFT = 64.0
TOP = 72.0
RIGHT = 72.0
LINE_GAP = 96.0
PROSE_H = 64.0
SPAN_GAP = 16.0
SEED = 2026051301


def image_tag(path: Path, x: float, y: float, w: float, h: float, opacity: float = 0.98) -> str:
    return (
        f'<image href="{data_uri(path)}" x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" opacity="{opacity:.3f}"/>'
    )


def _display_size(path: Path, target_h: float) -> tuple[float, float]:
    w, h = png_size(path)
    width = w * (target_h / h) if h else target_h
    return width, target_h


def _insert_render_span(
    con: sqlite3.Connection,
    job_id: int,
    span_id: str,
    kind: str,
    route: str,
    text: str,
    artifact_path: str | None,
    exact: bool,
) -> None:
    con.execute(
        """
        INSERT INTO render_spans(job_id, span_id, kind, route, text, artifact_path, exact_text_certified)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, span_id, kind, route, text, artifact_path, 1 if exact else 0),
    )


def render_document(
    writer_id: str,
    source_path: Path,
    out_dir: Path,
    *,
    max_words: int = 6,
    max_chars: int | None = None,
) -> dict[str, Any]:
    db.init_db()
    source_text = source_path.read_text(encoding="utf-8")
    plan = parse_source(source_text)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, Any]] = []
    rng = random.Random(SEED)

    with db.connect() as con:
        db.require_writer(con, writer_id)
        style = db.latest_style_ref(con, writer_id)
        if style is None:
            raise RuntimeError(f"writer {writer_id!r} has no style reference; run `handgen style add` first")
        style_ref = Path(style["path"])
        math_needed = any(span.kind == "math" for span in plan.spans)
        worksheet = db.latest_worksheet(con, writer_id)
        glyph_bank: dict[str, list[Path]] = {}
        worksheet_manifest: dict[str, Any] | None = None
        if math_needed:
            if worksheet is None:
                warning = {
                    "code": "missing_symbol_bank",
                    "message": "Input contains math spans but no worksheet glyph bank has been ingested.",
                    "action": "Run `handgen worksheet ingest writer_001 /path/to/worksheet.pdf` and render again.",
                }
                warnings.append(warning)
                write_json(out_dir / "warnings.json", warnings)
                raise RuntimeError(warning["message"])
            worksheet_manifest = read_json(Path(worksheet["manifest_path"]))
            glyph_bank = glyph_bank_from_manifest(worksheet_manifest)

        con.execute(
            "INSERT INTO documents(writer_id, source_path, source_sha256, created_at) VALUES (?, ?, ?, ?)",
            (writer_id, str(source_path), sha_text(plan.source_text), now_iso()),
        )
        document_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.execute(
            "INSERT INTO render_jobs(document_id, out_dir, status, created_at) VALUES (?, ?, ?, ?)",
            (document_id, str(out_dir), "running", now_iso()),
        )
        job_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])

    write_json(out_dir / "parse_plan.json", plan_to_dict(plan))
    runner = DiffBrushRunner()
    style_profile = math_style_profile(style_ref, worksheet_manifest, prose_height=PROSE_H)
    variable_labels = sorted(
        {
            ch
            for span in plan.spans
            if span.kind == "math"
            for ch in span.text
            if ch.isascii() and ch.isalpha()
        }
    )
    variable_bank = {
        label: runner.generate_variable(
            label=label,
            style_ref=style_ref,
            out_dir=out_dir,
            seed=SEED + 7000 + ord(label[0]) * 17,
            writer_id=writer_id,
        )
        for label in variable_labels
    }

    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">',
        "<title>Handgen document</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    span_records: list[dict[str, Any]] = []
    line_records: list[dict[str, Any]] = []
    prose_route = f"diffbrush_chunked_prose_chars_{max_chars}" if max_chars is not None else "diffbrush_chunked_prose_words"
    y = TOP
    max_x = PAGE_W - RIGHT
    overflow = False

    for line in plan.lines:
        x = LEFT
        line_span_ids: list[str] = []
        for span in line.spans:
            if span.kind == "prose":
                chunks = chunk_prose(span.text, max_words=max_words, max_chars=max_chars)
                if not chunks and span.text:
                    x += 10
                    continue
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{span.id}_chunk{chunk_idx:02d}"
                    run = runner.generate_chunk(
                        span_id=chunk_id,
                        text=chunk,
                        style_ref=style_ref,
                        out_dir=out_dir,
                        seed=SEED + line.index * 101 + chunk_idx,
                        writer_id=writer_id,
                    )
                    crop = Path(run["styled_crop"]["path"])
                    width, height = _display_size(crop, style_profile["target_prose_height"])
                    if x + width > max_x:
                        x = LEFT
                        y += LINE_GAP
                    svg.append(image_tag(crop, x, y, width, height))
                    record = {
                        "id": chunk_id,
                        "source_span_id": span.id,
                        "line_index": line.index,
                        "type": "prose",
                        "route": prose_route,
                        "text": chunk,
                        "bbox": {"x": round(x, 2), "y": round(y, 2), "width": round(width, 2), "height": round(height, 2)},
                        "diffbrush": run,
                        "exactness_evidence": {
                            "exact_text_certified": False,
                            "method": "DiffBrush visual candidate; source text is preserved in parse_plan and manifest.",
                        },
                    }
                    span_records.append(record)
                    line_span_ids.append(chunk_id)
                    x += width + SPAN_GAP
            else:
                block = render_math_span(
                    span.id,
                    span.text,
                    glyph_bank,
                    out_dir,
                    rng,
                    variable_bank=variable_bank,
                    style_profile=style_profile,
                )
                png = Path(block["paths"]["png"])
                width, height = _display_size(png, block["height"])
                if x + width > max_x:
                    x = LEFT
                    y += LINE_GAP
                math_y = y + 8
                svg.append(image_tag(png, x, math_y, width, height))
                record = {
                    "id": span.id,
                    "line_index": line.index,
                    "type": "math",
                    "route": block["route"],
                    "text": span.text,
                    "bbox": {"x": round(x, 2), "y": round(math_y, 2), "width": round(width, 2), "height": round(height, 2)},
                    "math": block,
                    "exactness_evidence": {
                        "exact_text_certified": not any(ch.isascii() and ch.isalpha() for ch in span.text),
                        "method": "Each visible token is rendered from the parsed source string; ASCII English variables use DiffBrush context components and digits/Greek/symbols use worksheet glyph crops.",
                    },
                }
                span_records.append(record)
                line_span_ids.append(span.id)
                x += width + SPAN_GAP
        line_records.append({"index": line.index, "text": line.text, "span_ids": line_span_ids})
        y += LINE_GAP
        if y > PAGE_H - 80:
            overflow = True

    if overflow:
        warnings.append(
            {
                "code": "page_overflow",
                "message": "Content exceeded the single MVP page height; pagination is not implemented yet.",
            }
        )
    svg.append("</svg>")
    svg_path = out_dir / "document.svg"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    png_path = render_svg_to_png(svg_path)
    write_json(out_dir / "warnings.json", warnings)

    manifest = {
        "schema_version": "handgen_render_manifest_v1",
        "generated_at": now_iso(),
        "writer_id": writer_id,
        "source_path": str(source_path),
        "source_text": plan.source_text,
        "source_sha256": sha_text(plan.source_text),
        "routes_used": sorted({record["route"] for record in span_records}),
        "prose_chunking": {
            "mode": "chars" if max_chars is not None else "words",
            "max_words": max_words if max_chars is None else None,
            "max_chars": max_chars,
        },
        "style_reference": str(style_ref),
        "worksheet_manifest": str(worksheet["manifest_path"]) if math_needed and worksheet is not None else None,
        "math_style_profile": style_profile,
        "selected_variable_candidates": variable_bank,
        "lines": line_records,
        "spans": span_records,
        "outputs": {"svg": str(svg_path), "png": str(png_path)},
        "warnings": warnings,
        "known_limits": [
            "DiffBrush prose is not exact-text certified in MVP v1.",
            "Math exactness depends on worksheet glyph-label quality.",
            "ASCII English variables in math are DiffBrush-generated but not OCR-certified in MVP v1.",
            "Pagination is not implemented; long documents may overflow one page.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)

    with db.connect() as con:
        for record in span_records:
            artifact_path = None
            if record["type"] == "prose":
                artifact_path = record["diffbrush"]["styled_crop"]["path"]
            elif record["type"] == "math":
                artifact_path = record["math"]["paths"]["png"]
            _insert_render_span(
                con,
                job_id,
                record["id"],
                record["type"],
                record["route"],
                record["text"],
                artifact_path,
                bool(record["exactness_evidence"]["exact_text_certified"]),
            )
        db.insert_artifact(con, job_id, "svg", str(svg_path), sha_file(svg_path))
        db.insert_artifact(con, job_id, "png", str(png_path), sha_file(png_path))
        db.insert_artifact(con, job_id, "manifest", str(manifest_path), sha_file(manifest_path))
        con.execute("UPDATE render_jobs SET status = ?, completed_at = ? WHERE id = ?", ("done", now_iso(), job_id))

    return manifest
