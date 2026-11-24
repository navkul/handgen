from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .. import db
from ..chunking import balanced_dp
from ..diffbrush.runner import DiffBrushRunner
from ..eval.source_contract import verify_source_contract
from ..ingest.style import writer_style_lines, writer_word_exemplars
from ..ingest.worksheet import glyph_bank_from_manifest, load_worksheet_manifest
from ..parser import ParsePlan, Span, chunk_prose, parse_source, plan_to_dict
from ..utils import (
    now_iso,
    publish_files,
    render_svg_to_png,
    sha_file,
    sha_text,
    write_json,
)
from .assets import InkPiece, WriterAssets
from .carriers import carrier_prose_crops
from .content import infer_math_spans, math_slices, prose_slices
from .layout import PAGE_H, PAGE_W, PROSE_H, PageLayout
from .math import math_style_profile, missing_glyphs

SEED = 2026051301


def _render_span(span: Span, assets: WriterAssets, split, carrier_crop) -> Iterator[InkPiece]:
    if carrier_crop is not None:
        path, details = carrier_crop
        for part in re.findall(r"^\s+|\S(?:.*\S)?|\s+$", span.text):
            if part.isspace():
                yield assets.space(part)
            else:
                yield assets.generated_piece(part, path, "diffbrush_carrier_prose", "ocr_match",
                                             {"carrier_crop": details})
    elif span.kind == "math":
        for i, part in enumerate(math_slices(span.text)):
            yield assets.worksheet(part, f"{span.id}_{i:02d}") if part.strip() else assets.space(part)
    else:
        for i, part in enumerate(prose_slices(span.text, split)):
            yield from assets.prose(part, f"{span.id}_{i:02d}")


def _compose(plan: ParsePlan, assets: WriterAssets, split, carrier_layout: bool,
             require_verified: bool) -> tuple[PageLayout, list[dict], list[dict], list[dict]]:
    layout = PageLayout()
    records, lines, carriers = [], [], []
    for line in plan.lines:
        crops = {}
        if carrier_layout and (assets.mode == "diffbrush" or (assets.mode == "auto" and assets.diffbrush_available())):
            crops, carrier = carrier_prose_crops(line, assets)
            if carrier:
                carriers.append({"line_index": line.index, **carrier})
        ids = []
        for span in line.spans:
            pos = span.source_start
            for i, piece in enumerate(_render_span(span, assets, split, crops.get(span.id))):
                piece_id = f"{span.id}_piece{i:03d}"
                if require_verified and piece.evidence in {"best_effort", "unverified_model_digit"}:
                    raise RuntimeError(f"no verified handwriting for {piece.text!r} on line {line.index + 1}")
                if piece.evidence in {"best_effort", "unverified_model_digit"}:
                    generated = piece.details.get("vatr") or piece.details.get("diffbrush") or {}
                    assets.warnings.append({"code": "uncertified_prose" if span.kind == "prose" else "uncertified_digit",
                                            "span_id": piece_id, "expected": piece.text, "route": piece.route,
                                            "ocr": generated.get("ocr_verification", {}).get("actual")})
                record = {
                    "id": piece_id, "source_span_id": span.id, "source_start": pos, "source_end": pos + len(piece.text),
                    "line_index": line.index, "type": span.kind if piece.path else "space", "text": piece.text,
                    "route": piece.route, "artifact": str(piece.path) if piece.path else None,
                    "exactness_evidence": {
                        "source_text_certified": True, "visual_text_certified": piece.evidence == "ocr_match",
                        "method": piece.evidence,
                    }, **piece.details,
                }
                if piece.path:
                    layout.place(piece.path, piece.width, piece.height, piece.baseline, record)
                else:
                    layout.space(piece.width)
                    record["advance"] = piece.width
                records.append(record)
                ids.append(piece_id)
                pos += len(piece.text)
        lines.append({"index": line.index, "text": line.text, "ending": line.ending, "span_ids": ids})
        layout.newline()
    return layout, records, lines, carriers


def _save_pages(layout: PageLayout, work: Path) -> list[dict]:
    pages = []
    for number in range(1, len(layout.pages) + 1):
        path = work / ("document.svg" if number == 1 else f"document-{number:03d}.svg")
        path.write_text(layout.svg(number), encoding="utf-8")
        png = render_svg_to_png(path)
        pages.append({"number": number, "width": PAGE_W, "height": PAGE_H, "svg": str(path), "png": str(png)})
    return pages


def render_document(
    writer_id: str,
    source_path: Path,
    out_dir: Path,
    *,
    max_words: int = 6,
    max_chars: int | None = None,
    chunking_strategy: Callable[[str], list[str]] | None = None,
    chunking_strategy_name: str | None = None,
    prose_mode: str = "auto",
    infer_math: bool = False,
    carrier_layout: bool = False,
    require_verified: bool = False,
    adapter: Path | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    if prose_mode not in {"auto", "samples", "diffbrush", "vatr"}:
        raise ValueError(f"unknown prose mode {prose_mode!r}")
    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars must be positive")
    if adapter and prose_mode not in {"auto", "vatr"}:
        raise ValueError("A fitted writer adapter requires automatic or VATr++ prose.")
    if adapter and not adapter.is_file():
        raise FileNotFoundError(f"Missing fitted writer adapter: {adapter}")
    source_path, out_dir = source_path.resolve(), out_dir.resolve()
    raw = source_path.read_bytes()
    plan = parse_source(raw.decode("utf-8"))
    if infer_math:
        known: set[str] = set()
        lines = tuple(infer_math_spans(line, known) for line in plan.lines)
        plan = ParsePlan(plan.source_text, lines, tuple(s for line in lines for s in line.spans))
    split = balanced_dp
    strategy_name = "balanced_dp"
    if chunking_strategy:
        split = chunking_strategy
        strategy_name = chunking_strategy_name or "custom"
    elif max_chars is not None:
        def split(text):
            return chunk_prose(text, max_words=max_words, max_chars=max_chars)
        strategy_name = f"legacy_chars_{max_chars}"

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    db.init_db()
    with db.connect() as con:
        db.require_writer(con, writer_id)
        refs = writer_style_lines(con, writer_id)
        if not refs:
            raise RuntimeError(f"writer {writer_id!r} has no readable handwriting references; run handgen style add")
        words = writer_word_exemplars(con, writer_id)
        sheet = db.latest_worksheet(con, writer_id)
        doc = con.execute("INSERT INTO documents(writer_id, source_path, source_sha256, created_at) VALUES (?, ?, ?, ?)",
                          (writer_id, str(source_path), sha_text(plan.source_text), now_iso())).lastrowid
        job = con.execute("INSERT INTO render_jobs(document_id, out_dir, status, created_at) VALUES (?, ?, 'running', ?)",
                          (doc, str(out_dir), now_iso())).lastrowid
        # Remains unique after a database reset or across writer catalogs.
        work = Path(tempfile.mkdtemp(prefix=f"{job:05d}-", dir=runs_dir))
    try:
        (work / "source.txt").write_bytes(raw)
        write_json(work / "parse_plan.json", plan_to_dict(plan))
        adapter_record = None
        if adapter:
            snapshot = work / "writer.safetensors"
            shutil.copyfile(adapter, snapshot)
            adapter_record = {"source": str(adapter.resolve()), "path": str(snapshot), "sha256": sha_file(snapshot)}
            adapter = snapshot
        worksheet = load_worksheet_manifest(Path(sheet["manifest_path"])) if sheet else None
        glyphs = glyph_bank_from_manifest(worksheet) if worksheet else {}
        profile = math_style_profile(refs[0], worksheet, prose_height=PROSE_H)
        runner = DiffBrushRunner()
        runner.style_refs = tuple(refs)
        assets = WriterAssets(writer_id, work, refs, profile, glyphs, words, runner, prose_mode, seed, adapter)
        coverage = []
        for span in plan.spans:
            if span.kind == "math":
                allow_digit = bool(worksheet) and prose_mode != "samples" and not require_verified and assets.diffbrush_available()
                missing = missing_glyphs(span.text, glyphs, allow_digit_fallback=allow_digit)
            elif prose_mode == "samples":
                missing = assets.sample_missing(span.text)
            else:
                continue
            if missing:
                coverage.append({"span_id": span.id, "line": span.line_index + 1, "characters": missing})
        if coverage:
            write_json(work / "missing_glyphs.json", coverage)
            missing_chars = sorted({c for item in coverage for c in item["characters"]})
            raise ValueError(f"missing handwriting samples for {', '.join(map(repr, missing_chars))}. "
                             f"Check worksheet labels or transcribed words for this writer. See {work / 'missing_glyphs.json'}")
        layout, spans, lines, carriers = _compose(plan, assets, split, carrier_layout, require_verified)
        manifest = {
            "schema_version": "handgen_render_manifest_v2", "writer_id": writer_id, "generated_at": now_iso(),
            "source_path": str(source_path), "source_file": str(work / "source.txt"),
            "source_file_sha256": sha_file(work / "source.txt"), "source_text": plan.source_text,
            "source_sha256": sha_text(plan.source_text), "seed": seed, "prose_mode": prose_mode,
            "prose_adapter": adapter_record,
            "prose_chunking": {"strategy": strategy_name, "max_chars_override": max_chars},
            "style_reference": str(refs[0]), "style_references": list(map(str, refs)),
            "worksheet_manifest": sheet["manifest_path"] if sheet else None,
            "math_style_profile": profile, "prose_metrics": assets.metrics, "space_width": assets.word_gap,
            "carrier_plan": carriers, "lines": lines, "spans": spans,
            "routes_used": sorted({s["route"] for s in spans}), "selected_variable_candidates": assets.digits,
            "sample_replacements": assets.attempts, "warnings": assets.warnings,
            "source_preservation_policy": {"require_verified": require_verified, "infer_math": infer_math,
                                           "carrier_layout": carrier_layout},
        }
        contract = verify_source_contract(manifest)
        manifest["source_contract"] = contract
        if not contract["passed"]:
            write_json(work / "source_contract_failed.json", contract)
            raise RuntimeError(f"source preservation contract failed. See {work / 'source_contract_failed.json'}")
        pages = _save_pages(layout, work)
        manifest["outputs"] = {"svg": pages[0]["svg"], "png": pages[0]["png"], "pages": pages}
        write_json(work / "warnings.json", assets.warnings)
        write_json(work / "manifest.json", manifest)
        with db.connect() as con:
            con.executemany(
                "INSERT INTO render_spans(job_id, span_id, kind, route, text, artifact_path, exact_text_certified) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(job, s["id"], s["type"], s["route"], s["text"], s["artifact"],
                  int(s["exactness_evidence"]["visual_text_certified"])) for s in spans],
            )
            for page in pages:
                for kind in ("svg", "png"):
                    db.insert_artifact(con, job, kind, page[kind], sha_file(Path(page[kind])))
            db.insert_artifact(con, job, "manifest", str(work / "manifest.json"), sha_file(work / "manifest.json"))
            con.execute("UPDATE render_jobs SET status = 'done', completed_at = ? WHERE id = ?", (now_iso(), job))
            publish_files(work, out_dir, ("document.svg", "document.png", "parse_plan.json", "warnings.json", "manifest.json"))
        return manifest
    except BaseException as error:
        write_json(work / "failure.json", {"error": str(error), "job_id": job})
        with db.connect() as con:
            con.execute("UPDATE render_jobs SET status = 'failed', completed_at = ? WHERE id = ?", (now_iso(), job))
        raise
