from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import db
from .eval.harness import (
    CARRIER_PROMPT_BENCHMARKS,
    PLACEHOLDER_BENCHMARK_TEMPLATES,
    PLACEHOLDER_BENCHMARK_WORDS,
    render_carrier_prompt_benchmark,
    render_eval_suite,
    render_placeholder_benchmark,
)
from .ingest.worksheet import ingest_worksheet
from .paths import DATA_DIR
from .render.document import render_document
from .utils import now_iso, png_size, sha_file, write_json


def cmd_init(_: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    print(f"initialized {DATA_DIR}")
    return 0


def cmd_writer_create(args: argparse.Namespace) -> int:
    db.create_writer(args.writer_id)
    print(f"writer ready: {args.writer_id}")
    return 0


def cmd_style_add(args: argparse.Namespace) -> int:
    db.init_db()
    source = Path(args.path).resolve()
    if not source.exists():
        raise RuntimeError(f"style reference not found: {source}")
    with db.connect() as con:
        db.require_writer(con, args.writer_id)
    dst = canonical_asset(source, DATA_DIR / "style_refs" / args.writer_id)
    width = height = None
    if dst.suffix.lower() == ".png":
        width, height = png_size(dst)
    with db.connect() as con:
        con.execute(
            """
            INSERT INTO style_refs(writer_id, path, sha256, width, height, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (args.writer_id, str(dst), sha_file(dst), width, height, now_iso()),
        )
    print(f"style reference registered: {dst}")
    return 0


def cmd_worksheet_ingest(args: argparse.Namespace) -> int:
    db.init_db()
    source = Path(args.path).resolve()
    if not source.exists():
        raise RuntimeError(f"worksheet not found: {source}")
    with db.connect() as con:
        db.require_writer(con, args.writer_id)
    source = canonical_asset(source, DATA_DIR / "worksheets" / args.writer_id)
    out_dir = source.with_suffix("")
    manifest = ingest_worksheet(args.writer_id, source, out_dir, dpi=args.dpi)
    manifest_path = out_dir / "manifest.json"
    with db.connect() as con:
        prior_ids = [
            row[0]
            for row in con.execute(
                "SELECT id FROM worksheets WHERE writer_id = ?", (args.writer_id,)
            ).fetchall()
        ]
        if prior_ids:
            placeholders = ",".join("?" for _ in prior_ids)
            con.execute(f"DELETE FROM glyphs WHERE worksheet_id IN ({placeholders})", prior_ids)
            con.execute(f"DELETE FROM worksheets WHERE id IN ({placeholders})", prior_ids)
        con.execute(
            "INSERT INTO worksheets(writer_id, source_path, manifest_path, created_at) VALUES (?, ?, ?, ?)",
            (args.writer_id, str(source), str(manifest_path), now_iso()),
        )
        worksheet_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        for label, crops in manifest.get("glyphs", {}).items():
            for crop in crops:
                if crop.get("missing_crop") or not crop.get("path"):
                    continue
                con.execute(
                    """
                    INSERT INTO glyphs(worksheet_id, label, path, sha256, width, height, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (worksheet_id, label, crop["path"], crop["sha256"], crop.get("width"), crop.get("height"), now_iso()),
                )
    print(f"worksheet ingested: {manifest_path}")
    return 0


def canonical_asset(source: Path, canonical_dir: Path) -> Path:
    canonical_dir = canonical_dir.resolve()
    source = source.resolve()
    canonical_dir.mkdir(parents=True, exist_ok=True)
    try:
        source.relative_to(canonical_dir)
        return source
    except ValueError:
        dst = canonical_dir / source.name
        if source != dst:
            shutil.copyfile(source, dst)
        return dst


def cmd_document_render(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    if not source.exists():
        raise RuntimeError(f"source text not found: {source}")
    out_dir = Path(args.out).resolve()
    manifest = render_document(args.writer_id, source, out_dir, max_words=args.max_words, max_chars=args.max_chars)
    print(write_summary(manifest))
    return 0


def cmd_eval_render(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve() if args.source else None
    if source is not None and not source.exists():
        raise RuntimeError(f"eval source text not found: {source}")
    metrics = render_eval_suite(
        args.writer_id,
        Path(args.out).resolve(),
        source_path=source,
        max_lines=args.max_lines,
    )
    print(f"metrics: {metrics['metrics_path']}")
    print(f"contact_sheet: {metrics['contact_sheet']}")
    print(f"png: {metrics['outputs'].get('png')}")
    print(f"carrier_mean_ocr_score: {metrics.get('carrier_mean_ocr_score')}")
    print(f"quality_gates: {metrics.get('quality_gates')}")
    print(f"warnings: {metrics.get('warning_codes')}")
    return 0


def cmd_eval_placeholders(args: argparse.Namespace) -> int:
    placeholders = tuple(args.placeholders.split(",")) if args.placeholders else PLACEHOLDER_BENCHMARK_WORDS
    if args.template_limit is None:
        templates = PLACEHOLDER_BENCHMARK_TEMPLATES
    else:
        templates = PLACEHOLDER_BENCHMARK_TEMPLATES[: args.template_limit]
    metrics = render_placeholder_benchmark(
        args.writer_id,
        Path(args.out).resolve(),
        placeholders=placeholders,
        templates=templates,
        seeds=args.seeds,
    )
    print(f"metrics: {metrics['metrics_path']}")
    print(f"contact_sheet: {metrics['contact_sheet']}")
    print(f"placeholder_policy: {metrics['placeholder_policy']}")
    print(f"recommended_placeholder_order: {metrics['recommended_placeholder_order']}")
    return 0


def cmd_eval_carrier_prompts(args: argparse.Namespace) -> int:
    placeholders = tuple(args.placeholders.split(",")) if args.placeholders else ("number",)
    contexts = tuple(args.contexts.split(",")) if args.contexts else None
    metrics = render_carrier_prompt_benchmark(
        args.writer_id,
        Path(args.out).resolve(),
        placeholders=placeholders,
        contexts=contexts,
        seeds=args.seeds,
        top_k=args.top_k,
    )
    print(f"metrics: {metrics['metrics_path']}")
    print(f"contact_sheet: {metrics['contact_sheet']}")
    print(f"carrier_prompt_policy: {metrics['carrier_prompt_policy']}")
    print(f"contexts: {metrics['contexts']}")
    return 0


def write_summary(manifest: dict) -> str:
    summary = {
        "svg": manifest["outputs"]["svg"],
        "png": manifest["outputs"]["png"],
        "manifest": str(Path(manifest["outputs"]["svg"]).with_name("manifest.json")),
        "warnings": manifest.get("warnings", []),
    }
    return "\n".join([f"{key}: {value}" for key, value in summary.items()])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handgen")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init")
    init_p.set_defaults(func=cmd_init)

    writer = sub.add_parser("writer")
    writer_sub = writer.add_subparsers(dest="writer_cmd", required=True)
    writer_create = writer_sub.add_parser("create")
    writer_create.add_argument("writer_id")
    writer_create.set_defaults(func=cmd_writer_create)

    style = sub.add_parser("style")
    style_sub = style.add_subparsers(dest="style_cmd", required=True)
    style_add = style_sub.add_parser("add")
    style_add.add_argument("writer_id")
    style_add.add_argument("path")
    style_add.set_defaults(func=cmd_style_add)

    worksheet = sub.add_parser("worksheet")
    worksheet_sub = worksheet.add_subparsers(dest="worksheet_cmd", required=True)
    worksheet_ingest = worksheet_sub.add_parser("ingest")
    worksheet_ingest.add_argument("writer_id")
    worksheet_ingest.add_argument("path")
    worksheet_ingest.add_argument("--dpi", type=int, default=220)
    worksheet_ingest.set_defaults(func=cmd_worksheet_ingest)

    document = sub.add_parser("document")
    doc_sub = document.add_subparsers(dest="document_cmd", required=True)
    doc_render = doc_sub.add_parser("render")
    doc_render.add_argument("writer_id")
    doc_render.add_argument("source")
    doc_render.add_argument("--out", required=True)
    doc_render.add_argument("--max-words", type=int, default=6, help="Deprecated for carrier rendering; kept for prose-only chunking.")
    doc_render.add_argument("--max-chars", type=int)
    doc_render.set_defaults(func=cmd_document_render)

    eval_p = sub.add_parser("eval")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_render = eval_sub.add_parser("render")
    eval_render.add_argument("writer_id")
    eval_render.add_argument("--out", required=True)
    eval_render.add_argument("--source")
    eval_render.add_argument("--max-lines", type=int, default=10)
    eval_render.set_defaults(func=cmd_eval_render)
    eval_placeholders = eval_sub.add_parser("placeholders")
    eval_placeholders.add_argument("writer_id")
    eval_placeholders.add_argument("--out", required=True)
    eval_placeholders.add_argument("--seeds", type=int, default=2)
    eval_placeholders.add_argument("--template-limit", type=int)
    eval_placeholders.add_argument("--placeholders", help="Comma-separated placeholder words to benchmark.")
    eval_placeholders.set_defaults(func=cmd_eval_placeholders)
    eval_carrier_prompts = eval_sub.add_parser("carrier-prompts")
    eval_carrier_prompts.add_argument("writer_id")
    eval_carrier_prompts.add_argument("--out", required=True)
    eval_carrier_prompts.add_argument("--seeds", type=int, default=1)
    eval_carrier_prompts.add_argument("--top-k", type=int, default=3)
    eval_carrier_prompts.add_argument("--placeholders", help="Comma-separated placeholder words to benchmark.")
    eval_carrier_prompts.add_argument("--contexts", help=f"Comma-separated contexts. Available: {','.join(CARRIER_PROMPT_BENCHMARKS)}")
    eval_carrier_prompts.set_defaults(func=cmd_eval_carrier_prompts)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
