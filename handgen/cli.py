from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
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
from .ingest.style import register_style
from .ingest.worksheet import ingest_worksheet
from .paths import DATA_DIR
from .render.document import SEED, render_document
from .utils import now_iso, sha_file


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
    transcript = Path(args.transcript).read_text(encoding="utf-8-sig") if args.transcript else None
    crop = tuple(args.crop) if args.crop else None
    with db.connect() as con:
        db.require_writer(con, args.writer_id)
        manifest_path, manifest = register_style(con, args.writer_id, source, DATA_DIR / "style_refs" / args.writer_id,
                                                transcript, crop=crop)
    print(f"Prepared {len(manifest['windows'])} reference windows from {len(manifest['lines'])} handwritten lines.")
    if manifest["transcript"] is not None:
        print(f"Learned {len(manifest['words'])} word exemplars from the supplied transcript.")
        for line in manifest["lines"]:
            if line["alignment"] != "user transcript aligned to isolated word ink":
                print(f"Skipped words on line {line['line_index'] + 1}. {line['alignment']}.")
    print(manifest_path)
    return 0


def cmd_style_fit(args: argparse.Namespace) -> int:
    from .vatr.fit import fit_writer

    test_reference = tuple(map(Path, args.test_reference)) if args.test_reference else None
    result = fit_writer(args.writer_id, Path(args.out).resolve(), steps=args.steps, seed=args.seed,
                        test_reference=test_reference)
    print(f"Selected step {result['selected_step']} with validation loss {result['selected_loss']:.4f} "
          f"(initial {result['baseline_loss']:.4f}).")
    print(result["adapter"])
    print(result["comparison"])
    if "independent_test" in result:
        test = result["independent_test"]
        print(f"Separate reference, {len(test['words'])} word samples: "
              f"{test['baseline_loss']:.4f} before fitting, {test['selected_loss']:.4f} after.")
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
    layout = Path(args.layout) if args.layout else None
    manifest = ingest_worksheet(args.writer_id, source, out_dir, dpi=args.dpi, layout=layout)
    manifest_path = out_dir / manifest["snapshot_name"]
    with db.connect() as con:
        con.execute(
            "DELETE FROM glyphs WHERE worksheet_id IN (SELECT id FROM worksheets WHERE writer_id = ?)",
            (args.writer_id,),
        )
        con.execute("DELETE FROM worksheets WHERE writer_id = ?", (args.writer_id,))
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
    if source.is_relative_to(canonical_dir):
        return source
    # Keep the first scan's name and bytes. A second upload with the same
    # basename gets its own source path, including when two imports race.
    with tempfile.NamedTemporaryFile(dir=canonical_dir) as copied:
        with source.open("rb") as incoming:
            shutil.copyfileobj(incoming, copied)
        copied.flush()
        digest = sha_file(Path(copied.name))
        for name in (source.name, f"{source.stem}-{digest}{source.suffix}"):
            dest = canonical_dir / name
            try:
                os.link(copied.name, dest)
            except FileExistsError:
                if dest.is_file() and sha_file(dest) == digest:
                    return dest
            else:
                return dest
    raise RuntimeError(f"Cannot store distinct worksheet assets under {canonical_dir}")


def cmd_document_render(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    if not source.exists():
        raise RuntimeError(f"source text not found: {source}")
    out_dir = Path(args.out).resolve()
    manifest = render_document(args.writer_id, source, out_dir, max_words=args.max_words, max_chars=args.max_chars,
                               prose_mode=args.prose, infer_math=args.infer_math, carrier_layout=args.carriers,
                               require_verified=args.require_verified, seed=args.seed,
                               adapter=Path(args.adapter).resolve() if args.adapter else None)
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
        prose_mode=args.prose,
        adapter=Path(args.adapter).resolve() if args.adapter else None,
        seed=args.seed, infer_math=args.infer_math, carrier_layout=args.carriers, require_verified=args.require_verified,
    )
    print(f"metrics: {metrics['metrics_path']}")
    print(f"contact_sheet: {metrics['contact_sheet']}")
    print(f"png: {metrics['outputs'].get('png')}")
    print(f"page_prose_character_error_rate: {metrics['page_readback']['character_error_rate']}")
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
        "pages": len(manifest["outputs"]["pages"]),
        "svg": manifest["outputs"]["svg"],
        "png": manifest["outputs"]["png"],
        "manifest": str(Path(manifest["outputs"]["svg"]).with_name("manifest.json")),
        "warnings": f"{len(manifest.get('warnings', []))} (details in warnings.json)",
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
    style_add.add_argument("path", help="Natural handwritten line or single-column page image.")
    style_add.add_argument("--transcript", help="UTF-8 text file with one line per handwritten line. Learn word shapes at clear spaces.")
    style_add.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                           help="Use a rectangular handwriting region in EXIF-oriented source pixels.")
    style_add.set_defaults(func=cmd_style_add)
    style_fit = style_sub.add_parser("fit", help="Fit VATr++ style conditioning on transcribed reference words.")
    style_fit.add_argument("writer_id")
    style_fit.add_argument("--out", required=True)
    style_fit.add_argument("--steps", type=int, default=100)
    style_fit.add_argument("--seed", type=int, default=732)
    style_fit.add_argument("--test-reference", nargs=2, metavar=("IMAGE", "TRANSCRIPT"),
                           help="Measure the selected model on unseen words from a separate handwriting image.")
    style_fit.set_defaults(func=cmd_style_fit)

    worksheet = sub.add_parser("worksheet")
    worksheet_sub = worksheet.add_subparsers(dest="worksheet_cmd", required=True)
    worksheet_ingest = worksheet_sub.add_parser("ingest")
    worksheet_ingest.add_argument("writer_id")
    worksheet_ingest.add_argument("path")
    worksheet_ingest.add_argument("--dpi", type=int, default=220)
    worksheet_ingest.add_argument("--layout", help="JSON file with ordered worksheet sections and their handwritten labels")
    worksheet_ingest.set_defaults(func=cmd_worksheet_ingest)

    document = sub.add_parser("document")
    doc_sub = document.add_subparsers(dest="document_cmd", required=True)
    doc_render = doc_sub.add_parser("render")
    doc_render.add_argument("writer_id")
    doc_render.add_argument("source")
    doc_render.add_argument("--out", required=True)
    doc_render.add_argument("--prose", choices=("auto", "samples", "diffbrush", "vatr"), default="auto",
                            help="Use reference words and sample repairs (auto), sample composition, DiffBrush, or VATr++.")
    doc_render.add_argument("--infer-math", action="store_true",
                            help="Also infer simple variable expressions in unmarked lines.")
    doc_render.add_argument("--carriers", action="store_true",
                            help="Try verified DiffBrush slot carriers on short mixed lines.")
    doc_render.add_argument("--require-verified", action="store_true",
                            help="Require sample-backed or OCR-matched ink for every visible span.")
    doc_render.add_argument("--seed", type=int, default=SEED)
    doc_render.add_argument("--adapter", help="VATr++ writer.safetensors produced by style fit.")
    doc_render.add_argument(
        "--max-words",
        type=int,
        default=6,
        help="Maximum words per greedy chunk when --max-chars is supplied.",
    )
    doc_render.add_argument(
        "--max-chars",
        type=int,
        help="Use the greedy splitter with this character limit.",
    )
    doc_render.set_defaults(func=cmd_document_render)

    eval_p = sub.add_parser("eval")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_render = eval_sub.add_parser("render")
    eval_render.add_argument("writer_id")
    eval_render.add_argument("--out", required=True)
    eval_render.add_argument("--source")
    eval_render.add_argument("--max-lines", type=int, default=10)
    eval_render.add_argument("--prose", choices=("auto", "samples", "diffbrush", "vatr"), default="auto")
    eval_render.add_argument("--adapter")
    eval_render.add_argument("--seed", type=int, default=SEED)
    eval_render.add_argument("--carriers", action="store_true")
    eval_render.add_argument("--infer-math", action="store_true")
    eval_render.add_argument("--require-verified", action="store_true")
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
