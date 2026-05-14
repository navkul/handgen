from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import db
from ..diffbrush.runner import DiffBrushRunner
from ..render.document import PLACEHOLDER_VOCABULARY, render_document
from ..utils import read_json, require_pillow_numpy, sha_text, write_json


DEFAULT_MIXED_LINES = [
    "Show that p + q is odd.",
    "If m < n, then z is positive.",
    "Let F = 8. Then G / K is useful.",
    "Check that u * v is greater than w.",
    "Suppose R - S is even.",
    "Given h = 3, then J is small.",
    "When c > d, then e is true.",
    "For L / M, the value is fixed.",
    "Assume t + s is prime.",
    "Then N - P is negative.",
]

PLACEHOLDER_BENCHMARK_WORDS = ("term", "value", "number", "thing", "letter", "amount")
PLACEHOLDER_BENCHMARK_TEMPLATES = (
    "If {placeholder} is true.",
    "Let {placeholder} be useful.",
    "Then {placeholder} is useful.",
)
PLACEHOLDER_BENCHMARK_CONTEXTS = {
    "If {placeholder} is true.": "if_short_slot",
    "Let {placeholder} be useful.": "let_short_slot",
    "Then {placeholder} is useful.": "then_slot_clause",
}
CARRIER_PROMPT_BENCHMARKS = {
    "if_short_slot": {
        "text": "If {placeholder},",
        "prompt_templates": (
            "If {placeholder}",
            "If {placeholder} is positive",
            "If {placeholder} is true",
            "If {placeholder} is useful",
            "If {placeholder} is small",
        ),
    },
    "let_short_slot": {
        "text": "Let {placeholder}.",
        "prompt_templates": (
            "Let {placeholder} be useful",
            "Let {placeholder} be fixed",
            "Let {placeholder}",
        ),
    },
    "then_slot_clause": {
        "text": "Then {placeholder} is useful.",
        "prompt_templates": (
            "Then {placeholder} is useful.",
            "Then {placeholder} is useful",
            "Then {placeholder} is positive",
            "Then {placeholder} is true",
        ),
    },
    "and_slot_clause": {
        "text": "and {placeholder} is small.",
        "prompt_templates": (
            "And {placeholder} is small.",
            "And {placeholder} is small",
            "And {placeholder} is useful",
            "And {placeholder} is positive",
        ),
    },
    "suppose_slot_clause": {
        "text": "Suppose {placeholder} is positive,",
        "prompt_templates": (
            "Suppose {placeholder} is positive",
            "Suppose {placeholder} is useful",
            "Suppose {placeholder} is true",
            "Suppose {placeholder}",
        ),
    },
}


def render_eval_suite(
    writer_id: str,
    out_dir: Path,
    *,
    source_path: Path | None = None,
    max_lines: int = 10,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = source_path or _write_default_source(out_dir, max_lines=max_lines)
    render_out = out_dir / "render"
    manifest = render_document(writer_id, source, render_out)
    metrics = summarize_manifest(manifest)
    metrics["source_path"] = str(source)
    metrics["render_manifest"] = str(render_out / "manifest.json")
    metrics["contact_sheet"] = str(_write_contact_sheet(manifest, out_dir / "contact_sheet.png"))
    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def render_placeholder_benchmark(
    writer_id: str,
    out_dir: Path,
    *,
    placeholders: tuple[str, ...] = PLACEHOLDER_BENCHMARK_WORDS,
    templates: tuple[str, ...] = PLACEHOLDER_BENCHMARK_TEMPLATES,
    seeds: int = 2,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    db.init_db()
    with db.connect() as con:
        db.require_writer(con, writer_id)
        style = db.latest_style_ref(con, writer_id)
    if style is None:
        raise RuntimeError(f"writer {writer_id!r} has no style reference; run `handgen style add` first")
    style_ref = Path(style["path"])
    runner = DiffBrushRunner()
    results: list[dict[str, Any]] = []
    for placeholder in placeholders:
        for template_index, template in enumerate(templates):
            text = template.format(placeholder=placeholder)
            for seed_index in range(seeds):
                span_id = f"placeholder_{placeholder}_{template_index:02d}_{seed_index:02d}_{sha_text(text)[:8]}"
                run = runner.generate_chunk(
                    span_id=span_id,
                    text=text,
                    style_ref=style_ref,
                    out_dir=out_dir,
                    seed=2026051400 + template_index * 100_000 + seed_index * 10_000 + sum(ord(ch) for ch in placeholder),
                    writer_id=writer_id,
                )
                verification = run.get("ocr_verification", {})
                results.append(
                    {
                        "placeholder": placeholder,
                        "context": PLACEHOLDER_BENCHMARK_CONTEXTS.get(template, "generic_slot_clause"),
                        "template": template,
                        "text": text,
                        "seed_index": seed_index,
                        "route": run.get("route"),
                        "exactness_certified": bool(run.get("exactness_certified")),
                        "match_score": float(verification.get("match_score", 0.0) or 0.0),
                        "ocr_text": verification.get("actual", ""),
                        "candidate_count": run.get("candidate_count"),
                        "accepted_candidate": run.get("accepted_candidate"),
                        "styled_crop": run.get("styled_crop"),
                    }
                )
    summary = _summarize_placeholder_results(results)
    context_summary = _summarize_placeholder_results_by_context(results)
    policy = _placeholder_policy_from_summary(summary, context_summary=context_summary)
    policy_path = out_dir / "placeholder_policy.json"
    write_json(policy_path, policy)
    contact_sheet = _write_placeholder_contact_sheet(results, out_dir / "placeholder_contact_sheet.png")
    metrics = {
        "schema_version": "handgen_placeholder_benchmark_v1",
        "writer_id": writer_id,
        "style_reference": str(style_ref),
        "placeholders": list(placeholders),
        "templates": list(templates),
        "seeds_per_template": seeds,
        "results": results,
        "summary": summary,
        "context_summary": context_summary,
        "recommended_placeholder_order": [item["placeholder"] for item in summary],
        "placeholder_policy": str(policy_path),
        "render_env": {"HANDGEN_PLACEHOLDER_POLICY": str(policy_path)},
        "contact_sheet": str(contact_sheet),
    }
    metrics_path = out_dir / "placeholder_metrics.json"
    write_json(metrics_path, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def render_carrier_prompt_benchmark(
    writer_id: str,
    out_dir: Path,
    *,
    placeholders: tuple[str, ...] = ("number",),
    contexts: tuple[str, ...] | None = None,
    seeds: int = 1,
    top_k: int = 3,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    db.init_db()
    with db.connect() as con:
        db.require_writer(con, writer_id)
        style = db.latest_style_ref(con, writer_id)
    if style is None:
        raise RuntimeError(f"writer {writer_id!r} has no style reference; run `handgen style add` first")
    style_ref = Path(style["path"])
    runner = DiffBrushRunner()
    selected_contexts = contexts or tuple(CARRIER_PROMPT_BENCHMARKS)
    results: list[dict[str, Any]] = []
    for context in selected_contexts:
        spec = CARRIER_PROMPT_BENCHMARKS[context]
        for placeholder in placeholders:
            text = spec["text"].format(placeholder=placeholder)
            for template_index, template in enumerate(spec["prompt_templates"]):
                prompt = template.format(placeholder=placeholder)
                for seed_index in range(seeds):
                    span_id = f"carrier_{context}_{placeholder}_{template_index:02d}_{seed_index:02d}_{sha_text(prompt)[:8]}"
                    run = runner.generate_chunk(
                        span_id=span_id,
                        text=text,
                        prompt_text=prompt,
                        style_ref=style_ref,
                        out_dir=out_dir,
                        seed=2026051500 + template_index * 100_000 + seed_index * 10_000 + sum(ord(ch) for ch in f"{context}:{placeholder}"),
                        writer_id=writer_id,
                    )
                    verification = run.get("ocr_verification", {})
                    results.append(
                        {
                            "context": context,
                            "placeholder": placeholder,
                            "text": text,
                            "prompt_template": template,
                            "prompt": prompt,
                            "seed_index": seed_index,
                            "route": run.get("route"),
                            "exactness_certified": bool(run.get("exactness_certified")),
                            "match_score": float(verification.get("match_score", 0.0) or 0.0),
                            "ocr_text": verification.get("actual", ""),
                            "candidate_count": run.get("candidate_count"),
                            "accepted_candidate": run.get("accepted_candidate"),
                            "styled_crop": run.get("styled_crop"),
                        }
                    )
    summary = _summarize_carrier_prompt_results(results)
    policy = _carrier_prompt_policy_from_summary(summary, top_k=top_k)
    policy_path = out_dir / "carrier_prompt_policy.json"
    write_json(policy_path, policy)
    contact_sheet = _write_carrier_prompt_contact_sheet(results, out_dir / "carrier_prompt_contact_sheet.png")
    metrics = {
        "schema_version": "handgen_carrier_prompt_benchmark_v1",
        "writer_id": writer_id,
        "style_reference": str(style_ref),
        "placeholders": list(placeholders),
        "contexts": list(selected_contexts),
        "seeds_per_prompt": seeds,
        "top_k": top_k,
        "results": results,
        "summary": summary,
        "carrier_prompt_policy": str(policy_path),
        "render_env": {"HANDGEN_CARRIER_PROMPT_POLICY": str(policy_path)},
        "contact_sheet": str(contact_sheet),
    }
    metrics_path = out_dir / "carrier_prompt_metrics.json"
    write_json(metrics_path, metrics)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    carrier_spans = [span for span in manifest.get("spans", []) if span.get("type") in {"slot_carrier_prose", "prose"}]
    math_slots = [span for span in manifest.get("spans", []) if span.get("type") == "math_slot_replacement"]
    route_counts: dict[str, int] = {}
    for span in carrier_spans:
        route = str(span.get("route", "unknown"))
        route_counts[route] = route_counts.get(route, 0) + 1
    carrier_scores = [
        float(span.get("diffbrush", {}).get("ocr_verification", {}).get("match_score", 0.0) or 0.0)
        for span in carrier_spans
    ]
    carrier_effective_floors = [
        float(span.get("diffbrush", {}).get("best_effort_effective_min_score", span.get("diffbrush", {}).get("best_effort_min_score", 0.0)) or 0.0)
        for span in carrier_spans
        if span.get("diffbrush", {}).get("best_effort_effective_min_score") is not None or span.get("diffbrush", {}).get("best_effort_min_score") is not None
    ]
    exact_count = sum(1 for span in carrier_spans if span.get("diffbrush", {}).get("exactness_certified"))
    slot_sizes = [
        {
            "id": span.get("id"),
            "text": span.get("text"),
            "width": span.get("bbox", {}).get("width"),
            "height": span.get("bbox", {}).get("height"),
            "slot_word_box_source": span.get("slot_word_box_source"),
            "visual_detection_method": (span.get("slot_word_box") or {}).get("visual_detection", {}).get("method"),
            "target_visible_height": span.get("slot_geometry", {}).get("target_visible_height"),
        }
        for span in math_slots
    ]
    slot_source_counts: dict[str, int] = {}
    slot_visual_detection_counts: dict[str, int] = {}
    for slot in slot_sizes:
        source = str(slot.get("slot_word_box_source") or "unknown")
        slot_source_counts[source] = slot_source_counts.get(source, 0) + 1
        method = str(slot.get("visual_detection_method") or "none")
        slot_visual_detection_counts[method] = slot_visual_detection_counts.get(method, 0) + 1
    placeholder_usage = []
    for line in manifest.get("carrier_plan", []):
        for insertion in line.get("insertions", []):
            placeholder = insertion.get("slot_placeholder") or {}
            placeholder_usage.append(
                {
                    "source_text": insertion.get("text"),
                    "placeholder": placeholder.get("text"),
                    "line": line.get("source_text"),
                }
            )
    source_contract = manifest.get("source_contract") or {}
    min_score = min(carrier_scores) if carrier_scores else None
    effective_floor = min(carrier_effective_floors) if carrier_effective_floors else None
    last_resort_count = route_counts.get("diffbrush_last_resort_prose_token", 0)
    quality_gates = {
        "source_contract_passed": bool(source_contract.get("passed")),
        "no_last_resort_carriers": last_resort_count == 0,
        "carrier_min_score_meets_effective_floor": (
            True if min_score is None or effective_floor is None else min_score >= effective_floor
        ),
        "all_slots_have_visual_detection": bool(slot_sizes) and all(slot.get("visual_detection_method") for slot in slot_sizes),
    }
    quality_gates["passed"] = all(quality_gates.values())
    return {
        "schema_version": "handgen_eval_metrics_v1",
        "source_contract": source_contract,
        "routes_used": manifest.get("routes_used", []),
        "warning_codes": [warning.get("code") for warning in manifest.get("warnings", [])],
        "carrier_count": len(carrier_spans),
        "carrier_exact_count": exact_count,
        "carrier_route_counts": route_counts,
        "carrier_last_resort_count": last_resort_count,
        "carrier_best_effort_count": route_counts.get("diffbrush_best_effort_prose_token", 0),
        "carrier_min_ocr_score": min_score,
        "carrier_mean_ocr_score": round(sum(carrier_scores) / len(carrier_scores), 4) if carrier_scores else None,
        "carrier_best_effort_effective_min_score": effective_floor,
        "quality_gates": quality_gates,
        "placeholder_vocabulary": list(PLACEHOLDER_VOCABULARY),
        "placeholder_usage": placeholder_usage,
        "slot_word_box_source_counts": slot_source_counts,
        "slot_visual_detection_counts": slot_visual_detection_counts,
        "slot_sizes": slot_sizes,
        "outputs": manifest.get("outputs", {}),
    }


def _summarize_placeholder_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_placeholder: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_placeholder.setdefault(result["placeholder"], []).append(result)
    summary = []
    for placeholder, rows in by_placeholder.items():
        scores = [float(row.get("match_score", 0.0) or 0.0) for row in rows]
        exact_count = sum(1 for row in rows if row.get("exactness_certified"))
        last_resort_count = sum(1 for row in rows if row.get("route") == "diffbrush_last_resort_prose_token")
        summary.append(
            {
                "placeholder": placeholder,
                "sample_count": len(rows),
                "mean_match_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "max_match_score": max(scores) if scores else 0.0,
                "exact_count": exact_count,
                "last_resort_count": last_resort_count,
            }
        )
    summary.sort(key=lambda item: (item["exact_count"], item["mean_match_score"], -item["last_resort_count"]), reverse=True)
    return summary


def _summarize_placeholder_results_by_context(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_context_placeholder: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        by_context_placeholder.setdefault((str(result.get("context", "generic_slot_clause")), str(result["placeholder"])), []).append(result)
    summary = []
    for (context, placeholder), rows in by_context_placeholder.items():
        scores = [float(row.get("match_score", 0.0) or 0.0) for row in rows]
        exact_count = sum(1 for row in rows if row.get("exactness_certified"))
        last_resort_count = sum(1 for row in rows if row.get("route") == "diffbrush_last_resort_prose_token")
        summary.append(
            {
                "context": context,
                "placeholder": placeholder,
                "sample_count": len(rows),
                "mean_match_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "max_match_score": max(scores) if scores else 0.0,
                "exact_count": exact_count,
                "last_resort_count": last_resort_count,
            }
        )
    summary.sort(key=lambda item: (item["context"], -item["exact_count"], -item["mean_match_score"], item["last_resort_count"], item["placeholder"]))
    return summary


def _placeholder_policy_from_summary(summary: list[dict[str, Any]], *, context_summary: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ranked = [str(item["placeholder"]) for item in summary]
    best = ranked[0] if ranked else "number"
    by_context: dict[str, list[dict[str, Any]]] = {}
    for item in context_summary or []:
        by_context.setdefault(str(item["context"]), []).append(item)
    context_policy = {
        context: [
            {"max_visible_len": 1, "placeholder": str(rows[0]["placeholder"])},
            {"max_visible_len": 3, "placeholder": str(rows[0]["placeholder"])},
            {"max_visible_len": 5, "placeholder": str(rows[0]["placeholder"])},
            {"max_visible_len": 10000, "placeholder": str(rows[0]["placeholder"])},
        ]
        for context, rows in by_context.items()
        if rows
    }
    return {
        "schema_version": "handgen_placeholder_policy_v1",
        "ranked_placeholders": ranked,
        "slot_placeholders": [
            {"max_visible_len": 1, "placeholder": best},
            {"max_visible_len": 3, "placeholder": best},
            {"max_visible_len": 5, "placeholder": best},
            {"max_visible_len": 10000, "placeholder": best},
        ],
        "slot_placeholders_by_context": context_policy,
        "method": "Derived from placeholder benchmark OCR exactness, mean match score, and last-resort count; uses context-specific winners when available.",
    }


def _summarize_carrier_prompt_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((result["context"], result["placeholder"], result["prompt_template"]), []).append(result)
    summary: list[dict[str, Any]] = []
    for (context, placeholder, prompt_template), rows in grouped.items():
        scores = [float(row.get("match_score", 0.0) or 0.0) for row in rows]
        exact_count = sum(1 for row in rows if row.get("exactness_certified"))
        last_resort_count = sum(1 for row in rows if row.get("route") == "diffbrush_last_resort_prose_token")
        summary.append(
            {
                "context": context,
                "placeholder": placeholder,
                "prompt_template": prompt_template,
                "sample_count": len(rows),
                "mean_match_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "max_match_score": max(scores) if scores else 0.0,
                "exact_count": exact_count,
                "last_resort_count": last_resort_count,
            }
        )
    summary.sort(key=lambda item: (item["context"], item["placeholder"], -item["exact_count"], -item["mean_match_score"], item["last_resort_count"], item["prompt_template"]))
    return summary


def _carrier_prompt_policy_from_summary(summary: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    contexts: dict[str, dict[str, Any]] = {}
    for item in summary:
        context = str(item["context"])
        placeholder = str(item.get("placeholder", ""))
        entry = contexts.setdefault(
            context,
            {
                "prompt_templates": [],
                "ranked_prompt_templates": [],
                "prompt_templates_by_placeholder": {},
                "ranked_prompt_templates_by_placeholder": {},
            },
        )
        template = str(item["prompt_template"])
        ranked_item = {
            "prompt_template": template,
            "placeholder": placeholder,
            "mean_match_score": item["mean_match_score"],
            "exact_count": item["exact_count"],
            "last_resort_count": item["last_resort_count"],
        }
        entry["ranked_prompt_templates"].append(ranked_item)
        if len(entry["prompt_templates"]) < top_k:
            entry["prompt_templates"].append(template)
        if placeholder:
            placeholder_templates = entry["prompt_templates_by_placeholder"].setdefault(placeholder.casefold(), [])
            placeholder_ranked = entry["ranked_prompt_templates_by_placeholder"].setdefault(placeholder.casefold(), [])
            placeholder_ranked.append(ranked_item)
            if len(placeholder_templates) < top_k:
                placeholder_templates.append(template)
    return {
        "schema_version": "handgen_carrier_prompt_policy_v1",
        "contexts": contexts,
        "method": "Derived from context- and placeholder-specific carrier prompt OCR match score, exact count, and last-resort count.",
    }


def _write_default_source(out_dir: Path, *, max_lines: int) -> Path:
    source = out_dir / "mixed_eval_source.txt"
    source.write_text("\n".join(DEFAULT_MIXED_LINES[:max_lines]) + "\n", encoding="utf-8")
    return source


def _write_contact_sheet(manifest: dict[str, Any], path: Path) -> Path:
    Image, _ = require_pillow_numpy()
    image_paths: list[Path] = []
    output_png = manifest.get("outputs", {}).get("png")
    if output_png:
        image_paths.append(Path(output_png))
    for span in manifest.get("spans", []):
        crop = span.get("diffbrush", {}).get("styled_crop", {}).get("path")
        if crop:
            image_paths.append(Path(crop))
    thumbs = []
    for image_path in image_paths:
        if not image_path.exists():
            continue
        image = _image_on_white(Image.open(image_path))
        image.thumbnail((460, 180))
        thumbs.append((image_path.name, image.copy()))
    if not thumbs:
        raise RuntimeError("no images available for eval contact sheet")
    cols = 2
    cell_w = 500
    cell_h = 230
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, (_, thumb) in enumerate(thumbs):
        x = (idx % cols) * cell_w + 20
        y = (idx // cols) * cell_h + 20
        sheet.paste(thumb, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def _write_placeholder_contact_sheet(results: list[dict[str, Any]], path: Path) -> Path:
    Image, _ = require_pillow_numpy()
    thumbs = []
    for result in results:
        crop = (result.get("styled_crop") or {}).get("path")
        if not crop or not Path(crop).exists():
            continue
        image = _image_on_white(Image.open(crop))
        image.thumbnail((360, 120))
        label = f"{result['placeholder']} {result['match_score']:.2f} {result['route']}"
        thumbs.append((label, image.copy()))
    if not thumbs:
        raise RuntimeError("no placeholder benchmark images available for contact sheet")
    cols = 3
    cell_w = 400
    cell_h = 160
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, (_, thumb) in enumerate(thumbs):
        x = (idx % cols) * cell_w + 20
        y = (idx // cols) * cell_h + 20
        sheet.paste(thumb, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def _write_carrier_prompt_contact_sheet(results: list[dict[str, Any]], path: Path) -> Path:
    Image, _ = require_pillow_numpy()
    thumbs = []
    for result in results:
        crop = (result.get("styled_crop") or {}).get("path")
        if not crop or not Path(crop).exists():
            continue
        image = _image_on_white(Image.open(crop))
        image.thumbnail((360, 120))
        label = f"{result['context']} {result['match_score']:.2f} {result['prompt']}"
        thumbs.append((label, image.copy()))
    if not thumbs:
        raise RuntimeError("no carrier prompt benchmark images available")
    cols = 2
    cell_w = 430
    cell_h = 160
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, (_, thumb) in enumerate(thumbs):
        x = (idx % cols) * cell_w + 18
        y = (idx // cols) * cell_h + 18
        sheet.paste(thumb, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def _image_on_white(image: Any) -> Any:
    Image, _ = require_pillow_numpy()
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def load_eval_metrics(path: Path) -> dict[str, Any]:
    return read_json(path)
