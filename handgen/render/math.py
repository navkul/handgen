from __future__ import annotations

import hashlib
import math
import random
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from ..ingest.paper import components, paper_luminance
from ..ingest.worksheet import dark_mask, glyph_bank_from_manifest, slug
from ..utils import data_uri, escape_xml, render_svg_to_png, sha_file
from .ink import INK_RGB, grayscale_ink_profile

GREEK_CHAR_LABELS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "σ": "sigma",
}
ALIASES = {**GREEK_CHAR_LABELS, "−": "-", "·": ".", "⋅": ".", "′": "'", "″": '"'}
COMPOSITES = {
    "≤": ("<", "-"),
    "≥": (">", "-"),
    "±": ("+", "-"),
    "≠": ("=", "/"),
    "×": ("+",),
    "÷": ("-", "."),
    "^": ("<",),
    "∧": ("<",),
    "∨": ("<",),
    "∞": ("8",),
    "→": ("-", ">"),
    "←": ("-", "<"),
}
SCRIPTS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789", strict=True))
BODY_LETTERS = set("acemnorsuvwxz")
DESCENDERS = {"g", "j", "p", "q", "y", "beta", "gamma", "mu"}

DEFAULT_MATH_STYLE = {
    "target_prose_height": 64.0,
    "target_variable_height": 52.0,
    "target_digit_height": 50.0,
    "target_operator_height": 36.0,
    "target_body_height": 36.0,
    "target_math_canvas_height": 78.0,
    "max_diffbrush_variable_upscale": 1.45,
    "symbol_alpha_scale": 0.82,
    "target_space_advance": 19.0,
    "target_token_gap": 5.0,
    "target_letter_gap": 3.0,
}


def _parts(character: str, bank: dict[str, list[Path]], prose: bool) -> tuple[str, ...]:
    if bank.get(character):
        return (character,)
    if character in GREEK_CHAR_LABELS or not prose:
        if character in COMPOSITES:
            return COMPOSITES[character]
        return (SCRIPTS.get(character, ALIASES.get(character, character)),)
    return (character,)


def missing_glyphs(
    text: str,
    glyph_bank: dict[str, list[Path]],
    *,
    allow_digit_fallback: bool = False,
    prose: bool = False,
) -> list[str]:
    """List uncovered source characters, accounting for supported stroke compositions."""
    return sorted(
        {
            ch
            for ch in text
            if not ch.isspace()
            and not (allow_digit_fallback and not prose and ch in "0123456789")
            and any(not glyph_bank.get(label) for label in _parts(ch, glyph_bank, prose))
        }
    )


@lru_cache(maxsize=384)
def _glyph_sample(path: Path, mtime_ns: int, size: int) -> tuple[Image.Image, dict[str, Any], str]:
    # The v1 bank contains raw crops, including faint disconnected dot-grid ink.
    # Use strong connected strokes to recover their soft edges, then trim once.
    with Image.open(path) as image:
        gray = (
            paper_luminance(image)
            if "A" in image.getbands() or "transparency" in image.info
            else image.convert("L")
        )
    mask = dark_mask(gray)
    boxes = components(mask)
    if not boxes:
        raise ValueError(f"Worksheet glyph crop is blank: {path}")
    largest = max(box["area"] for box in boxes)
    keep = np.zeros_like(mask)
    for box in boxes:
        if box["area"] >= largest * 0.010:
            region = np.s_[box["y0"] : box["y1"], box["x0"] : box["x1"]]
            keep[region] |= mask[region]
    support = np.asarray(Image.fromarray(keep).convert("L").filter(ImageFilter.MaxFilter(3)))
    alpha = np.where(support > 0, 255 - np.asarray(gray), 0).astype(np.uint8)
    alpha[alpha < 8] = 0
    ys, xs = np.nonzero(alpha)
    alpha = alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    darkness = alpha[alpha > 0] / 255
    profile = {
        "path": str(path),
        "ink_bbox_width": alpha.shape[1],
        "ink_bbox_height": alpha.shape[0],
        "ink_density": round(float(np.mean(alpha > 0)), 4),
        "mean_darkness": round(float(darkness.mean()), 4),
        "darkness_p75": round(float(np.quantile(darkness, 0.75)), 4),
    }
    return Image.fromarray(alpha), profile, sha_file(path)


def _sample(path: Path) -> tuple[Image.Image, dict[str, Any], str]:
    stat = path.stat()
    return _glyph_sample(path, stat.st_mtime_ns, stat.st_size)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def math_style_profile(
    style_ref: Path,
    worksheet_manifest: dict[str, Any] | None = None,
    prose_height: float = 64.0,
) -> dict[str, Any]:
    style_ink = grayscale_ink_profile(style_ref) if style_ref.exists() else {}
    samples = {
        label: [_sample(path)[1] for path in paths]
        for label, paths in glyph_bank_from_manifest(worksheet_manifest or {}).items()
    }

    def heights(labels: str | set[str]) -> list[int]:
        return [s["ink_bbox_height"] for label in sorted(labels) for s in samples.get(label, [])]

    caps = heights("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = heights("0123456789")
    body = heights(BODY_LETTERS)
    operators = heights("<>+")
    darkness = [s["mean_darkness"] for items in samples.values() for s in items]
    sheet_darkness = median(darkness) if darkness else 0.0
    style_darkness = style_ink.get("mean_darkness") or 0.8
    cap_h = prose_height * 0.81
    reference = float(median(caps or digits)) if caps or digits else (median(body) / 0.70 if body else 0.0)
    body_ratio = _clamp(median(body) / reference, 0.50, 0.86) if body and reference else 0.70
    op_ratio = _clamp(median(operators) / reference, 0.50, 0.90) if operators and reference else 0.66
    digit_ratio = _clamp(median(digits) / reference, 0.75, 1.10) if digits and reference else 0.94
    return {
        **DEFAULT_MATH_STYLE,
        "style_reference": str(style_ref),
        "measured_style_ink": style_ink,
        "measured_style_ink_height": style_ink.get("ink_bbox_height"),
        "measured_operator_ink_heights": operators,
        "measured_digit_ink_heights": digits,
        "measured_math_variable_ink_heights": heights("abcdefghijklmnopqrstuvwxyz"),
        "measured_worksheet_symbol_darkness": round(sheet_darkness, 4),
        "worksheet_reference_height": reference,
        "target_prose_height": float(prose_height),
        "target_variable_height": round(cap_h, 3),
        "target_body_height": round(body_ratio * cap_h, 3),
        "target_digit_height": round(digit_ratio * cap_h, 3),
        "target_operator_height": round(op_ratio * cap_h, 3),
        "target_math_canvas_height": round(prose_height * 1.22, 3),
        "symbol_alpha_scale": round(_clamp(style_darkness / sheet_darkness, 0.44, 1.50), 4)
        if sheet_darkness
        else 0.82,
        "target_space_advance": round(prose_height * 0.31, 3),
        "target_token_gap": round(prose_height * 0.08, 3),
        "target_letter_gap": round(prose_height * 0.054, 3),
        "operator_to_prose_ratio": round(op_ratio * cap_h / prose_height, 4),
        "method": "Worksheet scale measured from the writer's capitals, digits and x-height letters",
    }


def glyph_metrics(
    label: str, source_ink: dict[str, Any], style_profile: dict[str, Any]
) -> tuple[float, float]:
    """Return display ink height and baseline offset measured from the top of that ink."""
    cap = float(style_profile["target_variable_height"])
    body = float(style_profile.get("target_body_height", cap * 0.70))
    width = max(1, source_ink["ink_bbox_width"])
    height = max(1, source_ink["ink_bbox_height"])
    reference = style_profile.get("worksheet_reference_height") or 0
    measured = cap * height / reference if reference else None
    if label in BODY_LETTERS or label in {"alpha", "pi", "sigma"}:
        h = _clamp(measured or body, 0.70 * body, 1.20 * body)
    elif label in {"-", "−", "_", "=", "<", ">", "+", "*"}:
        natural_width = cap * width / reference if reference else cap * 0.67
        # Scale a horizontal mark by its width to retain its actual stroke height.
        h = _clamp(_clamp(natural_width, 0.34 * cap, 1.02 * cap) * height / width, 0.035 * cap, 0.87 * cap)
    elif label in {".", "·", "⋅"}:
        h = _clamp(measured or 0.14 * cap, 0.07 * cap, 0.23 * cap)
    elif label in {",", "'", '"', "′", "″"}:
        h = _clamp(measured or 0.30 * cap, 0.15 * cap, 0.44 * cap)
    elif label in {":", ";"}:
        h = _clamp(measured or 0.54 * cap, 0.32 * cap, 0.73 * cap)
    elif label == "^":
        h = cap * 0.34
    elif label in COMPOSITES:
        h = _clamp(0.80 * cap * height / width, 0.5 * cap, 0.97 * cap)
    elif label in "0123456789" or label in SCRIPTS:
        h = _clamp(measured or style_profile["target_digit_height"], 0.73 * cap, 1.12 * cap)
    else:
        h = _clamp(measured or cap, 0.67 * cap, (1.28 if label in DESCENDERS else 1.18) * cap)

    if label in SCRIPTS:
        h *= 0.63
        return h, h + (body * 0.66 if label in "⁰¹²³⁴⁵⁶⁷⁸⁹" else -body * 0.37)
    if label in {"^", "′", "″"}:
        return h, cap
    if label in {"-", "−", "·", "⋅", "+", "=", "<", ">", "*", "/"} or label in COMPOSITES:
        return h, (body + h) / 2
    if label in {"(", ")", "[", "]", "{", "}", "|"}:
        return h, (cap * 0.80 + h) / 2
    if label in {"'", '"'}:
        return h, cap
    if label == "_":
        return h, -0.065 * cap
    if label in {",", ";"}:
        return h, h - min(0.16 * cap, 0.52 * h)
    if label in DESCENDERS:
        return h, max(0.50 * h, h - min(0.27 * cap, 0.31 * h))
    return h, h


def _scaled_ink(source: Image.Image, *, width: int | None = None, height: int | None = None) -> Image.Image:
    if width is None:
        width = max(1, round(source.width * height / source.height))
    if height is None:
        height = max(1, round(source.height * width / source.width))
    return source.resize((width, height), Image.Resampling.LANCZOS)


def _compose_mark(ch: str, masks: list[Image.Image]) -> Image.Image:
    if ch in {"^", "∧", "∨", "∞"}:
        turn = Image.Transpose.ROTATE_90 if ch in {"∨", "∞"} else Image.Transpose.ROTATE_270
        return masks[0].transpose(turn)
    if ch == "×":
        rotated = masks[0].rotate(45, resample=Image.Resampling.BICUBIC, expand=True)
        return rotated.crop(rotated.getbbox())
    first = _scaled_ink(masks[0], width=100)
    if ch == "≠":
        slash = _scaled_ink(masks[1], height=max(round(first.height * 1.52), 105))
        result = Image.new("L", (max(first.width, slash.width), max(first.height, slash.height)), 0)
        result.paste(first, ((result.width - first.width) // 2, (result.height - first.height) // 2))
        overlay = Image.new("L", result.size, 0)
        overlay.paste(slash, ((result.width - slash.width) // 2, (result.height - slash.height) // 2))
        return ImageChops.lighter(result, overlay)
    if ch in {"→", "←"}:
        head = _scaled_ink(masks[1], height=44)
        # The shaft meets the pointed end of the writer's angle.
        arrow = Image.new("L", (first.width + 6, max(44, first.height)), 0)
        shaft_start = 6 if ch == "←" else 0
        arrow.paste(first, (shaft_start, (arrow.height - first.height) // 2))
        overlay = Image.new("L", arrow.size, 0)
        head_start = 0 if ch == "←" else arrow.width - head.width
        overlay.paste(head, (head_start, (arrow.height - head.height) // 2))
        return ImageChops.lighter(arrow, overlay)
    if ch == "÷":
        dot = _scaled_ink(masks[1], width=19)
        layers = [dot, first, dot]
    else:
        layers = [first, _scaled_ink(masks[1], width=100)]
    gap = 12
    result = Image.new("L", (100, sum(layer.height for layer in layers) + gap * (len(layers) - 1)), 0)
    y = 0
    for layer in layers:
        result.paste(layer, ((100 - layer.width) // 2, y))
        y += layer.height + gap
    return result


def _worksheet_asset(
    ch: str,
    labels: tuple[str, ...],
    paths: tuple[Path, ...],
    out_dir: Path,
    render_profile: dict[str, Any],
    style_profile: dict[str, Any],
) -> dict[str, Any]:
    samples = [_sample(path) for path in paths]
    masks = [sample[0] for sample in samples]
    composed = ch in COMPOSITES and labels != (ch,)
    mask = _compose_mark(ch, masks) if composed else masks[0]
    profile = dict(samples[0][1], ink_bbox_width=mask.width, ink_bbox_height=mask.height)
    carrier = render_profile.get("selected_diffbrush_ink_profile") or {}
    rgb = tuple(int(_clamp(v, 0, 255)) for v in (carrier.get("mean_rgb") or INK_RGB)[:3])
    target_alpha = carrier.get("alpha_p90") or carrier.get("mean_alpha")
    alpha_scale = (
        _clamp(target_alpha / profile["darkness_p75"], 0.35, 2.25)
        if target_alpha
        else render_profile.get("symbol_alpha_scale", style_profile["symbol_alpha_scale"])
    )
    # The same normalized raster is reused across words. Per-use variation lives
    # in the SVG transform, so a long page doesn't allocate a file per letter.
    alpha = mask.point(lambda v: min(255, round(v * alpha_scale)))
    signature = hashlib.sha256(alpha.tobytes())
    signature.update(str((alpha.size, rgb)).encode())
    path = out_dir / "glyph_rgba" / f"{slug(ch if composed else labels[0])}_{signature.hexdigest()[:16]}.png"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        rgba = Image.new("RGBA", mask.size, rgb + (0,))
        rgba.putalpha(alpha)
        rgba.save(path)
    return {
        "path": str(path),
        "width": mask.width,
        "height": mask.height,
        "source_ink_profile": profile,
        "source_crop": str(paths[0]),
        "source_crops": [str(p) for p in paths],
        "component_labels": list(labels),
        "source_sha256": [sample[2] for sample in samples],
        "route": "worksheet_stroke_composition" if composed else "worksheet_glyph_bank_symbol",
    }


def render_math_span(
    span_id: str,
    text: str,
    glyph_bank: dict[str, list[Path]],
    out_dir: Path,
    rng: random.Random,
    *,
    variable_bank: dict[str, dict[str, Any]] | None = None,
    style_profile: dict[str, Any] | None = None,
    ink_match_profile: dict[str, Any] | None = None,
    prose: bool = False,
) -> dict[str, Any]:
    variable_bank = variable_bank or {}
    style = style_profile or DEFAULT_MATH_STYLE
    render_profile = ink_match_profile or style
    cached: dict[tuple[Any, ...], dict[str, Any]] = {}
    tokens: list[dict[str, Any]] = []
    canvas_h = float(style["target_math_canvas_height"])
    baseline = canvas_h * 0.73
    gap = float(style.get("target_letter_gap", 3) if prose else style["target_token_gap"])
    pen = 0.0
    left = top = 0.0
    right, bottom = 0.0, canvas_h
    for i, ch in enumerate(text):
        if ch.isspace():
            width = float(style["target_space_advance"]) * (4 if ch == "\t" else 1)
            tokens.append({"label": "space", "source_character": ch, "x": pen, "width": width})
            pen += width
            continue
        if i and not text[i - 1].isspace():
            pen += gap * rng.uniform(0.88, 1.12)
        labels = _parts(ch, glyph_bank, prose)
        if all(glyph_bank.get(label) for label in labels):
            paths = tuple(rng.choice(sorted(glyph_bank[label])) for label in labels)
            key = (ch, paths)
            if key not in cached:
                cached[key] = _worksheet_asset(ch, labels, paths, out_dir, render_profile, style)
            asset = cached[key]
            metric_label = GREEK_CHAR_LABELS.get(ch, ch)
            target_h, from_top = glyph_metrics(metric_label, asset["source_ink_profile"], style)
        elif ch in "0123456789" and ch in variable_bank and not prose:
            fallback = variable_bank[ch]
            raw = fallback["styled_crop"]
            asset = dict(
                raw,
                source_crop=raw["path"],
                source_crops=[raw["path"]],
                component_labels=[ch],
                source_ink_profile=None,
                source_sha256=[raw.get("sha256")],
                route=fallback.get("route", "diffbrush_context_component"),
            )
            target_h = min(
                float(style["target_digit_height"]), raw["height"] * style["max_diffbrush_variable_upscale"]
            )
            from_top = target_h
        else:
            raise KeyError(f"Missing worksheet glyph for {ch!r} (requires {', '.join(labels)})")
        scale = rng.uniform(0.978, 1.025)
        h = target_h * scale
        w = asset["width"] * h / asset["height"]
        jitter = style["target_prose_height"] * 0.016
        token_baseline = baseline + rng.uniform(-jitter, jitter)
        x, y = pen, token_baseline - from_top * scale
        angle = rng.uniform(-1.6, 1.6)
        shear = rng.uniform(-0.015, 0.015)
        sine, cosine = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        half_w = (w * abs(cosine) + h * abs(cosine * shear - sine)) / 2
        half_h = (w * abs(sine) + h * abs(sine * shear + cosine)) / 2
        left, top = min(left, x + w / 2 - half_w), min(top, y + h / 2 - half_h)
        right, bottom = max(right, x + w / 2 + half_w), max(bottom, y + h / 2 + half_h)
        tokens.append(
            {
                "label": ch if ch in COMPOSITES or ch in SCRIPTS else labels[0],
                "source_character": ch,
                "asset": asset["path"],
                "base_asset": asset["path"],
                "route": asset["route"],
                "source_crop": asset["source_crop"],
                "source_crops": asset["source_crops"],
                "component_labels": asset["component_labels"],
                "source_sha256": asset["source_sha256"],
                "source_ink_profile": asset["source_ink_profile"],
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "baseline": token_baseline,
                "target_height_before_jitter": round(target_h, 3),
                "baseline_from_ink_top": round(from_top * scale, 3),
                "scale": round(scale, 5),
                "rotation_deg": round(angle, 5),
                "shear_x": round(shear, 5),
                "opacity": round(rng.uniform(0.963, 1.0), 4),
            }
        )
        pen += w
    # Allow antialiasing and the transformed corners at either edge of the run.
    dx, dy = 1 - math.floor(left), 1 - math.floor(top)
    width = max(1, math.ceil(max(pen, right)) + dx + 1)
    height = math.ceil(bottom) + dy + 1
    pieces = []
    images: dict[str, str] = {}
    for token in tokens:
        token["x"] = round(token["x"] + dx, 3)
        if token["label"] == "space":
            continue
        token["y"] = round(token["y"] + dy, 3)
        token["baseline"] = round(token["baseline"] + dy, 3)
        x, y, w, h = token["x"], token["y"], token["width"], token["height"]
        cx, cy = x + w / 2, y + h / 2
        skew_degrees = math.degrees(math.atan(token["shear_x"]))
        if token["asset"] not in images:
            images[token["asset"]] = data_uri(Path(token["asset"]))
        pieces.append(
            f'<image href="{images[token["asset"]]}" x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" '
            f'opacity="{token["opacity"]:.4f}" transform="translate({cx:.3f} {cy:.3f}) rotate({token["rotation_deg"]}) '
            f'skewX({skew_degrees:.4f}) translate({-cx:.3f} {-cy:.3f})"/>'
        )
        token["width"], token["height"] = round(w, 3), round(h, 3)
    svg_path = out_dir / ("worksheet_prose" if prose else "math") / f"{span_id}.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                f"<title>{escape_xml(text)}</title>",
                *pieces,
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    png_path = render_svg_to_png(svg_path)
    with Image.open(png_path) as png:
        bounds = png.convert("RGBA").getchannel("A").getbbox() or (0, 0, 0, 0)
    x0, y0, x1, y1 = bounds
    generated_digit = any(str(t.get("route", "")).startswith("diffbrush_") for t in tokens)
    route = "diffbrush_components_with_worksheet_symbol_math" if generated_digit else "worksheet_symbol_math"
    return {
        "id": span_id,
        "route": "worksheet_prose" if prose else route,
        "text": text,
        "tokens": tokens,
        "paths": {"svg": str(svg_path), "png": str(png_path)},
        "width": width,
        "height": height,
        "baseline": round(baseline + dy, 3),
        "ink_bbox": {"x": x0, "y": y0, "right": x1, "bottom": y1, "width": x1 - x0, "height": y1 - y0},
        "style_normalization": {
            "shared_ink_rgb": list(INK_RGB),
            "target_display_height": canvas_h,
            "style_profile": style,
            "ink_match_profile": ink_match_profile,
            "transparent_background": True,
            "per_character_jitter": True,
        },
    }
