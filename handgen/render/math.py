from __future__ import annotations

import random
from pathlib import Path
from statistics import median
from typing import Any

from ..utils import data_uri, escape_xml, png_size, render_svg_to_png, sha_file, slugify
from .ink import INK_RGB, ink_bbox_height, scanned_symbol_ink


LABEL_SLUGS = {
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


DEFAULT_MATH_STYLE = {
    "target_prose_height": 64.0,
    "target_variable_height": 64.0,
    "target_digit_height": 64.0,
    "target_operator_height": 56.0,
    "target_math_canvas_height": 78.0,
    "max_diffbrush_variable_upscale": 1.45,
    "symbol_alpha_scale": 0.62,
}


def label_slug(label: str) -> str:
    return LABEL_SLUGS.get(label, slugify(label))


def math_style_profile(style_ref: Path, worksheet_manifest: dict[str, Any] | None = None, prose_height: float = 64.0) -> dict[str, Any]:
    style_ink_h = ink_bbox_height(style_ref) if style_ref.exists() else int(prose_height)
    operator_heights: list[int] = []
    digit_heights: list[int] = []
    variable_heights: list[int] = []
    if worksheet_manifest:
        for label in ("<", ">", "+"):
            for crop in worksheet_manifest.get("glyphs", {}).get(label, []):
                if not crop.get("missing_crop") and crop.get("path"):
                    operator_heights.append(ink_bbox_height(Path(crop["path"])))
        for label in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            for crop in worksheet_manifest.get("glyphs", {}).get(label, []):
                if not crop.get("missing_crop") and crop.get("path"):
                    digit_heights.append(ink_bbox_height(Path(crop["path"])))
        for label in ("x", "y", "z", "i", "j", "k", "n", "m", "f", "g", "h"):
            for crop in worksheet_manifest.get("glyphs", {}).get(label, []):
                if not crop.get("missing_crop") and crop.get("path") and "math_letters" in crop["path"]:
                    variable_heights.append(ink_bbox_height(Path(crop["path"])))
    operator_ratio = median(operator_heights) / max(1, style_ink_h) if operator_heights else 0.9
    target_variable_h = prose_height
    target_digit_h = max(prose_height * 0.88, min(prose_height * 1.08, target_variable_h))
    target_operator_h = max(prose_height * 0.72, min(prose_height * 1.05, prose_height * operator_ratio))
    return {
        **DEFAULT_MATH_STYLE,
        "style_reference": str(style_ref),
        "measured_style_ink_height": style_ink_h,
        "measured_operator_ink_heights": operator_heights,
        "measured_digit_ink_heights": digit_heights,
        "measured_math_variable_ink_heights": variable_heights,
        "target_prose_height": round(prose_height, 2),
        "target_variable_height": round(target_variable_h, 2),
        "target_digit_height": round(target_digit_h, 2),
        "target_operator_height": round(target_operator_h, 2),
        "target_math_canvas_height": round(max(target_variable_h, target_digit_h, target_operator_h) * 1.22, 2),
        "operator_to_prose_ratio": round(operator_ratio, 4),
        "method": "derived from style-reference and worksheet glyph ink bounding boxes",
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
) -> dict[str, Any]:
    variable_bank = variable_bank or {}
    style_profile = style_profile or DEFAULT_MATH_STYLE
    prepared: dict[Path, dict[str, Any]] = {}
    pieces: list[str] = []
    tokens: list[dict[str, Any]] = []
    x = 0.0
    target_h = float(style_profile["target_math_canvas_height"])
    baseline = target_h * 0.75
    for ch in text:
        if ch == " ":
            width = style_profile["target_variable_height"] * rng.uniform(0.16, 0.25)
            tokens.append({"label": "space", "x": round(x, 3), "width": round(width, 3)})
            x += width
            continue
        label = GREEK_CHAR_LABELS.get(ch, ch)
        if ch.isascii() and ch.isalpha() and ch in variable_bank:
            variable = variable_bank[label]
            asset = variable["styled_crop"]
            asset_path = Path(asset["path"])
            source_crop = asset["path"]
            route = "diffbrush_variable_context_component"
            token_target_h = min(
                float(style_profile["target_variable_height"]),
                asset["height"] * float(style_profile["max_diffbrush_variable_upscale"]),
            )
        elif label not in glyph_bank:
            raise KeyError(f"missing worksheet glyph for math character {ch!r}")
        else:
            source = sorted(glyph_bank[label], key=lambda p: p.name)[rng.randrange(len(glyph_bank[label]))]
            if source not in prepared:
                asset_path = out_dir / "glyph_rgba" / f"{label_slug(label)}_{sha_file(source)[:8]}.png"
                prepared[source] = scanned_symbol_ink(
                    source,
                    asset_path,
                    threshold=218,
                    pad=5,
                    alpha_scale=float(style_profile["symbol_alpha_scale"]),
                )
            asset = prepared[source]
            asset_path = Path(asset["path"])
            source_crop = str(source)
            route = "worksheet_glyph_bank_symbol"
            token_target_h = (
                float(style_profile["target_operator_height"])
                if label in {"<", ">", "+", "=", "*", "/", "_", "|"}
                else float(style_profile["target_digit_height"])
            )
        scale = rng.uniform(0.94, 1.07)
        h = token_target_h * scale
        w = asset["width"] * (h / asset["height"])
        dx = rng.uniform(-1.4, 1.4)
        dy = rng.uniform(-3.2, 2.8)
        y = baseline - h * rng.uniform(0.72, 0.81) + dy
        angle = rng.uniform(-2.2, 2.2)
        opacity = rng.uniform(0.88, 1.0)
        cx = x + dx + w / 2
        cy = y + h / 2
        pieces.append(
            f'<image href="{data_uri(asset_path)}" x="{x + dx:.3f}" y="{y:.3f}" '
            f'width="{w:.3f}" height="{h:.3f}" opacity="{opacity:.3f}" '
            f'transform="rotate({angle:.3f} {cx:.3f} {cy:.3f})"/>'
        )
        tokens.append(
            {
                "label": label,
                "source_character": ch,
                "asset": str(asset_path),
                "source_crop": source_crop,
                "route": route,
                "x": round(x + dx, 3),
                "y": round(y, 3),
                "width": round(w, 3),
                "height": round(h, 3),
                "target_height_before_jitter": round(token_target_h, 3),
                "scale": round(scale, 4),
                "rotation_deg": round(angle, 4),
                "opacity": round(opacity, 4),
            }
        )
        x += w + float(style_profile["target_variable_height"]) * rng.uniform(0.08, 0.15)
    width = max(1.0, x)
    svg_path = out_dir / "math" / f"{span_id}.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}" height="{target_h:.3f}" viewBox="0 0 {width:.3f} {target_h:.3f}">',
                f"<title>{escape_xml(span_id)}</title>",
                *pieces,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    png_path = render_svg_to_png(svg_path)
    png_w, png_h = png_size(png_path)
    return {
        "id": span_id,
        "route": "diffbrush_variables_with_worksheet_symbol_math",
        "text": text,
        "tokens": tokens,
        "paths": {"svg": str(svg_path), "png": str(png_path)},
        "width": png_w,
        "height": png_h,
        "style_normalization": {
            "shared_ink_rgb": list(INK_RGB),
            "target_display_height": target_h,
            "style_profile": style_profile,
            "transparent_background": True,
            "per_character_jitter": True,
        },
    }
