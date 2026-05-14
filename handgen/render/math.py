from __future__ import annotations

import random
from pathlib import Path
from statistics import median
from typing import Any

from ..utils import data_uri, escape_xml, png_size, render_svg_to_png, sha_file, slugify
from .ink import INK_RGB, grayscale_ink_profile, ink_bbox_height, rgba_ink_profile, scanned_symbol_ink, varied_rgba_ink


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
    "target_space_advance": 17.0,
    "target_token_gap": 7.0,
}


def label_slug(label: str) -> str:
    return LABEL_SLUGS.get(label, slugify(label))


def math_style_profile(style_ref: Path, worksheet_manifest: dict[str, Any] | None = None, prose_height: float = 64.0) -> dict[str, Any]:
    style_ink = grayscale_ink_profile(style_ref) if style_ref.exists() else {}
    style_ink_h = int(style_ink.get("ink_bbox_height", prose_height))
    operator_heights: list[int] = []
    digit_heights: list[int] = []
    variable_heights: list[int] = []
    symbol_darkness: list[float] = []
    if worksheet_manifest:
        for label in ("<", ">", "+"):
            for crop in worksheet_manifest.get("glyphs", {}).get(label, []):
                if not crop.get("missing_crop") and crop.get("path"):
                    path = Path(crop["path"])
                    operator_heights.append(ink_bbox_height(path))
                    symbol_darkness.append(float(grayscale_ink_profile(path).get("mean_darkness", 0.0)))
        for label in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            for crop in worksheet_manifest.get("glyphs", {}).get(label, []):
                if not crop.get("missing_crop") and crop.get("path"):
                    path = Path(crop["path"])
                    digit_heights.append(ink_bbox_height(path))
                    symbol_darkness.append(float(grayscale_ink_profile(path).get("mean_darkness", 0.0)))
        for label in ("x", "y", "z", "i", "j", "k", "n", "m", "f", "g", "h"):
            for crop in worksheet_manifest.get("glyphs", {}).get(label, []):
                row_id = str(crop.get("row_id", ""))
                if not crop.get("missing_crop") and crop.get("path") and row_id in {"lowercase_letters", "uppercase_letters"}:
                    path = Path(crop["path"])
                    variable_heights.append(ink_bbox_height(path))
                    symbol_darkness.append(float(grayscale_ink_profile(path).get("mean_darkness", 0.0)))
    operator_ratio = median(operator_heights) / max(1, style_ink_h) if operator_heights else 0.9
    worksheet_darkness = median([value for value in symbol_darkness if value > 0.0]) if symbol_darkness else 0.0
    style_darkness = float(style_ink.get("mean_darkness", 0.58) or 0.58)
    alpha_scale = 0.62
    if worksheet_darkness:
        alpha_scale = max(0.35, min(1.45, 0.9 * style_darkness / worksheet_darkness))
    target_variable_h = prose_height
    target_digit_h = max(prose_height * 0.88, min(prose_height * 1.08, target_variable_h))
    target_operator_h = max(prose_height * 0.72, min(prose_height * 1.05, prose_height * operator_ratio))
    token_gap = max(4.0, min(10.0, prose_height * 0.11))
    return {
        **DEFAULT_MATH_STYLE,
        "style_reference": str(style_ref),
        "measured_style_ink": style_ink,
        "measured_style_ink_height": style_ink_h,
        "measured_operator_ink_heights": operator_heights,
        "measured_digit_ink_heights": digit_heights,
        "measured_math_variable_ink_heights": variable_heights,
        "measured_worksheet_symbol_darkness": round(worksheet_darkness, 4),
        "target_prose_height": round(prose_height, 2),
        "target_variable_height": round(target_variable_h, 2),
        "target_digit_height": round(target_digit_h, 2),
        "target_operator_height": round(target_operator_h, 2),
        "target_math_canvas_height": round(max(target_variable_h, target_digit_h, target_operator_h) * 1.22, 2),
        "symbol_alpha_scale": round(alpha_scale, 4),
        "target_space_advance": round(prose_height * 0.27, 2),
        "target_token_gap": round(token_gap, 2),
        "operator_to_prose_ratio": round(operator_ratio, 4),
        "method": "derived from style-reference ink profile and worksheet glyph ink measurements",
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
) -> dict[str, Any]:
    variable_bank = variable_bank or {}
    style_profile = style_profile or DEFAULT_MATH_STYLE
    render_profile = ink_match_profile or style_profile
    prepared: dict[tuple[Path, str], dict[str, Any]] = {}
    pieces: list[str] = []
    tokens: list[dict[str, Any]] = []
    x = 0.0
    target_h = float(style_profile["target_math_canvas_height"])
    baseline = target_h * 0.75
    for token_index, ch in enumerate(text):
        if ch == " ":
            width = float(style_profile["target_space_advance"]) * rng.uniform(0.88, 1.12)
            tokens.append({"label": "space", "source_character": ch, "x": round(x, 3), "width": round(width, 3)})
            x += width
            continue
        label = GREEK_CHAR_LABELS.get(ch, ch)
        source_ink: dict[str, Any] | None = None
        ink_variant: dict[str, Any] | None = None
        base_asset_path: str | None = None
        if ch in variable_bank:
            variable = variable_bank[label]
            asset = variable["styled_crop"]
            asset_path = Path(asset["path"])
            source_crop = asset["path"]
            route = variable.get("route", "diffbrush_context_component")
            token_target_h = min(
                float(style_profile["target_digit_height"] if ch.isdigit() else style_profile["target_variable_height"]),
                asset["height"] * float(style_profile["max_diffbrush_variable_upscale"]),
            )
        elif label not in glyph_bank:
            raise KeyError(f"missing worksheet glyph for math character {ch!r}")
        else:
            source = sorted(glyph_bank[label], key=lambda p: p.name)[rng.randrange(len(glyph_bank[label]))]
            source_ink = grayscale_ink_profile(source)
            base_alpha_scale = _alpha_scale_for_source(source_ink, render_profile, style_profile)
            carrier_ink_rgb = _ink_rgb_for_profile(render_profile)
            alpha_tag = f"{base_alpha_scale:.3f}".replace(".", "p")
            rgb_tag = "_".join(str(v) for v in carrier_ink_rgb)
            cache_key = (source, f"{alpha_tag}_{rgb_tag}")
            if cache_key not in prepared:
                asset_path = out_dir / "glyph_rgba" / f"{label_slug(label)}_{sha_file(source)[:8]}_a{alpha_tag}_rgb{rgb_tag}.png"
                prepared[cache_key] = scanned_symbol_ink(
                    source,
                    asset_path,
                    threshold=218,
                    pad=5,
                    alpha_scale=base_alpha_scale,
                    ink_rgb=carrier_ink_rgb,
                )
            base_asset = prepared[cache_key]
            variant = _variant_for_token(base_asset, out_dir, span_id, token_index, label, carrier_ink_rgb, rng)
            asset = variant
            asset_path = Path(variant["path"])
            base_asset_path = base_asset["path"]
            ink_variant = variant
            source_crop = str(source)
            route = "worksheet_glyph_bank_symbol"
            if label in {"<", ">", "+", "=", "*", "/", "_", "|"}:
                token_target_h = float(style_profile["target_operator_height"])
            elif ch.isascii() and ch.isalpha():
                token_target_h = float(style_profile["target_variable_height"])
            else:
                token_target_h = float(style_profile["target_digit_height"])
        scale = rng.uniform(0.94, 1.07)
        h = token_target_h * scale
        w = asset["width"] * (h / asset["height"])
        dx = rng.uniform(-1.4, 1.4)
        dy = rng.uniform(-3.2, 2.8)
        y = baseline - h * rng.uniform(0.72, 0.81) + dy
        angle = rng.uniform(-2.2, 2.2)
        opacity = rng.uniform(0.985, 1.0) if route == "worksheet_glyph_bank_symbol" else rng.uniform(0.88, 1.0)
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
                "base_asset": base_asset_path,
                "source_crop": source_crop,
                "route": route,
                "source_ink_profile": source_ink,
                "ink_variant": ink_variant,
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
        x += w + float(style_profile["target_token_gap"]) * rng.uniform(0.82, 1.18)
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
    visible_tokens = [token for token in tokens if token.get("label") != "space"]
    if visible_tokens:
        ink_bbox = {
            "x": round(min(float(token["x"]) for token in visible_tokens), 3),
            "y": round(min(float(token["y"]) for token in visible_tokens), 3),
            "right": round(max(float(token["x"]) + float(token["width"]) for token in visible_tokens), 3),
            "bottom": round(max(float(token["y"]) + float(token["height"]) for token in visible_tokens), 3),
        }
        ink_bbox["width"] = round(ink_bbox["right"] - ink_bbox["x"], 3)
        ink_bbox["height"] = round(ink_bbox["bottom"] - ink_bbox["y"], 3)
    else:
        ink_bbox = {"x": 0.0, "y": 0.0, "right": width, "bottom": target_h, "width": width, "height": target_h}
    return {
        "id": span_id,
        "route": "diffbrush_components_with_worksheet_symbol_math"
        if any(str(token.get("route", "")).startswith("diffbrush_") for token in tokens)
        else "worksheet_symbol_math",
        "text": text,
        "tokens": tokens,
        "paths": {"svg": str(svg_path), "png": str(png_path)},
        "width": png_w,
        "height": png_h,
        "ink_bbox": ink_bbox,
        "style_normalization": {
            "shared_ink_rgb": list(INK_RGB),
            "target_display_height": target_h,
            "style_profile": style_profile,
            "ink_match_profile": render_profile if ink_match_profile is not None else None,
            "transparent_background": True,
            "per_character_jitter": True,
        },
    }


def _alpha_scale_for_source(source_ink: dict[str, Any], render_profile: dict[str, Any], style_profile: dict[str, Any]) -> float:
    carrier_profile = render_profile.get("selected_diffbrush_ink_profile") or {}
    target_alpha = float(carrier_profile.get("alpha_p90", 0.0) or carrier_profile.get("mean_alpha", 0.0) or 0.0)
    source_darkness = float(source_ink.get("darkness_p75", 0.0) or source_ink.get("mean_darkness", 0.0) or 0.0)
    if target_alpha > 0.0 and source_darkness > 0.0:
        return max(0.35, min(2.25, target_alpha / source_darkness))
    return float(render_profile.get("symbol_alpha_scale", style_profile["symbol_alpha_scale"]))


def _ink_rgb_for_profile(render_profile: dict[str, Any]) -> list[int]:
    carrier_profile = render_profile.get("selected_diffbrush_ink_profile") or {}
    rgb = carrier_profile.get("mean_rgb") or list(INK_RGB)
    return [int(max(0, min(255, value))) for value in rgb[:3]]


def _variant_for_token(
    base_asset: dict[str, Any],
    out_dir: Path,
    span_id: str,
    token_index: int,
    label: str,
    ink_rgb: list[int],
    rng: random.Random,
) -> dict[str, Any]:
    alpha_jitter = rng.uniform(0.985, 1.015)
    shear_x = rng.uniform(-0.012, 0.012)
    thicken = 0
    variant_path = out_dir / "glyph_variants" / f"{span_id}_{token_index:02d}_{label_slug(label)}.png"
    variant = varied_rgba_ink(
        Path(base_asset["path"]),
        variant_path,
        alpha_scale=alpha_jitter,
        shear_x=shear_x,
        thicken=thicken,
        ink_rgb=ink_rgb,
    )
    return variant | {
        "base_asset": base_asset["path"],
        "base_ink_profile": rgba_ink_profile(Path(base_asset["path"])),
        "variant_policy": "per_token_alpha_shear_and_stroke_jitter",
    }
