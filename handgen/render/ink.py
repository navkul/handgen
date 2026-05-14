from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import require_pillow_numpy, sha_file


INK_RGB = (31, 36, 43)


def transparent_ink(
    src: Path,
    dst: Path,
    threshold: int = 244,
    pad: int = 4,
    alpha_scale: float = 1.6,
    ink_mask: Any | None = None,
) -> dict[str, Any]:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("L")
    arr = np.asarray(image)
    mask = ink_mask if ink_mask is not None else arr < threshold
    ys, xs = np.where(mask)
    if len(xs):
        box = (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
        image = image.crop(box)
        arr = np.asarray(image)
        mask = mask[box[1] : box[3], box[0] : box[2]]
    alpha = np.clip((255 - arr) * alpha_scale, 0, 255).astype("uint8")
    alpha = np.where(mask, alpha, 0).astype("uint8")
    alpha = np.where(alpha > 10, alpha, 0).astype("uint8")
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype="uint8")
    rgba[..., 0] = INK_RGB[0]
    rgba[..., 1] = INK_RGB[1]
    rgba[..., 2] = INK_RGB[2]
    rgba[..., 3] = alpha
    out = Image.fromarray(rgba, mode="RGBA")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return {
        "path": str(dst),
        "width": out.width,
        "height": out.height,
        "sha256": sha_file(dst),
        "ink_rgb": list(INK_RGB),
        "alpha_mode": "threshold_alpha_scaled",
        "alpha_scale": alpha_scale,
    }


def ink_bbox_height(src: Path, threshold: int = 224) -> int:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("L")
    arr = np.asarray(image)
    ys, _ = np.where(arr < threshold)
    if len(ys) == 0:
        return image.height
    return int(ys.max() - ys.min() + 1)


def grayscale_ink_profile(src: Path, threshold: int = 224) -> dict[str, Any]:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("L")
    arr = np.asarray(image)
    mask = arr < threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "path": str(src),
            "ink_bbox_height": image.height,
            "ink_bbox_width": image.width,
            "ink_density": 0.0,
            "mean_darkness": 0.0,
        }
    bbox = arr[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    bbox_mask = bbox < threshold
    darkness = (255 - bbox[bbox_mask]) / 255.0
    return {
        "path": str(src),
        "ink_bbox_height": int(ys.max() - ys.min() + 1),
        "ink_bbox_width": int(xs.max() - xs.min() + 1),
        "ink_density": round(float(bbox_mask.sum() / max(1, bbox_mask.size)), 4),
        "mean_darkness": round(float(darkness.mean()) if darkness.size else 0.0, 4),
        "darkness_p75": round(float(np.quantile(darkness, 0.75)) if darkness.size else 0.0, 4),
        "darkness_p90": round(float(np.quantile(darkness, 0.90)) if darkness.size else 0.0, 4),
    }


def rgba_ink_profile(src: Path, alpha_threshold: int = 10) -> dict[str, Any]:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("RGBA")
    rgb = np.asarray(image)[..., :3]
    alpha = np.asarray(image.getchannel("A"))
    mask = alpha > alpha_threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "path": str(src),
            "ink_bbox_height": image.height,
            "ink_bbox_width": image.width,
            "ink_density": 0.0,
            "mean_alpha": 0.0,
            "alpha_p75": 0.0,
            "alpha_p90": 0.0,
            "alpha_p95": 0.0,
            "mean_rgb": list(INK_RGB),
        }
    bbox_alpha = alpha[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    bbox_mask = bbox_alpha > alpha_threshold
    alpha_values = bbox_alpha[bbox_mask] / 255.0
    rgb_values = rgb[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1][bbox_mask]
    return {
        "path": str(src),
        "ink_bbox_height": int(ys.max() - ys.min() + 1),
        "ink_bbox_width": int(xs.max() - xs.min() + 1),
        "ink_density": round(float(bbox_mask.sum() / max(1, bbox_mask.size)), 4),
        "mean_alpha": round(float(alpha_values.mean()) if alpha_values.size else 0.0, 4),
        "alpha_p75": round(float(np.quantile(alpha_values, 0.75)) if alpha_values.size else 0.0, 4),
        "alpha_p90": round(float(np.quantile(alpha_values, 0.90)) if alpha_values.size else 0.0, 4),
        "alpha_p95": round(float(np.quantile(alpha_values, 0.95)) if alpha_values.size else 0.0, 4),
        "mean_rgb": [int(round(v)) for v in rgb_values.mean(axis=0).tolist()] if len(rgb_values) else list(INK_RGB),
    }


def soft_diffbrush_ink(src: Path, dst: Path, threshold: int = 236, pad: int = 3, alpha_scale: float = 1.0) -> dict[str, Any]:
    return transparent_ink(src, dst, threshold=threshold, pad=pad, alpha_scale=alpha_scale) | {
        "alpha_mode": "soft_diffbrush_alpha"
    }


def source_locked_token_ink(src: Path, dst: Path, threshold: int = 236, pad: int = 3, alpha_scale: float = 1.0) -> dict[str, Any]:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("L")
    arr = np.asarray(image)
    mask = arr < threshold
    keep_mask, clusters = _first_word_cluster_mask(mask)
    return transparent_ink(src, dst, threshold=threshold, pad=pad, alpha_scale=alpha_scale, ink_mask=keep_mask) | {
        "alpha_mode": "source_locked_token_alpha",
        "cluster_policy": "first_word_cluster",
        "detected_cluster_count": len(clusters),
    }


def scanned_symbol_ink(
    src: Path,
    dst: Path,
    threshold: int = 218,
    pad: int = 4,
    alpha_scale: float = 0.62,
    ink_rgb: tuple[int, int, int] | list[int] = INK_RGB,
) -> dict[str, Any]:
    asset = transparent_ink(src, dst, threshold=threshold, pad=pad, alpha_scale=alpha_scale)
    if tuple(ink_rgb) != INK_RGB:
        asset = recolor_rgba_ink(Path(asset["path"]), dst, ink_rgb)
    return asset | {
        "alpha_mode": "scanned_symbol_alpha_scaled"
    }


def recolor_rgba_ink(src: Path, dst: Path, ink_rgb: tuple[int, int, int] | list[int]) -> dict[str, Any]:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("RGBA")
    arr = np.asarray(image).copy()
    rgb = tuple(int(max(0, min(255, value))) for value in ink_rgb)
    arr[..., 0] = rgb[0]
    arr[..., 1] = rgb[1]
    arr[..., 2] = rgb[2]
    out = Image.fromarray(arr, mode="RGBA")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return {
        "path": str(dst),
        "width": out.width,
        "height": out.height,
        "sha256": sha_file(dst),
        "ink_rgb": list(rgb),
        "alpha_mode": "recolored_rgba",
    }


def varied_rgba_ink(
    src: Path,
    dst: Path,
    *,
    alpha_scale: float = 1.0,
    shear_x: float = 0.0,
    thicken: int = 0,
    ink_rgb: tuple[int, int, int] | list[int] | None = None,
) -> dict[str, Any]:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("RGBA")
    arr = np.asarray(image).copy()
    arr[..., 3] = np.clip(arr[..., 3].astype("float32") * alpha_scale, 0, 255).astype("uint8")
    if ink_rgb is not None:
        rgb = tuple(int(max(0, min(255, value))) for value in ink_rgb)
        arr[..., 0] = rgb[0]
        arr[..., 1] = rgb[1]
        arr[..., 2] = rgb[2]
    image = Image.fromarray(arr, mode="RGBA")
    if thicken:
        from PIL import ImageFilter  # type: ignore

        channels = image.split()
        alpha = channels[3]
        filt = ImageFilter.MaxFilter(3) if thicken > 0 else ImageFilter.MinFilter(3)
        image = Image.merge("RGBA", (*channels[:3], alpha.filter(filt)))
    if abs(shear_x) > 0.001:
        extra = int(abs(shear_x) * image.height) + 4
        new_w = image.width + extra
        offset = extra // 2 if shear_x > 0 else 0
        image = image.transform(
            (new_w, image.height),
            Image.Transform.AFFINE,
            (1, -shear_x, offset, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
        )
    image = _trim_rgba_alpha(image, pad=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)
    return {
        "path": str(dst),
        "width": image.width,
        "height": image.height,
        "sha256": sha_file(dst),
        "variant_of": str(src),
        "alpha_scale": round(alpha_scale, 4),
        "shear_x": round(shear_x, 4),
        "thicken": thicken,
        "ink_rgb": list(ink_rgb) if ink_rgb is not None else None,
        "alpha_mode": "per_use_varied_rgba",
    }


def trim_rgba_right(src: Path, dst: Path, right_x: int, pad: int = 4) -> dict[str, Any]:
    Image, _ = require_pillow_numpy()
    image = Image.open(src).convert("RGBA")
    crop_right = max(1, min(image.width, int(right_x) + pad))
    out = image.crop((0, 0, crop_right, image.height))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return {
        "path": str(dst),
        "width": out.width,
        "height": out.height,
        "sha256": sha_file(dst),
        "trimmed_from": str(src),
        "trim_right_x": crop_right,
    }


def _trim_rgba_alpha(image: Any, pad: int = 1) -> Any:
    _, np = require_pillow_numpy()
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return image
    return image.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
    )


def _first_word_cluster_mask(mask: Any) -> tuple[Any, list[dict[str, Any]]]:
    _, np = require_pillow_numpy()
    components = _mask_components(mask)
    if not components:
        return mask, []
    heights = sorted(comp["height"] for comp in components)
    median_h = heights[len(heights) // 2]
    max_gap = max(18, int(median_h * 0.9))
    clusters: list[dict[str, Any]] = []
    for comp in sorted(components, key=lambda item: item["x0"]):
        if not clusters or comp["x0"] - clusters[-1]["x1"] > max_gap:
            clusters.append({"x0": comp["x0"], "x1": comp["x1"], "components": [comp]})
        else:
            clusters[-1]["x1"] = max(clusters[-1]["x1"], comp["x1"])
            clusters[-1]["components"].append(comp)
    keep = np.zeros(mask.shape, dtype=bool)
    for comp in clusters[0]["components"]:
        keep |= comp["mask"]
    return keep, clusters


def _mask_components(mask: Any) -> list[dict[str, Any]]:
    _, np = require_pillow_numpy()
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[dict[str, Any]] = []
    for y in range(h):
        for x0 in np.where(mask[y] & ~seen[y])[0].tolist():
            if seen[y, x0] or not mask[y, x0]:
                continue
            stack = [(x0, y)]
            seen[y, x0] = True
            xs: list[int] = []
            ys: list[int] = []
            comp_mask = np.zeros(mask.shape, dtype=bool)
            while stack:
                x, yy = stack.pop()
                xs.append(x)
                ys.append(yy)
                comp_mask[yy, x] = True
                for nx in (x - 1, x, x + 1):
                    for ny in (yy - 1, yy, yy + 1):
                        if nx == x and ny == yy:
                            continue
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            if len(xs) >= 8:
                out.append(
                    {
                        "x0": min(xs),
                        "x1": max(xs) + 1,
                        "height": max(ys) - min(ys) + 1,
                        "area": len(xs),
                        "mask": comp_mask,
                    }
                )
    return out
