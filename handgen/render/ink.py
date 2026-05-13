from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import require_pillow_numpy, sha_file


INK_RGB = (31, 36, 43)


def transparent_ink(src: Path, dst: Path, threshold: int = 244, pad: int = 4, alpha_scale: float = 1.6) -> dict[str, Any]:
    Image, np = require_pillow_numpy()
    image = Image.open(src).convert("L")
    arr = np.asarray(image)
    mask = arr < threshold
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
    alpha = np.clip((255 - arr) * alpha_scale, 0, 255).astype("uint8")
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


def soft_diffbrush_ink(src: Path, dst: Path, threshold: int = 236, pad: int = 3, alpha_scale: float = 1.0) -> dict[str, Any]:
    return transparent_ink(src, dst, threshold=threshold, pad=pad, alpha_scale=alpha_scale) | {
        "alpha_mode": "soft_diffbrush_alpha"
    }


def scanned_symbol_ink(src: Path, dst: Path, threshold: int = 218, pad: int = 4, alpha_scale: float = 0.62) -> dict[str, Any]:
    return transparent_ink(src, dst, threshold=threshold, pad=pad, alpha_scale=alpha_scale) | {
        "alpha_mode": "scanned_symbol_alpha_scaled"
    }
