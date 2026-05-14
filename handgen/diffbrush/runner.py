from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

from ..paths import (
    DEFAULT_DIFFBRUSH_CHECKPOINT,
    DEFAULT_DIFFBRUSH_ROOT,
)
from ..utils import read_json, require_pillow_numpy, sha_text, sha_file
from ..render.ink import soft_diffbrush_ink
from .run_single import (
    DiffBrushSingleRunner,
    find_style_dir,
    seed_everything,
    select_device,
)


class DiffBrushRunner:
    """In-process DiffBrush runner.

    Builds models once on first use and reuses them for every subsequent
    chunk/variable call within the lifetime of the instance.
    """

    def __init__(self) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.root = Path(os.environ.get("DIFFBRUSH_ROOT", str(DEFAULT_DIFFBRUSH_ROOT)))
        self.checkpoint = Path(os.environ.get("DIFFBRUSH_CHECKPOINT", str(DEFAULT_DIFFBRUSH_CHECKPOINT)))
        self.device_request = os.environ.get("DIFFBRUSH_DEVICE", "auto")
        self.steps = int(os.environ.get("DIFFBRUSH_STEPS", "20"))
        self.eta = float(os.environ.get("DIFFBRUSH_ETA", "0.0"))
        self._inner: DiffBrushSingleRunner | None = None
        self._inner_style_dir: Path | None = None
        self._device = None

    def validate(self) -> None:
        missing = [p for p in (self.root, self.checkpoint) if not p.exists()]
        if missing:
            raise RuntimeError(f"DiffBrush dependency path does not exist: {', '.join(str(p) for p in missing)}")

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_chunk(
        self,
        *,
        span_id: str,
        text: str,
        style_ref: Path,
        out_dir: Path,
        seed: int,
        writer_id: str,
    ) -> dict[str, Any]:
        result_path = self._run(
            span_id=span_id,
            prompt=text,
            style_ref=style_ref,
            out_dir=out_dir,
            seed=seed,
            writer_id=writer_id,
            conditioning_mode="handgen_mvp_chunked_prose",
        )
        result = read_json(result_path)
        sample = Path(result["outputs"]["sample"])
        styled_path = out_dir / "prose" / f"{span_id}.png"
        styled = soft_diffbrush_ink(sample, styled_path, threshold=236, pad=2, alpha_scale=1.0)
        return {
            "id": span_id,
            "display_text": text,
            "conditioning_prompt": text,
            "result_json": str(result_path),
            "sample": str(sample),
            "seed": result.get("seed", seed),
            "steps": result.get("steps", self.steps),
            "device": result.get("device", self._device.type if self._device else self.device_request),
            "source_text_sha256": sha_text(text),
            "exactness_certified": False,
            "styled_crop": styled,
        }

    def generate_variable(
        self,
        *,
        label: str,
        style_ref: Path,
        out_dir: Path,
        seed: int,
        writer_id: str,
    ) -> dict[str, Any]:
        prompt = f"{label} + {label} is positive."
        span_id = f"var_u{ord(label):04x}_{sha_text(prompt)[:8]}"
        result_path = self._run(
            span_id=span_id,
            prompt=prompt,
            style_ref=style_ref,
            out_dir=out_dir,
            seed=seed,
            writer_id=writer_id,
            conditioning_mode="handgen_mvp_variable_context",
        )
        result = read_json(result_path)
        sample = Path(result["outputs"]["sample"])
        variants = self._variable_component_variants(label, sample, out_dir)
        if not variants:
            raise RuntimeError(f"DiffBrush produced no usable variable component for {label!r}")
        variants.sort(key=lambda item: item["score"], reverse=True)
        best = variants[0]
        return {
            "id": span_id,
            "label": label,
            "conditioning_prompt": prompt,
            "result_json": str(result_path),
            "sample": str(sample),
            "seed": result.get("seed", seed),
            "steps": result.get("steps", self.steps),
            "device": result.get("device", self._device.type if self._device else self.device_request),
            "source_text_sha256": sha_text(label),
            "exactness_certified": False,
            "styled_crop": best["styled_crop"],
            "component": best["component"],
            "component_variants": variants,
            "score": best["score"],
        }

    # ------------------------------------------------------------------
    # Core run method
    # ------------------------------------------------------------------

    def _ensure_inner(self, style_ref: Path):
        style_dir = find_style_dir(style_ref)
        if self._inner is not None and self._inner_style_dir == style_dir:
            return self._inner
        self.validate()
        device = select_device(self.device_request)
        try:
            inner = DiffBrushSingleRunner(self.root, self.checkpoint, style_dir, device)
        except Exception:
            if device.type != "mps":
                raise
            gc.collect()
            import torch
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
            device = torch.device("cpu")
            inner = DiffBrushSingleRunner(self.root, self.checkpoint, style_dir, device)
        self._inner = inner
        self._inner_style_dir = style_dir
        self._device = device
        return inner

    def _run(
        self,
        *,
        span_id: str,
        prompt: str,
        style_ref: Path,
        out_dir: Path,
        seed: int,
        writer_id: str,
        conditioning_mode: str,
    ) -> Path:
        run_dir = out_dir / "diffbrush_runs" / span_id
        result_path = run_dir / "result.json"
        if result_path.exists():
            return result_path

        run_dir.mkdir(parents=True, exist_ok=True)
        inner = self._ensure_inner(style_ref)
        sample_path = run_dir / "sample.png"
        seed_everything(seed)
        inner.generate_one(
            prompt,
            style_ref,
            sample_path,
            sampling_timesteps=self.steps,
            eta=self.eta,
        )
        result = {
            "prompt": prompt,
            "prompt_id": span_id,
            "run_id": span_id,
            "writer_id": writer_id,
            "conditioning_mode": conditioning_mode,
            "seed": seed,
            "steps": self.steps,
            "device": self._device.type,
            "repo_root": str(self.root),
            "checkpoint": str(self.checkpoint),
            "style_image": str(style_ref),
            "outputs": {"sample": str(sample_path)},
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result_path

    # ------------------------------------------------------------------
    # Variable component extraction (unchanged behavior)
    # ------------------------------------------------------------------

    def _variable_component_variants(self, label: str, sample: Path, out_dir: Path) -> list[dict[str, Any]]:
        Image, np = require_pillow_numpy()
        image = Image.open(sample).convert("L")
        arr = np.asarray(image)
        mask = arr < 205
        components = _components(mask)
        plausible = [
            comp
            for comp in components
            if comp["area"] >= 60 and 8 <= comp["width"] <= 50 and 18 <= comp["height"] <= 64
        ]
        plausible.sort(key=lambda item: item["x0"])
        variants: list[dict[str, Any]] = []
        for slot, comp in enumerate(plausible[:10]):
            crop_box = (
                max(0, comp["x0"] - 6),
                max(0, comp["y0"] - 6),
                min(image.width, comp["x1"] + 6),
                min(image.height, comp["y1"] + 6),
            )
            crop = image.crop(crop_box)
            raw_path = out_dir / "variables" / f"{label}_{slot:02d}_raw.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(raw_path)
            styled_path = out_dir / "variables" / f"{label}_{slot:02d}.png"
            styled = soft_diffbrush_ink(raw_path, styled_path, threshold=232, pad=2, alpha_scale=1.0)
            aspect = comp["width"] / max(1, comp["height"])
            aspect_score = max(0.0, min(1.0, 1.0 - abs(aspect - 0.55) / 0.55))
            slot_score = 1.0 if slot in {0, 2} else 0.0
            native_height_score = max(0.0, min(1.0, 1.0 - abs(styled["height"] - 34.0) / 34.0))
            density = _alpha_density(Path(styled["path"]))
            density_score = max(0.0, min(1.0, 1.0 - abs(density - 0.22) / 0.22))
            score = round(0.36 * slot_score + 0.24 * native_height_score + 0.22 * density_score + 0.18 * aspect_score, 4)
            variants.append(
                {
                    "component_slot": slot,
                    "component": {
                        "source_box": [comp["x0"], comp["y0"], comp["x1"], comp["y1"]],
                        "width": comp["width"],
                        "height": comp["height"],
                        "area": comp["area"],
                    },
                    "raw_crop": {"path": str(raw_path), "sha256": sha_file(raw_path), "box": list(crop_box)},
                    "styled_crop": styled,
                    "ink_density": round(density, 4),
                    "score": score,
                }
            )
        return variants


def _components(mask: Any) -> list[dict[str, int]]:
    _, np = require_pillow_numpy()
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[dict[str, int]] = []
    for y in range(h):
        for x0 in np.where(mask[y] & ~seen[y])[0].tolist():
            if seen[y, x0] or not mask[y, x0]:
                continue
            stack = [(x0, y)]
            seen[y, x0] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                x, yy = stack.pop()
                xs.append(x)
                ys.append(yy)
                for nx in (x - 1, x, x + 1):
                    for ny in (yy - 1, yy, yy + 1):
                        if nx == x and ny == yy:
                            continue
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            x1 = max(xs) + 1
            y1 = max(ys) + 1
            out.append(
                {
                    "x0": min(xs),
                    "y0": min(ys),
                    "x1": x1,
                    "y1": y1,
                    "width": x1 - min(xs),
                    "height": y1 - min(ys),
                    "area": len(xs),
                }
            )
    return out


def _alpha_density(path: Path) -> float:
    Image, np = require_pillow_numpy()
    alpha = np.asarray(Image.open(path).convert("RGBA").getchannel("A"))
    return float((alpha > 0).sum() / max(1, alpha.size))
