from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..eval.ocr import ocr_image, ocr_legibility_score, ocr_word_boxes, verify_ocr_text
from ..paths import (
    DEFAULT_DIFFBRUSH_CHECKPOINT,
    DEFAULT_DIFFBRUSH_ROOT,
)
from ..utils import read_json, require_pillow_numpy, sha_text, sha_file
from ..render.ink import rgba_ink_profile, soft_diffbrush_ink, trim_rgba_right


def _normalize_prompt_variants(prompt_text: str | list[str] | tuple[str, ...] | None, text: str) -> list[str]:
    raw = [text] if prompt_text is None else ([prompt_text] if isinstance(prompt_text, str) else list(prompt_text))
    variants: list[str] = []
    for item in raw:
        prompt = str(item).strip()
        if prompt and prompt not in variants:
            variants.append(prompt)
    return variants or [text]


def _prompt_seed_offset(prompt: str) -> int:
    return int(sha_text(prompt)[:8], 16) % 100_003


def _viable_best_effort_candidates(candidates: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    return [
        item
        for item in candidates
        if not item.get("quality_reject_reasons") and float(item.get("score", 0.0) or 0.0) >= min_score
    ]


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
        self._inner: Any | None = None
        self._inner_style_dir: Path | None = None
        self._device = None
        self.prose_candidates = int(os.environ.get("HANDGEN_DIFFBRUSH_PROSE_CANDIDATES", "4"))
        self.allow_best_effort = os.environ.get("HANDGEN_DIFFBRUSH_ALLOW_BEST_EFFORT", "1") != "0"
        self.min_best_effort_score = float(os.environ.get("HANDGEN_DIFFBRUSH_MIN_BEST_EFFORT_SCORE", "0.12"))
        self.style_legibility_floor_ratio = float(os.environ.get("HANDGEN_DIFFBRUSH_STYLE_LEGIBILITY_FLOOR_RATIO", "0.8"))
        self.last_resort_retries = int(os.environ.get("HANDGEN_DIFFBRUSH_LAST_RESORT_RETRIES", "2"))

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
        prompt_text: str | list[str] | tuple[str, ...] | None = None,
        style_ref: Path,
        out_dir: Path,
        seed: int,
        writer_id: str,
    ) -> dict[str, Any]:
        prompt_variants = _normalize_prompt_variants(prompt_text, text)
        candidates: list[dict[str, Any]] = []
        style_legibility = ocr_legibility_score(style_ref, psm=7)
        effective_min_score = self._effective_best_effort_score(style_legibility)
        for prompt_idx, candidate_prompt in enumerate(prompt_variants):
            for idx in range(max(1, self.prose_candidates)):
                candidate_seed = seed + _prompt_seed_offset(candidate_prompt) + idx * 1009
                candidate_id = f"{span_id}_cand{idx:02d}" if len(prompt_variants) == 1 else f"{span_id}_p{prompt_idx:02d}_cand{idx:02d}"
                candidate = self._generate_candidate(
                    candidate_id=candidate_id,
                    span_id=span_id,
                    text=text,
                    prompt_text=candidate_prompt,
                    style_ref=style_ref,
                    out_dir=out_dir,
                    seed=candidate_seed,
                    writer_id=writer_id,
                )
                candidates.append(candidate)
                if candidate["accepted"]:
                    return self._finalize_candidate(
                        span_id=span_id,
                        text=text,
                        candidate=candidate,
                        style_legibility=style_legibility,
                        candidates=candidates,
                        exactness_certified=True,
                        route="diffbrush_ocr_verified_prose_token",
                    )
        if self.allow_best_effort and candidates:
            if self.last_resort_retries > 0 and not _viable_best_effort_candidates(candidates, effective_min_score):
                retry_start = max(1, self.prose_candidates)
                for prompt_idx, candidate_prompt in enumerate(prompt_variants):
                    for retry_idx in range(self.last_resort_retries):
                        idx = retry_start + retry_idx
                        candidate_seed = seed + _prompt_seed_offset(candidate_prompt) + idx * 1009
                        candidate_id = f"{span_id}_cand{idx:02d}" if len(prompt_variants) == 1 else f"{span_id}_p{prompt_idx:02d}_cand{idx:02d}"
                        candidate = self._generate_candidate(
                            candidate_id=candidate_id,
                            span_id=span_id,
                            text=text,
                            prompt_text=candidate_prompt,
                            style_ref=style_ref,
                            out_dir=out_dir,
                            seed=candidate_seed,
                            writer_id=writer_id,
                        )
                        candidates.append(candidate)
                        if candidate["accepted"]:
                            return self._finalize_candidate(
                                span_id=span_id,
                                text=text,
                                candidate=candidate,
                                style_legibility=style_legibility,
                                candidates=candidates,
                                exactness_certified=True,
                                route="diffbrush_ocr_verified_prose_token",
                            )
            candidates.sort(key=lambda item: (item.get("selection_score", item.get("score", 0.0)), len(item["ocr"]["text"])), reverse=True)
            clean_candidates = [item for item in candidates if not item.get("quality_reject_reasons")]
            viable = [item for item in clean_candidates if float(item.get("score", 0.0) or 0.0) >= effective_min_score]
            best = viable[0] if viable else (clean_candidates[0] if clean_candidates else candidates[0])
            last_resort = not viable
            run = self._finalize_candidate(
                span_id=span_id,
                text=text,
                candidate=best,
                style_legibility=style_legibility,
                candidates=candidates,
                exactness_certified=False,
                route="diffbrush_last_resort_prose_token" if last_resort else "diffbrush_best_effort_prose_token",
            )
            run["best_effort_min_score"] = self.min_best_effort_score
            run["best_effort_effective_min_score"] = effective_min_score
            run["adaptive_last_resort_retries"] = self.last_resort_retries
            run["best_effort_reason"] = (
                "No candidate passed OCR exactness; selected the highest token-match candidate above the configured best-effort floor."
                if not last_resort
                else "No candidate passed OCR exactness or the configured best-effort floor; selected the highest non-quality-rejected DiffBrush candidate when available as a last resort so the render contains no non-DiffBrush natural-language fallback."
            )
            return run
        reject_summary = [
            {
                "id": item["id"],
                "seed": item["seed"],
                "ocr_text": item["ocr"]["text"],
                "reject_reasons": item["verification"]["reject_reasons"],
            }
            for item in candidates
        ]
        raise RuntimeError(f"DiffBrush produced no OCR-verified prose candidate for {text!r}: {reject_summary}")

    def _effective_best_effort_score(self, style_legibility: dict[str, Any]) -> float:
        style_score = float(style_legibility.get("score", 0.0) or 0.0)
        if style_score <= 0.0 or self.style_legibility_floor_ratio <= 0.0:
            return self.min_best_effort_score
        return round(max(self.min_best_effort_score, style_score * self.style_legibility_floor_ratio), 4)

    def _generate_candidate(
        self,
        *,
        candidate_id: str,
        span_id: str,
        text: str,
        prompt_text: str,
        style_ref: Path,
        out_dir: Path,
        seed: int,
        writer_id: str,
    ) -> dict[str, Any]:
        result_path = self._run(
            span_id=candidate_id,
            prompt=prompt_text,
            style_ref=style_ref,
            out_dir=out_dir,
            seed=seed,
            writer_id=writer_id,
            conditioning_mode="handgen_mvp_source_locked_prose_candidate",
        )
        result = read_json(result_path)
        sample = Path(result["outputs"]["sample"])
        styled_path = out_dir / "prose_candidates" / span_id / f"{candidate_id}.png"
        styled = soft_diffbrush_ink(sample, styled_path, threshold=236, pad=2, alpha_scale=1.0)
        ink_profile = rgba_ink_profile(Path(styled["path"]))
        quality_reject_reasons = _quality_reject_reasons(text, styled, ink_profile)
        ocr = ocr_image(Path(styled["path"]), psm=7)
        prompt_verification = verify_ocr_text(prompt_text, ocr["text"])
        verification = verify_ocr_text(text, ocr["text"]) if prompt_text == text else dict(prompt_verification) | {
            "passed": False,
            "reject_reasons": sorted(set(prompt_verification["reject_reasons"] + ["requires_padding_trim"])),
        }
        trim_evidence = None
        candidate_styled = styled
        should_try_trim = prompt_text != text or (not verification["passed"] and verification["prefix_matches_expected"])
        if should_try_trim:
            word_boxes = ocr_word_boxes(Path(styled["path"]), psm=7)
            trimmed = _trim_verified_prefix(
                expected_token_count=len(verify_ocr_text(text, text)["expected_tokens"]),
                styled=styled,
                word_boxes=word_boxes,
                out_dir=out_dir,
                span_id=span_id,
                candidate_id=candidate_id,
            )
            if trimmed is not None:
                trimmed_ocr = ocr_image(Path(trimmed["styled_crop"]["path"]), psm=7)
                trimmed_verification = verify_ocr_text(text, trimmed_ocr["text"])
                trim_evidence = {
                    "word_boxes": word_boxes,
                    "trimmed_crop": trimmed["styled_crop"],
                    "trimmed_ocr": trimmed_ocr,
                    "trimmed_verification": trimmed_verification,
                }
                if trimmed_verification["passed"]:
                    verification = trimmed_verification | {
                        "accepted_after_suffix_trim": True,
                        "trimmed_suffix_tokens": verify_ocr_text(text, ocr["text"])["trimmable_suffix_tokens"],
                        "prompt_verification_before_trim": prompt_verification,
                    }
                    ocr = trimmed_ocr
                    candidate_styled = trimmed["styled_crop"]
        if prompt_text != text and candidate_styled["path"] == styled["path"]:
            estimated_trim = _trim_estimated_prefix(
                text=text,
                prompt_text=prompt_text,
                styled=styled,
                out_dir=out_dir,
                span_id=span_id,
                candidate_id=candidate_id,
            )
            estimated_ocr = ocr_image(Path(estimated_trim["styled_crop"]["path"]), psm=7)
            estimated_verification = verify_ocr_text(text, estimated_ocr["text"])
            estimated_ink_profile = rgba_ink_profile(Path(estimated_trim["styled_crop"]["path"]))
            estimated_quality_reject_reasons = _quality_reject_reasons(text, estimated_trim["styled_crop"], estimated_ink_profile)
            trim_evidence = (trim_evidence or {}) | {
                "estimated_trimmed_crop": estimated_trim["styled_crop"],
                "estimated_trimmed_ocr": estimated_ocr,
                "estimated_trimmed_verification": estimated_verification,
                "estimated_trimmed_ink_profile": estimated_ink_profile,
                "estimated_trimmed_quality_reject_reasons": estimated_quality_reject_reasons,
                "estimated_trim_reason": "Prompt padding must not be rendered; cropped by expected/prompt character ratio because OCR prefix trim did not certify a boundary.",
            }
            verification = estimated_verification | {
                "estimated_padding_trim": True,
                "prompt_verification_before_trim": prompt_verification,
            }
            ocr = estimated_ocr
            candidate_styled = estimated_trim["styled_crop"]
            ink_profile = estimated_ink_profile
            quality_reject_reasons = estimated_quality_reject_reasons
        if not verification["passed"]:
            overwide_trim = _trim_overwide_suffix(
                text=text,
                styled=candidate_styled,
                out_dir=out_dir,
                span_id=span_id,
                candidate_id=candidate_id,
            )
            if overwide_trim is not None:
                overwide_ocr = ocr_image(Path(overwide_trim["styled_crop"]["path"]), psm=7)
                overwide_verification = verify_ocr_text(text, overwide_ocr["text"])
                overwide_ink_profile = rgba_ink_profile(Path(overwide_trim["styled_crop"]["path"]))
                overwide_quality_reject_reasons = _quality_reject_reasons(text, overwide_trim["styled_crop"], overwide_ink_profile)
                trim_evidence = (trim_evidence or {}) | {
                    "overwide_trimmed_crop": overwide_trim["styled_crop"],
                    "overwide_trimmed_ocr": overwide_ocr,
                    "overwide_trimmed_verification": overwide_verification,
                    "overwide_trimmed_ink_profile": overwide_ink_profile,
                    "overwide_trimmed_quality_reject_reasons": overwide_quality_reject_reasons,
                    "overwide_trim_reason": "Cropped a right-edge DiffBrush hallucination tail from an overwide carrier crop; no fallback text was inserted.",
                }
                verification = overwide_verification | {"overwide_suffix_trim": True}
                ocr = overwide_ocr
                candidate_styled = overwide_trim["styled_crop"]
                ink_profile = overwide_ink_profile
                quality_reject_reasons = overwide_quality_reject_reasons
        accepted = verification["passed"] and not quality_reject_reasons
        selection_score = float(verification.get("match_score", 0.0) or 0.0)
        if quality_reject_reasons:
            selection_score -= 1.0
        return {
            "id": candidate_id,
            "prompt_text": prompt_text,
            "result_json": str(result_path),
            "sample": str(sample),
            "seed": result.get("seed", seed),
            "steps": result.get("steps", self.steps),
            "device": result.get("device", self._device.type if self._device else self.device_request),
            "styled_crop": candidate_styled,
            "ink_profile": ink_profile,
            "quality_reject_reasons": quality_reject_reasons,
            "ocr": ocr,
            "verification": verification,
            "trim_evidence": trim_evidence,
            "score": verification.get("match_score", 0.0),
            "selection_score": round(selection_score, 4),
            "accepted": accepted,
        }

    def _finalize_candidate(
        self,
        *,
        span_id: str,
        text: str,
        candidate: dict[str, Any],
        style_legibility: dict[str, Any],
        candidates: list[dict[str, Any]],
        exactness_certified: bool,
        route: str,
    ) -> dict[str, Any]:
        final_path = Path(candidate["styled_crop"]["path"]).parents[2] / "prose" / f"{span_id}.png"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate["styled_crop"]["path"], final_path)
        final_styled = dict(candidate["styled_crop"]) | {"path": str(final_path), "sha256": sha_file(final_path)}
        final_word_boxes = ocr_word_boxes(final_path, psm=7)
        return {
            "id": span_id,
            "display_text": text,
            "conditioning_prompt": text,
            "diffbrush_prompt": candidate["prompt_text"],
            "route": route,
            "source_text_sha256": sha_text(text),
            "exactness_certified": exactness_certified,
            "result_json": candidate["result_json"],
            "sample": candidate["sample"],
            "seed": candidate["seed"],
            "steps": candidate["steps"],
            "device": candidate["device"],
            "style_ocr_legibility": style_legibility,
            "styled_crop": final_styled,
            "ocr_verification": candidate["verification"],
            "ocr_word_boxes": final_word_boxes,
            "accepted_candidate": candidate["id"],
            "candidate_count": len(candidates),
            "rejected_candidates": [item for item in candidates if not item["accepted"]] if exactness_certified else candidates,
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
        try:
            from .run_single import DiffBrushSingleRunner, find_style_dir, select_device
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise RuntimeError("DiffBrush in-process runner requires torch in the active Python environment.") from error
            raise
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
        from .run_single import seed_everything

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
            raw_path = out_dir / "variables" / f"{label}_{slot:02d}_raw.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            _save_isolated_component(arr, comp, crop_box, raw_path)
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
                        "isolation": "connected_component_mask",
                    },
                    "raw_crop": {"path": str(raw_path), "sha256": sha_file(raw_path), "box": list(crop_box)},
                    "styled_crop": styled,
                    "ink_density": round(density, 4),
                    "score": score,
                }
            )
        return variants


def _components(mask: Any) -> list[dict[str, Any]]:
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
                    "pixels": list(zip(xs, ys)),
                }
            )
    return out


def _save_isolated_component(arr: Any, comp: dict[str, Any], crop_box: tuple[int, int, int, int], dst: Path) -> None:
    Image, np = require_pillow_numpy()
    x0, y0, x1, y1 = crop_box
    crop_arr = np.full((y1 - y0, x1 - x0), 255, dtype="uint8")
    for x, y in comp["pixels"]:
        if x0 <= x < x1 and y0 <= y < y1:
            crop_arr[y - y0, x - x0] = arr[y, x]
    Image.fromarray(crop_arr, mode="L").save(dst)


def _trim_verified_prefix(
    *,
    expected_token_count: int,
    styled: dict[str, Any],
    word_boxes: dict[str, Any],
    out_dir: Path,
    span_id: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    words = word_boxes.get("words", [])
    if expected_token_count <= 0 or len(words) <= expected_token_count:
        return None
    last_expected = words[expected_token_count - 1]
    first_extra = words[expected_token_count]
    if first_extra["left"] <= last_expected["right"]:
        return None
    trim_x = int((last_expected["right"] + first_extra["left"]) / 2)
    trimmed_path = out_dir / "prose_candidates" / span_id / f"{candidate_id}_trimmed.png"
    trimmed = trim_rgba_right(Path(styled["path"]), trimmed_path, trim_x, pad=0)
    return {"styled_crop": dict(styled) | trimmed | {"trim_policy": "ocr_verified_prefix_suffix_trim"}}


def _trim_estimated_prefix(
    *,
    text: str,
    prompt_text: str,
    styled: dict[str, Any],
    out_dir: Path,
    span_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    width = int(styled["width"])
    ratio = len(text.rstrip()) / max(1, len(prompt_text.rstrip()))
    right_x = max(1, min(width, int(width * ratio) + 6))
    trimmed_path = out_dir / "prose_candidates" / span_id / f"{candidate_id}_estimated_prefix.png"
    trimmed = trim_rgba_right(Path(styled["path"]), trimmed_path, right_x=right_x, pad=2)
    return {
        "styled_crop": dict(styled) | trimmed | {"trim_policy": "estimated_padding_suffix_trim"},
        "crop_right_x": right_x,
        "ratio": round(ratio, 4),
    }


def _trim_overwide_suffix(
    *,
    text: str,
    styled: dict[str, Any],
    out_dir: Path,
    span_id: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    width = int(styled["width"])
    text_len = len(text.strip())
    if text_len <= 0:
        return None
    target_right = int(max(130, min(width, text_len * 30 + 24)))
    if width <= target_right * 1.22:
        return None
    trimmed_path = out_dir / "prose_candidates" / span_id / f"{candidate_id}_overwide_trimmed.png"
    trimmed = trim_rgba_right(Path(styled["path"]), trimmed_path, right_x=target_right, pad=2)
    return {
        "styled_crop": dict(styled) | trimmed | {
            "trim_policy": "overwide_diffbrush_suffix_trim",
            "crop_right_x": target_right,
            "original_width": width,
        }
    }


def _alpha_density(path: Path) -> float:
    Image, np = require_pillow_numpy()
    alpha = np.asarray(Image.open(path).convert("RGBA").getchannel("A"))
    return float((alpha > 0).sum() / max(1, alpha.size))


def _quality_reject_reasons(text: str, styled: dict[str, Any], ink_profile: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    density = float(ink_profile.get("ink_density", 0.0) or 0.0)
    mean_alpha = float(ink_profile.get("mean_alpha", 0.0) or 0.0)
    width = float(styled.get("width", 0.0) or 0.0)
    expected_len = max(1, len(text.strip()))
    if density > 0.42:
        reasons.append("excessive_ink_density")
    if mean_alpha <= 0.02:
        reasons.append("empty_or_nearly_transparent_ink")
    if expected_len <= 12 and width > 420:
        reasons.append("excessive_width_for_short_prompt")
    return reasons
