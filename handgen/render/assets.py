"""Produce source-indexed pieces of ink suitable for baseline-aligned layout."""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..diffbrush import DiffBrushUnavailable
from ..diffbrush.prompt import IAM_LETTERS
from ..diffbrush.runner import DiffBrushRunner, ProseVerificationError
from ..eval.ocr import OCRUnavailable
from ..ingest.style_metrics import (
    estimated_space_width,
    estimated_word_width,
    writing_metrics,
)
from ..utils import png_size, sha_text
from ..vatr.runner import VATrRunner, checkpoint_path
from .layout import PROSE_H, ink_geometry
from .math import missing_glyphs, render_math_span


@dataclass
class InkPiece:
    text: str
    route: str
    evidence: str
    path: Path | None = None
    width: float = 0
    height: float = 0
    baseline: float = 0
    details: dict[str, Any] = field(default_factory=dict)


class WriterAssets:
    def __init__(self, writer_id: str, out_dir: Path, style_lines: list[Path], profile: dict,
                 glyphs: dict[str, list[Path]], exemplars: dict[str, list[dict]], runner: DiffBrushRunner,
                 mode: str, seed: int, adapter: Path | None = None):
        self.writer_id: str = writer_id
        self.out_dir: Path = out_dir
        self.style_lines: list[Path] = style_lines
        self.profile: dict = profile
        self.glyphs: dict[str, list[Path]] = glyphs
        self.words: dict[str, list[dict]] = exemplars
        self.runner: DiffBrushRunner = runner
        self.mode: str = mode
        self.seed: int = seed
        self.rng = random.Random(seed)
        self.digits: dict[str, dict] = {}
        self.generated: dict[str, dict] = {}
        self.metrics = writing_metrics(exemplars)
        self.word_gap = estimated_space_width(self.metrics, body_height=profile["target_body_height"],
                                              fallback=profile["target_space_advance"])
        self.vatr: VATrRunner | None = None
        self.adapter = adapter
        self.attempts: list[dict] = []
        self.warnings: list[dict] = []
        self.generation_unavailable: str | None = None

    def diffbrush_available(self) -> bool:
        if self.generation_unavailable:
            return False
        checkpoint = getattr(self.runner, "checkpoint", None)
        return checkpoint is None or (checkpoint.is_file() and all(
            find_spec(name) is not None for name in ("torch", "diffusers", "timm", "lmdb")))

    def space(self, text: str) -> InkPiece:
        width = sum(4 if c == "\t" else 1 for c in text) * self.word_gap
        return InkPiece(text, "source_whitespace", "whitespace", width=width)

    def worksheet(self, text: str, piece_id: str, *, prose: bool = False) -> InkPiece:
        if not prose:
            for ch in sorted(set(text) & set("0123456789")):
                if ch not in self.glyphs and ch not in self.digits:
                    self.digits[ch] = self.runner.generate_variable(
                        label=ch, style_ref=self.style_lines[0], out_dir=self.out_dir,
                        seed=self.seed + 9100 + ord(ch) * 19, writer_id=self.writer_id,
                    )
        block = render_math_span(piece_id, text, self.glyphs, self.out_dir, self.rng,
                                 variable_bank=self.digits if not prose else None, style_profile=self.profile, prose=prose)
        synthesized_digit = any(str(t.get("route", "")).startswith("diffbrush") for t in block["tokens"])
        evidence = "unverified_model_digit" if synthesized_digit else "worksheet_labels"
        route = "worksheet_prose" if prose else block["route"]
        width = self.word_width(text, block["width"]) if prose else block["width"]
        return InkPiece(text, route, evidence, Path(block["paths"]["png"]),
                        width, block["height"], block["baseline"], {"glyph_run" if prose else "math": block})

    def word_width(self, word: str, width: float) -> float:
        estimate = estimated_word_width(word, self.metrics, body_height=self.profile["target_body_height"])
        return width if estimate is None else min(1.18 * width, max(.84 * width, estimate))

    def sample_parts(self, text: str) -> Iterator[str]:
        for word in re.findall(r"\s+|\S+", text):
            # A transcribed "fox" may also be used before a worksheet comma.
            match = re.fullmatch(r"(\W*)([\w']+)(\W*)", word)
            if word not in self.words and match and match[2] in self.words:
                yield from (p for p in match.groups() if p)
            else:
                yield word

    def sample_missing(self, text: str) -> list[str]:
        return sorted({ch for part in self.sample_parts(text) if part not in self.words
                       for ch in missing_glyphs(part, self.glyphs, prose=True)})

    def from_samples(self, text: str, span_id: str) -> Iterator[InkPiece]:
        for i, part in enumerate(self.sample_parts(text)):
            if part.isspace():
                yield self.space(part)
            elif part in self.words:
                word = self.rng.choice(self.words[part])
                scale = self.profile["target_body_height"] / word["body_height"]
                yield InkPiece(part, "transcribed_word_sample", "transcript_label", Path(word["path"]),
                               word["width"] * scale, word["height"] * scale, word["baseline"] * scale,
                               {"word_sample": {**word, "text": part}})
            else:
                yield self.worksheet(part, f"{span_id}_word{i:03d}", prose=True)

    def generate(self, text: str, *, prompt_text=None, carrier=False) -> dict:
        tag = ("carrier_" if carrier else "prose_") + sha_text(text + repr(prompt_text))[:16]
        if tag not in self.generated:
            try:
                run = self.runner.generate_chunk(span_id=tag, text=text, prompt_text=prompt_text, style_ref=self.style_lines[0],
                                                 out_dir=self.out_dir, seed=self.seed + int(tag[-8:], 16) % 1_000_003,
                                                 writer_id=self.writer_id)
            except (DiffBrushUnavailable, OCRUnavailable) as err:
                self.generation_unavailable = str(err)
                raise
            self.generated[tag] = run
        return self.generated[tag]

    def generated_piece(self, text: str, path: Path, route: str, evidence: str, details: dict) -> InkPiece:
        w, h = png_size(path)
        baseline, body_height = ink_geometry(path)
        # Keep stroke sizes consistent across ascenders and descenders.
        # Sparse punctuation and isolated capitals have a weak x-height estimate.
        if any(ch.islower() for ch in text):
            scale = min(2.0, PROSE_H * 2.1 / h, self.profile["target_body_height"] / body_height)
        else:
            scale = min(PROSE_H / max(1, h), 1.6)
        words = text.split()
        estimates = [estimated_word_width(word, self.metrics, body_height=self.profile["target_body_height"])
                     for word in words]
        if estimates and all(advance is not None for advance in estimates):
            target = sum(estimates) + (len(words) - 1) * self.word_gap
            # Aspect and body-size estimates both become noisy on short runs.
            # Bring an unusually wide crop closer to this writer's advances.
            scale = min(scale, 1.15 * target / w)
        return InkPiece(text, route, evidence, path, w * scale, h * scale, baseline * scale, details)

    def vatr_word(self, word: str) -> InkPiece:
        if self.vatr is None:
            self.vatr = VATrRunner(self.style_lines, self.seed, self.writer_id, self.adapter)
        run = self.vatr.word(word, self.out_dir)
        evidence = "ocr_match" if run["ocr_verification"]["passed"] else "best_effort"
        piece = self.generated_piece(word, Path(run["styled_crop"]["path"]), "vatr_word", evidence, {"vatr": run})
        piece.width = self.word_width(word, piece.width)
        return piece

    def replacement_word(self, word: str, piece_id: str) -> list[InkPiece]:
        covered = not self.sample_missing(word)
        has_exemplar = any(part in self.words for part in self.sample_parts(word))
        if covered and (not self.adapter or has_exemplar):
            return list(self.from_samples(word, piece_id))
        word_model_available = self.adapter or (checkpoint_path().is_file() and all(
            find_spec(name) is not None for name in ("torch", "torchvision", "safetensors")))
        if word.isascii() and word.isprintable() and word_model_available:
            pieces = [self.vatr_word(word[i:i + 32]) for i in range(0, len(word), 32)]
            if not covered or all(p.evidence == "ocr_match" for p in pieces):
                return pieces
            self.attempts.append({"span_id": piece_id, "expected": word, "used": ["worksheet_prose"],
                                  "rejected_vatr": [p.details["vatr"] for p in pieces]})
        return list(self.from_samples(word, piece_id)) if covered else []

    def repair(self, text: str, piece_id: str, run: dict | None) -> list[InkPiece]:
        from .repair import recover_prose_runs

        recovered = recover_prose_runs(text, run, self.out_dir) if run else {}
        pieces = []
        cursor = word_index = 0
        words = list(re.finditer(r"\S+", text))
        while word_index < len(words):
            match = words[word_index]
            if match.start() > cursor:
                pieces.append(self.space(text[cursor:match.start()]))
            word = match.group()
            if word_index in recovered:
                region = recovered[word_index]
                end = region["end_word"]
                substring = text[match.start():words[end - 1].end()]
                pieces.append(self.generated_piece(substring, Path(region["path"]), "diffbrush_verified_crop",
                                                   "ocr_match", {"recovered_crop": region}))
                word_index = end
            else:
                replacement = self.replacement_word(word, f"{piece_id}_repair{word_index:03d}")
                if not replacement:
                    return []
                pieces.extend(replacement)
                word_index += 1
            cursor = words[word_index - 1].end()
        if cursor < len(text):
            pieces.append(self.space(text[cursor:]))
        return pieces

    def prose(self, text: str, piece_id: str) -> Iterator[InkPiece]:
        if text.isspace():
            yield self.space(text)
            return
        if self.mode == "vatr":
            for word in re.findall(r"\s+|\S+", text):
                if word.isspace():
                    yield self.space(word)
                else:
                    yield from (self.vatr_word(word[i:i + 32]) for i in range(0, len(word), 32))
            return
        has_words = all(word in self.words for word in text.split())
        if self.mode == "samples" or (self.mode == "auto" and has_words):
            yield from self.from_samples(text, piece_id)
            return

        attempt = {"span_id": piece_id, "expected": text}
        run = None
        unsupported = sorted(set(" ".join(text.split())) - set(IAM_LETTERS))
        if self.mode == "auto" and (unsupported or not self.diffbrush_available()):
            attempt["reason"] = (f"Outside DiffBrush's alphabet: {unsupported}" if unsupported
                                 else self.generation_unavailable or "DiffBrush checkpoint or runtime unavailable")
        else:
            try:
                run = self.generate(" ".join(text.split()))
            except (ProseVerificationError, DiffBrushUnavailable, OCRUnavailable) as err:
                if self.mode != "auto":
                    raise
                attempt["reason"] = str(err)
        if self.mode == "auto" and (run is None or not run["exactness_certified"]):
            repairs = self.repair(text, piece_id, run)
            # A sample or a newly verified crop contributes zero unresolved
            # edits. Choose the word model only when it improves a failed line.
            edits = sum(p.details["vatr"]["ocr_verification"]["character_edit_distance"]
                        for p in repairs if p.evidence == "best_effort")
            if repairs and (not edits or run is None or edits < run["ocr_verification"]["character_edit_distance"]):
                attempt.update(used=sorted({p.route for p in repairs if p.path}), rejected_generation=run)
                self.attempts.append(attempt)
                yield from repairs
                return
        if run is None:
            raise RuntimeError(f"No handwriting route covered {text!r}. {attempt.get('reason', '')}")
        evidence = "ocr_match" if run["exactness_certified"] else "best_effort"
        yield self.generated_piece(text, Path(run["styled_crop"]["path"]), run["route"], evidence, {"diffbrush": run})
