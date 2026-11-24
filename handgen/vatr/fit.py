"""Fit a small writer-specific VATr++ conditioning vector on transcribed words."""

from __future__ import annotations

import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file
from torch.nn import functional as F

from .. import db
from ..ingest.style import prepare_style, writer_word_exemplars
from ..utils import publish_files, sha_file, sha_text, write_json
from .model import VATrModel
from .runner import checkpoint_path, device_name


def _paper(record: dict) -> Image.Image:
    with Image.open(record["path"]) as source:
        rgba = source.convert("RGBA")
    white = Image.new("L", rgba.size, 255)
    white.paste(rgba.convert("L"), mask=rgba.getchannel("A"))
    return white


def _normalized_ink(ink, width):
    # Align visible extents before comparing. VATr's training images have
    # a fixed advance per character, while writers choose their own widths.
    rows, columns = torch.where(ink.detach().cpu() > .21)
    if columns.numel():
        ink = ink[max(0, int(rows.min()) - 1):int(rows.max()) + 2,
                  max(0, int(columns.min()) - 1):int(columns.max()) + 2]
    return F.interpolate(ink[None, None], size=(32, width), mode="bilinear", align_corners=False)


def _shape_loss(model, memory, word, target):
    ink = _normalized_ink(1 - model.pixels(memory, word), target.shape[-1])
    loss = ink.new_zeros(())
    for kernel, weight in ((1, .10), (3, .35), (7, .55)):
        pred = F.avg_pool2d(ink, kernel, stride=1, padding=kernel // 2)
        ref = F.avg_pool2d(target, kernel, stride=1, padding=kernel // 2)
        overlap = (2 * (pred * ref).sum() + .01) / (pred.square().sum() + ref.square().sum() + .01)
        loss = loss + weight * (1 - overlap)
    return loss + 1.25 * (ink.mean() - target.mean()).abs() + .30 * (ink.square().mean() - target.square().mean()).abs()


def _target(record, word, device):
    ink = torch.tensor((255 - np.array(_paper(record))).copy(), device=device) / 255
    return _normalized_ink(ink, len(word) * 16)


@torch.no_grad()
def _comparison(model, baseline, fitted, words, bank, path, *, show_novel=True):
    preview_words = words[:12] + ([w for w in ("Therefore", "Monday", "study") if w not in bank] if show_novel else [])
    row_height, label_width, col_width = 60, 120, 210
    sheet = Image.new("RGB", (label_width + 3 * col_width, 44 + len(preview_words) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for x, title in zip((12, label_width, label_width + col_width, label_width + 2 * col_width),
                        ("Word", "Writer sample", "Before fitting", "Selected fit")):
        draw.text((x, 12), title, fill="#36556b", font=font)
    for i, word in enumerate(preview_words):
        y = 40 + row_height * i
        draw.line((10, y, sheet.width - 12, y), fill="#dce3e8")
        draw.text((12, y + 19), word, fill="black", font=font)
        images = [_paper(bank[word][0]) if word in bank else None]
        images.extend(Image.fromarray((model.pixels(memory, word) * 255).round().byte().cpu().numpy())
                      for memory in (baseline, fitted))
        for j, image in enumerate(images):
            if image is None:
                continue
            bounds = image.point(lambda v: 255 if v < 210 else 0).getbbox()
            if bounds:
                image = image.crop(bounds)
            scale = min((col_width - 20) / image.width, 38 / image.height)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                 Image.Resampling.LANCZOS)
            sheet.paste(image, (label_width + col_width * j, y + (row_height - image.height) // 2))
    sheet.save(path)


@torch.no_grad()
def _external_test(model, baseline, fitted, reference, enrolled, stage):
    image, transcript = reference
    enrolled_sources = {r["source_sha256"] for samples in enrolled.values() for r in samples}
    if sha_file(image) in enrolled_sources:
        raise ValueError("Use a separate handwriting image for --test-reference. This one supplied fitting words.")
    directory = stage / "test_crops"
    manifest = prepare_style(image, directory, transcript.read_text(encoding="utf-8-sig"))
    seen = {word.casefold() for word in enrolled}
    results, bank = [], {}
    for record in manifest["words"]:
        word = record["text"]
        if word.casefold() in seen or not (word.isascii() and word.isalpha() and len(word) <= 32):
            continue
        sample = record | {"path": str(directory / record["path"])}
        target = _target(sample, word, model.device)
        results.append({"word": word, "source_bbox": record["source_bbox"], "sample_sha256": record["sha256"],
                        "baseline_loss": _shape_loss(model, baseline, word, target).item(),
                        "selected_loss": _shape_loss(model, fitted, word, target).item()})
        bank.setdefault(word, []).append(sample)
    if not results:
        raise ValueError("The test reference needs an aligned, previously unseen ASCII word.")
    _comparison(model, baseline, fitted, list(bank), bank, stage / "test_comparison.png", show_novel=False)
    return {"source_sha256": manifest["source_sha256"], "transcript_sha256": sha_file(transcript),
            "words": results, "baseline_loss": float(np.mean([r["baseline_loss"] for r in results])),
            "selected_loss": float(np.mean([r["selected_loss"] for r in results]))}


def fit_writer(writer_id: str, out_dir: Path, *, steps: int = 100, seed: int = 732,
               test_reference: tuple[Path, Path] | None = None) -> dict:
    """Train only the 512-channel style offset. Validate on separate word types."""
    if steps < 1:
        raise ValueError("steps must be positive")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VATr++ checkpoint missing at {checkpoint}. See README.md.")
    db.init_db()
    with db.connect() as con:
        db.require_writer(con, writer_id)
        bank = writer_word_exemplars(con, writer_id)
    words = sorted((w for w in bank if w.isalpha() and w.isascii() and len(w) <= 32),
                   key=lambda w: (sha_text(w.casefold()), w))
    # Fold case variants together so e.g. 'Brand' and 'brand' cannot leak
    # between the training and validation splits.
    families = list(dict.fromkeys(w.casefold() for w in words))
    if len(families) < 4:
        raise ValueError("Fitting needs at least four distinct transcribed words for separate training and validation.")
    validation_set = set(families[::4])
    train = [w for w in words if w.casefold() not in validation_set]
    validation = [w for w in words if w.casefold() in validation_set]
    model = VATrModel(checkpoint, device_name())
    samples = [(word, record) for word in words for record in bank[word]]
    targets = {
        record["sha256"]: _target(record, word, model.device)
        for word, record in samples
    }
    training = [(w, rec) for w, rec in samples if w in train]
    held_out = [(w, rec) for w, rec in samples if w in validation]
    baseline = model.encode([_paper(record) for _, record in training], seed)

    @torch.no_grad()
    def validate(memory):
        return float(np.mean([_shape_loss(model, memory, w, targets[rec["sha256"]]).item() for w, rec in held_out]))

    delta = torch.zeros(1, 1, 512, device=model.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=.038)
    rng = random.Random(seed)
    best_score = baseline_score = validate(baseline)
    selected = baseline
    selected_step = 0
    history = []
    for step in range(1, steps + 1):
        word, record = rng.choice(training)
        optimizer.zero_grad()
        loss = _shape_loss(model, baseline + delta, word, targets[record["sha256"]]) + .08 * delta.square().mean()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps:
            memory = (baseline + delta).detach()
            score = validate(memory)
            history.append({"step": step, "validation_loss": score})
            if score < best_score:
                best_score, selected_step, selected = score, step, memory

    out_dir.mkdir(parents=True, exist_ok=True)
    digest = sha_file(checkpoint)
    adapter = out_dir / "writer.safetensors"
    with TemporaryDirectory(dir=out_dir) as tmp:
        stage = Path(tmp)
        save_file({"memory": selected.cpu().contiguous()}, str(stage / adapter.name), metadata={
            "writer_id": writer_id, "checkpoint_sha256": digest, "format": "handgen_vatr_memory_v1",
        })
        _comparison(model, baseline, selected, validation, bank, stage / "comparison.png")
        metrics = {"writer_id": writer_id, "device": str(model.device), "seed": seed, "steps": steps,
                   "train_words": train, "validation_words": validation,
                   "source_images": sorted({r["source_sha256"] for _, r in samples}),
                   "word_crop_hashes": [r["sha256"] for _, r in samples],
                   "trainable_parameters": delta.numel(), "metric": "multiscale_soft_dice_and_ink_density_loss",
                   "baseline_loss": baseline_score, "selected_loss": best_score, "selected_step": selected_step,
                   "validation_history": history, "checkpoint_sha256": digest, "adapter": str(adapter),
                   "comparison": str(out_dir / "comparison.png")}
        outputs = [adapter.name, "comparison.png"]
        if test_reference:
            metrics["independent_test"] = _external_test(model, baseline, selected, test_reference, bank, stage)
            metrics["independent_test"]["comparison"] = str(out_dir / "test_comparison.png")
            outputs.append("test_comparison.png")
        write_json(stage / "fit.json", metrics)
        publish_files(stage, out_dir, [*outputs, "fit.json"])
    return metrics
