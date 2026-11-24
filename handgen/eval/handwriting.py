"""Optional local line recognizers. Neither backend receives the expected text."""

from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..paths import PACKAGE_ROOT


def _trocr_path() -> Path:
    return Path(os.environ.get("HANDGEN_TROCR_MODEL", PACKAGE_ROOT / ".cache" / "trocr-base-handwritten")).expanduser()


def select_engine(*, boxes: bool = False, line_path: Path | None = None) -> str:
    name = os.environ.get("HANDGEN_OCR_ENGINE", "auto").strip().lower()
    if name not in {"auto", "vision", "trocr", "tesseract"}:
        raise ValueError("HANDGEN_OCR_ENGINE must be auto, vision, trocr or tesseract")
    if name == "tesseract" or (name == "auto" and os.environ.get("HANDGEN_TESSERACT")):
        return "tesseract"
    can_use_vision = sys.platform == "darwin" and importlib.util.find_spec("Vision") is not None
    if name == "vision" and not can_use_vision:
        raise RuntimeError("Apple Vision OCR requires macOS and the handgen[vision] extra")
    if name == "vision" or (boxes and can_use_vision):
        return "vision"
    if boxes:
        return "tesseract"
    if name == "trocr":
        return "trocr"
    has_trocr = (_trocr_path() / "model.safetensors").is_file() and importlib.util.find_spec("transformers") is not None
    if has_trocr:
        if not can_use_vision:
            return "trocr"
        if line_path is not None:
            with Image.open(line_path) as im:
                if im.width >= 12 * im.height:
                    return "trocr"
    return "vision" if can_use_vision else "tesseract"


def _on_paper(path: Path) -> Image.Image:
    with Image.open(path) as source:
        ink = source.convert("RGBA")
    paper = Image.new("RGBA", ink.size, "white")
    paper.alpha_composite(ink)
    return paper.convert("RGB")


@lru_cache(maxsize=1)
def _trocr(model_dir: Path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True).eval()
    return processor, model


def read_trocr(path: Path) -> dict[str, Any]:
    import torch

    model_dir = _trocr_path()
    if not model_dir.is_dir():
        raise RuntimeError(f"TrOCR needs a local checkpoint at {model_dir} or HANDGEN_TROCR_MODEL")
    image = _on_paper(path)
    ink = image.convert("L").point(lambda x: 255 if x < 220 else 0).getbbox()
    text = ""
    if ink:
        processor, model = _trocr(model_dir)
        image = ImageOps.expand(image.crop(ink), border=3, fill="white")
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt").pixel_values
            ids = model.generate(pixel_values=inputs, num_beams=1, do_sample=False, max_new_tokens=96, use_cache=True)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return {"engine": "trocr", "available": True, "model": str(model_dir), "text": text}


def read_vision(path: Path) -> dict[str, Any]:
    import objc
    import Vision
    from Foundation import NSData

    source = _on_paper(path)
    bounds = source.convert("L").point(lambda x: 255 if x < 230 else 0).getbbox()
    if bounds is None:
        return {"engine": "vision", "available": True, "text": "", "words": []}
    origin_x, origin_y = bounds[:2]
    source = source.crop(bounds)
    # Vision's text detector needs paper outside tightly cropped, short words.
    scale = max(1, min(4, 96 / source.height))
    w, h = max(1, round(source.width * scale)), max(1, round(source.height * scale))
    pad = 12
    image = ImageOps.expand(source.resize((w, h), Image.Resampling.LANCZOS), pad, fill="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    strings, words = [], []

    def to_x(value: float) -> int:
        return origin_x + max(0, min(source.width, round((value * image.width - pad) * source.width / w)))

    def to_y(value: float) -> int:
        return origin_y + max(0, min(source.height, round(((1 - value) * image.height - pad) * source.height / h)))

    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setRecognitionLanguages_(["en-US"])
        req.setUsesLanguageCorrection_(True)
        data = NSData.dataWithBytes_length_(payload, len(payload))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
        ok, error = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"Apple Vision OCR failed: {error}")
        for observation in sorted(req.results(), key=lambda item: item.boundingBox().origin.x):
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            top = candidates[0]
            line = str(top.string())
            strings.append(line)
            for match in re.finditer(r"\S+", line):
                # NSRange counts UTF-16 code units.
                start = len(line[:match.start()].encode("utf-16-le")) // 2
                length = len(match.group().encode("utf-16-le")) // 2
                box, _ = top.boundingBoxForRange_error_((start, length), None)
                if box is None:
                    continue
                rect = box.boundingBox()
                left, right = to_x(rect.origin.x), to_x(rect.origin.x + rect.size.width)
                upper, lower = to_y(rect.origin.y + rect.size.height), to_y(rect.origin.y)
                if right > left and lower > upper:
                    words.append({"text": match.group(), "left": left, "right": right, "top": upper, "bottom": lower,
                                  "width": right - left, "height": lower - upper,
                                  "confidence": float(top.confidence()) * 100})
    return {"engine": "vision", "available": True, "text": " ".join(strings), "words": words,
            "ink_size": [source.width, source.height]}
