from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from ..eval.ocr import ocr_image, verify_ocr_text
from ..ingest.style import style_word_crops
from ..paths import PACKAGE_ROOT
from ..render.ink import soft_diffbrush_ink
from ..utils import sha_file, sha_text


def checkpoint_path() -> Path:
    return Path(os.environ.get("HANDGEN_VATR_CHECKPOINT", PACKAGE_ROOT / "models/vatrpp/model.safetensors"))


def device_name() -> str:
    import torch

    requested = os.getenv("HANDGEN_VATR_DEVICE", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


class VATrRunner:
    def __init__(self, style_lines: list[Path], seed: int, writer_id: str, adapter: Path | None):
        checkpoint = checkpoint_path()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VATr++ checkpoint missing at {checkpoint}. See the VATr++ setup in README.md.")
        from .model import VATrModel
        self.adapter = adapter
        memory = None
        if adapter:
            from safetensors import safe_open
            with safe_open(adapter, framework="pt") as weights:
                meta = weights.metadata() or {}
                if (meta.get("writer_id") != writer_id or meta.get("format") != "handgen_vatr_memory_v1"
                        or meta.get("checkpoint_sha256") != sha_file(checkpoint)):
                    raise ValueError("The fitted VATr++ memory belongs to a different writer or checkpoint.")
                memory = weights.get_tensor("memory")
                if memory.ndim != 3 or memory.shape[0] < 1 or tuple(memory.shape[1:]) != (1, 512):
                    raise ValueError("The fitted VATr++ memory has an invalid shape.")
        self.model = VATrModel(checkpoint, device_name())
        self.memory = (memory.to(self.model.device) if memory is not None
                       else self.model.encode(style_word_crops(style_lines), seed))
        self.cache: dict[str, dict] = {}

    def word(self, word: str, work: Path) -> dict:
        if word not in self.cache:
            import torch
            with torch.no_grad():
                pixels = self.model.pixels(self.memory, word)
                gray = Image.fromarray((pixels * 255).round().byte().cpu().numpy())
            folder = work / "vatr"
            folder.mkdir(exist_ok=True)
            tag = sha_text(word)[:16]
            raw = folder / f"{tag}_raw.png"
            gray.save(raw)
            crop = soft_diffbrush_ink(raw, folder / f"{tag}.png", threshold=226, pad=1)
            ocr = ocr_image(Path(crop["path"]), psm=8)
            verification = verify_ocr_text(word, ocr["text"])
            self.cache[word] = {
                "target": word, "styled_crop": crop, "raw": str(raw),
                "ocr_verification": {**verification, "engine": ocr["engine"]},
                "checkpoint": str(checkpoint_path()), "device": str(self.model.device),
                "adapter": str(self.adapter) if self.adapter else None,
            }
        return self.cache[word]
