from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def publish_files(source: Path, destination: Path, names: Sequence[str]) -> None:
    """Stage all convenience copies, then publish the manifest last under a writer lock."""
    import fcntl

    with tempfile.TemporaryDirectory(dir=destination) as tmp:
        stage = Path(tmp)
        for name in names:
            shutil.copyfile(source / name, stage / name)
        with (destination / ".publish.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for name in names:
                os.replace(stage / name, destination / name)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/svg+xml" if suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", data[16:24])


def render_svg_to_png(svg_path: Path) -> Path:
    converter = shutil.which("rsvg-convert")
    if not converter:
        raise RuntimeError("rsvg-convert is required to export PNG")
    png_path = svg_path.with_suffix(".png")
    subprocess.run([converter, "-o", str(png_path), str(svg_path)], check=True)
    return png_path


def slugify(text: str) -> str:
    out = []
    for ch in text.strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "item"


def require_pillow_numpy() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This command requires Pillow and NumPy. Install the repo dependencies "
            "or run with an environment that has them available."
        ) from exc
    return Image, np
