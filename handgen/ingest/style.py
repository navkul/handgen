"""Prepare single-column natural handwriting samples and optional labeled words."""

from __future__ import annotations

import math
import os
import shutil
import sqlite3
from itertools import combinations, pairwise, zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from ..utils import now_iso, read_json, sha_file, sha_text, write_json
from .style_lines import (
    body_metrics,
    condition_image,
    handwriting_lines,
    ink_gaps,
    model_windows,
    primary_condition_image,
    source_box,
    style_word_crops,
)

__all__ = ["prepare_style", "primary_condition_image", "register_style", "style_word_crops",
           "writer_style_lines", "writer_word_exemplars"]

# A change to preprocessing must yield fresh model inputs and exemplars.
PACK_VERSION = 2


def _word_ranges(mask: np.ndarray, tokens: list[str], body_height: int) -> tuple[list[tuple[int, int]], str]:
    """Align labels at real ink gaps, abstaining when geometry is ambiguous."""
    gaps = sorted(ink_gaps(mask), key=lambda pair: pair[1] - pair[0], reverse=True)
    needed = len(tokens) - 1
    if len(gaps) < needed:
        return [], "fewer clear ink gaps than transcript spaces"
    widths = [b - a for a, b in gaps]
    if needed == 0:
        if widths and widths[0] >= max(4, body_height * 0.85):
            return [], "the single transcript word covers a likely word space"
        return [(0, mask.shape[1])], ""
    smallest = widths[needed - 1]
    floor = max(2, body_height * 0.45)
    unclear = "word spaces cannot be separated confidently from gaps between pen strokes"
    if smallest < floor:
        return [], unclear
    # If one internal pen lift is as wide as a word space, word lengths from
    # the transcript can disambiguate it. Only compare genuinely plausible
    # gaps. Competing fits of similar quality leave the whole line unlabeled.
    options = [gaps[:needed]]
    if len(widths) > needed and smallest < max(1.3 * widths[needed], widths[needed] + 1):
        certain = [gap for gap, width in zip(gaps, widths) if width > smallest * 1.35]
        uncertain = [gap for gap, width in zip(gaps, widths) if max(floor, smallest / 1.35) <= width <= smallest * 1.35]
        n = needed - len(certain)
        if math.comb(len(uncertain), n) > 256:
            return [], unclear
        options = [certain + list(choice) for choice in combinations(uncertain, n)]
    units = [sum(0.55 if c in "ilI.,'!:;|" else 1.3 if c in "mwMW" else 1 for c in word) for word in tokens]
    xs = np.flatnonzero(mask.any(axis=0))
    fits = []
    for spaces in options:
        gaps_lr = sorted(spaces)
        # Blank space carries no ink; evaluate each observed ink width.
        starts = [xs[0], *(b for _, b in gaps_lr)]
        ends = [*(a for a, _ in gaps_lr), xs[-1] + 1]
        advances = (np.array(ends) - starts) / units
        relative = advances / np.median(advances)
        if relative.min() >= 0.29 and relative.max() <= 3.2:
            cost = float(np.square(np.log(relative)).sum())
            fits.append((cost, gaps_lr))
    if not fits:
        return [], "word widths disagree with the transcript"
    fits.sort()
    if len(fits) > 1 and fits[1][0] - fits[0][0] < 0.6:
        return [], unclear
    cuts = [0, *((a + b) // 2 for a, b in fits[0][1]), mask.shape[1]]
    return list(pairwise(cuts)), ""


def prepare_style(source: Path, output: Path, transcript: str | None = None, *,
                  crop: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Extract cleaned conditioning windows and crop supported transcript words."""
    lines, geometry = handwriting_lines(source, crop)
    text_lines = [line.split() for line in transcript.splitlines() if line.strip()] if transcript is not None else []
    if transcript is not None and len(text_lines) != len(lines):
        raise ValueError(f"Transcript has {len(text_lines)} nonblank lines, handwriting has {len(lines)}. "
                         "Supply one transcript line per handwritten line.")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": PACK_VERSION, "source_file": source.name, "source_sha256": sha_file(source),
        **geometry, "transcript": transcript, "lines": [], "windows": [], "words": [],
    }
    for i, line_sample in enumerate(lines):
        x, y, right, bottom = line_sample.box
        line_ink, line_mask = line_sample.ink, line_sample.mask
        baseline, body_height = body_metrics(line_mask)
        line = {"line_index": i, "source_bbox": source_box((x, y, right, bottom), geometry),
                "analysis_bbox": [x, y, right, bottom], "baseline": baseline, "body_height": body_height}
        manifest["lines"].append(line)
        for j, (left, end) in enumerate(model_windows(line_mask, body_height)):
            window = condition_image(line_ink[:, left:end])
            path = output / f"line_{i:03d}_{j:02d}.png"
            window.save(path)
            manifest["windows"].append({
                "path": path.name, "sha256": sha_file(path), "width": window.width, "height": window.height,
                "line_index": i, "window_index": j, "source_bbox": source_box((x + left, y, x + end, bottom), geometry),
            })
        if transcript is None:
            continue
        ranges, reason = _word_ranges(line_mask, text_lines[i], body_height)
        if reason:
            line["alignment"] = reason
            continue
        line["alignment"] = "user transcript aligned to isolated word ink"
        previous_end = None
        for j, ((a, b), token) in enumerate(zip(ranges, text_lines[i])):
            ys, xs = np.nonzero(line_ink[:, a:b])
            tx, ty = a + int(xs.min()), int(ys.min())
            ex, ey = a + int(xs.max()) + 1, int(ys.max()) + 1
            alpha = Image.fromarray(line_ink[ty:ey, tx:ex])
            sample = Image.new("RGBA", alpha.size, (8, 8, 8, 0))
            sample.putalpha(alpha)
            path = output / f"word_{i:03d}_{j:03d}.png"
            sample.save(path)
            manifest["words"].append({
                "text": token, "path": path.name, "sha256": sha_file(path), "width": sample.width, "height": sample.height,
                "baseline": baseline - ty, "body_height": body_height, "line_index": i, "word_index": j,
                "gap_before": tx - previous_end if previous_end is not None else None,
                "source_bbox": source_box((x + tx, y + ty, x + ex, y + ey), geometry),
            })
            previous_end = ex
    write_json(output / "manifest.json", manifest)
    return manifest


def _store_original(source: Path, snapshot: Path, writer_dir: Path, digest: str) -> Path:
    if source.is_relative_to(writer_dir) and sha_file(source) == digest:
        return source
    for target in (writer_dir / source.name, writer_dir / "sources" / digest / source.name):
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(snapshot, target)
        except FileExistsError:
            if not target.is_file() or sha_file(target) != digest:
                continue
        return target
    raise RuntimeError(f"A stored source at the content-addressed path differs from {source}")


def register_style(
    con: sqlite3.Connection, writer_id: str, source: Path, writer_dir: Path, transcript: str | None = None,
    *, crop: tuple[int, int, int, int] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish the original and complete, content-addressed pack before registering it."""
    writer_dir = writer_dir.resolve()
    writer_dir.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    region = list(crop) if crop else None
    with TemporaryDirectory(dir=writer_dir) as temporary:
        stage = Path(temporary)
        snapshot = stage / "source" / source.name
        snapshot.parent.mkdir()
        shutil.copyfile(source, snapshot)
        digest = sha_file(snapshot)
        if transcript is None:
            previous = con.execute("""
                SELECT a.manifest_path FROM style_artifacts AS a
                JOIN style_refs AS s ON s.id = a.style_ref_id
                WHERE s.writer_id = ? AND s.sha256 = ? ORDER BY s.id DESC LIMIT 1
            """, (writer_id, digest)).fetchone()
            if previous and (writer_dir / previous["manifest_path"]).is_file():
                old = read_json(writer_dir / previous["manifest_path"])
                if old.get("crop") == region:
                    transcript = old.get("transcript")
        pack_stage = stage / "prepared"
        manifest = prepare_style(snapshot, pack_stage, transcript, crop=crop)
        key = f"v{PACK_VERSION}-{digest}"
        if crop:
            key += "-r" + "_".join(str(v) for v in crop)
        if transcript is not None:
            key += "-" + sha_text(transcript)
        target = writer_dir / "prepared" / key
        target.mkdir(parents=True, exist_ok=True)
        raw = _store_original(source, snapshot, writer_dir, digest)
        # Identical packs produce identical crops. Each replace is atomic and
        # the manifest is published last, including on retries or concurrent adds.
        for record in manifest["windows"] + manifest["words"]:
            name = record["path"]
            os.replace(pack_stage / name, target / name)
        manifest_path = target / "manifest.json"
        os.replace(pack_stage / "manifest.json", manifest_path)

    if not con.in_transaction:
        con.execute("BEGIN IMMEDIATE")
    row = con.execute("""
        SELECT s.id, a.manifest_path FROM style_refs AS s
        LEFT JOIN style_artifacts AS a ON a.style_ref_id = s.id
        WHERE s.writer_id = ? AND s.sha256 = ? ORDER BY s.id DESC LIMIT 1
    """, (writer_id, digest)).fetchone()
    if row is None:
        cur = con.execute("""
            INSERT INTO style_refs(writer_id, path, sha256, width, height, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (writer_id, str(raw), digest, *manifest["source_size"], now_iso()))
        style_id = cur.lastrowid
    else:
        style_id = row["id"]
        con.execute("UPDATE style_refs SET path = ?, width = ?, height = ? WHERE id = ?",
                    (str(raw), *manifest["source_size"], style_id))
        if transcript is None and row["manifest_path"]:
            previous = writer_dir / row["manifest_path"]
            if previous.is_file():
                annotated = read_json(previous)
                if (annotated.get("transcript") is not None and annotated.get("version") == PACK_VERSION
                        and annotated.get("crop") == region):
                    return previous, annotated
    con.execute("""
        INSERT INTO style_artifacts(style_ref_id, manifest_path) VALUES (?, ?)
        ON CONFLICT(style_ref_id) DO UPDATE SET manifest_path = excluded.manifest_path
    """, (style_id, str(manifest_path.relative_to(writer_dir))))
    return manifest_path, manifest


def _registered_samples(con: sqlite3.Connection, writer_id: str):
    db_file = con.execute("SELECT file FROM pragma_database_list WHERE name = 'main'").fetchone()[0]
    writer_dir = Path(db_file).parent / "style_refs" / writer_id
    rows = con.execute("""
        SELECT s.path, s.sha256, a.manifest_path FROM style_refs AS s
        LEFT JOIN style_artifacts AS a ON a.style_ref_id = s.id
        WHERE s.writer_id = ? ORDER BY s.id DESC
    """, (writer_id,))
    seen: set[str] = set()
    available = False
    unavailable_legacy = []
    for row in rows:
        if row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        if row["manifest_path"]:
            path = writer_dir / row["manifest_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared reference {path}. Re-register the source with handgen style add.")
            manifest = read_json(path)
            if manifest.get("source_sha256") != row["sha256"]:
                raise ValueError(f"Style pack {path} differs from the registered source. Re-register the style image.")
            available = True
            yield path.parent, manifest
        else:
            # Pre-pack registrations pointed directly at a line image, often
            # using an absolute path from the repository's previous location.
            raw = Path(row["path"])
            for path in (raw, writer_dir / raw.name, writer_dir / "sources" / row["sha256"] / raw.name):
                if path.is_file() and sha_file(path) == row["sha256"]:
                    available = True
                    yield path.parent, {"windows": [{"path": path.name}], "words": []}
                    break
            else:
                # Older releases could overwrite a previous upload with the
                # same basename. Its obsolete row must not block newer samples.
                unavailable_legacy.append(raw)
    if unavailable_legacy and not available:
        raise FileNotFoundError(f"Missing registered style reference {unavailable_legacy[0]}. "
                                "Re-register it with handgen style add.")


def writer_style_lines(con: sqlite3.Connection, writer_id: str) -> list[Path]:
    """Return prepared lines interleaved across this writer's registered sources."""
    by_source = []
    for root, manifest in _registered_samples(con, writer_id):
        windows = sorted(manifest["windows"], key=lambda item: -item.get("width", 0))
        by_source.append([_verified_crop(root, entry) for entry in windows])
    paths = [path for group in zip_longest(*by_source) for path in group if path is not None]
    return paths


def _verified_crop(root: Path, entry: dict) -> Path:
    path = root / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing style crop {path}. Re-register the source with handgen style add.")
    if entry.get("sha256") and sha_file(path) != entry["sha256"]:
        raise ValueError(f"Style crop {path} has changed. Re-register the source with handgen style add.")
    return path


def writer_word_exemplars(con: sqlite3.Connection, writer_id: str) -> dict[str, list[dict[str, Any]]]:
    bank: dict[str, list[dict[str, Any]]] = {}
    for root, manifest in _registered_samples(con, writer_id):
        for word in manifest["words"]:
            path = _verified_crop(root, word)
            record = dict(word, path=str(path), source_sha256=manifest["source_sha256"])
            bank.setdefault(word["text"], []).append(record)
    return bank
