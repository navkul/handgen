import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from handgen import db
from handgen.diffbrush.runner import DiffBrushRunner
from handgen.parser import chunk_prose, parse_source
from handgen.render.assets import InkPiece, WriterAssets
from handgen.render.content import infer_math_spans, prose_slices
from handgen.render.document import render_document
from handgen.render.layout import BOTTOM, LEFT, PAGE_H, PAGE_W, RIGHT, TOP
from handgen.render.math import DEFAULT_MATH_STYLE
from handgen.utils import now_iso, sha_file


def test_source_slices_preserve_markup_boundaries_line_endings_and_long_tokens():
    source = "  Let [[A[i]]] = [[x²]].\r\n\r\nI wonder why.  \n"
    plan = parse_source(source)
    assert plan.source_text == "  Let A[i] = x².\r\n\r\nI wonder why.  \n"
    assert [s.text for s in plan.spans if s.kind == "math"] == ["A[i]", "x²"]
    assert all(plan.source_text[s.source_start:s.source_end] == s.text for s in plan.spans)
    indices = parse_source("Use [[A[B[i]] + 1]], [[x in [0,1)]], and [[x]].")
    assert [s.text for s in indices.spans if s.kind == "math"] == ["A[B[i]] + 1", "x in [0,1)", "x"]
    assert infer_math_spans(plan.lines[0], set()) == plan.lines[0]
    text = " \tabcdefghijklmnop\t\tq  r "
    assert "".join(prose_slices(text, lambda s: chunk_prose(s, max_chars=6))) == text
    long_token = "x" * 230
    slices = list(prose_slices(long_token, lambda text: [text]))
    assert "".join(slices) == long_token and max(map(len, slices)) <= 52


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "handgen.sqlite3")
    db.init_db()
    db.create_writer("notes")
    img = Image.new("L", (32, 32), "white")
    draw = ImageDraw.Draw(img)
    draw.line((6, 5, 24, 26), fill=0, width=3)
    draw.line((23, 6, 7, 25), fill=0, width=3)
    tile = tmp_path / "x.png"
    img.save(tile)
    sheet = tmp_path / "manifest.json"
    sheet.write_text(json.dumps({"glyphs": {"x": [{"path": str(tile)}]}}))
    with db.connect() as con:
        con.execute("INSERT INTO style_refs(writer_id,path,sha256,created_at) VALUES (?,?,?,?)",
                    ("notes", str(tile), sha_file(tile), now_iso()))
        con.execute("INSERT INTO worksheets(writer_id,source_path,manifest_path,created_at) VALUES (?,?,?,?)",
                    ("notes", str(tile), str(sheet), now_iso()))
    return tmp_path


@pytest.mark.skipif(not shutil.which("rsvg-convert"), reason="requires the SVG rasterizer")
def test_later_pages_are_real_and_a_failed_retry_preserves_the_previous_render(catalog):
    source = catalog / "note.txt"
    raw = ("  x\t[[x]]  \r\n" * 36 + "\r\n").encode()
    source.write_bytes(raw)
    out = catalog / "output"
    result = render_document("notes", source, out, prose_mode="samples", require_verified=True)
    assert Path(result["source_file"]).read_bytes() == raw
    assert result["source_contract"]["rendered_source_text"] == raw.decode().replace("[[x]]", "x")
    pages = result["outputs"]["pages"]
    assert len(pages) == 2
    for page in pages:
        bbox = Image.open(page["png"]).convert("L").point(lambda g: 255 if g < 220 else 0).getbbox()
        assert bbox and LEFT <= bbox[0] < bbox[2] <= PAGE_W - RIGHT
        assert TOP <= bbox[1] < bbox[3] <= PAGE_H - BOTTOM
        placed = {s["id"] for s in result["spans"] if s.get("page") == page["number"]}
        svg = ET.parse(page["svg"])
        assert {im.attrib["id"] for im in svg.findall(".//{*}image")} == placed
    assert all(s["route"] == "worksheet_symbol_math" for s in result["spans"] if s["type"] == "math")
    source.write_text("x [[θ]]", encoding="utf-8")
    with pytest.raises(ValueError, match="θ"):
        render_document("notes", source, out, prose_mode="samples")
    assert json.loads((out / "manifest.json").read_text()) == result
    with db.connect() as con:
        assert [row[0] for row in con.execute("SELECT status FROM render_jobs ORDER BY id")] == ["done", "failed"]
    missing, = out.glob("runs/*/missing_glyphs.json")
    assert json.loads(missing.read_text())[0]["characters"] == ["θ"]


@pytest.mark.skipif(not shutil.which("rsvg-convert"), reason="requires the SVG rasterizer")
def test_auto_uses_available_worksheet_for_unmodeled_prose_and_mixed_lines(catalog, monkeypatch):
    monkeypatch.setenv("DIFFBRUSH_CHECKPOINT", str(catalog / "missing-weights.pt"))
    worksheet = catalog / "manifest.json"
    sheet = json.loads(worksheet.read_text())
    sheet["glyphs"]["@"] = sheet["glyphs"]["x"]
    worksheet.write_text(json.dumps(sheet))
    source = catalog / "source.txt"
    source.write_text("x@x [[x]]")
    result = render_document("notes", source, catalog / "fallback-output", carrier_layout=True, require_verified=True)
    assert result["source_contract"]["passed"]
    assert {s["route"] for s in result["spans"] if s["type"] != "space"} == {"worksheet_symbol_math", "worksheet_prose"}
    assert result["warnings"] == []


def test_automatic_adapter_keeps_readable_words_and_replaces_failed_ones(tmp_path, monkeypatch):
    writer = WriterAssets("notes", tmp_path, [], DEFAULT_MATH_STYLE, {"x": [tmp_path / "x.png"]}, {},
                          DiffBrushRunner(), "auto", 12, tmp_path / "adapter.safetensors")
    sampled = InkPiece("x", "worksheet_prose", "worksheet_labels")
    fitted = InkPiece("x", "vatr_word", "ocr_match", details={"vatr": {"target": "x"}})
    monkeypatch.setattr(writer, "from_samples", lambda *_: iter([sampled]))
    monkeypatch.setattr(writer, "vatr_word", lambda _: fitted)
    assert writer.replacement_word("x", "span0") == [fitted]
    fitted.evidence = "best_effort"
    assert writer.replacement_word("x", "span0") == [sampled]
    assert writer.attempts[0]["rejected_vatr"] == [{"target": "x"}]
