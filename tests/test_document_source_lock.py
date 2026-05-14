import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from handgen import db
from handgen.render.document import render_document
from handgen.utils import now_iso, sha_file

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


class FakeDiffBrushRunner:
    def generate_chunk(self, *, span_id, text, prompt_text=None, style_ref, out_dir, seed, writer_id):
        path = out_dir / "prose" / f"{span_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (24, 12), (31, 36, 43, 255)).save(path)
        return {
            "id": span_id,
            "display_text": text,
            "conditioning_prompt": text,
            "route": "diffbrush_ocr_verified_prose_token",
            "source_text_sha256": sha_file(path),
            "exactness_certified": True,
            "styled_crop": {"path": str(path), "width": 24, "height": 12, "sha256": sha_file(path)},
            "ocr_verification": {"passed": True, "actual": text, "expected": text},
        }

    def generate_variable(self, **kwargs):
        raise AssertionError("This test source contains no math variables")


class DocumentSourceLockTests(unittest.TestCase):
    def test_render_document_routes_prose_as_source_preserving_chunks(self):
        if Image is None:
            raise unittest.SkipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "handgen.sqlite3"
            style = root / "style.png"
            source = root / "source.txt"
            out_dir = root / "out"
            Image.new("L", (80, 30), 255).save(style)
            source.write_text("one  two", encoding="utf-8")

            with patch("handgen.db.DB_PATH", db_path):
                db.init_db()
                db.create_writer("writer_test")
                with db.connect() as con:
                    con.execute(
                        "INSERT INTO style_refs(writer_id, path, sha256, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        ("writer_test", str(style), sha_file(style), 80, 30, now_iso()),
                    )
                with patch("handgen.render.document.DiffBrushRunner", FakeDiffBrushRunner), patch(
                    "handgen.render.document.render_svg_to_png", self._fake_render_svg_to_png
                ):
                    manifest = render_document("writer_test", source, out_dir)

            self.assertTrue(manifest["source_contract"]["passed"], manifest["source_contract"]["errors"])
            self.assertEqual(manifest["source_contract"]["rendered_source_text"], "one  two")
            prose_spans = [span for span in manifest["spans"] if span["type"] == "prose"]
            self.assertEqual([span["text"] for span in prose_spans], ["one  two"])
            self.assertTrue(all(span["route"] == "diffbrush_ocr_verified_prose_token" for span in prose_spans))
            self.assertTrue(all(span["exactness_evidence"]["visual_text_certified"] for span in prose_spans))

    @staticmethod
    def _fake_render_svg_to_png(svg_path):
        png_path = svg_path.with_suffix(".png")
        Image.new("RGBA", (32, 16), (255, 255, 255, 255)).save(png_path)
        return png_path


if __name__ == "__main__":
    unittest.main()
