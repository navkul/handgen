import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handgen.ingest import worksheet
from handgen.ingest.worksheet import (
    ROW_SPECS,
    glyph_bank_from_manifest,
    ingest_worksheet,
    load_worksheet_manifest,
)

SOURCE = Path(__file__).resolve().parents[1] / "data/worksheets/writer_001/writer_001_worksheet_01.png"


def silhouette(path):
    with Image.open(path) as image:
        mask = image.convert("L").point(lambda value: 255 if value < 150 else 0)
    mask = mask.crop(mask.getbbox())
    return np.asarray(mask.resize((48, 48), Image.Resampling.BILINEAR)) > 110


class WorksheetSpecTests(unittest.TestCase):
    def test_symbol_worksheet_includes_letter_rows_after_symbols(self):
        row_ids = [row["id"] for row in ROW_SPECS]
        self.assertEqual(
            row_ids,
            ["digits", "punctuation_operators", "greek_symbols", "lowercase_letters", "uppercase_letters"],
        )
        self.assertEqual(ROW_SPECS[3]["labels"], list("abcdefghijklmnopqrstuvwxyz"))
        self.assertEqual(ROW_SPECS[4]["labels"], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    def test_wrapped_rows_assign_across_all_sections(self):
        for counts, expected in [
            ([10, 18, 5, 8, 26, 26], [(0, 1), (1, 3), (3, 4), (4, 5), (5, 6)]),
            # A full Greek alphabet section wrapping after seven entries.
            ([10, 23, 7, 1, 26, 26], [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)]),
        ]:
            rows = [{"boxes": [None] * n} for n in counts]
            self.assertEqual(worksheet._assign_rows(rows, ROW_SPECS, default_layout=True), expected)

    def test_missing_gamma_does_not_shift_later_greek_labels(self):
        labels, missing = worksheet._labels_for_groups(ROW_SPECS[2], 7)
        self.assertEqual(labels, ["alpha", "beta", "theta", "lambda", "mu", "pi", "sigma"])
        self.assertEqual(missing, ["gamma"])

    def test_rotated_shaded_jpeg_matches_labeled_original_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Image.open(SOURCE) as source:
                gray = source.convert("L").resize((909, 1286), Image.Resampling.LANCZOS)
            dark = (255 - np.asarray(gray)) * 0.53
            paper = np.linspace(0.70, 0.96, gray.width)
            page = Image.fromarray(((255 - dark) * paper).round().astype(np.uint8))
            page.rotate(5.25, Image.Resampling.BICUBIC, expand=True, fillcolor=238).save(
                root / "scan.jpg", quality=90
            )
            clean = load_worksheet_manifest(SOURCE.with_suffix("") / "manifest.json")
            scanned = ingest_worksheet("writer", root / "scan.jpg", root / "bank")
            bank = glyph_bank_from_manifest(scanned)
            originals = glyph_bank_from_manifest(clean)
            self.assertEqual(bank.keys(), originals.keys())
            self.assertEqual(len(originals), 92)
            self.assertEqual(scanned["coverage"]["missing_labels"], ["gamma"])
            self.assertAlmostEqual(scanned["deskew_degrees"], -5.25, delta=0.6)
            self.assertEqual([r["physical_row_count"] for r in scanned["rows"]], [1, 2, 1, 3, 3])
            # Compare each label's silhouette, not just the number of boxes.
            ious = []
            for label, paths in bank.items():
                a, b = silhouette(paths[0]), silhouette(originals[label][0])
                ious.append(np.sum(a & b) / np.sum(a | b))
            self.assertGreater(np.median(ious), 0.62)
            self.assertGreater(min(ious), 0.43)

    def test_dark_ruled_paper_preserves_worksheet_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Image.open(SOURCE) as source:
                gray = source.convert("L")
            lines = Image.new("L", gray.size, 255)
            pen = ImageDraw.Draw(lines)
            for y in range(30, gray.height, 60):
                pen.line((15, y, gray.width - 15, y), fill=115, width=2)
            for x in range(40, gray.width, 75):
                pen.line((x, 15, x, gray.height - 15), fill=160, width=1)
            page = Image.fromarray(np.minimum(gray, lines))
            page.thumbnail((1200, 1600), Image.Resampling.LANCZOS)
            page.rotate(4.13, Image.Resampling.BICUBIC, expand=True, fillcolor=255).save(root / "grid.png")
            m = ingest_worksheet("writer", root / "grid.png", root / "bank")
            self.assertAlmostEqual(m["deskew_degrees"], -4.13, delta=0.12)
            self.assertEqual(m["coverage"]["missing_labels"], ["gamma"])
            self.assertEqual(len(m["coverage"]["available_labels"]), 92)

    def test_variants_portable_paths_and_failed_reingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = Image.new("RGBA", (240, 100), (0, 0, 0, 0))
            pen = ImageDraw.Draw(page)
            pen.line([(20, 70), (42, 20), (65, 70)], fill="black", width=5)
            pen.line([(130, 68), (147, 25), (174, 68)], fill="black", width=4)
            page.save(root / "samples.png")
            layout = root / "layout.json"
            layout.write_text(json.dumps({"rows": [{"id": "caps", "labels": ["A", "A"]}]}))
            bankdir = root / "bank"
            good = ingest_worksheet("writer", root / "samples.png", bankdir, layout=layout)
            self.assertEqual(len(good["glyphs"]["A"]), 2)
            manifest_path = bankdir / "manifest.json"
            saved = manifest_path.read_bytes()
            stored = json.loads(saved)
            self.assertFalse(Path(stored["glyphs"]["A"][0]["path"]).is_absolute())
            self.assertEqual(Path(stored["glyphs"]["A"][0]["path"]).parent.name, "a")
            layout.write_text(json.dumps({"rows": [{"id": "caps", "labels": ["A", "B", "C"]}]}))
            with self.assertRaisesRegex(ValueError, "Detected glyph counts"):
                ingest_worksheet("writer", root / "samples.png", bankdir, layout=layout)
            self.assertEqual(manifest_path.read_bytes(), saved)
            relocated = root / "moved"
            bankdir.rename(relocated)
            bank = glyph_bank_from_manifest(load_worksheet_manifest(relocated / "manifest.json"))
            self.assertEqual(len(set(bank["A"])), 2)
            self.assertTrue(all(path.is_relative_to(relocated.resolve()) for path in bank["A"]))


if __name__ == "__main__":
    unittest.main()
