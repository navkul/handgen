import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handgen import db
from handgen.ingest.paper import paper_luminance
from handgen.ingest.style import (
    prepare_style,
    register_style,
    style_word_crops,
    writer_style_lines,
    writer_word_exemplars,
)
from handgen.ingest.style_metrics import (
    estimated_space_width,
    estimated_word_width,
    writing_metrics,
)
from handgen.utils import now_iso, sha_file

ROOT = Path(__file__).resolve().parents[1]
PANGRAM = ROOT / "data/style_refs/writer_002/natural_prose_primary.png"
TEXT = "The quick brown fox jumps over the lazy dog"


class StyleLearningTests(unittest.TestCase):
    def test_bundled_line_recovers_case_and_known_word_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            m = prepare_style(PANGRAM, pack, TEXT)
            self.assertEqual([w["text"] for w in m["words"]], TEXT.split())
            # Centers of manually inspected words in the original 1024 x 64 strip.
            for word, expected_x in zip(m["words"], [35, 111, 208, 295, 385, 484, 563, 655, 745]):
                x0, y0, x1, y1 = word["source_bbox"]
                self.assertAlmostEqual((x0 + x1) / 2, expected_x, delta=2)
                with Image.open(pack / word["path"]) as img:
                    self.assertEqual(img.mode, "RGBA")
                    self.assertEqual(img.size, (x1 - x0, y1 - y0))
            refs = [pack / w["path"] for w in m["windows"]]
            self.assertTrue(all(512 <= w["width"] <= 1024 and w["height"] == 64 for w in m["windows"]))
            self.assertEqual(len(style_word_crops(refs)), 9)

    def test_ruled_tilted_photo_with_uneven_light_and_dust(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Image.open(PANGRAM) as img:
                line = np.minimum(np.asarray(img.convert("L"), dtype=float) / 236, 1)[:, :790]
            page = np.tile(np.linspace(183, 243, 1080), (650, 1))
            page += 7 * np.sin(np.linspace(0, 3 * np.pi, 650))[:, None]
            page[41:45, 56:60] = 20
            for y in range(60, 590, 55):
                page[y:y + 2, 30:-30] -= 32
            for x, y in [(62, 112), (135, 315), (101, 471)]:
                page[y:y + 64, x:x + 790] *= line
            photo = Image.fromarray(page.clip(0, 255).astype(np.uint8)).rotate(
                -3.25, Image.Resampling.BICUBIC, expand=True, fillcolor=220)
            photo.save(root / "photo.jpg", quality=92)
            clean = prepare_style(PANGRAM, root / "clean", TEXT)
            scanned = prepare_style(root / "photo.jpg", root / "scanned", "\n".join([TEXT] * 3))
            self.assertAlmostEqual(scanned["deskew_degrees"], 3.25, delta=0.25)
            self.assertEqual([w["text"] for w in scanned["words"]], TEXT.split() * 3)
            for i, observed in enumerate(scanned["words"]):
                with Image.open(root / "clean" / clean["words"][i % 9]["path"]) as clean_img:
                    truth = np.asarray(clean_img.getchannel("A")) > 64
                with Image.open(root / "scanned" / observed["path"]) as scan_img:
                    raster = scan_img.getchannel("A").resize((truth.shape[1], truth.shape[0]), Image.Resampling.BILINEAR)
                pred = np.asarray(raster) > 64
                dice = 2 * np.count_nonzero(pred & truth) / (pred.sum() + truth.sum())
                self.assertGreater(dice, 0.77, observed["text"])

    def test_ambiguous_words_remain_unlabeled_and_blank_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = Image.new("RGBA", (150, 70), (0, 0, 0, 0))
            im.save(root / "blank.png")
            with self.assertRaisesRegex(ValueError, "No handwriting"):
                prepare_style(root / "blank.png", root / "blank")
            Image.new("L", im.size).save(root / "dark.png")
            with self.assertRaisesRegex(ValueError, "Too little visible paper"):
                prepare_style(root / "dark.png", root / "dark")
            draw = ImageDraw.Draw(im)
            for x in (25, 55, 85):
                draw.ellipse((x, 20, x + 15, 42), outline=(0, 0, 0, 255), width=2)
            im.save(root / "unclear.png")
            m = prepare_style(root / "unclear.png", root / "pack", "oo oo")
            self.assertEqual(m["words"], [])
            self.assertGreater(len(m["windows"]), 0)
            with self.assertRaisesRegex(ValueError, "one transcript line"):
                prepare_style(root / "unclear.png", root / "mismatch", "one\ntwo")
            self.assertEqual(paper_luminance(im).getpixel((0, 0)), 255)

    def test_crop_assigns_whole_strokes_when_neighboring_line_bounds_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Image.open(PANGRAM) as img:
                pixels = np.asarray(img.convert("L"), dtype=float)
            sample = np.clip((pixels[:, :790] - 20) * 255 / 216, 0, 255).astype(np.uint8)
            page = np.full((320, 1000), 255, dtype=np.uint8)
            for x, y in [(75, 35), (70, 71), (74, 108), (73, 226)]:
                region = page[y:y + 64, x:x + 790]
                region[:] = np.minimum(region, sample)
            Image.fromarray(page).save(root / "page.png")
            crop = (48, 29, 950, 184)
            m = prepare_style(root / "page.png", root / "pack", "\n".join([TEXT] * 3), crop=crop)
            self.assertEqual(m["crop"], list(crop))
            self.assertEqual([w["text"] for w in m["words"]], TEXT.split() * 3)
            boxes = [line["source_bbox"] for line in m["lines"]]
            self.assertGreater(boxes[0][3], boxes[1][1])
            self.assertGreater(boxes[1][3], boxes[2][1])
            self.assertAlmostEqual(boxes[0][0], 81, delta=2)
            with self.assertRaisesRegex(ValueError, "oriented source image"):
                prepare_style(root / "page.png", root / "outside", crop=(-1, 0, 950, 320))

    def test_reregister_preserves_labels_and_same_basename_keeps_both_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            catalog = data / "handgen.sqlite3"
            db.init_db(catalog)
            writer_dir = data / "style_refs" / "w"
            other = ROOT / "data/style_refs/writer_001/natural_prose_primary.png"
            with db.connect(catalog) as con:
                con.execute("INSERT INTO writers(id, created_at) VALUES (?, ?)", ("w", now_iso()))
                register_style(con, "w", PANGRAM, writer_dir, TEXT)
                register_style(con, "w", other, writer_dir)
                register_style(con, "w", PANGRAM, writer_dir)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM style_refs").fetchone()[0], 2)
                self.assertEqual(sha_file(writer_dir / PANGRAM.name), sha_file(PANGRAM))
                self.assertEqual(set(writer_word_exemplars(con, "w")), set(TEXT.split()))
                con.execute("INSERT INTO style_refs(writer_id, path, sha256, created_at) VALUES (?, ?, ?, ?)",
                            ("w", str(writer_dir / "lost.png"), "obsolete-sha", now_iso()))
            moved = Path(tmp) / "moved"
            data.rename(moved)
            with db.connect(moved / catalog.name) as con:
                refs = writer_style_lines(con, "w")
                self.assertEqual(len(refs), 3)
                self.assertNotEqual(refs[0].parent, refs[1].parent)
                self.assertTrue(all(p.is_file() and p.is_relative_to(moved.resolve()) for p in refs))

    def test_width_calibration_holds_out_whole_words_and_keeps_the_better_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            for writer, references in [("001", ["primary", "alt01", "alt04"]), ("002", ["primary"])]:
                bank = {}
                for reference in references:
                    stem = f"writer_{writer}_{reference}"
                    transcript = (ROOT / f"examples/transcripts/{stem}.txt").read_text()
                    sample = ROOT / f"data/style_refs/writer_{writer}/natural_prose_{reference}.png"
                    pack = prepare_style(sample, Path(tmp) / stem, transcript)
                    self.assertEqual([r["text"] for r in pack["words"]], transcript.split())
                    for word in pack["words"]:
                        bank.setdefault(word["text"], []).append(word)
                metrics = writing_metrics(bank)
                if writer == "001":
                    self.assertEqual(metrics["heldout_word_families"], 19)
                    self.assertLess(metrics["heldout_fitted_mape"], metrics["heldout_fixed_mape"] - 0.04)
                    self.assertLess(estimated_word_width("little", metrics, body_height=32),
                                    estimated_word_width("mammal", metrics, body_height=32))
                else:
                    self.assertEqual(metrics["heldout_word_families"], 8)
                    self.assertGreater(metrics["heldout_fitted_mape"], metrics["heldout_fixed_mape"])
                    self.assertIsNone(estimated_word_width("rabbit", metrics, body_height=32))
                gap = estimated_space_width(metrics, body_height=32, fallback=16)
                self.assertTrue(16 < gap < metrics["median_space"] * 32)
            empty = writing_metrics({})
            self.assertIsNone(estimated_word_width("sample", empty, body_height=32))
            self.assertEqual(estimated_space_width(empty, body_height=32, fallback=16), 16)


if __name__ == "__main__":
    unittest.main()
