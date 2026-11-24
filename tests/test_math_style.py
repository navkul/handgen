import random
import shutil
import tempfile
import unittest
from pathlib import Path

from handgen.ingest.worksheet import glyph_bank_from_manifest, load_worksheet_manifest
from handgen.render.math import math_style_profile, missing_glyphs, render_math_span

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None


def ink_png(path: Path, box: tuple[int, int, int, int], size: tuple[int, int] = (80, 80)) -> None:
    if Image is None or ImageDraw is None:
        raise unittest.SkipTest("Pillow is not installed")
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill=20)
    image.save(path)


class MathStyleTests(unittest.TestCase):
    def test_operator_target_uses_measured_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            style = root / "style.png"
            op = root / "lt.png"
            digit = root / "two.png"
            var = root / "x.png"
            ink_png(style, (10, 10, 60, 70))
            ink_png(op, (10, 20, 50, 55))
            ink_png(digit, (10, 5, 50, 72))
            ink_png(var, (10, 5, 50, 72))
            manifest = {
                "glyphs": {
                    "<": [{"path": str(op)}],
                    "2": [{"path": str(digit)}],
                    "x": [{"path": str(var), "row_id": "lowercase_letters"}],
                }
            }
            profile = math_style_profile(style, manifest, prose_height=64.0)
            self.assertLess(profile["target_operator_height"], profile["target_variable_height"])
            self.assertIn("operator_to_prose_ratio", profile)
            self.assertEqual(profile["max_diffbrush_variable_upscale"], 1.45)
            self.assertIn("symbol_alpha_scale", profile)
            self.assertIn("target_token_gap", profile)

    def test_preflight_accepts_real_stroke_compositions_and_respects_missing_sources(self):
        bank = {label: [Path("glyph.png")] for label in ["1", "2", "<", "+", "-", ".", "alpha"]}
        self.assertEqual(missing_glyphs("α ≤ 1 − 2² ± 2 ÷ 2", bank), [])
        self.assertEqual(missing_glyphs("α ≤ 1 ≠ γ₃", bank), sorted(["≠", "γ", "₃"]))
        self.assertEqual(missing_glyphs("3", bank, allow_digit_fallback=True), [])
        self.assertEqual(missing_glyphs("3", bank, allow_digit_fallback=True, prose=True), ["3"])
        self.assertEqual(missing_glyphs("≠", {"≠": [Path("custom.png")]}), [])

    @unittest.skipUnless(shutil.which("rsvg-convert"), "rsvg-convert is required to rasterize the test run")
    def test_worksheet_digits_diacritics_and_operators_use_baselines_and_source_strokes(self):
        fixtures = Path(__file__).resolve().parents[1] / "data"
        manifest = load_worksheet_manifest(
            fixtures / "worksheets/writer_001/writer_001_worksheet_01/manifest.json"
        )
        bank = glyph_bank_from_manifest(manifest)
        profile = math_style_profile(fixtures / "style_refs/writer_001/natural_prose_primary.png", manifest)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "x² - 2.1 + y₂ ≤ α ÷ 1 ≠ 0 → ∞: ^ ∨ ∧ x′"
            run = render_math_span(
                "eq",
                text,
                bank,
                root,
                random.Random(48),
                style_profile=profile,
                variable_bank={"2": {"styled_crop": {"path": "unused.png"}}},
            )
            self.assertEqual("".join(t["source_character"] for t in run["tokens"]), text)
            by_label = {t["source_character"]: t for t in run["tokens"]}
            self.assertEqual(by_label["2"]["route"], "worksheet_glyph_bank_symbol")
            self.assertEqual(by_label["≤"]["component_labels"], ["<", "-"])
            self.assertEqual(by_label["÷"]["component_labels"], ["-", "."])
            self.assertEqual(by_label["≠"]["route"], "worksheet_stroke_composition")
            self.assertEqual(by_label["∞"]["component_labels"], ["8"])
            self.assertEqual(by_label["→"]["component_labels"], ["-", ">"])
            self.assertLess(by_label["^"]["y"], by_label["∨"]["y"])
            self.assertEqual(by_label["α"]["label"], "alpha")
            self.assertLess(by_label["-"]["height"], 0.30 * by_label["x"]["height"])
            self.assertLess(by_label["."]["height"], 0.36 * by_label["1"]["height"])
            self.assertLess(by_label["²"]["y"] + by_label["²"]["height"], by_label["x"]["baseline"] - 10)
            self.assertGreater(by_label["₂"]["y"] + by_label["₂"]["height"], by_label["x"]["baseline"] + 7)
            self.assertGreater(by_label["y"]["y"] + by_label["y"]["height"], by_label["x"]["baseline"] + 6)
            self.assertGreater(run["ink_bbox"]["x"], 0)
            self.assertLess(run["ink_bbox"]["right"], run["width"])
            self.assertTrue(all(Path(p).is_file() for t in run["tokens"] for p in t.get("source_crops", [])))
            repeat = render_math_span(
                "prose", "bobbing.", bank, root, random.Random(48), prose=True, style_profile=profile
            )
            self.assertEqual(repeat["route"], "worksheet_prose")
            bees = [t for t in repeat["tokens"] if t["source_character"] == "b"]
            self.assertEqual(len({b["asset"] for b in bees}), 1)
            self.assertGreater(len({b["rotation_deg"] for b in bees}), 1)
            self.assertTrue(all(b["baseline"] > 0 for b in bees))


if __name__ == "__main__":
    unittest.main()
