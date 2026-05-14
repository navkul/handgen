import tempfile
import unittest
from pathlib import Path

from handgen.render.math import math_style_profile

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


if __name__ == "__main__":
    unittest.main()
