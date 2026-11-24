import tempfile
import unittest
from pathlib import Path

from handgen.render.ink import (
    soft_diffbrush_ink,
    source_locked_token_ink,
    trim_rgba_right,
    varied_rgba_ink,
)

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None


class InkTests(unittest.TestCase):
    def test_diffusion_paper_tone_does_not_bound_crop_and_trim_uses_retained_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = Image.new("L", (125, 54), 255)
            draw = ImageDraw.Draw(im)
            draw.rectangle((3, 1, 116, 52), fill=226)
            draw.line((25, 24, 39, 29), fill=35, width=3)
            draw.line((75, 12, 81, 44), fill=30, width=3)
            im.save(root / "raw.png")
            crop = soft_diffbrush_ink(root / "raw.png", root / "ink.png", pad=2)
            self.assertLess(crop["width"], 68)
            prefix = trim_rgba_right(root / "ink.png", root / "prefix.png", 38, pad=0)
            self.assertLess(prefix["height"], 16)
            self.assertGreater(prefix["height"], 5)

    def test_source_locked_token_ink_keeps_first_word_cluster(self):
        if Image is None or ImageDraw is None:
            raise unittest.SkipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "sample.png"
            dst = root / "token.png"
            image = Image.new("L", (140, 40), 255)
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 28, 28), fill=20)
            draw.rectangle((36, 10, 52, 28), fill=20)
            draw.rectangle((104, 10, 124, 28), fill=20)
            image.save(src)

            styled = source_locked_token_ink(src, dst, threshold=236, pad=2, alpha_scale=1.0)

            self.assertEqual(styled["detected_cluster_count"], 2)
            self.assertLess(styled["width"], 70)

    def test_varied_rgba_ink_records_per_use_parameters(self):
        if Image is None or ImageDraw is None:
            raise unittest.SkipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "glyph.png"
            dst = root / "variant.png"
            image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((7, 3, 11, 16), fill=(31, 36, 43, 160))
            image.save(src)

            varied = varied_rgba_ink(src, dst, alpha_scale=1.1, shear_x=0.03, thicken=1)

            self.assertTrue(dst.exists())
            self.assertEqual(varied["alpha_mode"], "per_use_varied_rgba")
            self.assertEqual(varied["thicken"], 1)
            self.assertGreater(varied["width"], 0)


if __name__ == "__main__":
    unittest.main()
