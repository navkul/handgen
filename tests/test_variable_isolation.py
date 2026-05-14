import tempfile
import unittest
from pathlib import Path

from handgen.diffbrush.runner import DiffBrushRunner

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None


class VariableIsolationTests(unittest.TestCase):
    def test_variable_variant_masks_out_disconnected_neighbor_ink(self):
        if Image is None or ImageDraw is None:
            raise unittest.SkipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            image = Image.new("L", (80, 70), 255)
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 18, 36, 54), fill=20)
            draw.rectangle((40, 28, 42, 34), fill=20)
            image.save(sample)

            variants = DiffBrushRunner()._variable_component_variants("p", sample, root)

            self.assertTrue(variants)
            raw = Image.open(variants[0]["raw_crop"]["path"]).convert("L")
            pixels = raw.load()
            self.assertEqual(pixels[24, 16], 255)
            self.assertEqual(variants[0]["component"]["isolation"], "connected_component_mask")


if __name__ == "__main__":
    unittest.main()
