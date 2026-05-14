import tempfile
import unittest
from pathlib import Path

from handgen.render.document import _slot_box_for_display

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None


class SlotDetectionTests(unittest.TestCase):
    def test_slot_box_uses_placeholder_ink_run_not_neighboring_ink(self):
        if Image is None or ImageDraw is None:
            raise unittest.SkipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            crop = Path(tmp) / "carrier.png"
            image = Image.new("RGBA", (140, 44), (255, 255, 255, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((30, 14, 42, 30), fill=(31, 36, 43, 220))
            draw.rectangle((55, 10, 80, 33), fill=(31, 36, 43, 220))
            image.save(crop)

            box, source = _slot_box_for_display(
                crop,
                {
                    "text": "number",
                    "left": 44.0,
                    "right": 86.0,
                    "top": 4.0,
                    "bottom": 38.0,
                    "width": 42.0,
                    "height": 34.0,
                },
                rendered_width=140.0,
                rendered_height=44.0,
                source="estimated_from_slot_carrier_text",
            )

        self.assertEqual(source, "visual_alpha_refined_from_estimated_from_slot_carrier_text")
        self.assertGreater(box["left"], 45.0)
        self.assertLess(box["right"], 90.0)
        self.assertEqual(box["visual_detection"]["method"], "alpha_column_run_overlap")
        self.assertEqual(box["visual_detection"]["run_count"], 2)


if __name__ == "__main__":
    unittest.main()
