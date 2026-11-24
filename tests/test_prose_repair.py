import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from handgen.render.repair import recover_prose_runs


def line_asset(root, words, *, touching=False):
    width = 32 * len(words)
    image = Image.new("RGBA", (width, 22))
    draw = ImageDraw.Draw(image)
    boxes = []
    for i, word in enumerate(words):
        left = 32 * i
        draw.rectangle((left + 2, 3, left + (33 if touching else 23), 19), fill=(31, 36, 43, 100 + i * 10))
        boxes.append({"left": left, "right": left + 25, "text": word})
    path = root / "line.png"
    image.save(path)
    return {"styled_crop": {"path": str(path)}, "ocr_word_boxes": {"words": boxes}}


def read_marker_words(path, words):
    # This mock reads the saved crop. Markers make included neighbor ink
    # observable independently of the word boxes that proposed the crop.
    with Image.open(path) as image:
        values = np.unique(image.getchannel("A"))
    return [words[(int(value) - 100) // 10] for value in values if value]


class ProseRepairTests(unittest.TestCase):
    def test_retains_verified_substrings_and_exact_source_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            words = ["Wrong", "good,", "pen", "older", "again!"]
            run = line_asset(root, words)

            def reader(path, *, psm):
                return {"text": " ".join(read_marker_words(path, words)), "engine": "fixture"}

            source = "Bad  good,\tpen other again!"
            with patch("handgen.render.repair.ocr_image", side_effect=reader):
                recovered = recover_prose_runs(source, run, root)
            self.assertEqual(list(recovered), [1, 4])
            first = recovered[1]
            self.assertEqual(first["text"], "good,\tpen")
            self.assertEqual((first["end_word"], recovered[4]["end_word"]), (3, 5))
            for run in recovered.values():
                self.assertEqual(source[run["source_start"] : run["source_end"]], run["text"])
                self.assertTrue(run["ocr_verification"]["separate_crop_read"])
                self.assertEqual(read_marker_words(Path(run["path"]), words), run["text"].split())

    def test_touching_neighbor_cannot_be_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = line_asset(root, ["Yes", "haze"], touching=True)
            with patch("handgen.render.repair.ocr_image", side_effect=AssertionError("unsafe cut")):
                self.assertEqual(recover_prose_runs("Yes here", run, root), {})

    def test_refuses_context_only_punctuation_and_recovers_neighbor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = line_asset(root, ["Read", "this.", "foreign"])
            actual = ["Read", "this", "foreign"]

            def reader(path, *, psm):
                return {"text": " ".join(read_marker_words(path, actual)), "engine": "fixture"}

            with patch("handgen.render.repair.ocr_image", side_effect=reader):
                recovered = recover_prose_runs("Read this. please", run, root)
            self.assertEqual(list(recovered), [0])
            self.assertEqual(recovered[0]["text"], "Read")
            self.assertEqual(recovered[0]["end_word"], 1)


if __name__ == "__main__":
    unittest.main()
