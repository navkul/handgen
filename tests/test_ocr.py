import unittest

from handgen.eval.ocr import normalize_text, verify_ocr_text


class OcrVerificationTests(unittest.TestCase):
    def test_accepts_exact_normalized_text(self):
        result = verify_ocr_text("Pick", "pick\n")
        self.assertTrue(result["passed"], result["reject_reasons"])
        self.assertEqual(normalize_text("Pick."), "pick")

    def test_rejects_repeated_or_hallucinated_text(self):
        repeated = verify_ocr_text("Pick", "Pick Pick")
        hallucinated = verify_ocr_text("Pick", "Pick numbers")

        self.assertFalse(repeated["passed"])
        self.assertIn("repeated_tokens", repeated["reject_reasons"])
        self.assertFalse(hallucinated["passed"])
        self.assertIn("hallucinated_tokens", hallucinated["reject_reasons"])

    def test_marks_extra_suffix_as_trimmable_prefix(self):
        result = verify_ocr_text("Prove that x y is", "Prove that x y is cqy")
        self.assertFalse(result["passed"])
        self.assertTrue(result["prefix_matches_expected"])
        self.assertEqual(result["trimmable_suffix_tokens"], ["cqy"])


if __name__ == "__main__":
    unittest.main()
