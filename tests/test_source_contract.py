import unittest

from handgen.eval.source_contract import verify_source_contract


class SourceContractTests(unittest.TestCase):
    def test_manifest_reconstructs_source_from_rendered_spans(self):
        manifest = {
            "source_text": "Pick 0 < a",
            "lines": [{"index": 0, "text": "Pick 0 < a", "span_ids": ["p0", "s0", "p1", "s1", "m0"]}],
            "spans": [
                {"id": "p0", "type": "prose", "text": "Pick"},
                {"id": "s0", "type": "prose_space", "text": " "},
                {"id": "p1", "type": "prose", "text": "0"},
                {"id": "s1", "type": "prose_space", "text": " "},
                {
                    "id": "m0",
                    "type": "math",
                    "text": "< a",
                    "math": {
                        "tokens": [
                            {"label": "<", "source_character": "<"},
                            {"label": "space", "source_character": " "},
                            {"label": "a", "source_character": "a"},
                        ]
                    },
                },
            ],
        }

        result = verify_source_contract(manifest)

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["rendered_source_text"], "Pick 0 < a")

    def test_manifest_fails_when_rendered_line_mismatches_source(self):
        manifest = {
            "source_text": "extra words",
            "lines": [{"index": 0, "text": "extra words", "span_ids": ["p0"]}],
            "spans": [{"id": "p0", "type": "prose", "text": "extra"}],
        }

        result = verify_source_contract(manifest)

        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"][0]["code"], "line_reconstruction_mismatch")


if __name__ == "__main__":
    unittest.main()
