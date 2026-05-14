import unittest

from handgen.parser import chunk_prose, parse_source


class ParserTests(unittest.TestCase):
    def test_parse_mixed_source_preserves_text(self):
        plan = parse_source("Pick [[0 < a < b]]\nDone")
        self.assertEqual(plan.source_text, "Pick 0 < a < b\nDone")
        self.assertEqual([span.kind for span in plan.spans], ["prose", "math", "prose"])
        self.assertEqual(plan.spans[1].text, "0 < a < b")

    def test_unclosed_math_delimiter_errors(self):
        with self.assertRaisesRegex(ValueError, "unclosed math delimiter"):
            parse_source("Bad [[x")

    def test_chunk_prose_by_words(self):
        self.assertEqual(chunk_prose("one two three four five", max_words=2), ["one two", "three four", "five"])

    def test_chunk_prose_by_chars_preserves_words_when_possible(self):
        self.assertEqual(
            chunk_prose("alpha beta gamma delta", max_chars=12),
            ["alpha beta", "gamma delta"],
        )

    def test_chunk_prose_by_chars_splits_overlong_token(self):
        self.assertEqual(
            chunk_prose("extraordinary proof", max_chars=8),
            ["extraord", "inary", "proof"],
        )


if __name__ == "__main__":
    unittest.main()
