import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from handgen.render.document import _carrier_for_line, _chunk_carrier


class CarrierPlanningTests(unittest.TestCase):
    def test_fragmented_math_example_uses_blank_carriers(self):
        known = set()
        cases = [
            ("Prove that x + y is odd.", "Prove that blank is odd.", [("Prove that blank is odd.", "Prove that blank is odd. and then the line")]),
            (
                "If x is true, then y is false.",
                "If blank is true, then blank is false.",
                [("If blank is true, then blank is false.", "If blank is true, then blank is false.")],
            ),
            (
                "Let a = 6. Then a must be even.",
                "Let blank. Then blank must be even.",
                [("Let blank. Then blank must be even.", "Let blank. Then blank must be even. and then")],
            ),
            (
                "There exists a number that is prime.",
                "There exists a number that is prime.",
                [("There exists a number that is prime.", "There exists a number that is prime.")],
            ),
        ]

        for source, slot_carrier_text, chunks in cases:
            carrier = _carrier_for_line(source, known)

            self.assertEqual(carrier["slot_carrier_text"], slot_carrier_text)
            self.assertEqual([(chunk["text"], chunk["prompt_text"]) for chunk in _chunk_carrier(slot_carrier_text)], chunks)

    def test_single_variables_use_letter_placeholder_and_comparison_context(self):
        known = set()
        carrier = _carrier_for_line("Show that p + q is greater than r.", known)

        self.assertEqual(carrier["slot_carrier_text"], "Show that blank is greater than blank.")
        self.assertEqual([item["text"] for item in carrier["insertions"]], ["p + q", "r"])

    def test_short_blank_carrier_stays_whole_and_uses_prompt_padding(self):
        chunks = _chunk_carrier("Let blank. Then blank is useful.")

        self.assertEqual([chunk["text"] for chunk in chunks], ["Let blank. Then blank is useful."])
        self.assertEqual([chunk["prompt_text"] for chunk in chunks], ["Let blank. Then blank is useful. and then the"])

    def test_short_blank_clause_stays_whole(self):
        chunks = _chunk_carrier("If blank, then blank is positive.")

        self.assertEqual([chunk["text"] for chunk in chunks], ["If blank, then blank is positive."])
        self.assertEqual([chunk["prompt_text"] for chunk in chunks], ["If blank, then blank is positive. and then"])

    def test_short_tail_chunk_is_merged_back(self):
        chunks = _chunk_carrier("Suppose blank is positive, and blank is small.")

        self.assertEqual([chunk["text"] for chunk in chunks], ["Suppose blank is positive, and blank is small."])
        self.assertEqual([chunk["prompt_text"] for chunk in chunks], ["Suppose blank is positive, and blank is small."])

    def test_let_and_then_slot_predicates_stay_whole(self):
        chunks = _chunk_carrier("Let amount be fixed. Then term is useful.")

        self.assertEqual([chunk["text"] for chunk in chunks], ["Let amount be fixed. Then term is useful."])
        self.assertEqual([chunk["prompt_text"] for chunk in chunks], ["Let amount be fixed. Then term is useful."])

    def test_placeholder_policy_can_be_loaded_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.json"
            policy.write_text(
                """{
                  "slot_placeholders": [
                    {"max_visible_len": 1, "placeholder": "term"},
                    {"max_visible_len": 3, "placeholder": "thing"},
                    {"max_visible_len": 10000, "placeholder": "amount"}
                  ]
                }""",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"HANDGEN_PLACEHOLDER_POLICY": str(policy)}):
                known = set()
                carrier = _carrier_for_line("Show that p + q is greater than r.", known)

        self.assertEqual(carrier["slot_carrier_text"], "Show that thing is greater than term.")

    def test_context_placeholder_policy_overrides_global_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.json"
            policy.write_text(
                """{
                  "slot_placeholders": [
                    {"max_visible_len": 10000, "placeholder": "number"}
                  ],
                  "slot_placeholders_by_context": {
                    "let_short_slot": [
                      {"max_visible_len": 10000, "placeholder": "value"}
                    ]
                  }
                }""",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"HANDGEN_PLACEHOLDER_POLICY": str(policy)}):
                known = set()
                carrier = _carrier_for_line("Let R - S be fixed. Then h = 3 is useful.", known)

        self.assertEqual(carrier["slot_carrier_text"], "Let value be fixed. Then number is useful.")
        self.assertEqual(carrier["insertions"][0]["slot_placeholder"]["context"], "let_short_slot")
        self.assertEqual(carrier["insertions"][1]["slot_placeholder"]["context"], "then_slot_clause")

    def test_carrier_prompt_policy_can_be_loaded_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "carrier_policy.json"
            policy.write_text(
                """{
                  "contexts": {
                    "if_short_slot": {
                      "prompt_templates": [
                        "If {placeholder}",
                        "If {placeholder} is useful"
                      ]
                    }
                  }
                }""",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"HANDGEN_CARRIER_PROMPT_POLICY": str(policy)}):
                chunks = _chunk_carrier("If value")

        self.assertEqual(chunks[0]["prompt_variants"], ["If value", "If value is useful"])

    def test_carrier_prompt_policy_uses_placeholder_specific_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "carrier_policy.json"
            policy.write_text(
                """{
                  "contexts": {
                    "then_slot_clause": {
                      "prompt_templates": [
                        "Then {placeholder} is useful"
                      ],
                      "prompt_templates_by_placeholder": {
                        "term": [
                          "Then {placeholder} is positive",
                          "Then {placeholder} is true"
                        ]
                      }
                    }
                  }
                }""",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"HANDGEN_CARRIER_PROMPT_POLICY": str(policy)}):
                term_chunks = _chunk_carrier("then term is true.")
                number_chunks = _chunk_carrier("then number is true.")

        self.assertEqual(term_chunks[0]["prompt_variants"], ["Then term is positive", "Then term is true"])
        self.assertEqual(number_chunks[0]["prompt_variants"], ["Then number is useful"])


if __name__ == "__main__":
    unittest.main()
