import unittest

from handgen.chunking import (
    STRATEGIES,
    balanced_dp,
    current,
    get_strategy,
    punctuation_first,
    variable_length_sampling,
)


BENCHMARK_SENTENCES = [
    "The small robot waited by the window while the rain tapped softly against the glass.",
    "After the lecture ended, Maya packed her notebook, checked the time, and hurried across campus to meet her lab partner.",
    "The old bookstore on the corner smelled like dust, coffee, and paper, but its narrow shelves always seemed to hide exactly the book someone needed.",
    "When the team tested the new handwriting model, they noticed that short prompts looked clean at first, yet longer sentences revealed repeated words and uneven spacing.",
    "For tomorrow's demo, we need a worksheet, a few typed answers, several handwriting samples, and a simple way to compare the rendered output against the original style.",
]


def _word_set(text: str) -> list[str]:
    return text.split()


class ChunkingStrategyTests(unittest.TestCase):
    def test_registry_exposes_all_four_strategies(self):
        self.assertEqual(
            set(STRATEGIES),
            {"current", "balanced_dp", "punctuation_first", "variable_length_sampling"},
        )

    def test_get_strategy_returns_callable_and_rejects_unknown(self):
        self.assertIs(get_strategy("balanced_dp"), balanced_dp)
        with self.assertRaisesRegex(ValueError, "unknown chunking strategy"):
            get_strategy("nope")

    def test_strategies_return_chunks_for_every_benchmark_sentence(self):
        for sentence in BENCHMARK_SENTENCES:
            for name, fn in STRATEGIES.items():
                chunks = fn(sentence)
                self.assertGreater(len(chunks), 0, msg=f"{name}: produced zero chunks for {sentence!r}")
                self.assertTrue(all(c.strip() for c in chunks), msg=f"{name}: emitted an empty chunk")

    def test_no_strategy_splits_words(self):
        # The concatenation of chunks (joined by spaces) must reproduce the
        # exact original whitespace-tokenized word sequence — proving no chunk
        # cut a word in half.
        for sentence in BENCHMARK_SENTENCES:
            expected_words = _word_set(sentence)
            for name, fn in STRATEGIES.items():
                if name == "current":
                    # current() may collapse runs of whitespace; normalise the
                    # source for comparison.
                    pass
                chunks = fn(sentence)
                rejoined = " ".join(chunks)
                self.assertEqual(
                    _word_set(rejoined),
                    expected_words,
                    msg=f"{name} altered token sequence: {chunks}",
                )

    def test_balanced_dp_respects_hard_max(self):
        for sentence in BENCHMARK_SENTENCES:
            for chunk in balanced_dp(sentence, target_chars=42, hard_max_chars=52):
                self.assertLessEqual(len(chunk), 52)

    def test_balanced_dp_avoids_tiny_orphan_tails(self):
        # current() produces a 12-char orphan on this sentence; the DP
        # splitter should rebalance so no chunk is under ~half the target.
        sentence = (
            "After the lecture ended, Maya packed her notebook, checked the "
            "time, and hurried across campus to meet her lab partner."
        )
        chunks = balanced_dp(sentence, target_chars=42, hard_max_chars=52)
        self.assertTrue(all(len(c) >= 20 for c in chunks), msg=f"orphan tail in {chunks}")

    def test_punctuation_first_respects_hard_max(self):
        for sentence in BENCHMARK_SENTENCES:
            for chunk in punctuation_first(sentence, target_chars=42, hard_max_chars=52):
                self.assertLessEqual(len(chunk), 52)

    def test_punctuation_first_actually_cuts_on_punctuation_when_useful(self):
        sentence = "First clause, second clause, third clause."
        chunks = punctuation_first(sentence, target_chars=20, hard_max_chars=24)
        # Every output chunk should end at a punctuation-aligned boundary or
        # be the final fragment.
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.rstrip().endswith((",", ";", ":", ".")))

    def test_variable_length_is_deterministic_for_fixed_seed(self):
        sentence = BENCHMARK_SENTENCES[2]
        first = variable_length_sampling(sentence, seed=42)
        second = variable_length_sampling(sentence, seed=42)
        self.assertEqual(first, second)
        # And different seeds (usually) produce different partitions; check at
        # least one of three other seeds differs to confirm the RNG is wired.
        diffs = [variable_length_sampling(sentence, seed=s) != first for s in (1, 2, 3)]
        self.assertTrue(any(diffs))

    def test_variable_length_respects_hard_max(self):
        for sentence in BENCHMARK_SENTENCES:
            for chunk in variable_length_sampling(sentence, hard_max_chars=55):
                self.assertLessEqual(len(chunk), 55)

    def test_current_preserved_behavior(self):
        # The "current" strategy must still match the production default.
        self.assertEqual(
            current("one two three four five six seven"),
            ["one two three four five six", "seven"],
        )

    def test_empty_input_returns_empty_list(self):
        for name, fn in STRATEGIES.items():
            self.assertEqual(fn(""), [], msg=f"{name} did not return [] for empty input")
            self.assertEqual(fn("   "), [], msg=f"{name} did not return [] for whitespace")


if __name__ == "__main__":
    unittest.main()
