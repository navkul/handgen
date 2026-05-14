import unittest

from handgen.ingest import worksheet
from handgen.ingest.worksheet import ROW_SPECS


class WorksheetSpecTests(unittest.TestCase):
    def test_symbol_worksheet_includes_letter_rows_after_symbols(self):
        row_ids = [row["id"] for row in ROW_SPECS]
        self.assertEqual(row_ids, ["digits", "punctuation_operators", "greek_symbols", "lowercase_letters", "uppercase_letters"])
        self.assertEqual(ROW_SPECS[3]["labels"], list("abcdefghijklmnopqrstuvwxyz"))
        self.assertEqual(ROW_SPECS[4]["labels"], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    def test_wrapped_punctuation_rows_are_consumed_as_one_logical_section(self):
        rows = [
            {"x0": 0, "y0": 0, "x1": 10, "y1": 1},
            {"x0": 0, "y0": 10, "x1": 18, "y1": 11},
            {"x0": 0, "y0": 20, "x1": 5, "y1": 21},
            {"x0": 0, "y0": 30, "x1": 8, "y1": 31},
            {"x0": 0, "y0": 40, "x1": 26, "y1": 41},
            {"x0": 0, "y0": 50, "x1": 26, "y1": 51},
        ]
        counts_by_y = {0: 10, 10: 18, 20: 5, 30: 8, 40: 26, 50: 26}
        original_x_groups = worksheet.x_groups
        worksheet.x_groups = lambda _mask, box: [(0, 1)] * counts_by_y[box[1]]
        try:
            digit_rows, next_idx = worksheet._consume_spec_rows(None, rows, 0, 0)
            operator_rows, next_idx = worksheet._consume_spec_rows(None, rows, next_idx, 1)
            greek_rows, next_idx = worksheet._consume_spec_rows(None, rows, next_idx, 2)
        finally:
            worksheet.x_groups = original_x_groups

        self.assertEqual([row["source_row_index"] for row in digit_rows], [0])
        self.assertEqual([row["source_row_index"] for row in operator_rows], [1, 2])
        self.assertEqual([row["source_row_index"] for row in greek_rows], [3])
        self.assertEqual(next_idx, 4)

    def test_missing_gamma_does_not_shift_later_greek_labels(self):
        labels, missing = worksheet._labels_for_groups(ROW_SPECS[2], 7)
        self.assertEqual(labels, ["alpha", "beta", "theta", "lambda", "mu", "pi", "sigma"])
        self.assertEqual(missing, ["gamma"])


if __name__ == "__main__":
    unittest.main()
