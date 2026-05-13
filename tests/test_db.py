from pathlib import Path
import unittest
from unittest import mock

from handgen.db import connect, create_writer, init_db


class DbTests(unittest.TestCase):
    def test_init_db_creates_writer_table(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite3"
            init_db(db_path)
            with connect(db_path) as con:
                tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            self.assertIn("writers", tables)
            self.assertIn("render_jobs", tables)

    def test_create_writer_uses_default_db(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "handgen.sqlite3"
            with mock.patch("handgen.db.DB_PATH", db_path):
                create_writer("writer_001")
                with connect(db_path) as con:
                    row = con.execute("SELECT id FROM writers WHERE id = 'writer_001'").fetchone()
            self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
