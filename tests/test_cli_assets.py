import tempfile
import unittest
from pathlib import Path

from handgen.cli import canonical_asset


class CliAssetTests(unittest.TestCase):
    def test_canonical_asset_keeps_existing_canonical_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_dir = root / "data" / "style_refs" / "writer_002"
            canonical_dir.mkdir(parents=True)
            source = canonical_dir / "natural_prose_primary.png"
            source.write_bytes(b"png")

            self.assertEqual(canonical_asset(source, canonical_dir), source.resolve())
            self.assertEqual(source.read_bytes(), b"png")

    def test_canonical_asset_copies_external_path_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "incoming.png"
            source.write_bytes(b"png")
            canonical_dir = root / "data" / "worksheets" / "writer_002"

            dst = canonical_asset(source, canonical_dir)

            self.assertEqual(dst, (canonical_dir / "incoming.png").resolve())
            self.assertEqual(dst.read_bytes(), b"png")


if __name__ == "__main__":
    unittest.main()
