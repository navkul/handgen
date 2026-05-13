import unittest

from handgen.utils import escape_xml


class XmlTests(unittest.TestCase):
    def test_escape_xml(self):
        self.assertEqual(escape_xml('a < b & "c"'), "a &lt; b &amp; &quot;c&quot;")


if __name__ == "__main__":
    unittest.main()
