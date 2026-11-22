from functools import reduce
from operator import xor
import unittest

from parsers.wbi2_sections import HEADER, RECORD, SECTOR, SEED, parse


def fixture():
    payload = b"a" * 3 + bytes(SECTOR - 3)
    # The leading zero-fill record occupies memory but no file sectors.
    records = [(0x2000, 0, SEED, 1, 512, 0x80002000), (0x1000, 3, SEED, 0, 3, 0x80001000)]
    table = b"".join(RECORD.pack(*r, reduce(xor, r, SEED), 0) for r in records)
    header = HEADER.pack(b"WBI2", 0, len(records), 0, 0x1001, 123, SECTOR,
                         len(payload), 0, 0, SEED, 515)
    return bytearray(header.ljust(SECTOR, b"\0") + payload + table.ljust(SECTOR, b"\0") + b"trailer")


class Wbi2SectionsTest(unittest.TestCase):
    def test_reverse_mapping_and_padding(self):
        image = parse(fixture())
        file = SECTOR + 1
        pa, va = 0x1001, 0x80001001
        expected = dict(section=1, offset_in_section=1, file=file, physical=pa, virtual=va)
        self.assertEqual(image.locate(file), expected)
        self.assertEqual(image.locate(pa, "physical"), expected)
        self.assertEqual(image.locate(va, "virtual"), expected)
        self.assertIsNone(image.locate(SECTOR + 3))
        self.assertIsNone(image.locate(image.table_start))
        self.assertEqual(image.trailer_start, 3 * SECTOR)
        self.assertIsNone(image.locate(0x2000, "physical")["file"])

    def test_bad_checksum(self):
        data = fixture()
        data[2 * SECTOR] ^= 1
        with self.assertRaisesRegex(ValueError, "descriptor checksum"):
            parse(data)

    def test_truncated_section_table(self):
        with self.assertRaisesRegex(ValueError, "section table truncated"):
            parse(fixture()[:2 * SECTOR + RECORD.size])

    def test_compression_not_supported(self):
        data = fixture()
        data[16] = 1
        with self.assertRaisesRegex(ValueError, "compressed"):
            parse(data)


if __name__ == "__main__":
    unittest.main()
