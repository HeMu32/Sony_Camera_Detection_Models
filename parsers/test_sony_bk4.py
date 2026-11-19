from struct import pack_into
import unittest

from parsers.sony_bk4 import Backup


def sample():
    data = bytearray(0x140)
    data[0x0c:0x10] = b"BK4\0"
    pack_into("<II", data, 0x10, 2, 4)
    pack_into("<HIHI", data, 0x100, 1, 0, 3, 1)
    for i, ref in enumerate((0x03000130, 0xff000133, 0x00000137, 0xffffffff)):
        pack_into("<HI", data, 0x10c + 6 * i, 0, ref)
    data[0x130:0x13e] = b"abc\x02\0de\x03\0\0\0fgh"
    return data


class SonyBackupTest(unittest.TestCase):
    def test_inline_and_prefixed_lengths(self):
        table = Backup(sample())
        self.assertEqual(table.lookup(0), (0x130, b"abc"))
        self.assertEqual(table.lookup(0x10000), (0x135, b"de"))
        self.assertEqual(table.lookup(0x10001), (0x13b, b"fgh"))
        self.assertIsNone(table.lookup(0x10002))
        self.assertIsNone(table.lookup(0x10100))
        self.assertIsNone(table.lookup(0x20000))

    def test_rejects_noncontiguous_index(self):
        data = sample()
        pack_into("<I", data, 0x108, 0)
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            Backup(data)

    def test_rejects_truncated_prefixed_item(self):
        data = sample()
        pack_into("<I", data, 0x114, 0xff00013f)
        with self.assertRaisesRegex(ValueError, "truncated"):
            Backup(data).lookup(0x10000)
