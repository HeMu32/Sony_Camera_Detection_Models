from hashlib import sha256
from struct import pack_into
import unittest

from sony_sa_post import members


def sample():
    data = bytearray(0x100)
    pack_into("<2sHHHII", data, 0, b"SP", 1, 0x40, 3, 0x10, 0x1c)
    pack_into("<6I", data, 0x10, 0x30, 0, 0x30, 0x50, 0, 0x50)
    pack_into("<IHH", data, 0x30, 0x91130aaa, 7, 0x50)
    data[0x39] = 0x40
    data[0x44:0x50] = b"SA_FDN__SA07"
    return data


class SonySaPostTest(unittest.TestCase):
    def test_aliases_and_empty_slots(self):
        data = sample()
        records = members(bytes(data))
        self.assertEqual(tuple(records), (0x40, 0x42))
        self.assertEqual(records[0x40].name, "SA_FDN__SA07")
        self.assertEqual(records[0x42].header_job_id, 0x40)
        self.assertEqual(records[0x42].offset, 0x30)
        self.assertEqual(records[0x42].sha256, sha256(data[0x30:0x80]).hexdigest())

    def test_rejects_invalid_extent(self):
        data = sample()
        pack_into("<I", data, 0x1c, len(data))
        with self.assertRaisesRegex(ValueError, "extent"):
            members(data)

    def test_rejects_overlapping_distinct_records(self):
        data = sample()
        pack_into("<I", data, 0x24, 0x60)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            members(data)


if __name__ == "__main__":
    unittest.main()
