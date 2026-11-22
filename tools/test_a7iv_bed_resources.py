from pathlib import Path
import unittest

from probe_a7iv_bed_resources import trace_resources


CP = Path(__file__).resolve().parents[1] / "extracted/7m4/cp/cpapp-b.bin"


@unittest.skipUnless(CP.is_file(), "requires the pinned A7 IV CP image")
class BedResourceTest(unittest.TestCase):
    def test_native_messages_use_distinct_resource_ids(self):
        separate = trace_resources(CP.read_bytes())
        self.assertEqual([(r["id"], r["filename"], r["capacity_check"], r["offset"]) for r in separate],
                         [(0x4e1, "/usr/sabin/bed_dic1.bin", 0x70500, 0),
                          (0x4e2, "/usr/sabin/bed_dic2.bin", 0x93a00, 0)])
        self.assertEqual([r["request_hex"] for r in separate],
                         ["20000000e104000000000000000000000000000055001100",
                          "20000000e204000000000000000000000000000055001100"])

    def test_response_physical_addresses_do_not_come_from_capacity(self):
        blob = CP.read_bytes()
        separate = trace_resources(blob)
        adjacent = trace_resources(blob, (0x500100000, 0x500170500))
        self.assertEqual(separate[0], adjacent[0])
        self.assertEqual(separate[1]["sa_word_fixture"], 0x400a0000)
        self.assertEqual(adjacent[1]["sa_word_fixture"], 0x40040000 + 0x70500 // 4)
        for key in ("id", "filename", "offset", "handle_fixture", "request_hex", "capacity_check"):
            self.assertEqual(separate[1][key], adjacent[1][key])


if __name__ == "__main__":
    unittest.main()
