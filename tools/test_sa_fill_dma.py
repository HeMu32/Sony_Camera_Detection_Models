from pathlib import Path
from struct import pack
import unittest

from probe_sa_fill_dma import probe_fill_local
from sa_local_dma import copy_local_rows


FILL = Path(__file__).resolve().parents[1] / "debug/a7iv_parts/SA_FILL_0x01.bin"


class LocalDmaTest(unittest.TestCase):
    def test_materializer_keeps_missing_sources_unknown(self):
        memory = {0x1000: 0x5a}
        spans = {"source_span": (0x1000, 0x1002), "destination_span": (0x2000, 0x2010)}
        with self.assertRaisesRegex(ValueError, "unknown source byte 0x1001"):
            copy_local_rows([(0x2000, 0xc0001000, 0x10000, 0x10000)], memory, **spans)
        self.assertEqual(memory, {0x1000: 0x5a})
        self.assertEqual(copy_local_rows([(0x2000, 0xc0001000, 0, 0)], memory, **spans), {0x2000: 0x5a})
        for invalid in ((0x2000, 0x80001000, 0, 0), (0x2000, 0xc0001000, 1, 1),
                        (0x2000, 0xc0001000, 0xffff0000, 0xffff0000)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                copy_local_rows([invalid], memory, **spans)

    @unittest.skipUnless(FILL.is_file(), "requires pinned SA_FILL member")
    def test_original_fill_mode_expands_nonzero_patterns_in_separate_local_requests(self):
        blob = FILL.read_bytes()
        cases = ((20, (0x03020100, 4), 1), (32, (0xdeadbeef, 0x78), 2), (48, (0x0cab84d1, 0x93), 3))
        for width, pattern, batches in cases:
            with self.subTest(width=width):
                result = probe_fill_local(blob, width=width, pattern=pattern)
                requests = result["local_requests"]
                self.assertEqual([r["submit_pc"] for r in requests], [0xb7a] * batches)
                self.assertEqual([r["rows"] for r in requests],
                                 [[[0x3000 + 40 * i, 0xc0002850, 0x270000, 0x270000]]
                                  for i in range(batches)])
                length_word = (40 * batches - 1) << 16
                self.assertEqual(result["external_request"]["rows"],
                                 [[0x20008000, 0x80003000, length_word, length_word]])
                self.assertEqual(bytes.fromhex(result["buffer_hex"]), pack("<IB", *pattern) * (8 * batches))


if __name__ == "__main__":
    unittest.main()
