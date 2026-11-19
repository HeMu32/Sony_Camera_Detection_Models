from random import Random
from struct import pack_into
import unittest

from parsers.xhd_cp_result import COUNT_OFFSET, Geometry, RAW, RAW_START, XhdResult, decode
from probe_sony_cp_xhd_result import XhdResultProbe
from test_sony_cp_region_scale import CP_IMAGES


def fixture(records):
    result = bytearray(RAW_START + RAW.size * len(records))
    pack_into("<I", result, COUNT_OFFSET, len(records))
    for index, record in enumerate(records):
        RAW.pack_into(result, RAW_START + index * RAW.size, *record)
    return bytes(result)


class XhdResultTest(unittest.TestCase):
    def test_record_order_clip_confidence_and_minimum_side(self):
        geometry = Geometry((640, 480), (320, 320), (160, 160), min_side_raw=20)
        payload = fixture([
            (2048, 4096, 8192, 8192, 0, 0x42, 16384),
            (2048, 2048, 2560, 2560, 3, 0x21, 32768),
            (2048, 2048, 8192, 2560, 3, 0x22, 32768),
            (-5120, 0, 1024, 1024, 255, 0x79, 65535),
        ])
        self.assertEqual(decode(payload, geometry), [
            XhdResult((40, 80, 160, 160), 0.5, 0, 0x42),
            XhdResult((40, 40, 50, 50), 0.0, 3, 0x21),
            XhdResult((40, 40, 160, 50), 0.0, 3, 0x22),
            XhdResult((0, 0, 20, 20), 65535 / 32768, 8, 0x79),
        ])
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode(payload[:-1], geometry)

    def test_independent_converter_matches_both_original_cp_result_slices(self):
        model_files = [image for image, _ in CP_IMAGES.values() if image.is_file()]
        if not model_files:
            self.skipTest("requires a pinned CP firmware image")
        rng = Random(145)
        probes = [XhdResultProbe(image.read_bytes()) for image in model_files]
        for case in range(24):
            geometry = Geometry((rng.randint(80, 1920), rng.randint(60, 1080)),
                                (rng.randint(65, 640), rng.randint(65, 640)),
                                (rng.randint(0, 1920), rng.randint(0, 1080)),
                                90 * (case % 4), rng.randint(0, 255))
            rows = [(*[rng.randint(-32768, 32767) for _ in range(4)],
                     rng.randrange(16), rng.randrange(256), rng.randrange(65536))
                    for _ in range((0, 1, 2, 8, 50, 50)[case % 6])]
            payload = fixture(rows)
            for probe in probes:
                with self.subTest(camera=probe.firmware.model, case=case):
                    self.assertEqual(probe.convert_fixture(payload, geometry), decode(payload, geometry))
        header = bytearray(RAW_START)
        pack_into("<I", header, COUNT_OFFSET, 51)
        payload = bytes(header)
        with self.assertRaisesRegex(ValueError, "exceeds firmware limit"):
            decode(payload, geometry)
        for probe in probes:
            with self.assertRaisesRegex(ValueError, "invalid XHD count branch"):
                probe.convert_fixture(payload, geometry)


if __name__ == "__main__":
    unittest.main()
