from pathlib import Path
from struct import pack, pack_into, unpack_from
import tarfile
import unittest

from probe_a7iv_fdn_cp_request import BACKUP_MEMBER
from probe_a7iv_keypoint_cp_result import (
    CANDIDATE_BYTES, LUT_BYTES, RECORD_BYTES, probe_keypoint, published_config,
    published_score_tables, score_resource_paths,
)


ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
FIRMWARE = ROOT / "firmware/7M4_v602/firmware.tar"
SABIN = ROOT / "extracted/7m4/sabin"


def candidate(mode, route):
    data = bytearray(range(CANDIDATE_BYTES))
    for i, offset in enumerate((0x08, 0x18, 0x28, 0x38)):
        pack_into("<Q", data, offset, 0x500030000 + i * 0x100)
    pack_into("<I", data, 0x58, mode)
    pack_into("<I", data, 0x60, route)
    return data


def record(score, flags):
    data = bytearray(pack("<22I", *(100 * i + 1 for i in range(RECORD_BYTES // 4))))
    pack_into("<i", data, 0x4c, score)
    pack_into("<I", data, 0x54, flags)
    return data


def table(values, *, lower_index=1):
    data = bytearray(LUT_BYTES)
    pack_into("<fHH", data, 0, 0, lower_index, 2)
    pack_into("<3f", data, 0x40, *values)
    return data


@unittest.skipUnless(CP.is_file(), "requires the pinned A7 IV 6.02 CP image")
class KeypointResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = CP.read_bytes()

    @unittest.skipUnless(FIRMWARE.is_file(), "requires the published firmware archive")
    def test_native_parameter_and_result_with_released_minna_settings(self):
        with tarfile.open(FIRMWARE, "r") as archive:
            config = published_config(self.cp, archive.extractfile(BACKUP_MEMBER).read())
        self.assertEqual(config[0xf0:0xf3], bytes((0, 250, 250)))
        self.assertEqual(unpack_from("<I", config, 0xf4)[0], 0)
        score = 0xd234
        for mode, route, route_flag in ((1, 1, 1), (2, 2, 2), (3, 3, 1), (4, 4, 2), (4, 5, 0)):
            for flags in (0, 1, 0x80000000):
                with self.subTest(mode=mode, route=route, flags=flags):
                    context, source = candidate(mode, route), record(score, flags)
                    run = probe_keypoint(self.cp, config, context, source)
                    param, result = run["parameter"], run["result"]
                    self.assertEqual((run["vtable"], run["methods"]),
                                     (0xc17eac, (0x2657ec, 0x265e30)))
                    self.assertEqual(run["input_route_flag"], route_flag)
                    self.assertEqual(unpack_from("<5I", param),
                                     (mode, 0x4000c000, 0x40008080, 0x40004040, 0x400040c0))
                    self.assertEqual(param[0x32:0x34], bytes((250, 250)))
                    self.assertEqual(param[0x3e:0x40], b"\0\0")
                    self.assertEqual(param[0x4a:], bytes(0x36))
                    self.assertEqual(unpack_from("<2i", result, 0x14), (score, score) if not flags else (0, 0))
                    self.assertEqual(result[0x1c:0x40], source[:0x24])
                    self.assertEqual(result[0x4c:0x58], source[0x30:0x3c])
                    self.assertEqual(result[0x58:0x5c], source[0x50:0x54])
                    self.assertEqual(result[0x60:], context)

    def test_factory_mask_is_an_allowed_bit_set_not_a_score_threshold(self):
        config = bytearray(0x100)
        pack_into("<I", config, 0xf4, 4)
        for flags, expected in ((0, 0x1234), (4, 0x1234), (5, 0), (1, 0)):
            with self.subTest(flags=flags):
                run = probe_keypoint(self.cp, config, candidate(1, 2), record(0x1234, flags))
                self.assertEqual(unpack_from("<2i", run["result"], 0x14), (expected, expected))

    def test_original_loader_calls_name_the_two_aed2_score_resources(self):
        paths = tuple(map(str, score_resource_paths(self.cp)))
        self.assertEqual(paths, ("/usr/sabin/aed2_dica_s2f.bin", "/usr/sabin/aed2_dicb_s2f.bin"))

    @unittest.skipUnless(all((SABIN / f"aed2_dic{c}_s2f.bin").is_file() for c in ("a", "b")),
                         "requires both published AED2 score resources")
    def test_released_score_tables_change_the_same_candidate_by_mode(self):
        resources = published_score_tables(self.cp, SABIN)
        for raw, expected in (
            (0, (26161, 28360, 0)),
            (32768, (53809, 61254, 32768)),
            (65000, (64374, 64221, 65000)),
            (65536, (65536, 65536, 65536)),
        ):
            for mode, converted in zip((2, 3, 4), expected):
                with self.subTest(mode=mode, raw=raw):
                    result = probe_keypoint(self.cp, bytes(0x100), candidate(mode, 1), record(raw, 0),
                                            fixture_tables=resources)["result"]
                    self.assertEqual(unpack_from("<2i", result, 0x14), (raw, converted))

    def test_only_modes_two_and_three_use_their_distinct_score_table(self):
        tables = (table((0.25, 0.375, 0.5)), table((0.75, 0.625, 0.5)))
        for mode, score, remapped in (
            (1, 0xc000, 0xc000), (4, 0xc000, 0xc000),
            (2, 0xc000, 0xb000), (3, 0xc000, 0x5000),
            (2, 0x10000, 0xc000), (3, 0x10000, 0x4000),
            (2, -1, 0x20000), (3, 0x10001, 0x20000),
        ):
            with self.subTest(mode=mode, score=score):
                run = probe_keypoint(self.cp, bytes(0x100), candidate(mode, 1), record(score, 0),
                                     fixture_tables=tables)
                self.assertEqual(unpack_from("<2i", run["result"], 0x14), (score, remapped))
        bad_index = table((0.25, 0.375, 0.5), lower_index=0)
        run = probe_keypoint(self.cp, bytes(0x100), candidate(2, 1), record(0xc000, 0),
                             fixture_tables=(bad_index, tables[1]))
        self.assertEqual(unpack_from("<2i", run["result"], 0x14), (0xc000, 0x20000))


if __name__ == "__main__":
    unittest.main()
