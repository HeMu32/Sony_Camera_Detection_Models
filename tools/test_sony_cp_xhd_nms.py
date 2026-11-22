from pathlib import Path
import tarfile
import unittest

from probe_sony_cp_xhd_nms import NmsProbe, TuningFixture
from test_sony_cp_region_scale import CP_IMAGES


COMMON = {
    8: (0, {0: (0, 1, 2, 3), 1: (5, 6, 7, 8)}),
    9: (2, {5: (0, 1, 2), 6: (5, 6, 7)}),
    10: (3, {7: (0, 1)}),
    11: (1, {2: (0, 1), 3: (3, 4), 4: (6, 7)}),
}
ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = {
    "A7R V": ("debug/r5_xhd_fw/firmware.tar", "0110_backup/SYSIPSX-DSLR/GL/CX89200_ALLLANG.bin",
              0xef000, "ea94b038ab3f5af04e9cdf7eb8dee82226a3d4c3f599478e10a27c7f68ce5a56", 218),
    "A1 II": ("debug/a1ii_fw/firmware.tar", "0110_backup/SYSIPSX-DSLR/HC/CX89900_ALLLANG.bin",
              0x100000, "8e30cbe8a1accb86ca1a5e822d55d77e484d93b5ff8e4cc78e4ce672e79889e0", 128),
}


class NativeNmsTest(unittest.TestCase):
    def test_region_caller_work_array_is_read_by_the_original_logger(self):
        images = [path for path, _ in CP_IMAGES.values() if path.is_file()]
        if not images:
            self.skipTest("requires a pinned CP firmware image")
        for image in images:
            probe = NmsProbe(image.read_bytes())
            for mode, expected in ((0, [(0x10, 3, 0x4bc)]),
                                   (1, [(0, 2, 0), (1, 2, 0xa8), (0x10, 3, 0x4bc)])):
                with self.subTest(camera=probe.firmware.model, mode=mode):
                    self.assertEqual(probe.trace_schedule_reader(mode), expected)

    def test_published_backup_records_feed_both_original_nms_packers(self):
        verified = 0
        for camera, (path, _) in CP_IMAGES.items():
            archive, member, offset, digest, threshold = PUBLISHED[camera]
            if not (path.is_file() and (ROOT / archive).is_file()):
                continue
            with self.subTest(camera=camera):
                with tarfile.open(ROOT / archive, "r:") as tar:
                    data = tar.extractfile(member).read()
                probe = NmsProbe(path.read_bytes())
                actual = probe.describe_from_published_backup(10, 4, data, prefill=0xcd)
                self.assertEqual(actual["configuration_id"], "0x60013")
                self.assertEqual(actual["bk4_member_offset"], hex(offset))
                self.assertEqual(actual["record_sha256"], digest)
                self.assertEqual(actual["threshold_rows"][7], [128, threshold, 0xcd, 0xcd])
                self.assertEqual(actual["threshold_per_rows"][7], [76, 76, 0xcd, 0xcd])
                self.assertEqual((actual["threshold_global"], actual["threshold_background_foreground"],
                                  actual["minimum_side_byte"], actual["skip_max_decode"]), (229, 129, 9, 0))
                other = probe.describe_from_published_backup(8, 1, data)
                self.assertEqual(other["configuration_id"], "0x60011")
                self.assertEqual(other["threshold_rows"][0], [128, 128, 0, 0])
                self.assertEqual((other["minimum_side_byte"], other["skip_max_decode"]), (18, 1))
                verified += 1
        if not verified:
            self.skipTest("requires a pinned CP image and its Sony firmware archive")

    def test_original_tuning_constructor_selects_configuration_destination(self):
        images = [path for path, _ in CP_IMAGES.values() if path.is_file()]
        if not images:
            self.skipTest("requires a pinned CP firmware image")
        for image in images:
            probe = NmsProbe(image.read_bytes())
            base = {"A7R V": 0x172407c, "A1 II": 0x1738f40}[probe.firmware.model]
            for target, index in enumerate((0, 1, 0, 2, 3, 4), start=1):
                with self.subTest(camera=probe.firmware.model, target=target):
                    self.assertEqual(probe.tuning_binding(target),
                                     (base + 0x600 * index, 0x60011 + index))

    def test_original_parameters_and_procedure_descriptors(self):
        images = [path for path, _ in CP_IMAGES.values() if path.is_file()]
        if not images:
            self.skipTest("requires a pinned CP firmware image")
        inputs = (0x61000200, 0x6800c500, 0x69000000)
        for image in images:
            probe = NmsProbe(image.read_bytes())
            cases = dict(COMMON)
            cases.update({16: COMMON[10], 17: COMMON[11], 18: (0, {})}
                         if probe.firmware.model == "A1 II" else {12: (0, {})})
            for ident, (nms_id, rows) in cases.items():
                for pairs, special, prefill in (
                        (tuple((34 + i, 163 - i) for i in range(10)), (13, 200, 9), 219),
                        (tuple((212 - i, 45 + i) for i in range(10)), (0, 5, 0), 161)):
                    with self.subTest(camera=probe.firmware.model, id=ident, prefill=prefill):
                        actual = probe.describe(ident, TuningFixture(pairs, *special),
                                                inputs=inputs, prefill=prefill)
                        self.assertEqual(actual["inputs"], inputs)
                        self.assertEqual(actual["accelerator_output_address"], inputs[1] + 0x32030)
                        self.assertEqual(actual["nms_id"], nms_id)
                        self.assertEqual(actual["skip_max_decode"], int(special[2] != 0))
                        self.assertEqual((actual["threshold_global"], actual["threshold_background_foreground"]),
                                         special[:2])
                        for row in range(8):
                            indices = rows.get(row, ())
                            self.assertEqual(actual["threshold_rows"][row],
                                             [pairs[i][0] for i in indices] + [prefill] * (4 - len(indices)))
                            self.assertEqual(actual["threshold_per_rows"][row],
                                             [pairs[i][1] for i in indices] + [prefill] * (4 - len(indices)))
                        self.assertEqual(actual["short_schedule"], [(0x10, 3, 0x4bc)])
                        self.assertEqual(actual["long_schedule"], [(0, 2, 0), (1, 2, 0xa8), (0x10, 3, 0x4bc)])
                        self.assertEqual(actual["parameter_view"], {"block_offset": 0, "size": 0x520})
                        self.assertEqual(actual["result_view"], {"block_offset": 0x520, "size": 0x260})


if __name__ == "__main__":
    unittest.main()
