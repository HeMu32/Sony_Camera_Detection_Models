import unittest

from probe_sony_cp_xhd_nms import NmsProbe, TuningFixture
from test_sony_cp_region_scale import CP_IMAGES


COMMON = {
    8: (0, {0: (0, 1, 2, 3), 1: (5, 6, 7, 8)}),
    9: (2, {5: (0, 1, 2), 6: (5, 6, 7)}),
    10: (3, {7: (0, 1)}),
    11: (1, {2: (0, 1), 3: (3, 4), 4: (6, 7)}),
}


class NativeNmsTest(unittest.TestCase):
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
