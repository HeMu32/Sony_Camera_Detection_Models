import unittest
from struct import pack, unpack

from probe_sony_cp_confidence import ConfidenceProbe, GETTER_NAMES
from test_sony_cp_region_scale import CP_IMAGES


def f32(value):
    return unpack("<f", pack("<f", value))[0]


def threshold(base, adjustment, fallback):
    expected_base = f32(base / 10000 if 1 <= base <= 10000 else fallback)
    expected_adjustment = f32(adjustment / 100000000 if 1 <= adjustment <= 9999 else 0)
    return f32(expected_base + expected_adjustment)


class SonyConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.probes = [ConfidenceProbe(image.read_bytes()) for image, _ in CP_IMAGES.values()
                       if image.is_file()]
        if not self.probes:
            self.skipTest("requires a pinned CP firmware image")

    def test_face_selects_sensitivity_fields_and_checks_both_u16_ranges(self):
        indices = (2, 0, 1, 2, 3, 4, 2)
        fixtures = ((0, 0), (1, 1), (10000, 9999), (10001, 10000), (65535, 0), (0, 4321))
        for probe in self.probes:
            for name, start, fallback in zip(GETTER_NAMES[:2], (0xac, 0xc0), (0.60, 0.98)):
                for sensitivity, index in enumerate(indices):
                    for b, a in fixtures:
                        with self.subTest(camera=probe.firmware.model, name=name, sensitivity=sensitivity, b=b, a=a):
                            result = probe.evaluate_fixture(name, 8, sensitivity, base_raw=b, adjustment_raw=a)
                            start_field = start + index * 4
                            self.assertEqual(result["fixture_field_offsets"], [hex(start_field), hex(start_field + 2)])
                            self.assertEqual(result["threshold"], threshold(b, a, fallback))
                            diagnostic = probe.firmware.model == "A1 II" and sensitivity == 6
                            self.assertEqual(result["suppressed_diagnostic_names"], [name + "Threshold"] if diagnostic else [])
                invalid = probe.evaluate_fixture(name, 8, 7, base_raw=6750)
                self.assertEqual((invalid["fixture_field_offsets"], invalid["threshold"]), ([], 0))

    def test_xhd_reads_both_original_fields_for_every_bounded_type_part_and_sensitivity(self):
        routes = (0, 0, 0, 1, 2, 0, 1, 0, 1)
        mapped = (2, 0, 1, 2, 3, 4, 2, 2)
        for probe in self.probes:
            config = CP_IMAGES[probe.firmware.model][1] + 0x1d0
            self.assertEqual(probe.targets, tuple(config + i * 0x600 for i in (0, 0, 1, 2, 2, 2, 3, 3, 4)))
            for name, start, adjustment in zip(GETTER_NAMES[2:], (0x120, 0x198), (0xd6, 0xea)):
                for kind, route in enumerate(routes):
                    for part in range(4):
                        for sensitivity, index in enumerate(mapped):
                            r = probe.evaluate_fixture(name, 3, sensitivity, xhd_type=kind, part=part,
                                                       base_raw=6543, adjustment_raw=9999)
                            fields = [start + 10 * (4 * route + part) + 2 * index, adjustment + 4 * index]
                            self.assertEqual(r["fixture_field_offsets"], [hex(f) for f in fields])
                            self.assertEqual(r["threshold"], threshold(6543, 9999, 0.5))
                for target in range(9):
                    r = probe.evaluate_fixture(name, target, 3, base_raw=10001, adjustment_raw=10000)
                    self.assertEqual(r["threshold"], 0.5)
                    self.assertEqual(r["configuration_va"], hex(probe.targets[target]))


if __name__ == "__main__":
    unittest.main()
