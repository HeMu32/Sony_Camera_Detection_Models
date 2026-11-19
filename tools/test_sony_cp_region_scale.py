from pathlib import Path
from struct import pack, unpack
import unittest

from probe_sony_cp_region_scale import RegionScale


ROOT = Path(__file__).resolve().parents[1]
CP_IMAGES = {
    "A7R V": (ROOT / "extracted/r5/cp/cpapp-b.bin", 0x0172407c),
    "A1 II": (ROOT / "debug/deepseek_pose_dispatch/fw/a1ii_cp/cpapp-b.bin", 0x01738f40),
}


def f32_bits(numbers):
    return unpack("<8I", pack("<8f", *numbers))


class RegionScaleTest(unittest.TestCase):
    def test_both_original_selectors_and_region_scale_getters(self):
        selected = (0, 1, 0, 2, 3, 4)
        modes = (None, 0, 1, 2, 2, 2, 3, 3, 4, 0)
        defaults = (0.666, 1.0, 1.5, 1.0, 1.5, 0.666, 1.0, 1.5)
        verified = 0
        for camera, (image, pointer) in CP_IMAGES.items():
            if not image.is_file():
                continue
            with self.subTest(camera=camera):
                probe = RegionScale(image.read_bytes())
                self.assertEqual(probe.firmware.model, camera)
                self.assertEqual(probe.table, tuple(pointer + 0x600 * i for i in selected))
                for mode, slot in enumerate(modes):
                    for target in range(1, 7):
                        self.assertEqual(probe.select(mode, target),
                                         pointer + 0x600 * (selected[target - 1] if slot is None else slot))
                self.assertEqual(probe.evaluate_fixture(0, 1, (0,) * 8), (pointer, f32_bits(defaults)))
                fields = (751, 1, 1250, 321, 65535, 523, 889, 1601)
                self.assertEqual(probe.evaluate_fixture(8, 1, fields),
                                 (pointer + 0x1800, f32_bits(tuple(v / 1000 for v in fields))))
                verified += 1
        if not verified:
            self.skipTest("requires a pinned CP firmware image")


if __name__ == "__main__":
    unittest.main()
