from pathlib import Path
import tarfile
import unittest

from probe_sony_adam_xhd import (
    AUX_BYTES, CLASS_BYTES, PROFILES, REGRESSION_BYTES, describe, packed_parameters, run, tensor_buffers,
)
from probe_sony_cp_xhd_nms import NmsProbe


ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = (
    ("r5", "A7R V", "extracted/r5/cp/cpapp-b.bin", "debug/r5_xhd_fw/firmware.tar",
     "0110_backup/SYSIPSX-DSLR/GL/CX89200_ALLLANG.bin", "r5_s10"),
    ("1m2", "A1 II", "debug/deepseek_pose_dispatch/fw/a1ii_cp/cpapp-b.bin", "debug/a1ii_fw/firmware.tar",
     "0110_backup/SYSIPSX-DSLR/HC/CX89900_ALLLANG.bin", "m2_s10"),
)
PLANE_RECORD = bytes.fromhex("01 00 00 00 66 12 cc 1c cc 27 00 25 07 01 f2 7f")
BIRD_RECORDS = bytes.fromhex(
    "02 00 00 00 33 18 00 24 99 20 99 2c 01 00 ea 6a 33 00 99 23 33 04 cc 2c 01 00 d5 55"
)


class AdamXhdTest(unittest.TestCase):
    def load_images(self, directory, cp_file):
        adam_file = ROOT / "extracted" / directory / "adam/adam_app.bin"
        if not adam_file.is_file() or not (ROOT / cp_file).is_file():
            self.skipTest("requires the pinned CP and Adam firmware images")
        return NmsProbe((ROOT / cp_file).read_bytes()), adam_file.read_bytes()

    def test_original_dispatch_and_cpu_writeback_on_explicit_empty_inputs(self):
        for directory, model, cp_file, *_ in FIRMWARE:
            with self.subTest(model=model):
                cp, adam = self.load_images(directory, cp_file)
                params, _ = packed_parameters(cp, 10)
                (procedure,) = cp.trace_schedule_reader(0)
                empty = (bytes(CLASS_BYTES), bytes(REGRESSION_BYTES), bytes(AUX_BYTES))
                self.assertEqual(run(adam, params, *empty, procedure=procedure), (model, bytes(4)))
                with self.assertRaisesRegex(ValueError, "procedure descriptor"):
                    run(adam, params, *empty, procedure=(0x10, 2, 0x4bc))

    def test_original_backends_on_bird_plane_and_skyline_outputs(self):
        for directory, model, cp_file, archive, member_name, plane_dir in FIRMWARE:
            cases = (
                (8, "r5_s08/Bird", BIRD_RECORDS, [(121, 180, 163, 223), (1, 178, 21, 224)]),
                (10, f"{plane_dir}/Flight", PLANE_RECORD, [(92, 144, 199, 185)]),
                (8, "r5_s08/Empty", bytes(4), []),
                (10, "r5_s10/Empty", bytes(4), []),
            )
            with self.subTest(model=model):
                cp, adam = self.load_images(directory, cp_file)
                if not (ROOT / archive).is_file():
                    self.skipTest("requires the matching published backup")
                with tarfile.open(ROOT / archive, "r:") as tar, tar.extractfile(member_name) as file:
                    member = file.read()
                for slot, name, expected, raster in cases:
                    with self.subTest(slot=slot, image=name):
                        prefix = ROOT / "debug/cpu_raw" / name
                        if not prefix.with_name(f"{prefix.name}.20.raw").is_file():
                            self.skipTest("requires the original model output fixture")
                        _, driver, target = PROFILES[slot]
                        params, min_side = packed_parameters(cp, driver, published=(target, member))
                        (procedure,) = cp.trace_schedule_reader(0)
                        self.assertEqual(params[0x10], int(slot == 8))
                        result = run(adam, params, *tensor_buffers(prefix, slot), procedure=procedure)
                        self.assertEqual(result, (model, expected))
                        self.assertEqual([r["xyxy_raster"] for r in describe(result[1], min_side)], raster)
                        if raster:
                            wrong = tensor_buffers(prefix, slot, transpose_aux=True)
                            self.assertEqual(run(adam, params, *wrong, procedure=procedure)[1], bytes(4))


if __name__ == "__main__":
    unittest.main()
