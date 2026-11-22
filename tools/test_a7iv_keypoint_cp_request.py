from pathlib import Path
import unittest

from parsers.sony_sa_post import members
from probe_a7iv_keypoint_cp_request import DRIVERS, probe_request


ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
SP = ROOT / "extracted/7m4/sabin/sa_sabin_cp_cam_post.bin"
MEMBERS = {
    "aed": (0x3e, "SA_AED__SA07", "3Aed6AedDrvE", 0x26bc98, 0x703471),
    "aed2": (0x43, "SA_AED2_SA07", "4Aed27Aed2DrvE", 0x26ee48, 0x708d55),
    "bed": (0x44, "SA_BED__SA07", "3Bed6BedDrvE", 0x26d570, 0x7060e3),
}


@unittest.skipUnless(CP.is_file(), "requires the pinned A7 IV 6.02 CP image")
class KeypointRequestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = CP.read_bytes()

    def test_original_vtable_builds_ordered_requests(self):
        for family, driver in DRIVERS.items():
            job, _, type_suffix, builder, callback = MEMBERS[family]
            fixtures = tuple((0x80001000 + i * 0x1000, 0x500002000 + i * 0x1000,
                              0x19 + i, 0x37 + i) for i in range(driver.max_rows))
            for count in (0, 1, driver.max_rows):
                for flag, option in ((False, 0x88), (True, 0)):
                    with self.subTest(family=family, count=count, flag=flag):
                        run = probe_request(self.cp, family, fixtures[:count], driver_flag=flag)
                        self.assertEqual((run["job"], run["request_builder"], run["queue"]),
                                         (job, builder, 0x930101))
                        self.assertEqual(run["type_name"], f"N6Camera5Minna8Keypoint{type_suffix}")
                        length = 0x10 + 0x20 * count
                        self.assertEqual(run["length"], length)
                        self.assertEqual(run["header"], (0, 0xf0, 0, 0, length, 0, 0))
                        expected = tuple(
                            (0xa0, 0xa9, 0, job, token, option, 0, 0, callback, 0, tag,
                             (param + 0x80000000) // 4, (result - 0x400000000) // 4, 0)
                            for param, result, tag, token in fixtures[:count]
                        )
                        self.assertEqual(run["rows"], expected)

    @unittest.skipUnless(SP.is_file(), "requires the original SA bundle")
    def test_submitted_job_selects_original_member(self):
        jobs = members(SP.read_bytes())
        for family, (expected, name, *_) in MEMBERS.items():
            with self.subTest(family=family):
                run = probe_request(self.cp, family, ((0x80001000, 0x500002000, 3, 1),))
                job = run["rows"][0][3]
                self.assertEqual((run["job"], job, jobs[job].header_job_id), (expected,) * 3)
                self.assertEqual(jobs[job].name, name)


if __name__ == "__main__":
    unittest.main()
