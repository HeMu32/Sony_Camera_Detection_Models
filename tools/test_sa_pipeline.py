from pathlib import Path
import unittest

from parsers.sony_sa_pipeline import analyze, timing_counts
from parsers.sony_sa_slot_fields import SlotFields


PARTS = Path(__file__).resolve().parents[1] / "debug/a7iv_parts"
MEMBERS = ("SA_ROT900x02", "SA_NV12_0x02", "SA_SHA250x02", "SA_FDN__SA07")


class SaPipelineTest(unittest.TestCase):
    def test_word_pairing_never_crosses_record_boundaries(self):
        idle = SlotFields((0,) * 6, 0, (15,) * 15, 11)
        selected = SlotFields((0,) * 6, 0, (6,) + (15,) * 14, 11)
        self.assertEqual(timing_counts([[idle], [selected]], 6, 34, -1)["compared"], 0)
        self.assertEqual(timing_counts([[idle, selected]], 6, 34, -1)["compared"], 1)

    @unittest.skipUnless(all((PARTS / f"{name}.bin").is_file() for name in MEMBERS),
                         "requires extracted members")
    def test_control_fit_predicts_candidate_bit_changes_on_fdn(self):
        report = analyze(PARTS)
        for code, bit in (("6", 34), ("7", 29)):
            with self.subTest(code=code):
                result = report[code]
                best = result["highest_ranked"][0]
                self.assertEqual((best["suffix_bit"], best["stored_word_offset"]), (bit, -3))
                self.assertGreater(best["correlation"], 0.79)
                for held_out in result["held_out_fdn"].values():
                    self.assertEqual(held_out["both"], held_out["selected"])
        nv12 = report["7"]["controls"]["nv12"]
        self.assertEqual((nv12["both"], nv12["selected"]), (8, 30))
        self.assertEqual(report["7"]["held_out_fdn"]["b62_head"]["both"], 6)
        self.assertEqual(report["6"]["held_out_fdn"]["b62_head"]["both"], 8)


if __name__ == "__main__":
    unittest.main()
