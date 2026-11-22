from pathlib import Path
import unittest

from parsers.sony_sa_program_planes import HEADER_SLOTS, PROGRAM_BASE, boot_programs
from parsers.sony_sa_slot_fields import SlotFields, analyze, slot_population


PARTS = Path(__file__).resolve().parents[1] / "debug/a7iv_parts"
NAMES = ("SA_SHA250x02", "SA_ROT900x02", "SA_NV12_0x02", "SA_FDN__SA07", "SA_PDN__SA07")


class SaSlotFieldsTest(unittest.TestCase):
    def test_raw_fields_include_the_cross_plane_sixth_field(self):
        default = SlotFields.decode(HEADER_SLOTS[1])
        self.assertEqual(default.fields13, (0x1f7,) * 6)
        self.assertEqual(default.selectors4, (15,) * 15)
        self.assertEqual(default.top4, 11)
        self.assertEqual(default.encode(), HEADER_SLOTS[1])
        q = SlotFields((0, 0, 0, 0, 0x1000, 0x1357), 0x123, (0,) * 15, 7)
        a, b, _ = q.encode()
        self.assertEqual(b & 0x3fff, 1 | (0x1357 << 1))
        self.assertEqual(a, 0)
        self.assertEqual(SlotFields.decode(q.encode()), q)

    @unittest.skipUnless(all((PARTS / f"{n}.bin").is_file() for n in NAMES), "requires extracts")
    def test_same_slot_field_boundaries_win_on_all_three_cohorts(self):
        report = analyze(PARTS)
        counts = {"controls": (147, 371, 372, 1),
                  "fdn": (298, 508, 515, 7), "pdn": (507, 646, 655, 9)}
        for name, (slots, diagonal, total, misses) in counts.items():
            with self.subTest(cohort=name):
                r = report[name]
                self.assertEqual(r["unique_slots"], slots)
                best, next_best = r["highest_ranked_boundaries"][:2]
                self.assertEqual((best["width"], best["start_bit"]), (13, 0))
                self.assertGreater(best["score"] - next_best["score"], 0.07)
                matrix = r["nondefault_by_code_and_field"]
                self.assertTrue(all(row.index(max(row)) == i and row.count(max(row)) == 1
                                    for i, row in enumerate(matrix)))
                self.assertEqual(sum(row[i] for i, row in enumerate(matrix)), diagonal)
                self.assertEqual(sum(r["occurrences_by_code"]), total)
                self.assertEqual(len(r["counterexamples"]), misses)

    @unittest.skipUnless((PARTS / "SA_SHA250x02.bin").is_file(), "requires extracted SHA")
    def test_sha_repetitions_preserve_field_high_bits_and_vary_other_bits(self):
        sha = (PARTS / "SA_SHA250x02.bin").read_bytes()
        image = boot_programs(sha)[PROGRAM_BASE + 0x1000]
        for aligned in ((15, 22, 30), (16, 23, 31), (17, 24, 32)):
            samples = [SlotFields.decode(image.at(i)).field_parts for i in aligned]
            for lane in (0, 2, 3, 4):
                self.assertEqual(len({s[lane][0] for s in samples}), 1)
        self.assertEqual([SlotFields.decode(image.at(i)).field_parts[0] for i in (15, 22, 30)],
                         [(0x1a, 8, 5), (0x1a, 6, 4), (0x1a, 8, 5)])
        for words in slot_population([(sha, (0x100,), False)]):
            self.assertEqual(SlotFields.decode(words).encode(), words)


if __name__ == "__main__":
    unittest.main()
