from pathlib import Path
import unittest

from probe_sa_interrupt_control import (STOP_BEFORE_TIME_LOG, TIMER_BIT, TIMER_WAIT,
                                        extract_interrupt_test, probe)


BUNDLE = Path(__file__).resolve().parents[1] / "extracted/7m4/sabin/sa_sabin_mp_cam_post.bin"


@unittest.skipUnless(BUNDLE.is_file(), "requires the pinned A7 IV MP camera-post bundle")
class SaInterruptControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.member = extract_interrupt_test(BUNDLE.read_bytes())

    def test_wrong_interrupt_does_not_read_cancel_word(self):
        trace = probe(self.member, interrupt_status=2, cancel_value=1)
        self.assertEqual(trace["wait"], TIMER_WAIT)
        self.assertEqual(trace["after_wake_stop"], STOP_BEFORE_TIME_LOG)
        self.assertIn("NOT timeout return [0x00000002]", trace["after_wake_log"])
        self.assertEqual(trace["event_reads"], [{"pc": 0x180, "address": 0x48200, "value": 2}])

    def test_timer_checks_for_cancel_value_one_not_any_nonzero_status(self):
        for cancel in (0, 1, 2, 0xffffffff):
            with self.subTest(cancel=cancel):
                trace = probe(self.member, interrupt_status=TIMER_BIT, cancel_value=cancel)
                self.assertTrue(trace["before_wait_log"].endswith("Waiting for Timer."))
                self.assertIn({"pc": 0x174, "address": 0x48208, "value": TIMER_BIT | 2, "size": 4},
                              trace["before_wait_registers"])
                self.assertEqual(trace["event_reads"], [
                    {"pc": 0x180, "address": 0x48200, "value": TIMER_BIT},
                    {"pc": 0x18e, "address": 0x481cc, "value": cancel},
                ])
                self.assertIn({"pc": 0x186, "address": 0x48200, "value": 0xfbffffff, "size": 4},
                              trace["after_wake_registers"])
                self.assertEqual(trace["after_wake_stop"], STOP_BEFORE_TIME_LOG if cancel == 1 else TIMER_WAIT)
                self.assertEqual(trace["after_wake_log"], "CANCELED\r\n*** END !! ***\r\n" if cancel == 1 else "")


if __name__ == "__main__":
    unittest.main()
