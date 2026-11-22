from pathlib import Path
import unittest

from probe_sa_rotation_arguments import probe


ROTATION = Path(__file__).resolve().parents[1] / "debug/a7iv_parts/SA_ROT900x02.bin"


@unittest.skipUnless(ROTATION.is_file(), "requires the extracted rotation member")
class RotationArgumentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rotation = ROTATION.read_bytes()

    def test_actual_validation_rejects_unaligned_width_and_unknown_mode(self):
        for width, height, mode, code in ((17, 16, 0, 1), (16, 17, 0, 2), (16, 16, 1, 5)):
            with self.subTest(width=width, height=height, mode=mode):
                run = probe(self.rotation, width, height, mode)
                self.assertEqual(run["validator_return"], 1)
                self.assertEqual([(e["address"], e["value"]) for e in run["validator_writes"]],
                                 [(0x48188, code)])
                self.assertNotIn("record", run)

    def test_original_dispatch_selects_size_and_mode_variants(self):
        for width, height, mode, offset, call in (
            (16, 16, 0, 0x35c, 0x6a4), (16, 16, 8, 0x3bc, 0x730),
            (16, 32, 0, 0x41c, 0x48c), (32, 16, 8, 0x47c, 0x5e4),
            (64, 64, 0, 0x41c, 0x48c), (64, 64, 8, 0x47c, 0x5e4),
        ):
            with self.subTest(width=width, height=height, mode=mode):
                run = probe(self.rotation, width, height, mode)
                self.assertEqual(run["validator_return"], 0)
                self.assertEqual(run["record"]["file_offset"], offset)
                self.assertEqual(run["call_pc"], call)
                self.assertEqual(run["first_unknown"],
                                 {"pc": 0x19c, "reason": "unsupported opcode", "opcode": 0x000a})
                body = run["isolated_wrapper_body"]
                self.assertEqual(body["stop"],
                                 {"pc": 0x1b6, "reason": "unsupported opcode", "opcode": 0x0009})
                actual = body["registers_at_0009"]
                self.assertEqual((actual["r0"], actual["r8"], actual["r9"], actual["r10"]),
                                 (offset - 0x340, 0xa0010000, width, height))

    def test_output_is_a_buffer_request_not_a_scalar_return(self):
        for mode in (0, 8):
            small = probe(self.rotation, 16, 16, mode)
            self.assertEqual(small["isolated_post_call_slice"], {
                "fixture_r0_values": [0, 0xdeadbeef], "wait_pcs": [0x6d4],
                "stop": {"pc": 0x6d4, "reason": "wfi"},
                "descriptor_words": [0x20020000, 0x80214000, 0x1f0030, 0x1f0f0f, 1, 0x40000000],
            })
            for width, height in ((16, 32), (32, 16), (32, 48), (48, 32), (64, 64)):
                with self.subTest(width=width, height=height, mode=mode):
                    run = probe(self.rotation, width, height, mode, src_stride=176, dst_stride=208)
                    src = run["image_descriptor_words"]
                    dst = run["isolated_post_call_slice"]["descriptor_words"]
                    local_offset = 2 * (64 - height) if mode == 0 else 128 * (64 - width)
                    self.assertEqual(src, [0x20010000, 0x00805800, ((2 * width - 1) << 16) | 44,
                                           ((2 * width - 1) << 16) | (height - 1) * 0x101,
                                           1, 0x40000000])
                    self.assertEqual(dst, [0x20020000, 0x80814000 + local_offset,
                                           ((2 * height - 1) << 16) | 52,
                                           ((2 * height - 1) << 16) | (width - 1) * 0x101,
                                           1, 0x40000000])

    def test_multi_tile_trace_stops_before_the_second_unknown_call(self):
        for mode, base in ((0, 0x20020000), (8, 0x20026000)):
            with self.subTest(mode=mode):
                trace = probe(self.rotation, 128, 64, mode)["isolated_post_call_slice"]
                self.assertEqual(trace["wait_pcs"], [0x4c8])
                self.assertEqual(trace["stop"],
                                 {"pc": 0x19c, "reason": "unsupported opcode", "opcode": 0x000a})
                self.assertEqual(trace["descriptor_words"],
                                 [base, 0x80814000, 0x7f0030, 0x7f3f3f, 1, 0x40000000])

    def test_mode_remaps_tags_but_does_not_prove_field_roles(self):
        for edge in (16, 64):
            with self.subTest(edge=edge):
                a, b = (probe(self.rotation, edge, edge, m)["record"]["body_words"] for m in (0, 8))
                self.assertEqual([i for i, (x, y) in enumerate(zip(a, b)) if x != y],
                                 [0, 6, 8, 9, 11, 12, 14, 16, 18, 19])
                self.assertEqual([i for i, (x, y) in enumerate(zip(a, b))
                                  if (x ^ y) & 0xffffff], [0])


if __name__ == "__main__":
    unittest.main()
