from pathlib import Path
import unittest

from probe_sa_compute_controls import nv12_control, sha_control, sha256_words


PARTS = Path(__file__).resolve().parents[1] / "debug/a7iv_parts"
SHA_FILE = PARTS / "SA_SHA250x02.bin"
NV12_FILE = PARTS / "SA_NV12_0x02.bin"


class SaComputeControlTest(unittest.TestCase):
    def test_sha_constants_exact_integer_roots(self):
        iv, k = sha256_words()
        self.assertEqual((len(iv), len(k)), (8, 64))
        self.assertEqual((iv[0], iv[-1], k[0], k[-1]),
                         (0x6a09e667, 0x5be0cd19, 0x428a2f98, 0xc67178f2))

    @unittest.skipUnless(SHA_FILE.is_file(), "requires pinned SHA member")
    def test_sha_distinguishes_two_original_paths(self):
        result = sha_control(SHA_FILE.read_bytes())
        long_hash = "be6091c8ed9bc6fb4f2239c21ad3332041ebf5603819b40ef5e6757e66563911"
        short_hash = "a2d6d3a2b68880ecb6829c2e9a1082b8e9ff170dc76da47adf58a274d3fe79d5"
        for i, worker in enumerate(result["workers"]):
            with self.subTest(worker=i):
                rounds = i % 2 == 0
                self.assertEqual(worker["sha_round_words_written"], 64 if rounds else 0)
                self.assertEqual(worker["record_size"], 232 if rounds else 168)
                self.assertEqual(worker["record_sha256"], long_hash if rounds else short_hash)
                regs = worker["isolated_wrapper_body"]["r0_to_r10"]
                values = [r["value"] for r in worker["mailbox_reads"]]
                mailbox = worker["fixture_mailbox"]
                if rounds:
                    self.assertEqual(regs[5:8], [mailbox + 0x14, *values])
                    self.assertEqual([r["address"] for r in worker["mailbox_reads"]],
                                     [mailbox + 0xc, mailbox + 0x10])
                else:
                    self.assertEqual(regs[5:8], values)
                    self.assertEqual([r["address"] for r in worker["mailbox_reads"]],
                                     [mailbox, mailbox + 4, mailbox + 8])
        self.assertEqual(len(result["workers"]), 5)
        self.assertEqual(result["isolated_outer_call"]["record_member"], 0x4bc)
        self.assertEqual(result["isolated_outer_call"]["call_pc"], 0x9f4)
        self.assertEqual(result["record_word_observations"], {
            "short_low24_sample": [0x071219, 0x11000f, 0x03130d, 0x0a0200],
            "long_complement_triplet_indices": [14, 21, 29],
            "long_bit31_words": [0x8eff0000, 0x95ff0000, 0x9dff0000, 0x95ff0000],
            "long_first_seen_7bit_indices": list(range(2, 49)),
        })

    @unittest.skipUnless(NV12_FILE.is_file(), "requires pinned NV12 member")
    def test_nv12_mailbox_values_and_optional_continuation(self):
        blob = NV12_FILE.read_bytes()
        off, on = (nv12_control(blob, followup) for followup in (False, True))
        self.assertEqual(off["first_call"], on["first_call"])
        self.assertEqual(off["first_call"]["record_dsp"], 0x2034)
        self.assertEqual(off["first_call"]["record_member"], 0xe54)
        self.assertEqual([(w["address"], w["value"]) for w in off["setup_writes"][1:]],
                         [(0x40000 + i * 4, x) for i, x in enumerate(off["fixture_words"][:4])])
        self.assertEqual(off["isolated_continuation"], off["wait"])
        self.assertIsNone(off["second_call"])
        self.assertEqual(on["second_call"]["call_pc"], 0x21d2)
        self.assertEqual(on["second_call"]["record_dsp"], 0x20bc)
        self.assertEqual(on["second_call"]["record_member"], 0xedc)
        self.assertEqual(on["second_call"]["leading_words"], [0x01000000] * 2)

    @unittest.skipUnless(SHA_FILE.is_file() and NV12_FILE.is_file(),
                         "requires pinned SHA and NV12 members")
    def test_mailbox_perturbation_separates_register_sources(self):
        sha, nv12 = SHA_FILE.read_bytes(), NV12_FILE.read_bytes()
        counts = [64, 0, 64, 0, 64]
        for i in (1, 2):
            with self.subTest(fixture=i):
                sha_words = tuple(i * j * 0x11000 for j in range(1, 9))
                result = sha_control(sha, mailbox_words=sha_words)
                self.assertEqual([w["sha_round_words_written"] for w in result["workers"]], counts)
                for worker in result["workers"]:
                    base = worker["fixture_mailbox"]
                    self.assertEqual([r["value"] for r in worker["mailbox_reads"]],
                                     [sha_words[(r["address"] - base) // 4]
                                      for r in worker["mailbox_reads"]])
                nv12_words = (i * 0x10000, i * 0x20000, i * 0x30000, i * 0x440, 5)
                result = nv12_control(nv12, False, mailbox_words=nv12_words)
                self.assertEqual([(w["address"], w["value"]) for w in result["setup_writes"][1:]],
                                 [(0x40000 + j * 4, w) for j, w in enumerate(nv12_words[:4])])


if __name__ == "__main__":
    unittest.main()
