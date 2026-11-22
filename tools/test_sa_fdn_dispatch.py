from hashlib import sha256
from pathlib import Path
from struct import pack
import unittest

from compare_sa_payload_records import MARKER, compare, spans
from extract_sa_fdn_payloads import payloads
from probe_sa_fdn_boot import FIXTURE_BASE, probe as boot_probe, probe_startup_control
from probe_sa_fdn_dispatch import MEMBER_SHA
from run_sa_fdn_dispatch import DispatchMachine, bank_entry, front_end


MEMBER = Path(__file__).resolve().parents[1] / "debug/a7iv_parts/SA_FDN__SA07.bin"
ROTATION = MEMBER.with_name("SA_ROT900x02.bin")
CP = MEMBER.parents[2] / "extracted/7m4/cp/cpapp-b.bin"

class SaScalarInstructionTest(unittest.TestCase):
    def make_instruction(self, opcode, lhs, rhs, flag):
        machine = DispatchMachine({"stage_dsp_address": "0", "first_transfer_destination": "0",
                                   "stack": "2", "mailbox_selector": "0x17800"}, pack("<H", opcode))
        machine.reg[1:3], machine.flag = [lhs, rhs], flag
        machine.step()
        self.assertEqual(machine.pc, 2)
        return machine.reg[1], machine.flag

    def test_lsls_uses_bit31_as_carry_and_wraps(self):
        for a, expected in ((0x80000000, (0, True)), (0x40000003, (0x80000006, False)),
                            (0xffffffff, (0xfffffffe, True))):
            with self.subTest(a=a):
                self.assertEqual(self.make_instruction(0x0071, a, 0, True), expected)

    def test_sbcs_preserves_unsigned_borrow_through_wraparound(self):
        for a, b, incoming, expected in ((8, 6, True, (1, False)), (0, 0, True, (0xffffffff, True)),
                                         (0xffffffff, 0xffffffff, True, (0xffffffff, True)),
                                         (0xffffffff, 0xffffffff, False, (0, False))):
            with self.subTest(a=a, b=b, incoming=incoming):
                self.assertEqual(self.make_instruction(0x2121, a, b, incoming), expected)

    def test_sign_extension_reads_only_the_selected_lane_without_changing_flags(self):
        for opcode, value, flag, expected in (
                (0x1921, 0xdeadbe80, True, 0xffffff80),
                (0x1921, 0xdeadbe7f, False, 0x7f),
                (0x1a21, 0xdead8000, True, 0xffff8000),
                (0x1a21, 0xdead7fff, False, 0x7fff)):
            with self.subTest(opcode=opcode, value=value):
                self.assertEqual(self.make_instruction(opcode, 0xaaaaaaaa, value, flag),
                                 (expected, flag))


class SaStartupControlTest(unittest.TestCase):
    @unittest.skipUnless(CP.is_file(), "requires the extracted A7 IV 6.02 CP image")
    def test_original_startup_reuses_mailbox_address_for_boot_descriptor(self):
        run = probe_startup_control(CP.read_bytes())
        self.assertEqual(run["first_stop"], {"pc": FIXTURE_BASE + 0x78,
                         "reason": "unmodeled read", "address": 0x48004, "size": 4})
        self.assertEqual(run["fixture_waits"], [{"pc": FIXTURE_BASE + pc, "reason": "wfi"}
                                               for pc in (0x104, 0x188)])
        self.assertEqual(run["scratch_address"], 0x17800)
        self.assertEqual(run["descriptor_words"],
                         [0x500101a0, 0x3ff85e06, 0x1f0000, 0, 1, 0x40000000])
        self.assertEqual(run["descriptor_pointer_writes"], [0x17800, 0x17818, 0])
        self.assertEqual(run["fixture_stage2"], {
            "src_block_offset": 0x200, "destination": 0x5ff00000, "size": 0x580,
        })
        self.assertEqual(run["stage2_stop"], {"pc": 0x5ff000e0, "reason": "unmodeled read",
                                             "address": 0x481d0, "size": 4})

    @unittest.skipUnless(CP.is_file() and MEMBER.is_file(),
                         "requires the extracted CP image and FDN member")
    def test_startup_calls_original_fdn_but_does_not_supply_a_detector_request(self):
        result = probe_startup_control(CP.read_bytes(), MEMBER.read_bytes())
        run = result["fdn_handoff_fixture"]
        self.assertEqual(
            [(e["pc"], e["address"], e["value"]) for e in run["fixture_startup_register_reads"][:3]],
            [(0x5ff000e0, 0x481d0, 0x40ff), (0x5ff000e2, 0x481d4, 0x45001000),
             (0x5ff000e4, 0x481d8, 0xffffffff)])
        self.assertEqual(run["header_request"], {
            "source_register_value": 0x45001000, "destination": 0x5ff20000, "size": 0x100,
        })
        self.assertEqual(run["before_header_copy"], {"pc": 0x5ff001a0, "reason": "unmodeled read",
                                                     "address": 0x5ff20018, "size": 4})
        self.assertEqual(run["fdn_entry_stop"], {"pc": 0x5ff20094, "reason": "unmodeled read",
                                                "address": 0x48004, "size": 4})
        self.assertEqual(run["fdn_mapped_bytes"], 0x31b10)
        self.assertEqual(run["fdn_boot_waits"], [{"pc": pc, "reason": "wfi"}
                                                for pc in (0x5ff20150, 0x5ff20288)])
        self.assertEqual(run["fdn_external_mapping"],
                         {"file_offset": 0x500, "destination": 0x50000000, "size": 0x3800})
        self.assertEqual(run["fdn_external_stop"], {"pc": 0x50003022, "reason": "unmodeled read",
                                                    "address": 0x49e00, "size": 4})
        self.assertEqual(run["isolated_bank_wait"], {"pc": 0x1150, "reason": "wfi"})
        self.assertEqual(run["stale_mailbox_reads"], [
            {"pc": 0x115c, "address": 0x17800, "size": 4, "value": FIXTURE_BASE + 0x1a0},
        ])
        self.assertEqual(run["after_stale_wake"], run["isolated_bank_wait"])
        self.assertEqual([(e["address"], e["value"]) for e in run["writes_after_stale_wake"]],
                         [(0x48200, 0xfffffffe), (0x48218, 1), (0x48208, 1)])

        producer = result["fdn_mailbox_producer_fixture"]
        self.assertEqual(producer["producer_stop"], {"pc": 0x50000d44, "reason": "slice boundary"})
        for index, bank in enumerate(producer["banks"]):
            with self.subTest(bank=index):
                base = 0x17800 + index * 0xe4
                expected = [1, 5 * (index + 1), 5, (0x4022 + 0x11 * index) << 16,
                            960, 0x40550000, 0x40660000, 30, 0x40440000, 30720]
                self.assertEqual(bank["mailbox"], base)
                self.assertEqual({e["address"]: e["value"] for e in producer["writes"]
                                  if base <= e["address"] < base + 0xe4},
                                 {base + 4 * i: v for i, v in enumerate(expected)})
                self.assertEqual({e["address"]: e["value"] for e in bank["reads"]},
                                 {base + 4 * i: v for i, v in enumerate(expected)})
                self.assertEqual(bank["wrapper_r4"], 0xc80 + index * 0x1000)
                self.assertEqual(bank["stop"], {"pc": 0x10fc + index * 0x1000,
                                               "reason": "unsupported opcode", "opcode": 0x000a})


class MarkerSpansTest(unittest.TestCase):
    def test_only_word_aligned_markers_split_sequence(self):
        first = b"\x99" + MARKER + b"\x99" * 3 + MARKER
        second = b"\x42" * 4 + MARKER
        self.assertEqual(list(spans(first + second)), [(0, first), (len(first), second)])

    def test_rejects_trailing_data_or_missing_final_marker(self):
        for raw in (b"", MARKER + b"\0", b"\x42" * 4):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                list(spans(raw))


@unittest.skipUnless(MEMBER.is_file(), "requires the extracted FDN member")
class FdnDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blob = MEMBER.read_bytes()
        if sha256(cls.blob).hexdigest() != MEMBER_SHA:
            raise ValueError("unexpected FDN member")

    def test_without_device_response_stops_at_first_wait(self):
        for block, entry in (("A", 0x1150), ("B", 0xfc8)):
            with self.subTest(block=block):
                result = front_end(self.blob, block)
                self.assertEqual(result["first_stop"], {"pc": entry, "reason": "wfi"})
                self.assertEqual(result["mailbox_reads"], [])
                self.assertEqual([(e["address"], e["value"]) for e in result["mmio_writes"]],
                                 [(0x48208, 1)])
                self.assertEqual(result["registers_at_stop"]["r15"], 0x17e0)

    def test_original_boot_prepares_descriptors_and_reaches_stage2_target(self):
        for block, waits, target in (("A", (0x150, 0x288), 0x50000000),
                                     ("B", (0x14c, 0x284), 0x50800000)):
            with self.subTest(block=block):
                first = boot_probe(self.blob, block)
                self.assertEqual(first["first_stop"], {"pc": FIXTURE_BASE + 0x94,
                                 "reason": "unmodeled read", "address": 0x48004, "size": 4})
                self.assertEqual(first["scratch_writes"], [])
                run = boot_probe(self.blob, block, fixture="header-only")
                self.assertEqual(run["bootstrap_descriptor_words"],
                                 ["0x500102a0", "0x18", "0x12f0000", "0x12f0000", "0x1", "0x40000000"])
                self.assertEqual(run["fixture_stops"], [
                    *[{"pc": FIXTURE_BASE + offset, "reason": "wfi"} for offset in waits],
                    {"pc": target, "reason": "unmapped fetch", "address": target, "size": 2},
                ])
                descriptor_bases = [w["value"] for w in run["mmio_writes"] if w["address"] == 0x48234]
                self.assertEqual(descriptor_bases, [0, 0x18, 0])

    def test_optional_external_table_is_requested_but_not_invented(self):
        for block, stop_at in (("A", 0x260), ("B", 0x25c)):
            with self.subTest(block=block):
                run = boot_probe(self.blob, block, fixture="with-external-table")
                self.assertEqual(run["bootstrap_descriptor_words"], [
                    "0x5ffffe00", "0x28", "0x15f0000", "0x15f0000",
                    "0x500102a0", "0x188", "0x12f0000", "0x12f0000", "0x1", "0x40000000",
                ])
                self.assertEqual(run["fixture_stops"][-1], {"pc": FIXTURE_BASE + stop_at,
                                 "reason": "unmodeled read", "address": 0x28, "size": 4})

    def test_all_selectors_reach_the_expected_handlers(self):
        first_targets = {
            "A": (0x11e8, 0x1264, 0x11dc, 0x11d0, 0x119c),
            "B": (0x1064, 0x1070, 0x1028, 0x101c, 0x1000),
        }
        for block, targets in first_targets.items():
            for selector, target in enumerate(targets):
                with self.subTest(block=block, selector=selector):
                    result = front_end(self.blob, block, selector)
                    self.assertIn(hex(target), result["reached_targets"])
                    self.assertEqual(result["mailbox_reads"][0]["value"], selector)

    def test_invalid_selector_reports_error_then_waits(self):
        for block in "AB":
            with self.subTest(block=block):
                result = front_end(self.blob, block, 5)
                self.assertEqual(result["after_fixture"], result["first_stop"])
                self.assertEqual(result["reached_targets"], [])
                self.assertEqual([(e["address"], e["value"]) for e in result["mmio_writes"]],
                                 [(0x48208, 1), (0x48200, 0xfffffffe),
                                  (0x48218, 1), (0x48208, 1)])

    def test_mailbox_copy_then_unsupported_entry(self):
        for block, selector, target, source_offsets in (
            ("A", 0, 0x10fc, (8, 0x1c, 0x28, 0x2c, 0x30, 0x34, 0x38, 0x3c)),
            ("B", 2, 0xf78, (8, 0x1c, 4, 0x24, 0x28, 0x20, 0x30)),
        ):
            with self.subTest(block=block):
                result = front_end(self.blob, block, selector)
                self.assertEqual(result["after_fixture"],
                                 {"pc": target, "reason": "unsupported opcode", "opcode": 0x000a})
                first = [e["value"] for e in result["mmio_writes"]
                         if 0x40000 <= e["address"] < 0x40000 + 4 * len(source_offsets)]
                self.assertEqual(first, [0x51000000 + offset for offset in source_offsets])

    def test_all_five_banks_select_identical_payload_bytes(self):
        expected = {
            "A": ((0, 1100, 0x3d34), (1, 1148, 0x4180)),
            "B": ((2, 1300, 0x1ea64),),
        }
        for block, entries in expected.items():
            actual = list(payloads(self.blob, block))
            self.assertEqual(tuple((m["selector"], len(raw), m["copies"][0]["file_offset"])
                                   for raw, m in actual), entries)
            for _, m in actual:
                self.assertEqual(len(m["copies"]), 5)
                self.assertEqual(len({c["selector_read"] for c in m["copies"]}), 5)

    def test_a4_reaches_wrapper_after_extra_mailbox_fields(self):
        for bank in range(5):
            with self.subTest(bank=bank):
                a1 = bank_entry(self.blob, "A", bank, 1)
                a4 = bank_entry(self.blob, "A", bank, 4)
                self.assertEqual(a4["after_fixture"], a1["after_fixture"])
                self.assertEqual(a4["registers_at_stop"]["r4"] - a1["registers_at_stop"]["r4"], 0x198)
                base = a4["first_read_after_wake"]["address"]
                self.assertEqual([event["address"] - base for event in a4["mailbox_reads"]],
                                 [0, 8, 0x1c, 4, 0x24, 0x64, 0x20, 0x6c, 0xc, 0x68, 0x14])
                writes = {e["address"]: e["value"] for e in a4["mmio_writes"]}
                self.assertEqual(writes[0x40010], 0x51000064)
                self.assertEqual(writes[0x40018], 0x5100006c)
                self.assertEqual(a4["registers_at_stop"]["r6"], 0x51000068)

    @unittest.skipUnless(ROTATION.is_file(), "requires the extracted rotation member")
    def test_rotation_control_and_identical_fdn_records(self):
        result = compare(self.blob, ROTATION.read_bytes())
        self.assertEqual(result["rotation_wrapper_dsp"], 0x19c)
        self.assertEqual(
            [(r["member_offset"], r["body_words"], r["dsp_call_pc"])
             for r in result["records"]["rotation"]],
            [(0x35c, 20, 0x6a4), (0x3bc, 20, 0x730), (0x41c, 20, 0x48c), (0x47c, 20, 0x5e4)])
        self.assertEqual([[r["body_words"] for r in result["records"][label]]
                          for label in ("A/0", "A/1", "B/2")],
                         [[39, 76, 68, 76], [30, 64, 33, 76, 64], [33, 64, 76, 64, 52, 12]])
        self.assertEqual(result["fdn_a4"][0]["member_offset"], 0x4318)
        self.assertEqual(result["identical_records"],
                         [["A/0:3", "A/1:3", "B/2:2"], ["A/1:4", "B/2:3"]])


if __name__ == "__main__":
    unittest.main()
