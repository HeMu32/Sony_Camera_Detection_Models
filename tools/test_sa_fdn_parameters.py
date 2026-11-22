from copy import deepcopy
from pathlib import Path
from random import Random
from struct import pack_into, unpack_from
import unittest

from parsers.fdn_geometry import APERTURE, SETUP_WORDS, setup_words, table_for_fdn_a
from parsers.fdn_image_dma import bind as bind_image_request
from parsers.fdn_transfers import project
from parsers.sony_sa_program_planes import join_record, overlays
from probe_a7iv_fdn_cp_request import probe as cp_probe
from probe_sa_fdn_parameters import SQUARE_WINDOW, probe, run_fdn_parameters


ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
FDN = ROOT / "debug/a7iv_parts/SA_FDN__SA07.bin"
DICTIONARIES = [ROOT / f"extracted/7m4/sabin/fdn_{name}_dict.bin" for name in ("front", "side")]


@unittest.skipUnless(CP.is_file() and FDN.is_file(), "requires the pinned CP and FDN images")
class FdnParametersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp, cls.fdn = CP.read_bytes(), FDN.read_bytes()

    def test_original_fdn_level_selection_carries_frame_address_to_dma(self):
        levels = [(0x500100000 + i * 0x100000, 640 - 64 * i, 480 - 48 * i, 704 - 64 * i)
                  for i in range(5)]
        for index, (lo, hi) in enumerate(((48, 95), (96, 191), (192, 383), (384, 767), (768, 1535))):
            for extent in (lo, hi):
                with self.subTest(extent=extent):
                    cp = cp_probe(self.cp, manager_id=3, search_type=0, usecase=1,
                                  window=SQUARE_WINDOW, pyramid=levels, candidate_extent=extent)
                    selection = cp["frame_selection"]
                    self.assertEqual(selection["level"], index)
                    self.assertEqual(bytes.fromhex(selection["copied_frame_record_hex"])[14:], b"\xa5\xa5")
                    trace = run_fdn_parameters(self.fdn, int(cp["sammu_parameter_address"], 16),
                                               bytes.fromhex(cp["parameter_block_hex"]), bytes(0x160),
                                               trace_setup=True)
                    self.assertEqual(trace["validator_mask"], 0)
                    request = bind_image_request(trace)
                    _, _, pitch = levels[index][1:]
                    self.assertEqual(request.source_pitch_bytes, 2 * pitch)
                    self.assertEqual(trace["converted_addresses"][0], 0xa0040000 + index * 0x40000)
                    self.assertEqual(request.descriptor_words[0],
                                     0x20040000 + index * 0x40000 + request.source_offset)
                    normalized = bytes.fromhex(trace["normalized_parameter_hex"])
                    self.assertEqual(unpack_from("<3H", normalized, 0x10), levels[index][1:])
        for extent in (0, 47, 1536, 65535):
            with self.subTest(rejected=extent), self.assertRaisesRegex(ValueError, "scale selector rejected"):
                cp_probe(self.cp, manager_id=3, search_type=0, usecase=1,
                         pyramid=levels, candidate_extent=extent)

    def test_fdn_checks_and_translates_the_raw_cp_block(self):
        run = probe(self.cp, self.fdn)
        self.assertEqual(run["validator_mask"], 0)
        self.assertEqual(run["converted_parameter_address"], 0xa0000400)
        self.assertEqual(run["stop"], {"pc": 0x50000b54, "reason": "slice boundary"})
        self.assertEqual(run["converted_addresses"],
                         [0xa0008000, 0xa0040000, 0xa0046500, 0xe0004040])
        raw = bytes.fromhex(run["cp_parameter_hex"])
        self.assertEqual(bytes.fromhex(run["normalized_parameter_hex"])[0x10:], raw[0x10:])

        for index in range(4):
            with self.subTest(index=index):
                modified = bytearray(raw)
                offset = index * 4
                pack_into("<I", modified, offset, unpack_from("<I", raw, offset)[0] + 4)
                rejected = run_fdn_parameters(self.fdn, run["cp_parameter_address"],
                                              modified, bytes(0x160))
                self.assertEqual(rejected["validator_mask"], 1 << index)
                self.assertEqual(rejected["mmio_writes"][-1],
                                 {"pc": 0x50000afe, "address": 0x48188,
                                  "size": 4, "value": 1 << index})

    def test_window_bounds_and_initial_image_request(self):
        for size, expected in ((47, 0x600), (48, 0), (64, 0), (95, 0), (96, 0x600)):
            with self.subTest(size=size):
                window = (*SQUARE_WINDOW[:5], size, size, *SQUARE_WINDOW[-2:])
                self.assertEqual(probe(self.cp, self.fdn, window=window)["validator_mask"], expected)
        for x, y, expected_source, expected_geometry in (
            (32, 32, 0x20013040, 0xc07f00b0), (48, 32, 0x20013060, 0xc07f00b0),
            (33, 32, 0x20013040, 0xc09f00b0), (32, 33, 0x200135c0, 0xc07f00b0),
        ):
            with self.subTest(x=x, y=y):
                window = (*SQUARE_WINDOW[:3], x, y, *SQUARE_WINDOW[5:])
                run = probe(self.cp, self.fdn, window=window, trace_setup=True)
                self.assertEqual(run["setup_stop"], {"pc": 0x50002524, "reason": "wfi"})
                words = {e["address"]: e["value"] for e in run["setup_scratch_writes"]}
                self.assertEqual((words[0x10], words[0x18]), (expected_source, expected_geometry))

    def test_original_image_descriptor_matches_aligned_packed_row_geometry(self):
        rng = Random(0x46444e)
        for i in range(72):
            width, height = rng.choice((60, 62, 64, 128, 638, 640)), rng.choice((48, 63, 128, 479, 480))
            edge = rng.randrange(48, min(95, width, height) + 1)
            pitch = (width + 3) & ~3
            pitch += 4 * rng.randrange(24)
            x = rng.choice((0, width - edge, rng.randrange(width - edge + 1)))
            y = rng.choice((0, height - edge, rng.randrange(height - edge + 1)))
            with self.subTest(x=x, y=y, edge=edge, width=width, height=height, pitch=pitch):
                run = probe(self.cp, self.fdn,
                            window=(width, height, pitch, x, y, edge, edge, i % 3, i % 15), trace_setup=True)
                request = bind_image_request(run)
                self.assertEqual(request.source_xy, ((x // 16) * 16, y))
                self.assertLessEqual(request.source_row_bytes, 112 * 2)
                self.assertGreaterEqual(request.source_offset + request.source_row_bytes,
                                        2 * (y * pitch + x + edge))
                self.assertEqual(request.source_rows, edge)
                self.assertEqual(request.source_pitch_bytes, pitch * 2)

    def test_original_image_validator_boundaries_are_not_arbitrary_jpeg_dimensions(self):
        for width, height, pitch, expected in (
                (60, 48, 60, 0), (640, 480, 640, 0), (58, 48, 60, 0x10), (61, 48, 64, 0x10),
                (642, 480, 644, 0x10), (640, 481, 640, 0x20), (640, 480, 638, 0x40),
                (640, 480, 636, 0x40), (640, 480, 65532, 0), (640, 480, 65535, 0x40)):
            with self.subTest(width=width, height=height, pitch=pitch):
                trace = probe(self.cp, self.fdn, window=(width, height, pitch, 0, 0, 48, 48, 0, 2),
                              trace_setup=True)
                self.assertEqual(trace["validator_mask"], expected)
                if expected == 0:
                    bind_image_request(trace)
                else:
                    with self.assertRaisesRegex(ValueError, "image request wait"):
                        bind_image_request(trace)

    def test_image_gather_preserves_packing_and_excludes_padding_between_rows(self):
        # Deliberately odd x, padded pitch, and unrelated bytes in the two lanes.
        window = (128, 96, 132, 17, 11, 49, 49, 0, 2)
        trace = probe(self.cp, self.fdn, window=window, trace_setup=True)
        request = bind_image_request(trace)
        frame = bytes(lane for y in range(96) for x in range(132) for lane in (x, y ^ 0x80))
        expected = bytes(lane for y in range(11, 60) for x in range(16, 80) for lane in (x, y ^ 0x80))
        self.assertEqual(request.gather(frame), expected)
        with self.assertRaisesRegex(ValueError, "truncated"):
            request.gather(frame[:2 * (59 * 132 + 80) - 1])
        bad = deepcopy(trace)
        row_word = next(w for w in bad["setup_scratch_writes"] if w["address"] == 0x14)
        row_word["value"] += 2
        with self.assertRaisesRegex(ValueError, "original aligned"):
            bind_image_request(bad)

    @unittest.skipUnless(all(p.is_file() for p in DICTIONARIES), "requires pinned dictionaries")
    def test_node_selects_dictionary_and_original_kernel_request(self):
        dictionaries = [p.read_bytes() for p in DICTIONARIES]
        hashes = {"front": "4edec39935f43381d2b57d4b66ac23d9d6743a50d22bd12e34d6197fd1b3d9e4",
                  "side": "884ac1da1858e959c6ece6265ff65dfc1596f7bf91daf93254664d356d1fa607"}
        for node in range(16):
            with self.subTest(node=node):
                run = probe(self.cp, self.fdn, window=(*SQUARE_WINDOW[:-1], node),
                            dictionaries=dictionaries)
                if node == 15:
                    self.assertEqual(run["validator_mask"], 0x1000)
                    self.assertNotIn("dictionary_fixture", run)
                    continue
                request = run["dictionary_fixture"]
                self.assertEqual(request["dictionary"], "front" if node < 3 else "side")
                self.assertEqual(request["sha256_in_request_order"], hashes[request["dictionary"]])
                self.assertEqual(len(request["requests"]), 72)
                self.assertEqual(request["requests"][0],
                                 {"offset": 0x4c3c, "destination": 0x3800, "size": 24})
                first = run["first_workload_fixture"]
                self.assertEqual(first["stop"], {"pc": 0x500008e4,
                                                  "reason": "unsupported opcode", "opcode": 0x000a})
                self.assertEqual(first["command_pointer_r4"], 0x5000001c)
                self.assertEqual(first["matching_command_bytes"], {
                    "outer_file_offset": 0x51c, "bank_file_offset": 0x3d34, "size": 2248,
                    "sha256": "7002f423d4b3497c776f97d041296f094d1baabec5ba803fe00d0375a35c3bb6",
                })
                self.assertEqual(len(first["mailbox_writes"]), 120)
                self.assertEqual(first["mailbox_heads"], [
                    {"address": 0x17800 + i * 0xe4, "selector": 0} for i in range(5)])
                self.assertEqual(
                    [b["stop"] for b in first["banks_after_fixture_wake"]],
                    [{"pc": pc, "reason": "unsupported opcode", "opcode": 0x000a}
                     for pc in (0x10fc, 0x20fc, 0x30fc, 0x88fc, 0x98fc)])
                self.assertEqual(first["caller_continuation"], {
                    "fixture_return_pc": 0x50001ea2,
                    "fixture_r0_values": [0, 0xdeadbeef],
                    "wait_pcs": [0x50001eac, 0x5000232c, 0x50002810, 0x50000f28],
                    "intermediate_descriptor_words":
                        [0x6000, 0xc000b000, 0x11ff0000, 0x11ff0000, 1, 0x40000000],
                    "followup_mailbox_heads":
                        [{"address": 0x17800 + i * 0xe4, "selector": 2} for i in range(5)],
                    "followup_workers":
                        [{"bank": i, "stop": {"pc": pc, "reason": "wfi"}, "mailbox_reads": 23,
                          "nonzero_payload_reads": 0}
                         for i, pc in enumerate((0x1150, 0x2150, 0x3150, 0x8950, 0x9950))],
                    "stop": {"pc": 0x500010c8, "reason": "unmodeled read",
                             "address": 0x48168, "size": 4},
                })

    @unittest.skipUnless(all(p.is_file() for p in DICTIONARIES), "requires pinned dictionaries")
    def test_original_a_boot_control_and_dictionary_reach_second_compute(self):
        dictionaries = [p.read_bytes() for p in DICTIONARIES]
        for node in (2, 3):
            with self.subTest(node=node):
                trace = probe(self.cp, self.fdn, window=(*SQUARE_WINDOW[:-1], node),
                              dictionaries=dictionaries, boot_control=True)["first_workload_fixture"]
                boot = trace["boot_control_fixture"]
                self.assertEqual(boot["table"], {"member_offset": 0x6b00, "destination": 0x7800, "size": 0x460})
                self.assertEqual(boot["first_registers_r4_to_r7"], [0x5000001c, 0x11000, 0x70, 0xb000])
                request = bind_image_request(probe(self.cp, self.fdn, trace_setup=True))
                self.assertEqual(boot["first_registers_r4_to_r7"][1:3],
                                 [request.local_address, request.local_pitch_operand])
                outer_setup = setup_words(table_for_fdn_a(self.fdn), 64, node)
                for i, worker in enumerate(trace["banks_after_fixture_wake"], 1):
                    self.assertEqual(worker["registers_r4_to_r7"][1:], [0x11000, 112, 0xb000])
                    writes = worker["aperture_writes"]
                    self.assertEqual([w["address"] for w in writes], list(range(0x40000, 0x40050, 4)))
                    self.assertEqual([w["value"] for w in writes],
                                     [*outer_setup[:18], i * 8, outer_setup[19]])
                second = boot["isolated_after_first_compute"]
                self.assertEqual(second["registers_r4_to_r7"], [0x500000c8, 0x6000, 0x3800, 0x11000])
                self.assertEqual(second["return_pc"], 0x50002bb0)
                self.assertEqual([w["pc"] for w in second["waits"]], [0x50001eac, 0x5000232c, 0x50002810])
                self.assertEqual(second["stop"], {"pc": 0x500008e4, "reason": "unsupported opcode", "opcode": 0xa})
                self.assertEqual(second["preceding_transfer_words"],
                                 [0x6000, 0xc000b000, 0x11ff0000, 0x11ff0000, 1, 0x40000000])
                setup = {w["address"]: w["value"] for w in second["aperture_writes"]}
                self.assertEqual([setup[k] for k in (0x40190, 0x40194, 0x40198, 0x4019c, 0x401a0,
                                                     0x401c4, 0x401d0, 0x40210, 0x40214)],
                                 [96, 4608, 24, 2, 92, 46, 46, 32767, 0xffff8000])
                self.assertEqual([w for w in second["aperture_writes"] if w["address"] == 0x40218],
                                 [{"pc": 0x50002b3e, "address": 0x40218, "size": 4, "value": 5}])
                data = dictionaries[node - 2]
                self.assertEqual(setup[0x40320], unpack_from("<I", data, 0x4ccc)[0])
                self.assertEqual([setup[k] for k in (0x4032c, 0x40328, 0x40330)],
                                 second["registers_r4_to_r7"][1:])
                workers = second["worker_calls"]
                self.assertEqual([w["stop"]["pc"] for w in workers], [0x10fc, 0x20fc, 0x30fc, 0x88fc, 0x98fc])
                self.assertEqual([w["stop"].get("opcode") for w in workers], [0xa] * 5)
                for i, (worker, weight) in enumerate(zip(workers, (0x4000, 0x4800, 0x5000, 0xa000, 0xa800)), 1):
                    args = [0xb000 + (i - 1) * 0x1200, weight, 0x11000 + i * (46 * 46 * 2)]
                    self.assertEqual(worker["registers_r4_to_r7"][1:], args)
                    ws = {w["address"]: w["value"] for w in worker["aperture_writes"]}
                    self.assertEqual([ws[k] for k in (0x4032c, 0x40328, 0x40330)], args)
                    self.assertEqual(ws[0x40218], 5)
                    self.assertEqual(ws[0x40320], unpack_from("<I", data, 0x4ccc + i * 4)[0])
                _, banks = overlays(self.fdn)
                first_record = self.fdn[0x51c:0x5c8]
                second_record = self.fdn[0x5c8:0x708]
                marker = bytes.fromhex("00000000000000010000000100000001")
                self.assertTrue(first_record.endswith(marker) and second_record.endswith(marker))
                self.assertEqual({word["index"] for word in join_record(banks[0][0], first_record)}, set(range(40)))
                self.assertEqual({word["index"] for word in join_record(banks[0][0], second_record)},
                                 {0, 1, *range(40, 108)})

    @unittest.skipUnless(all(p.is_file() for p in DICTIONARIES), "requires pinned dictionaries")
    def test_a_tail_binds_all_b_interior_planes_in_firmware_order(self):
        dictionaries = [p.read_bytes() for p in DICTIONARIES]
        for node, data in zip((2, 3), dictionaries):
            with self.subTest(node=node):
                result = probe(self.cp, self.fdn, window=(*SQUARE_WINDOW[:-1], node),
                               dictionaries=dictionaries, boot_control=True, trace_later_stages=True)
                first = result["first_workload_fixture"]["boot_control_fixture"]["isolated_after_first_compute"]
                trace = first["later_control"]
                self.assertEqual(trace["stop"], {"pc": 0x50000b5c, "reason": "unmodeled read",
                                                 "address": 0x48100, "size": 4})
                self.assertEqual([r["slots"] for r in trace["program_requests"]],
                                 [slots for slots in (66, 96, 87, 121) for _ in range(6)])
                self.assertEqual(len(trace["calls"]), 20)
                self.assertEqual([c["register_snapshot"]["0x40218"] for c in trace["calls"]],
                                 [12, 12, 8, 8, 8, 6, 11, 11, 12, 12, 12, 11, 11, 11, 11, 11, 11, 9, 9, 9])
                a66 = trace["calls"][:2]
                self.assertEqual([c["registers_r4_to_r7"][1:] for c in a66],
                                 [[0x11000, 0x381c, 0xe000], [0x11000, 0x38ac, 0xe400]])
                for call in a66:
                    self.assertEqual(call["program_slots"], 66)
                    cfg = call["register_snapshot"]
                    self.assertEqual([cfg[k] for k in ("0x40190", "0x40194", "0x401a0", "0x401a8",
                                                       "0x401c4", "0x401d0")], [92, 4232, 44, 184, 22, 22])
                self.assertEqual([c["register_snapshot"]["0x40320"] for c in a66],
                                 list(unpack_from("<2I", data, 0x53d8)))
                final = trace["calls"][-3:]
                self.assertEqual([c["registers_r4_to_r7"][0] for c in final], [0x500007d4] * 3)
                self.assertEqual([c["program_slots"] for c in final], [121] * 3)
                self.assertEqual([(c["register_snapshot"]["0x401a0"], c["register_snapshot"]["0x401c4"])
                                  for c in final], [(44, 20)] * 3)
                workers = [c for w in trace["final_workers"] for c in w["scheduled_calls"]]
                self.assertEqual(len(workers), 15)
                addresses = [c["register_snapshot"]["0x40330"] for c in final] + [
                    c["aperture_40330"] for c in workers]
                self.assertEqual(addresses,
                                 [0xb82e + i * (22 * 22 * 2) for i in range(18)])
                self.assertEqual([c["registers_r4_to_r7"][3] - c["aperture_40330"] for c in workers],
                                 [88 * bank for bank in range(1, 6) for _ in range(3)])
                biases = [c["register_snapshot"]["0x40320"] for c in final] + [c["bias_word"] for c in workers]
                self.assertEqual(biases, list(unpack_from("<18I", data, 0x16e10)))
                submit = trace["dictionary_submit"]
                self.assertEqual(submit["submit_pc"], 0x50002766)
                self.assertEqual(len(submit["rows"]), 48)
                coeff_base = 0x20040000 if node == 2 else 0x20080000
                self.assertIn([coeff_base + 0x16ab0, 0x3938, 0x8f0000, 0x8f0000], submit["rows"])

    @unittest.skipUnless(all(p.is_file() for p in DICTIONARIES), "requires pinned dictionaries")
    def test_dictionary_descriptors_repack_six_complete_blocks(self):
        dictionaries = [p.read_bytes() for p in DICTIONARIES]
        for node, data in zip((2, 3), dictionaries):
            with self.subTest(node=node):
                run = probe(self.cp, self.fdn, window=(*SQUARE_WINDOW[:-1], node),
                            dictionaries=dictionaries)["dictionary_fixture"]
                result = project(data, run["requests"])
                bases = (0x3800, 0x4000, 0x4800, 0x5000, 0xa000, 0xa800)
                self.assertEqual(result["sha256_in_request_order"], run["sha256_in_request_order"])
                self.assertEqual(result["covered_source_bytes"], 0x538 * len(bases))
                self.assertEqual(result["repeated_source_bytes"], 0)
                order = [("blocks", 6, 24, 4), ("blocks", 7, 288, 8), ("blocks", 8, 576, 8),
                         ("blocks", 4, 24, 4), ("blocks", 5, 288, 8),
                         ("extra_blocks", 0, 96, 8)]
                section_offset = 0
                for i, (group, block, coeff, bias) in enumerate(order):
                    for bank, base in enumerate(bases):
                        j = 2 * (i * len(bases) + bank)
                        for k, field, size, target_offset in (
                                (0, "coefficients", coeff, 0), (1, "bias", bias, coeff)):
                            row = result["requests"][j + k]
                            self.assertEqual((row["group"], row["block"], row["field"],
                                              row["field_offset"], row["size"], row["destination"]),
                                             (group, block, field, bank * size, size,
                                              base + section_offset + target_offset))
                    section_offset += coeff + bias
                self.assertEqual([(r["address"], r["size"]) for r in result["destination_regions"]],
                                 [(base, 0x538) for base in bases])
                touched = {(b["group"], b["block"]) for b in result["blocks"]
                           if b["coefficients"]["covered"] or b["bias"]["covered"]}
                self.assertEqual(touched, {(group, block) for group, block, _, _ in order})
                for b in result["blocks"]:
                    selected = (b["group"], b["block"]) in touched
                    self.assertEqual(b["header"]["covered"], 0)
                    for field in ("coefficients", "bias"):
                        self.assertEqual(b[field]["covered"], b[field]["size"] if selected else 0)

                with self.assertRaisesRegex(ValueError, "exactly one dictionary span"):
                    project(data, [{**run["requests"][0], "offset": 0x4c3b}])
                with self.assertRaisesRegex(ValueError, "destination overlap"):
                    project(data, run["requests"][:1] * 2)

    @unittest.skipUnless(all(p.is_file() for p in DICTIONARIES), "requires pinned dictionaries")
    def test_first_fdn_compute_uses_unpacked_byte_step_after_aligned_fetch(self):
        dictionaries = [p.read_bytes() for p in DICTIONARIES]
        for x, local in ((272, 0x11000), (273, 0x11001), (281, 0x11009), (288, 0x11000)):
            with self.subTest(x=x):
                trace = probe(self.cp, self.fdn, window=(640, 480, 704, x, 193, 64, 64, 0, 0),
                              dictionaries=dictionaries, boot_control=True)
                fetch = bind_image_request(trace)
                self.assertEqual(fetch.local_address, 0x11000)
                self.assertEqual(fetch.compute_local_address, local)
                first = trace["first_workload_fixture"]
                self.assertEqual(first["boot_control_fixture"]["first_registers_r4_to_r7"][1:3],
                                 [local, fetch.local_pitch_operand])
                for worker in first["banks_after_fixture_wake"]:
                    self.assertEqual(worker["registers_r4_to_r7"][1:3], [local, fetch.local_pitch_operand])

    @unittest.skipUnless(all(p.is_file() for p in DICTIONARIES), "requires pinned dictionaries")
    def test_geometry_setup_reproduces_the_first_aperture_writes(self):
        dictionaries = [p.read_bytes() for p in DICTIONARIES]
        table = table_for_fdn_a(self.fdn)
        self.assertEqual(table, self.fdn[0x8e00:0x8f20])
        cases = [(32, 32, 64, opt, node) for opt in range(3) for node in range(15)]
        cases += [(32, 32, edge, 0, node) for edge in (48, 63, 65, 95) for node in (0, 2, 14)]
        cases += [(33, 32, 64, 0, 2), (32, 47, 64, 0, 2)]
        for x, y, edge, option, node in cases:
            with self.subTest(x=x, y=y, edge=edge, option=option, node=node):
                window = (*SQUARE_WINDOW[:3], x, y, edge, edge, option, node)
                trace = probe(self.cp, self.fdn, window=window, dictionaries=dictionaries)
                writes = [w for w in trace["first_workload_fixture"]["register_writes"]
                          if APERTURE <= w["address"] < APERTURE + SETUP_WORDS * 4]
                self.assertEqual([(w["address"], w["size"]) for w in writes],
                                 [(APERTURE + 4 * i, 4) for i in range(SETUP_WORDS)])
                self.assertEqual(tuple(w["value"] for w in writes),
                                 setup_words(table, edge, node, option))


if __name__ == "__main__":
    unittest.main()
