from pathlib import Path
import unittest

from parsers.fdn_geometry import (
    APERTURE, NODE_INDEX, OPTION_INDEX_DELTA, pdn_geometry_words, table_for_fdn_a, table_for_pdn,
)
from parsers.sony_sa_operands import analyze, aperture_candidates, commands, fdn_aperture_candidates
from parsers.sony_sa_program_planes import overlays
from probe_sa_fdn_parameters import SQUARE_WINDOW, probe as fdn_probe
from probe_sa_pdn_shared import call_setup, first_stage_request, image_geometry_request, probe


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "debug/a7iv_parts"
PDN = PARTS / "SA_PDN__SA07.bin"
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
DICTIONARIES = [ROOT / f"extracted/7m4/sabin/fdn_{name}_dict.bin" for name in ("front", "side")]


@unittest.skipUnless(all((PARTS / f"{name}.bin").is_file()
                         for name in ("SA_FDN__SA07", "SA_PDN__SA07", "SA_SHA250x02")),
                     "requires pinned SA members")
class SaOperandsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze(PARTS)

    def test_sha_shift_values_change_byte_positions_in_tail(self):
        sha = self.report["sha"]
        self.assertEqual(sha["large_sigma_1_triplets_high_middle"], [14, 21, 29])
        self.assertEqual(sha["large_sigma_0_triplets_high_middle"], [])
        self.assertEqual(sha["large_sigma_0_observed_high_middle"], [(0, 30), (13, 19), (22, 10)])
        pairs = sha["long_tail_matching_pairs"]
        self.assertTrue(all(r["middle_unchanged"] and r["old_high_reappears_as_new_low"] for r in pairs))
        self.assertEqual([r["new_bytes"] for r in pairs], [(6, 7, 25), (4, 30, 0), (2, 19, 13), (1, 10, 22)])
        self.assertEqual(sha["short_schedule_bytes"], [(7, 18, 25), (17, 0, 15), (3, 19, 13), (10, 2, 0)])

    def test_fdn_does_not_use_only_direct_parameter_word_indices(self):
        a = self.report["fdn"]["a_first"]
        self.assertEqual(a["outside_written_word_indices"], [20, 21, 0x80000, 0x8000a])
        self.assertEqual(a["different_arguments_for_identical_slots"], [[4, 18]])
        self.assertEqual(self.report["fdn"]["b_compact"]["outside_written_word_indices"], [0x800])

    def test_second_fdn_kernel_matches_pdn_only_under_the_expected_overlay(self):
        kernel = self.report["fdn_pdn_second_kernel"]
        self.assertEqual((kernel["words"], kernel["matching_121"], kernel["matching_122"]), (80, 80, 17))
        self.assertTrue(kernel["same_121_table_all_banks"])
        self.assertEqual(kernel["pdn_record_copies"], [0x538, 0x7a64, 0x9114, 0xa7e4, 0xbeb4, 0xd584])
        observed = probe(PDN.read_bytes())
        one, two = (observed[key] for key in ("mode_one", "mode_two"))
        self.assertEqual(one["record_member_offset"], kernel["pdn_extent"][0])
        self.assertEqual(two["record_member_offset"], 0x788)
        self.assertEqual(one["registers_r4_to_r7"], [0x50000038, 0xe000, 0x4000, 0x14032])
        self.assertEqual(two["registers_r4_to_r7"], [0x50000288, 0xe000, 0x4000, 0x1401a])
        for result in (one, two):
            self.assertEqual(result["stop"],
                             {"pc": 0x50000b5c, "reason": "unsupported opcode", "opcode": 0xa})
            self.assertEqual([result["aperture"][key] for key in ("0x4032c", "0x40328", "0x40330")],
                             result["registers_r4_to_r7"][1:])
        load = observed["isolated_program_request"]
        self.assertEqual(load["member_offset"], kernel["pdn_121_offset"])
        self.assertEqual(load["rows"], [[0x50010180, 0x0400f000, 0x8b570000, 0x43c70200]])

    def test_same_pdn_command_keeps_86_as_a_candidate_for_a_variable_signed_byte(self):
        pdn = PDN.read_bytes()
        _, banks = overlays(pdn)
        record = commands(pdn, banks[0][0], (0x538, 0x678))
        control = call_setup(pdn, 0x50001fd4, bytes(0x100))
        for raw, expanded in ((5, 5), (17, 17), (0xf2, 0xfffffff2)):
            with self.subTest(raw=raw):
                fixture = bytearray(0x100)
                fixture[0x61] = raw
                trace = call_setup(pdn, 0x50001fd4, bytes(fixture))
                self.assertEqual(trace["record_member_offset"], control["record_member_offset"])
                self.assertEqual(trace["aperture"], {**control["aperture"], "0x40218": expanded})
                setup = {int(k, 16): v for k, v in trace["aperture"].items()}
                matches = [r for r in aperture_candidates(record, setup) if r["address"] == 0x40218]
                self.assertEqual(matches, [{"position": 43, "word": 0x29080086,
                                            "address": 0x40218, "last_written_value": expanded}])

    @unittest.skipUnless(CP.is_file() and all(p.is_file() for p in DICTIONARIES),
                         "requires pinned A7 IV CP and dictionaries")
    def test_fdn_low_byte_candidates_follow_the_actual_setup_at_multiple_calls(self):
        fdn, cp = (PARTS / "SA_FDN__SA07.bin").read_bytes(), CP.read_bytes()
        dictionaries = [p.read_bytes() for p in DICTIONARIES]
        for node, bias in ((2, 0xeed), (3, 0x3185)):
            with self.subTest(node=node):
                workload = fdn_probe(cp, fdn, window=(*SQUARE_WINDOW[:-1], node),
                                     dictionaries=dictionaries, boot_control=True,
                                     trace_later_stages=True)["first_workload_fixture"]
                first, second, *later = fdn_aperture_candidates(fdn, workload)
                self.assertEqual(first["record_offset"], 0x51c)
                self.assertEqual([r["position"] for r in first["operands"]
                                  if r["last_written_value"] is None], [0, 14, 17, 24])
                self.assertEqual(second["record_offset"], 0x5c8)
                operands = {r["position"]: r for r in second["operands"]}
                for i, arg, address, value in (
                        (8, 0x64, 0x40190, 96), (18, 0x24ca, 0x40328, 0x3800),
                        (35, 0xc8, 0x40320, bias), (43, 0x080086, 0x40218, 5),
                        (47, 0x84, 0x40210, 32767), (48, 0x1885, 0x40214, 0xffff8000),
                        (63, 0x74, 0x401d0, 46), (70, 0xcc, 0x40330, 0x11000),
                        (61, 0xfef8, 0x403e0, None)):
                    row = operands[i]
                    self.assertEqual((row["word"] & 0xffffff, row["address"], row["last_written_value"]),
                                     (arg, address, value))
                control_positions = {}
                for call in later:
                    for operand in call["operands"]:
                        if operand["address"] == 0x40218:
                            control_positions.setdefault(call["record_offset"], set()).add(operand["position"])
                self.assertEqual(control_positions, {0x708: {37}, 0x828: {43}, 0x9f0: {33},
                                                     0xb94: {43}, 0xcd4: {33}})

    def test_pdn_loads_related_first_stage_before_calling_its_record(self):
        first = self.report["fdn_pdn_first_stage"]
        self.assertEqual(first["words"], 39)
        self.assertEqual(first["identical_plane_positions"], [*range(9), 28, 38])
        self.assertEqual(first["changed_slot_positions"], [35, 36, 37, 38])
        self.assertEqual(first["changed_arguments"], [(29, 0x13, 0)])
        self.assertTrue(first["same_bit31"])
        load = probe(PDN.read_bytes())["first_stage_request"]
        self.assertEqual((load["table_offset"], load["slots"]), (first["pdn_table"], 39))
        self.assertEqual(load["wait"], {"pc": 0x500014b4, "reason": "wfi"})
        self.assertEqual(load["request"]["rows"], [[0x500132a0, 0x0400f000, 0x83a70000, 0x41370200]])
        self.assertEqual((load["return_pc"], load["record_member_offset"]),
                         (0x500014d2, first["pdn_extent"][0]))
        self.assertEqual(load["registers_r4_to_r7"], [0x50000ab0, 0x16600, 64, 0xb000])
        self.assertEqual(load["stop"], {"pc": 0x50000b5c, "reason": "unsupported opcode", "opcode": 0xa})

    def test_pdn_caller_uses_the_same_matrix_routes_as_fdn_before_compute(self):
        pdn = PDN.read_bytes()
        table = table_for_pdn(pdn)
        self.assertEqual(table, table_for_fdn_a((PARTS / "SA_FDN__SA07.bin").read_bytes()))
        for option, delta in enumerate(OPTION_INDEX_DELTA):
            for node, zero_index in enumerate(NODE_INDEX):
                index = zero_index + delta
                for extents in ((48, 48), (63, 71), (22, 1)):
                    with self.subTest(option=option, node=node, extents=extents):
                        trace = first_stage_request(pdn, option=option, node=node, extents=extents)
                        self.assertEqual(trace["geometry_call"], {
                            "registers_r4_to_r7": [0x16600, 64, *extents],
                            "stack_words": [0xb000, 22, 22, *divmod(index, 2)],
                        })
                        writes = trace["aperture_writes"]
                        self.assertEqual([w["address"] for w in writes],
                                         list(range(APERTURE, APERTURE + 0x4c, 4)))
                        self.assertEqual(tuple(w["value"] for w in writes),
                                         pdn_geometry_words(table, index, extents))

    def test_pdn_image_window_fetch_and_compute_local_alignment(self):
        pdn = PDN.read_bytes()
        cases = (
            ((281, 193), [266, 175, 29, 37], 0x20044480, 0xc05f00b0, 0x40172424, 0x1600a),
            ((282, 193), [267, 175, 29, 37], 0x20044480, 0xc05f00b0, 0x40172424, 0x1600b),
            ((287, 193), [272, 175, 29, 37], 0x200444a0, 0xc03f00b0, 0x400f2424, 0x16000),
            ((286, 194), [271, 176, 29, 37], 0x20044a00, 0xc05f00b0, 0x40172424, 0x1600f),
        )
        for (x, y), window, source, packed_pitch, area, local in cases:
            with self.subTest(x=x, y=y):
                trace = image_geometry_request(pdn, center_q18=(x << 18, y << 18))
                self.assertEqual(trace["window_xywh"], window)
                self.assertEqual(trace["frame"], [0xa0008000, 640, 480, 1408])
                self.assertEqual(trace["image_request"]["rows"],
                                 [[0, 0x4c000000, 0, 0], [source, 0x216000, packed_pitch, area]])
                self.assertEqual(trace["edge_requests"], [])
                self.assertEqual(trace["compute"]["geometry_call"]["stack_words"],
                                 [0xb000, 22, 22, 2, 1])
                self.assertEqual(trace["compute"]["registers_r4_to_r7"],
                                 [0x50000ab0, local, 64, 0xb000])

    def test_pdn_left_top_and_right_bottom_edges_wait_for_extra_dma(self):
        pdn = PDN.read_bytes()
        cases = (
            ((5, 8), [-9, -9, 29, 37],
             [0x20008000, 0x216250, 0xc03f00b0, 0x400f1b1b],
             [0x500011f8] * 16 + [0x5000115c] * 9, [0x16240, 0xc0416250, 8, 0x1b1b], 0x16007),
            ((635, 475), [620, 457, 29, 37],
             [0x200a5640, 0x216000, 0xc05f00b0, 0x40171616],
             [0x50001034] * 9 + [0x500010c8] * 14, [0x16020, 0xc041601f, 8, 0x1616], 0x1600c),
        )
        for (x, y), window, row, waits, first_pad, local in cases:
            with self.subTest(x=x, y=y):
                trace = image_geometry_request(pdn, center_q18=(x << 18, y << 18))
                self.assertEqual(trace["window_xywh"], window)
                self.assertEqual(trace["image_request"]["rows"][1], row)
                self.assertEqual(trace["edge_waits"], waits)
                self.assertEqual(len(trace["edge_requests"]), len(waits))
                self.assertEqual(trace["edge_requests"][0]["rows"], [first_pad])
                self.assertEqual(trace["compute"]["registers_r4_to_r7"],
                                 [0x50000ab0, local, 64, 0xb000])


if __name__ == "__main__":
    unittest.main()
