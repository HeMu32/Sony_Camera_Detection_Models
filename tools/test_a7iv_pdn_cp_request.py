from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from struct import pack, unpack
import unittest

import numpy as np

from parsers.pdn_image_dma import bind as bind_image_request
from parsers.pdn_sections import scalar_head_dot, scalar_head_quantized
from parsers.sony_sa_post import members
from parsers.sony_sa_program_planes import overlays, plane_slots
from probe_a7iv_pdn_cp_request import probe
from probe_sa_pdn_shared import bind_final_programs


ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
PDN = ROOT / "debug/a7iv_parts/SA_PDN__SA07.bin"
FDN = ROOT / "debug/a7iv_parts/SA_FDN__SA07.bin"
BUNDLE = ROOT / "extracted/7m4/sabin/sa_sabin_cp_cam_post.bin"
DICTIONARY = ROOT / "extracted/7m4/sabin/pdn_dic.bin"


class PdnCpRequestTest(unittest.TestCase):
    @unittest.skipUnless(CP.is_file() and PDN.is_file(), "requires pinned A7 IV CP and PDN")
    def test_original_result_level_selects_pdn_packed_frame(self):
        cp, pdn = CP.read_bytes(), PDN.read_bytes()
        levels = [(0x500100000 + i * 0x100000, 640 - 64 * i, 480 - 48 * i, 704 - 64 * i)
                  for i in range(5)]
        for index, (_, width, height, pitch) in enumerate(levels):
            with self.subTest(result_level=index):
                result = probe(cp, pdn, pyramid=levels, result_level=index)
                self.assertEqual(result["validator_mask"], 0)
                geometry = result["image_geometry"]
                self.assertEqual(geometry["frame"], [0xa0040000 + index * 0x40000, width, height, pitch * 2])
                request = bind_image_request(geometry)
                self.assertEqual(geometry["image_request"]["rows"][1][0],
                                 0x20040000 + index * 0x40000 + request.source_offset)

    @unittest.skipUnless(all(p.is_file() for p in (CP, PDN, DICTIONARY)), "requires pinned PDN inputs")
    def test_original_dictionary_supplies_distinct_bias_shift_and_control_operand(self):
        dictionary = DICTIONARY.read_bytes()
        trace = probe(CP.read_bytes(), PDN.read_bytes(), dictionary=dictionary)["isolated_weight_setup"]
        self.assertEqual(trace["dictionary_copies"][:2], [
            {"submit_pc": 0x50001d28, "offset": 0x11a20, "target": 0x17c00, "size": 0x68},
            {"submit_pc": 0x50001dcc, "offset": 0x11a88, "target": 0x17800, "size": 0x294},
        ])
        self.assertEqual([(r["offset"], r["target"], r["size"]) for r in trace["dictionary_copies"][2:]],
                         [(0x18 * i, dest, 0x18) for i, dest in enumerate(
                             (0xc1e0, 0xc800, 0xd000, 0xd800, 0xf360, 0xf800))]
                         + [(0x90, 0xc1f8, 0xc)])
        self.assertEqual([row[0] for row in trace["unexecuted_rows"] if row[1] == 0xc0017000],
                         [0x12000, 0x12800, 0x13000, 0x13800, 0x14000])
        self.assertEqual(trace["stop"], {"pc": 0x50000b5c, "reason": "unsupported opcode", "opcode": 0xa})
        self.assertEqual(trace["registers_r4_to_r7"], [0x50000038, 0x17000, 0xc1e0, 0xdad2])
        shift_byte, = unpack("<b", dictionary[0x11ab7:0x11ab8])
        control_byte, = unpack("<b", dictionary[0x11abc:0x11abd])
        bias, = unpack("<h", dictionary[0x90:0x92])
        self.assertEqual((shift_byte, control_byte, bias), (-11, 14, -4056))
        self.assertEqual(trace["parameter_reads"], [
            {"pc": 0x50001f62, "address": 0x1782f, "size": 1, "value": shift_byte},
            {"pc": 0x50001f66, "address": 0x17834, "size": 1, "value": control_byte},
            {"pc": 0x5000227c, "address": 0xc1f8, "size": 2, "value": bias},
        ])
        self.assertEqual(trace["aperture"]["0x40218"], control_byte)
        self.assertEqual(trace["aperture"]["0x40320"], (bias << -shift_byte) & 0xffffffff)

    @unittest.skipUnless(all(p.is_file() for p in (CP, PDN, DICTIONARY)), "requires pinned PDN inputs")
    def test_all_weight_workers_consume_original_producer_snapshots(self):
        pdn, dictionary = PDN.read_bytes(), DICTIONARY.read_bytes()
        trace = probe(CP.read_bytes(), pdn, dictionary=dictionary)["isolated_weight_setup"]
        self.assertEqual(trace["output_borders"], {
            "plane_bases": [0xdaa0 + i * 0x420 for i in range(6)], "row_bytes": 48,
            "logical_shape": [22, 22], "unwritten_interior_shape": [20, 20], "known_zero_bytes": 1008,
        })
        biases = unpack("<6h", dictionary[0x90:0x9c])
        refs = ((0x800, 0x7a64, 0xc800), (0x2800, 0x9114, 0xd000), (0x4800, 0xa7e4, 0xd800),
                (0x8000, 0xbeb4, 0xf360), (0xa000, 0xd584, 0xf800))
        for i, (worker, (base, record, coeff)) in enumerate(zip(trace["worker_calls"], refs, strict=True), 1):
            with self.subTest(worker=i):
                self.assertEqual(worker["local_base"], base)
                self.assertEqual(worker["crop_registers_r4_to_r7"], [base + 0xac8, 0x1600a, 64, 0x17000])
                self.assertEqual(worker["registers_r4_to_r7"], [
                    base + 0x44, 0x12000 + (i - 1) * 0x800, coeff, 0xdad2 + i * 0x420,
                ])
                self.assertEqual(worker["record_member_offset"], record)
                self.assertEqual(pdn[record:record + 0x140], pdn[0x538:0x678])
                self.assertEqual([(r["address"], r["value"]) for r in worker["parameter_reads"]],
                                 [(0x1782f, -11), (0x17834, 14), (0xc1f8 + 2 * i, biases[i])])
                ports = worker["aperture"]
                self.assertEqual((ports["0x40190"], ports["0x401a0"], ports["0x401c4"]),
                                 (44, 48, 20))
                self.assertEqual((ports["0x40218"], ports["0x40320"], ports["0x40330"]),
                                 (14, (biases[i] << 11) & 0xffffffff, 0xdad2 + i * 0x420))

    @unittest.skipUnless(all(p.is_file() for p in (CP, PDN, DICTIONARY)), "requires pinned PDN inputs")
    def test_map_stage_binds_all_48_call_targets_to_original_parameters(self):
        pdn, dictionary = PDN.read_bytes(), DICTIONARY.read_bytes()
        trace = probe(CP.read_bytes(), pdn, dictionary=dictionary, trace_later=True)
        trace = trace["isolated_later_control"]["map_stage"]
        coefficient_bases = (0xd0e0, 0xd800, 0xe000, 0xec40, 0xf000, 0xf800)
        table, banks = overlays(pdn)
        local_copy, program_copy, coefficients = trace["setup_dma_submissions"][-3:]
        self.assertEqual(local_copy["rows"], [[0x12000, 0xc000e6a0, 0x077f0000, 0x077f0000]])
        self.assertEqual(local_copy["submit_pc"], 0x500031f0)
        (src, dst, a, b), = program_copy["rows"]
        self.assertEqual((src - 0x50010000 + table, dst, plane_slots(a, b)),
                         (banks[0][0].offset, 0x0400f000, 121))
        self.assertEqual(coefficients["rows"],
                         [[0x20011da8 + 0x180 * i, dst, 0x017f0000, 0x017f0000]
                          for i, dst in enumerate(coefficient_bases)] +
                         [[0x200126a8, 0xd260, 0x005f0000, 0x005f0000]])
        self.assertEqual((trace["record_image_offset"], trace["record_body_words"]), (0x178, 64))
        record = pdn[0x678:0x788]
        biases = unpack("<48h", dictionary[0x26a8:0x2708])
        shift, control = (unpack("<b", dictionary[i:i + 1])[0] for i in (0x11bf8, 0x11bfd))
        self.assertEqual((shift, control), (-8, 13))
        self.assertEqual(trace["parameter_reads"], [
            {"pc": 0x5000210c, "address": 0x17970, "size": 1, "value": shift},
            {"pc": 0x50002110, "address": 0x17975, "size": 1, "value": control},
        ])
        refs = (0x7ba4, 0x9254, 0xa924, 0xbff4, 0xd6c4)
        for worker, offset in zip(trace["workers"], refs, strict=True):
            self.assertEqual(worker["record_member_offset"], offset)
            self.assertEqual(worker["record_body_words"], 64)
            self.assertEqual(pdn[offset:offset + len(record)], record)
            self.assertEqual([(r["address"], r["size"], r["value"]) for r in worker["parameter_reads"]],
                             [(0x17970, 1, shift), (0x17975, 1, control)])
        calls = [trace["controller_calls"], *(w["calls"] for w in trace["workers"])]
        for group, (batch, coeff) in enumerate(zip(calls, coefficient_bases, strict=True)):
            self.assertEqual(len(batch), 8)
            source = 0x12000 if group < 3 else 0xe6a0
            config = batch[0]["aperture"]
            self.assertEqual([config[hex(port)] for port in
                              (0x40190, 0x4019c, 0x401a0, 0x401b0, 0x401c4, 0x401d0, 0x40218)],
                             [288, 2, 16, 48, 5, 5, control])
            for i, call in enumerate(batch, 8 * group):
                destination = 0xc1e0 + i * 80
                self.assertEqual(call["registers_r4_to_r7"][1:],
                                 [source, coeff + (i % 8) * 48, destination])
                ports = call["aperture"]
                self.assertEqual([ports[hex(port)] for port in (0x40320, 0x40324, 0x4032c, 0x40330)],
                                 [(biases[i] << -shift) & 0xffffffff, source, source, destination])

    @unittest.skipUnless(FDN.is_file() and PDN.is_file(), "requires pinned FDN and PDN members")
    def test_pdn_map_stage_reuses_fdn_a_final_kernel(self):
        fdn, pdn = FDN.read_bytes(), PDN.read_bytes()
        self.assertEqual(fdn[0xcd4:0xde4], pdn[0x678:0x788])
        f = overlays(fdn)[1][0][4]
        for bank in overlays(pdn)[1]:
            self.assertEqual(bank[0].planes, f.planes)

    @unittest.skipUnless(all(p.is_file() for p in (CP, PDN, DICTIONARY)), "requires pinned PDN inputs")
    def test_final_batches_bind_native_worker_records_and_head_buffer(self):
        pdn = PDN.read_bytes()
        trace = probe(CP.read_bytes(), pdn, dictionary=DICTIONARY.read_bytes(), trace_later=True)
        trace = trace["isolated_later_control"]
        expected_stack = [0xc1e0, 0xe4e0, 0x17c00, 0x17800]
        weight_targets = [0xd0e0, 0xe800, 0x10000, 0x11800, 0x13000, 0x14800]
        refs = [(0x800, 0x83b4, 0x1b22, 0xd400), (0x2800, 0x9a64, 0x3b3a, 0xd800),
                (0x4800, 0xb134, 0x5b32, 0xdc00), (0x8000, 0xc804, 0x9336, 0xe000),
                (0xa000, 0xded4, 0xb336, 0xe400)]
        table, banks = overlays(pdn)
        self.assertNotEqual(banks[0][4].planes, banks[1][4].planes)
        self.assertEqual([banks[i][4].offset for i in range(6)],
                         [0x118e0] + [0x12d20] * 5)
        self.assertEqual(trace["final_program_bindings"][0], trace["final_program_bindings"][1])
        bindings = trace["final_program_bindings"][0]
        for bank, (binding, offset) in enumerate(zip(bindings, [0xe80, *(ref[1] for ref in refs)], strict=True)):
            words = unpack(f"<{72 if bank == 0 else 73}I",
                           pdn[offset:offset + (288 if bank == 0 else 292)])
            self.assertEqual(binding, {"bank": bank, "program_member_offset": banks[bank][4].offset,
                                       "record_member_offset": offset,
                                       "referenced_slots": sorted({w >> 24 & 0x7f for w in words})})

        for selector, call in enumerate(trace["calls"][-2:]):
            with self.subTest(selector=selector):
                self.assertEqual(call["return_pc"], 0x5000393e)
                setup = call["setup_since_previous_compute"]
                self.assertEqual((setup["record_image_offset"], setup["record_body_words"]), (0x980, 72))
                self.assertEqual(setup["stack_words"], expected_stack)
                self.assertEqual(setup["isolated_wrapper"]["r0_to_r7"][0], 0x50000980)
                self.assertEqual(setup["isolated_wrapper"]["caller_stack_offset"], 0x24)
                self.assertEqual(setup["isolated_wrapper"]["stop"],
                                 {"pc": 0x5000591e, "reason": "unsupported opcode", "opcode": 9})
                self.assertEqual(setup["aperture_writes"], [])
                self.assertEqual([(w["address"], w["value"]) for w in setup["mailbox_writes"]],
                                 [(0xc000, 10)] + list(zip(range(0xc1ac, 0xc1dc, 4),
                                                           [selector, 0xfc00, *weight_targets, *expected_stack],
                                                           strict=True)))
                rows = [row for request in setup["dma_submissions"] for row in request["rows"]]
                if selector == 0:
                    for bank, (src, dst, a, b) in enumerate(rows[:6]):
                        program = banks[bank][4]
                        self.assertEqual((src - 0x50010000 + table, dst, plane_slots(a, b)),
                                         (program.offset, 0x04000000 + bank * 0x1000, 62))
                    rows = rows[6:]
                self.assertEqual(rows,
                                 [[0x20012708 + 0x1400 * (selector * 6 + i), dst, 0x13ff0000, 0x13ff0000]
                                  for i, dst in enumerate(weight_targets)] +
                                 [[0x20021708 + 0x60 * selector, 0xfc00, 0x005f0000, 0x005f0000]])
                local_records = []
                for worker, (base, file_offset, return_pc, r7), mailbox_weight in zip(
                        setup["workers"], refs, weight_targets[1:], strict=True):
                    self.assertEqual((worker["local_base"], worker["record_member_offset"], worker["return_pc"]),
                                     (base, file_offset, return_pc))
                    self.assertEqual(worker["registers_r4_to_r7"], [base + 0x994, selector, 0xfc00, r7])
                    self.assertEqual(worker["stack_words"], expected_stack)
                    wrapper = worker["isolated_wrapper"]
                    self.assertEqual(wrapper["r0_to_r7"][0], base + 0x994)
                    self.assertEqual(wrapper["r0_to_r7"][4:], worker["registers_r4_to_r7"])
                    self.assertEqual(wrapper["caller_stack_offset"], 0x24)
                    self.assertEqual(wrapper["stop"]["pc"], worker["stop"]["pc"] + 0x16)
                    self.assertEqual(worker["aperture_writes"], [])
                    self.assertEqual(worker["record_body_words"], 73)
                    values = [r["value"] for r in worker["mailbox_reads"]]
                    self.assertEqual(values, [selector, 0xfc00, mailbox_weight, *expected_stack])
                    local_records.append(unpack("<73I", pdn[file_offset:file_offset + 73 * 4]))
                for first, second in zip(local_records, local_records[1:]):
                    self.assertEqual({i: b - a for i, (a, b) in enumerate(zip(first, second, strict=True)) if a != b},
                                     {0: 0x800, 5: 0x1400, 57: 0x80000})
        corrupted = deepcopy(trace["calls"][-2:])
        dma = corrupted[0]["setup_since_previous_compute"]["dma_submissions"]
        first_worker_load = next(row for request in dma for row in request["rows"] if row[1] == 0x04001000)
        first_worker_load[0] -= banks[1][4].offset - banks[0][4].offset
        with self.assertRaisesRegex(ValueError, "does not match the original overlay"):
            bind_final_programs(pdn, corrupted)
        self.assertEqual(trace["earlier_head_base_writes"],
                         [{"pc": 0x50004fa2, "address": 0xe4e0, "size": 2, "value": 0}])
        self.assertEqual(trace["unseeded_head_reads"],
                         [{"pc": 0x500039f0, "address": 0xe4e0, "size": 2, "value": 0}])

    @unittest.skipUnless(all(p.is_file() for p in (CP, PDN, DICTIONARY)), "requires pinned PDN inputs")
    def test_later_controller_stops_unseeded_but_scalar_dot_matches_explicit_fixtures(self):
        cp, pdn, dictionary = CP.read_bytes(), PDN.read_bytes(), DICTIONARY.read_bytes()
        weights = unpack("<96h", dictionary[0x11958:0x11a18])
        fixtures = (([0] * 96, 0),
                    ([1, *([0] * 95)], -511),
                    ([0, -3, *([0] * 94)], 2151),
                    ([*([0] * 95), 7], -10808),
                    ([32767 if w >= 0 else -32768 for w in weights], None),
                    ([-32768 if w >= 0 else 32767 for w in weights], None))
        for values, expected in fixtures:
            with self.subTest(expected=expected):
                result = probe(cp, pdn, dictionary=dictionary, trace_later=True, head_fixture=values)
                trace = result["isolated_later_control"]
                self.assertEqual((len(trace["calls"]), len(trace["dictionary_copies"]), len(trace["waits"])),
                                 (26, 70, 66))
                self.assertEqual(trace["scalar_head_entry"], {"pc": 0x500039e8, "reason": "slice boundary"})
                self.assertEqual(trace["scalar_head_dictionary_offset"], 0x11958)
                self.assertEqual(trace["unseeded_stop"],
                                 {"pc": 0x500039f0, "reason": "unmodeled read", "address": 0xe4e8, "size": 2})
                last = trace["calls"][-2:]
                self.assertEqual([r["registers_r4_to_r7"] for r in last],
                                 [[0x50000980, i, 0xfc00, 0xd0e0] for i in (0, 1)])
                actual = trace["scalar_head_on_explicit_fixture"]
                self.assertEqual(actual["dot"], scalar_head_dot(dictionary, values))
                self.assertEqual(actual["i16"], scalar_head_quantized(dictionary, values))
                fp32, = unpack("<f", pack("<I", actual["fp32_bits"]))
                one = np.float32(1)
                logit = np.float32(actual["i16"]) / np.float32(2048)
                self.assertAlmostEqual(fp32, one / (one + np.exp(-logit)), delta=2e-6)
                self.assertEqual(actual["written_address"], 0xe0020000)
                self.assertLessEqual(abs(actual["q30"] - fp32 * (1 << 30)), 1)
                if expected is not None:
                    self.assertEqual(actual["dot"], expected)
                if expected == 0:
                    self.assertEqual((actual["i16"], actual["q30"]), (55, 0x206dff00))

    @unittest.skipUnless(CP.is_file() and PDN.is_file(), "requires pinned A7 IV CP and PDN")
    def test_both_original_eye_branches_use_the_packed_coordinates(self):
        cp, pdn = CP.read_bytes(), PDN.read_bytes()
        left, right = (281 << 18, 193 << 18), (300 << 18, 190 << 18)
        for eye, center, window, local in (("left", left, [266, 178, 29, 29], 0x1600a),
                                          ("right", right, [285, 175, 29, 29], 0x1600d)):
            with self.subTest(eye=eye):
                r = probe(cp, pdn, eye=eye)
                self.assertEqual((r["cp_parameter_address"], r["converted_address"]),
                                 (0x40000400, 0xa0000400))
                self.assertEqual(unpack("<3I3H2x4I4HI", bytes.fromhex(r["cp_parameter_hex"])),
                                 (0x40008000, 0x40010000, 0x40020000, 640, 480, 704,
                                  *left, *right, 64, 64, 1, 8, 42))
                self.assertEqual(r["dispatch_stop"], {"pc": 0x500015a8, "reason": "slice boundary"})
                self.assertEqual(r["crop_call"], {"eye": eye, "r4_r7": [0xa0008000, 640, 480, 704],
                                                 "stack_words": [*center, 64, 64, 1, 8, 0x17000]})
                g = r["image_geometry"]
                self.assertEqual(g["window_xywh"], window)
                x, y, width, height = window
                source = 0x20008000 + y * 1408 + (x & ~15) * 2
                row_width = ((x + width + 15) & ~15) - (x & ~15)
                self.assertEqual(g["image_request"]["rows"], [
                    [0, 0x4c000000, 0, 0],
                    [source, 0x216000, 0xc00000b0 | (2 * row_width - 1) << 16,
                     0x40000000 | (row_width // 2 - 1) << 16 | (height - 1) * 0x101],
                ])
                self.assertEqual(g["compute"]["registers_r4_to_r7"],
                                 [0x50000ab0, local, 64, 0x17000])

    @unittest.skipUnless(CP.is_file() and PDN.is_file(), "requires pinned A7 IV CP and PDN")
    def test_original_pdn_clipped_dma_rows_and_border_continuations(self):
        cp, pdn = CP.read_bytes(), PDN.read_bytes()
        rng = Random(0x50444e)
        padding_cases = set()

        def value(x, y):
            return (x * 41 + y * 67) & 255

        for _ in range(48):
            w, h = rng.choice((60, 62, 128, 638, 640)), rng.choice((48, 96, 478, 480))
            pitch = ((w + 3) & ~3) + rng.randrange(5) * 4
            x = rng.choice((0, w - 1, rng.randrange(w)))
            y = rng.choice((0, h - 1, rng.randrange(h)))
            scale = rng.choice((48, 64, 95, rng.randrange(48, 96)))
            with self.subTest(frame=(w, h, pitch), center=(x, y), scale=scale):
                center = (x << 18, y << 18)
                result = probe(cp, pdn, frame=(w, h, pitch), left=center, right=center,
                               scales=(scale, scale))
                self.assertEqual(result["validator_mask"], 0)
                g = result["image_geometry"]
                request = bind_image_request(g)
                sx, sy = request.source_offset % (2 * pitch) // 2, request.source_offset // (2 * pitch)
                local = bytes(value(sx + col, sy + row) for row in range(request.source_rows)
                              for col in range(request.source_row_bytes // 2))
                bx, by, bw, bh = request.window_xywh
                padding_cases.add((bx < 0, by < 0, bx + bw > w, by + bh > h))
                expected = bytes(value(min(max(px, 0), w - 1), min(max(py, 0), h - 1))
                                 for py in range(by, by + bh) for px in range(bx, bx + bw))
                self.assertEqual(request.crop_from_local_rows(local), expected)
        self.assertTrue({(True, True, False, False), (False, False, True, True),
                         (True, False, False, True), (False, True, True, False),
                         (False, False, False, False)} <= padding_cases)

    @unittest.skipUnless(CP.is_file() and PDN.is_file(), "requires pinned A7 IV CP and PDN")
    def test_pdn_source_gather_keeps_both_lanes_and_rejects_wrong_edge_requests(self):
        cp, pdn = CP.read_bytes(), PDN.read_bytes()
        g = probe(cp, pdn, frame=(128, 96, 132), left=(127 << 18, 0), right=(0, 0),
                  scales=(95, 95))["image_geometry"]
        request = bind_image_request(g)
        self.assertEqual((request.local_origin, request.compute_address), (0x16540, 0x16009))
        self.assertEqual((request.source_offset, request.source_row_bytes, request.source_rows),
                         (192, 72, 23))
        frame = bytes(b for y in range(96) for x in range(132) for b in (x, y ^ 0x80))
        gathered = bytes(b for y in range(23) for x in range(96, 132) for b in (x, y ^ 0x80))
        self.assertEqual(request.gather(frame), gathered)
        with self.assertRaisesRegex(ValueError, "truncated packed source"):
            request.gather(frame[:22 * 264 + 263])
        with self.assertRaisesRegex(ValueError, "one-byte post-transfer rows"):
            request.crop_from_local_rows(gathered)
        bad = deepcopy(g)
        bad["edge_requests"][0]["rows"][0][1] ^= 1
        with self.assertRaisesRegex(ValueError, "edge DMA"):
            bind_image_request(bad)

    @unittest.skipUnless(CP.is_file() and PDN.is_file(), "requires pinned A7 IV CP and PDN")
    def test_validator_rejects_rectangles_and_checks_both_centers_before_cropping(self):
        cp, pdn = CP.read_bytes(), PDN.read_bytes()
        for fields, mask in (({"scales": (64, 80)}, 0xc00), ({"scales": (47, 47)}, 0xc00),
                             ({"scales": (96, 96)}, 0xc00), ({"frame": (640, 480, 512)}, 0x20),
                             ({"left": (-1, 1)}, 0x40), ({"right": (1, 480 << 18)}, 0x200),
                             ({"angle": 3}, 0x1000), ({"node": 15}, 0x2000)):
            with self.subTest(fields=fields):
                r = probe(cp, pdn, **fields)
                self.assertEqual(r["validator_mask"], mask)
                self.assertNotIn("image_geometry", r)
        for scale in (48, 95):
            result = probe(cp, pdn, scales=(scale, scale))
            self.assertEqual(result["validator_mask"], 0)
            self.assertIn("image_geometry", result)

    @unittest.skipUnless(BUNDLE.is_file(), "requires the A7 IV camera-post SP bundle")
    def test_exact_fdn_dma_helper_is_shared_with_four_detector_members(self):
        data = BUNDLE.read_bytes()
        records = members(data)
        expected = {0x3e: [0xaf0, 0x53be8], 0x40: [0x29b8], 0x42: [0x1c80],
                    0x43: [0xca0, 0x43d38], 0x44: [0xb64, 0x3f450]}
        origin = records[0x40].offset
        helper = data[origin + 0x29b8:origin + 0x2a60]
        self.assertEqual(sha256(helper).hexdigest(),
                         "badbf5d4b3475933b180070dd27a4352bc421d090972efd055f89da5729954d4")
        for job, offsets in expected.items():
            member = records[job]
            payload = data[member.offset:member.offset + member.size]
            self.assertEqual(payload.count(helper), len(offsets))
            for start in offsets:
                self.assertEqual(payload[start:start + len(helper)], helper)


if __name__ == "__main__":
    unittest.main()
