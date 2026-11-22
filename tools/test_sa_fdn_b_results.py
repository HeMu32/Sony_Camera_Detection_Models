from copy import deepcopy
from pathlib import Path
from random import Random
from struct import pack_into, unpack_from
import unittest

from parsers.fdn_geometry import i32, project_b_landmarks, table_for_fdn_a
from parsers.fdn_head_io import materialize
from parsers.fdn_sections import parse
from parsers.sony_sa_program_planes import PROGRAM_BASE, descriptor_slots, join_record, overlays
from parsers.sony_sa_slot_fields import SlotFields
from probe_a7iv_fdn_cp_result import probe as cp_result
from probe_sa_fdn_b_results import probe, probe_writer
from probe_sa_fdn_parameters import SQUARE_WINDOW


ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
FDN = ROOT / "debug/a7iv_parts/SA_FDN__SA07.bin"
DICT = [ROOT / f"extracted/7m4/sabin/fdn_{name}_dict.bin" for name in ("front", "side")]


def output_words(result):
    return {w["address"] - result["fixture_output"]: w["value"] for w in result["writes"]}


def params(x, y, edge, width=640, height=480):
    data = bytearray(0x28)
    pack_into("<9H", data, 0x10, width, height, width + 64, x, y, edge, edge, 0, 0)
    return data


@unittest.skipUnless(FDN.is_file(), "requires pinned FDN member")
class FdnBResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fdn = FDN.read_bytes()
        cls.table = table_for_fdn_a(cls.fdn)

    def test_integer_projection_matches_original_result_writer(self):
        self.assertEqual(self.table, self.fdn[0x23510:0x23630])
        rng = Random(1234)
        for i in range(250):
            edge = rng.randrange(48, 96)
            x, y, direction = rng.randrange(50), rng.randrange(80), rng.randrange(2)
            route, group = rng.randrange(6), rng.randrange(9)
            points = tuple(tuple(rng.randrange(-0x100000, 0x300001) for _ in range(2))
                           for _ in range(3))
            temporary = [0x80000123, 0x44445555] + [v & 0xffffffff for xy in points for v in xy]
            with self.subTest(i=i, route=route, group=group):
                actual = probe_writer(self.fdn, params(x, y, edge), temporary,
                                      direction=direction, route_mode=route, matrix_group=group)
                projected = project_b_landmarks(self.table, (640, 480), (x, y, edge),
                                                points, group, route)
                self.assertEqual(actual["words"], [temporary[0], direction, *(v for xy in projected for v in xy)])
                self.assertEqual([w["address"] for w in actual["writes"]],
                                 list(range(0xe0004040, 0xe0004060, 4)))

    @unittest.skipUnless(CP.is_file(), "requires pinned CP")
    def test_original_cp_reads_the_writer_q18_landmarks_and_signed_score(self):
        raw = [0xfffffff5, 0xabababab, 0x80000, 0xc0000, 0x180000, 0xc0000, 0x100000, 0x140000]
        fixture = probe_writer(self.fdn, params(32, 32, 64), raw, direction=0,
                               matrix_group=4, route_mode=2)
        words = fixture["words"]
        result = cp_result(CP.read_bytes(), (i32(words[0]), *words[1:], 0x12345678), (-12, 5))
        self.assertEqual((result["score"], result["passed"], result["other_direction"]), (-11, 1, 0))
        self.assertEqual(result["left_eye_raw"], (48 << 18, 56 << 18))
        self.assertEqual(result["right_eye_raw"], (80 << 18, 56 << 18))
        self.assertEqual(result["nose_root_raw"], (64 << 18, 72 << 18))

    @unittest.skipUnless(CP.is_file(), "requires pinned CP")
    def test_optional_branch_writes_zero_body_echoes_parameters_and_relocates(self):
        cp = CP.read_bytes()
        for delta, exec_id in ((0, 0x13579bdf), (0x20, 0x0cf00d12)):
            with self.subTest(delta=delta):
                result = probe(cp, self.fdn, branch=1, output_delta=delta, exec_id=exec_id)
                self.assertEqual(result["branch_reads"], [
                    {"pc": 0x50800f28, "address": 0x481cc, "value": 1}])
                self.assertEqual(result["fixture_output"], 0xe0004040 + delta)
                writes = output_words(result)
                self.assertEqual(writes[0x20], exec_id)
                self.assertEqual(result["writes"][0]["pc"], 0x50800f24)
                self.assertEqual(writes.keys(), set(range(0, 0x4c, 4)))
                self.assertEqual([writes[i] for i in range(0, 0x20, 4)], [0] * 8)
                self.assertEqual(b"".join(writes[i].to_bytes(4, "little") for i in range(0x24, 0x4c, 4)),
                                 bytes.fromhex(result["cp_parameters_hex"]))

    @unittest.skipUnless(all(p.is_file() for p in [CP, *DICT]), "requires pinned CP and dictionaries")
    def test_zero_branch_uses_original_dictionary_and_reaches_normal_writer(self):
        cp = CP.read_bytes()
        unavailable = probe(cp, self.fdn)
        self.assertEqual(unavailable["branch_reads"], [
            {"pc": 0x50800f28, "address": 0x481cc, "value": 0}])
        self.assertEqual(unavailable["stop"], {"pc": 0x508008f4, "reason": "unmodeled read",
                                              "address": 0xa005654c, "size": 4})
        self.assertEqual(output_words(unavailable), {0x20: 0x13579bdf})
        dictionaries = [p.read_bytes() for p in DICT]
        result = probe(cp, self.fdn, dictionaries=dictionaries)
        _, banks = overlays(self.fdn, 0x1a800)
        expected = [(PROGRAM_BASE + bank * 0x1000, banks[bank][i])
                    for i in (1, 2, 0, 3) for bank in range(6)]
        self.assertEqual(len(result["program_requests"]), len(expected))
        for request, (destination, image) in zip(result["program_requests"], expected):
            self.assertEqual((request["member_offset"], request["words"][1]), (image.offset, destination))
            self.assertEqual(descriptor_slots(*request["words"][1:]), image.slots)
        self.assertEqual(result["isolated_calls"], [
            {"r4": 0x5080024c, "return_pc": 0x50800b2a, "requests_before_call": 6},
            {"r4": 0x50800720, "return_pc": 0x508009aa, "requests_before_call": 24},
        ])
        self.assertEqual(result["stop"], {"pc": 0x5080079e, "reason": "unmodeled read",
                                         "address": 0x48174, "size": 4})
        writes = output_words(result)
        self.assertEqual(writes.keys(), set(range(0, 0x4c, 4)))
        self.assertEqual((writes[0], writes[4], writes[0x20]), (0, 0, 0x13579bdf))
        moved = (*SQUARE_WINDOW[:3], SQUARE_WINDOW[3] + 1, SQUARE_WINDOW[4] + 1, *SQUARE_WINDOW[5:])
        translated = output_words(probe(cp, self.fdn, dictionaries=dictionaries, window=moved))
        for coordinate in range(8, 0x20, 4):
            self.assertEqual(translated[coordinate] - writes[coordinate], 1 << 18)

    @unittest.skipUnless(all(p.is_file() for p in [CP, *DICT]), "requires pinned CP and dictionaries")
    def test_seven_compact_heads_are_bound_to_score_and_landmark_addresses(self):
        cp = CP.read_bytes()
        dictionaries = [p.read_bytes() for p in DICT]
        for dictionary, node, data in zip(("front", "side"), (2, 3), dictionaries):
            with self.subTest(dictionary=dictionary):
                run = probe(cp, self.fdn, dictionaries=dictionaries, window=(*SQUARE_WINDOW[:-1], node))
                heads = parse(data)["dense_candidates_144x1"]
                indices = (4, 0, 1, 2, 3, 5, 6)
                reads = run["dictionary_reads"]
                self.assertEqual(len(heads), 7)
                self.assertEqual([(r["dictionary"], r["offset"], r["size"]) for r in reads],
                                 [(dictionary, heads[i]["bias_offset"], 4) for i in indices])
                self.assertEqual([r["value"] for r in reads],
                                 [unpack_from("<I", data, heads[i]["bias_offset"])[0] for i in indices])
                base = run["writer_temporary_pointer"]
                self.assertEqual(run["score_head_arguments"]["temporary_result"], base)
                jobs = [item for batch in run["landmark_head_mailboxes"] for item in batch]
                self.assertEqual([j["temporary_result"] - base for j in jobs],
                                 [0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c])
                self.assertEqual([r["value"] for r in reads[1:]], [j["bias_word"] for j in jobs])

    @unittest.skipUnless(all(p.is_file() for p in [CP, *DICT]), "requires pinned CP and dictionaries")
    def test_original_dense_workers_and_weight_regions_match_all_seven_heads(self):
        cp = CP.read_bytes()
        dictionaries = [p.read_bytes() for p in DICT]
        for node, data in zip((2, 3), dictionaries):
            with self.subTest(node=node):
                run = probe(cp, self.fdn, dictionaries=dictionaries,
                            window=(*SQUARE_WINDOW[:-1], node), trace_heads=True)
                heads = materialize(data, run)
                self.assertEqual([(h.role, h.head) for h in heads],
                                 [("score", 4), ("left_x", 0), ("left_y", 1), ("right_x", 2),
                                  ("right_y", 3), ("nose_x", 5), ("nose_y", 6)])
                self.assertEqual([h.weight_pointer for h in heads], list(range(0x3800, 0x7000, 0x800)))
                self.assertEqual([h.activation_pointer for h in heads],
                                 [0x14400, 0x14c00, 0x15400, 0x15c00, 0x14c00, 0x15400, 0x15c00])
                self.assertEqual([h.weight_bytes for h in heads],
                                 [data[h.dictionary_offset:h.dictionary_offset + 288] for h in heads])
                self.assertEqual([h.temporary_pointer - run["writer_temporary_pointer"] for h in heads],
                                 [0, 8, 12, 16, 20, 24, 28])
                fixture = run["head_fixture"]
                locations = [c["record_member_offset"] for c in fixture["worker_calls"]]
                self.assertEqual(locations, [0x1ef38, 0x1fd98, 0x20be8] * 2)
                all_jobs = [fixture["outer_call"], *fixture["worker_calls"]]
                expected = self.fdn[0x1b320:0x1b360]
                self.assertTrue(all(bytes.fromhex(c["record_hex"]) == expected
                                    and c["count_word"] == 12 for c in all_jobs))
                _, banks = overlays(self.fdn, 0x1a800)
                indices = [*range(51, 59), 59, 58, 60, 61, 0, 1, 1, 1]
                for bank in range(6):
                    image = banks[bank][3]
                    self.assertEqual([r["index"] for r in join_record(image, expected)], indices)
                    once, again = image.at(57), image.at(58)
                    self.assertEqual(once[:2], again[:2])
                    self.assertEqual((once[2] ^ again[2]) & ((1 << 60) - 1), 0)
                    self.assertEqual((SlotFields.decode(once).top4, SlotFields.decode(again).top4), (2, 11))
                bad_trace = deepcopy(run)
                bad_trace["head_fixture"]["outer_call"]["registers_r5_to_r7"][1] += 2
                with self.assertRaisesRegex(ValueError, "does not match score"):
                    materialize(data, bad_trace)

    @unittest.skipUnless(all(p.is_file() for p in [CP, *DICT]), "requires pinned CP and dictionaries")
    def test_prehead_batches_keep_original_waits_and_isolate_the_gate_perturbation(self):
        cp = CP.read_bytes()
        dictionaries = [p.read_bytes() for p in DICT]
        for node, bank_base, data in zip((2, 3), (0x20040000, 0x20080000), dictionaries):
            with self.subTest(node=node):
                fixture = probe(cp, self.fdn, dictionaries=dictionaries,
                                window=(*SQUARE_WINDOW[:-1], node), trace_heads=True)["head_fixture"]
                batches = fixture["prehead_batches"]
                submits = [r for r in fixture["dma_submissions"]
                           if r["submit_pc"] in (0x50802898, 0x50802756)]
                self.assertEqual([b["submit_pc"] for b in batches], [0x50802898] * 3 + [0x50802756])
                self.assertEqual([len(s["rows"]) for s in submits], [10, 10, 10, 7])
                compact = submits[-1]
                self.assertEqual(compact["pointer_pc"], 0x508026fc)
                self.assertEqual([row[1:] for row in compact["rows"]],
                                 [[dst, 0x11f0000, 0x11f0000] for dst in range(0x3800, 0x7000, 0x800)])
                raw = next(s["rows"] for s in fixture["dma_submissions"] if s["submit_pc"] == 0x50802a02)
                self.assertEqual(raw, [[src, 0xc0012800, 0xffff0000, 0xffff0000]
                                       for src in range(0x14000, 0x17000, 0x800)])
                positions = [0x1ee58, 0x1fcb8, 0x20b08, 0x21ea8, 0x22d08]
                waits = [0xfc8, 0x1fc8, 0x2fc8, 0x87c8, 0x97c8]
                _, banks = overlays(self.fdn, 0x1a800)
                for batch, base_bias_offset in zip(batches, (0x12d00, 0x12d18, 0x12d30, 0x12d48)):
                    workers = batch["workers"]
                    self.assertEqual([w["gate_word"] for w in workers], [0] * 5)
                    self.assertEqual([w["stop"] for w in workers],
                                     [{"pc": wait, "reason": "wfi"} for wait in waits])
                    calls = [w["perturbed_gate_one"] for w in workers]
                    self.assertEqual([c["record_member_offset"] for c in calls], positions)
                    self.assertEqual(calls[0]["first_bias_read"],
                                     {"pc": 0x1532, "address": workers[0]["bias_pointer"],
                                      "value": unpack_from("<I", data, base_bias_offset)[0]})
                    for worker, call in zip(workers, calls):
                        self.assertEqual(call["registers_r5_to_r7"], [0, worker["weight_pointer"], 0])
                        record = bytes.fromhex(call["record_hex"])
                        self.assertEqual(len(record), 0x120)
                        self.assertEqual(record[-0x40:], self.fdn[0x1b320:0x1b360])
                        items = join_record(banks[worker["bank"] + 1][3], record[:-0x40])
                        self.assertEqual({item["index"] for item in items}, set(range(51)))
                self.assertEqual(compact["rows"][0][0], bank_base + 0x1642c)

    @unittest.skipUnless(all(p.is_file() for p in [CP, *DICT]), "requires pinned CP and dictionaries")
    def test_original_boot_table_and_a_dictionary_submit_reach_first_b_compute(self):
        cp = CP.read_bytes()
        dictionaries = [p.read_bytes() for p in DICT]
        for node, name, data in zip((2, 3), ("front", "side"), dictionaries):
            with self.subTest(node=node):
                result = probe(cp, self.fdn, dictionaries=dictionaries,
                               window=(*SQUARE_WINDOW[:-1], node), boot_control=True)
                fixture = result["boot_control_fixture"]
                self.assertEqual(fixture["boot_table"],
                                 {"member_offset": 0x21310, "destination": 0x7800, "size": 0x540})
                self.assertEqual(fixture["padding_source"],
                                 {"member_offset": 0x23630, "destination": 0x17f80, "size": 48})
                padding = next(r for r in fixture["b_submissions"] if r["submit_pc"] == 0x50802b12)
                self.assertEqual(padding["rows"], [
                    [0xb800 + 968 * plane + 44 * y, 0xc0017f80, 0x2b0000, 0x2b0000]
                    for y in (0, 21) for plane in range(18)])
                self.assertEqual(fixture["first_compute_buffer_coverage"],
                                 {"destination": 0xb800, "size": 17424, "dma_bytes": 1584,
                                  "scalar_written_bytes": 1440, "known_bytes": 3024,
                                  "unknown_bytes": 14400, "first_unknown": 0xb82e,
                                  "matches_zero_border_18x22x22": True})
                self.assertEqual(fixture["a_dictionary_submission"]["submit_pc"], 0x50002766)
                transfers = fixture["a_dictionary_transfers"]
                self.assertEqual(len(transfers), 48)
                self.assertEqual({(t["dictionary"], t["offset"], t["destination"], t["size"])
                                  for t in transfers if t["destination"] in (0x39d4, 0x3b3c)},
                                 {(name, 0x548, 0x39d4, 360), (name, 0xdb8, 0x3b3c, 20)})
                first = fixture["first_compute"]
                self.assertEqual(first["registers_r4_to_r7"], [0x5080001c, 0xb800, 0x39d4, 0x15018])
                self.assertEqual(first["record_member_offset"], 0x1ac1c)
                self.assertEqual(first["return_pc"], 0x50802d6c)
                self.assertEqual(first["runtime_bias_reads"],
                                 [{"pc": 0x50802d2e, "address": 0x3b3c,
                                   "value": unpack_from("<I", data, 0xdb8)[0]}])
                self.assertEqual(self.fdn[0x6e80:0x6f60], self.fdn[0x21310:0x213f0])
                record = self.fdn[0x1ac1c:0x1ae4c]
                _, banks = overlays(self.fdn, 0x1a800)
                active = join_record(banks[0][0], record)
                self.assertEqual({item["index"] for item in active}, set(range(116)))
                for offset in (0x1e834, 0x1f694, 0x204e4, 0x21884, 0x226e4):
                    self.assertTrue(self.fdn.startswith(record, offset))
                self.assertEqual(result["stop"],
                                 {"pc": 0x50800760, "reason": "unsupported opcode", "opcode": 10})
                self.assertEqual(result["writes"], [])

    @unittest.skipUnless(all(p.is_file() for p in [CP, *DICT]), "requires pinned CP and dictionaries")
    def test_original_boot_schedules_all_prehead_coefficients_without_writing_fake_results(self):
        cp = CP.read_bytes()
        dictionaries = [p.read_bytes() for p in DICT]
        for node, name, data in zip((2, 3), ("front", "side"), dictionaries):
            with self.subTest(node=node):
                run = probe(cp, self.fdn, dictionaries=dictionaries,
                            window=(*SQUARE_WINDOW[:-1], node), boot_control=True,
                            trace_later_stages=True)
                fixture = run["head_fixture"]
                self.assertEqual([descriptor_slots(*p["words"][1:]) for p in run["program_requests"]],
                                 [slots for slots in (87, 121, 116, 62) for _ in range(6)])
                copies, = (s["rows"] for s in fixture["dma_submissions"]
                           if s["submit_pc"] == 0x50802a02)
                self.assertEqual(copies, [[0x14000 + i * 0x800, 0xc0012800 + i * 720,
                                           719 << 16, 719 << 16] for i in range(6)])
                self.assertEqual([(w["pc"], w["value"]) for w in fixture["outer_40218_writes"]],
                                 [(0x50802cf4, 12), (0x50802f02, 11), (0x50802f02, 10),
                                  (0x50802cf4, 11), (0x50802f02, 10), (0x50802cf4, 14),
                                  (0x50802f02, 10), *[(0x50803098, 9)] * 4])
                batches = fixture["prehead_batches"]
                self.assertEqual(len(batches), 4)
                destinations = []
                for batch_index, batch in enumerate(batches):
                    workers = batch["workers"]
                    self.assertEqual([(w["bank"], w["selector"], w["subselector"], w["gate_word"])
                                      for w in workers], [(i, 3, 5, 6) for i in range(5)])
                    groups = [batch["outer_calls"], *(w["scheduled_calls"] for w in workers)]
                    for group, calls in enumerate(groups):
                        self.assertEqual(len(calls), 6)
                        for lane, call in enumerate(calls):
                            index = 24 * group + 6 * batch_index + lane
                            target = 0x14400 + index * 2
                            destinations.append(call["aperture_40330"])
                            self.assertEqual(call["aperture_40330"], target)
                            self.assertEqual(call["result_argument"], target)
                            self.assertEqual(call["aperture_40218"], 9)
                            self.assertEqual(call["input_pointer"],
                                             0x14000 + group * 0x800 + (360 if batch_index >= 2 else 0))
                            self.assertEqual((call["dictionary"], call["dictionary_offset"]),
                                             (name, 0x6220 + index * 360))
                            self.assertEqual(call["bias_word"], unpack_from("<I", data, 0x12ca0 + index * 4)[0])
                self.assertEqual(sorted(destinations), list(range(0x14400, 0x14520, 2)))
                heads = materialize(data, run)
                self.assertEqual([h.activation_pointer for h in heads],
                                 [0x14400, 0x14c00, 0x15400, 0x15c00, 0x14c00, 0x15400, 0x15c00])
                self.assertEqual(run["stop"], {"pc": 0x50801de4, "reason": "slice boundary"})
                self.assertEqual(output_words(run), {0x20: 0x13579bdf})

    @unittest.skipUnless(all(p.is_file() for p in [CP, *DICT]), "requires pinned CP and dictionaries")
    def test_original_b_plane_schedule_covers_the_six_dma_sources(self):
        cp = CP.read_bytes()
        dictionaries = [p.read_bytes() for p in DICT]
        for node, name, data in zip((2, 3), ("front", "side"), dictionaries):
            with self.subTest(node=node):
                run = probe(cp, self.fdn, dictionaries=dictionaries,
                            window=(*SQUARE_WINDOW[:-1], node), boot_control=True,
                            trace_later_stages=True)
                stage = run["head_fixture"]["packed_plane_stage"]
                workers = stage["workers"]
                self.assertEqual([(w["selector"], w["subselector"], w["gate_word"]) for w in workers],
                                 [(1, 5, 12)] * 5)
                self.assertEqual([w["scheduled_calls"][0]["record_member_offset"] for w in workers],
                                 [0x1e954, 0x1f7b4, 0x20604, 0x219a4, 0x22804])
                groups = [stage["outer_calls"], *(w["scheduled_calls"] for w in workers)]
                for group, (calls, input_pointer, skew) in enumerate(zip(
                        groups, (0x13a20, 0x14260, 0x14aa0, 0x13e40, 0x14680, 0x13a20),
                        (0, 24, 48, 12, 36, 0))):
                    self.assertEqual(len(calls), 12)
                    for plane, call in enumerate(calls):
                        index = 12 * group + plane
                        address = 0x12800 + index * 60
                        self.assertEqual(call["input_pointer"], input_pointer)
                        self.assertEqual(call["aperture_40330"], address)
                        self.assertEqual(call["result_argument"], address + skew)
                        self.assertEqual(call["aperture_40218"], 10)
                        self.assertEqual((call["dictionary"], call["dictionary_offset"]),
                                         (name, 0x12f14 + index * 168))
                        self.assertEqual(call["bias_word"], unpack_from("<I", data, 0x15e54 + index * 4)[0])


if __name__ == "__main__":
    unittest.main()
