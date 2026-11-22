from pathlib import Path
from struct import pack_into
import unittest

from compare_sa_payload_records import rotation, spans
from parsers.sony_sa_post import members
from parsers.sony_sa_program_planes import (
    HEADER_SLOTS, PROGRAM_BASE, boot_programs, descriptor_slots, join_record, overlays,
    packed_boot_programs, program, summarize,
)
from parsers.sony_sa_slot_fields import rank_boundaries
from probe_sa_defdet_boot import DEFDET_SHA, controller_request, probe as probe_defdet
from probe_sa_fdn_parameters import probe


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "debug/a7iv_parts"
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
FDN = PARTS / "SA_FDN__SA07.bin"
PDN = PARTS / "SA_PDN__SA07.bin"
DICT = [ROOT / f"extracted/7m4/sabin/fdn_{name}_dict.bin" for name in ("front", "side")]
SA = ROOT / "extracted/7m4/sabin"


def sa_members(bundle):
    data = bundle.read_bytes()
    for record in members(data).values():
        yield record, data[record.offset:record.offset + record.size]


class SaProgramPlanesTest(unittest.TestCase):
    @unittest.skipUnless((SA / "sa_sabin_cp_cam_post.bin").is_file(), "requires CP camera SP bundle")
    def test_defdet_controller_checks_each_gate_and_resolves_parameter_pointer(self):
        data = next(b for m, b in sa_members(SA / "sa_sabin_cp_cam_post.bin") if m.job_id == 2)
        good = controller_request(data)
        self.assertEqual(good["parameter_pointer_writes"], [
            {"pc": 0xa48, "address": 0x23a4, "size": 4, "value": 0xc0000000}])
        self.assertEqual(good["gate_writes"], [
            {"pc": 0xd9a, "address": 0x11840, "size": 2, "value": 240},
            {"pc": 0xe0a, "address": 0x1184c, "size": 2, "value": 240},
        ])
        self.assertEqual(good["first_stop"], {"pc": 0x1230, "reason": "wfi"})
        self.assertEqual(good["submissions"], [
            {"pointer_pc": 0x1228, "pointer": 0xa100, "submit_pc": 0x122a,
             "rows": [[0x13000, 0xc038460a, 0x003f0000, 0],
                      [0x13800, 0xc038460a, 0x003f0000, 0]]}])
        for gate in ("field18", "field24"):
            result = controller_request(data, **{gate: 239})
            self.assertEqual((result["first_stop"], result["r0"], result["submissions"]),
                             ({"pc": 0xa00, "reason": "wfi"}, 2, []))
        unmatched = controller_request(data, register=0x12340)
        self.assertEqual(unmatched["parameter_pointer_writes"][0]["value"], 0xffffffff)
        self.assertEqual(unmatched["first_stop"],
                         {"pc": 0xdb2, "reason": "unmodeled read", "address": 0xffffffff, "size": 4})

    @unittest.skipUnless((SA / "sa_sabin_cp_cam_post.bin").is_file(), "requires CP camera SP bundle")
    def test_defdet_controller_uses_parameter_mode_not_dma_reply_for_compute_branch(self):
        data = next(b for m, b in sa_members(SA / "sa_sabin_cp_cam_post.bin") if m.job_id == 2)
        for reply in ((0, 0), (0x2e, 0xa5a5003d), (2, 2)):
            with self.subTest(reply=reply):
                plain = controller_request(data, reply_words=reply)
                self.assertEqual(plain["second_stop"], {"pc": 0x126c, "reason": "wfi"})
                self.assertEqual(plain["controller_branch"], {"r3": 0, "target": 0x127c})
                self.assertEqual(plain["continuation_wait_pcs"],
                                 [0x1bec, 0x1c34, 0x1c7c, 0x1d00, 0xa00])
                self.assertNotIn("compute_entry", plain)
                self.assertEqual([(w["address"], w["value"]) for w in plain["reply_field_writes"]],
                                 [(0x13038, reply[0] >> 2 & 1), (0x13838, reply[1] >> 2 & 1),
                                  (0x1303a, reply[0] & 3), (0x1383a, reply[1] & 3)])
                compute = controller_request(data, field28=1, reply_words=reply)
                self.assertEqual(compute["controller_branch"], {"r3": 2, "target": 0x1f94})
                self.assertEqual(compute["continuation_wait_pcs"], [0x20f0])
                self.assertEqual(compute["continuation_stop"],
                                 {"pc": 0x95c, "reason": "unsupported opcode", "opcode": 0xa})
                self.assertEqual(compute["submissions"][1],
                                 {"pointer_pc": 0x20b6, "pointer": 0xa100, "submit_pc": 0x20bc,
                                  "rows": [[0x2003a400, 0x03f81a00, 0x01ff003c, 1]]})
                entry = compute["compute_entry"]
                self.assertEqual((entry["return_pc"], entry["r4_r7"]),
                                 (0xb2e, [0x1c, 0, 0x6800, 0x12800]))
                self.assertEqual((entry["record"]["member_offset"], entry["record"]["max_index"]),
                                 (0x4dc, 72))
        other = controller_request(data, field28=2)
        self.assertEqual(other["first_stop"],
                         {"pc": 0x1dcc, "reason": "unmodeled read", "address": 0x49414, "size": 4})

    @unittest.skipUnless((SA / "sa_sabin_cp_cam_post.bin").is_file(), "requires CP camera SP bundle")
    def test_defdet_packed_rows_and_original_staging_request(self):
        member, data = next((m, b) for m, b in sa_members(SA / "sa_sabin_cp_cam_post.bin")
                            if m.job_id == 2)
        self.assertEqual((member.name, member.sha256), ("DEFDET__0x02", DEFDET_SHA))
        cases = (
            (0x100, 0x398, 0x120, 0x500103c0,
             ((0x2710000, 0x3fd0, 79), (0x2710400, 0x3fd0, 79),
              (0x2910800, 0x4738, 83), (0x2910c00, 0x4f00, 83),
              (0x2991000, 0x56c8, 84), (0x2991400, 0x5ea8, 84))),
            (0x6700, 0x6998, 0x110, 0x500103b0,
             ((0x2710000, 0xba30, 79), (0x2710400, 0xba30, 79),
              (0x3610800, 0xc198, 109), (0x3510c00, 0xcbd0, 107),
              (0x3591000, 0xd5d8, 108), (0x3591400, 0xdff8, 108))),
        )
        cohorts = []
        population = set()
        for block, table, size, source, expected in cases:
            with self.subTest(block=block):
                programs = packed_boot_programs(data, block)
                self.assertEqual(boot_programs(data, block), {})
                self.assertEqual(tuple((dst, image.offset, image.slots)
                                       for dst, image in programs.items()), expected)
                cohorts.append(programs[0x2710000].planes)
                for image in programs.values():
                    population.update(zip(*image.planes))
                trace = probe_defdet(data, block)
                self.assertEqual((trace["table_offset"], trace["table_bytes"]), (table, size))
                self.assertEqual(trace["first_wait"], {"pc": 0x5001015c, "reason": "wfi"})
                self.assertEqual(trace["after_synthetic_wake"],
                                 {"pc": 0x5001027c, "reason": "wfi"})
                self.assertEqual(trace["staging_register_writes"][-2:], [
                    {"pc": 0x5001012a, "address": 0x48230, "size": 4, "value": 1},
                    {"pc": 0x50010132, "address": 0x48238, "size": 4, "value": source},
                ])
                local = trace["isolated_first_local_slice"]
                expected_slice = ((11, 0x4c0, 0x23b0, 0x230e) if block == 0x100 else
                                  (10, 0x6ab0, 0x3590, 0x2ea6))
                self.assertEqual((local["rows"], local["offset"], local["bytes"]), expected_slice[:3])
                self.assertEqual(local["stop"], {"pc": expected_slice[-1], "reason": "unmodeled read",
                                                 "address": 0x49e00, "size": 4})
        self.assertEqual(cohorts[0], cohorts[1])
        self.assertEqual(len(population), 375)
        best = rank_boundaries(population)[0]
        self.assertEqual((best["width"], best["start_bit"]), (13, 0))

    @unittest.skipUnless((SA / "sa_sabin_cp_cam_post.bin").is_file(), "requires CP camera SP bundle")
    def test_defdet_worker_command_slots_match_the_six_loaded_tables(self):
        data = next(b for m, b in sa_members(SA / "sa_sabin_cp_cam_post.bin") if m.job_id == 2)
        for block, shift, selector, extra_max in (
                (0x100, 0, 0x118ea, [82, 82, 83, 83]),
                (0x6700, 0x10, 0x11996, [108, 106, 107, 107])):
            with self.subTest(block=block):
                trace = probe_defdet(data, block)["isolated_compute_calls"]
                calls = trace["controller_call_slices"]
                self.assertEqual([r["call_pc"] for r in calls],
                                 [pc + shift for pc in (0xb2c, 0xb86, 0xbe2, 0xc94, 0xcee, 0xd36)])
                self.assertEqual([r["r4"] for r in calls],
                                 [0x1c, 0x1c, 0x1c, 0x20c, 0x20c, 0x4cc])
                self.assertEqual(calls[4]["slice_pc"], 0xce4 + shift)
                workers = trace["workers"]
                self.assertEqual([r["base"] for r in workers], [0x4800, 0x5800, 0x6000, 0x8800, 0x9000])
                modes = workers[0]["runs"]
                self.assertEqual([r["selector_fixture"] for r in modes], [0, 1, 2, 3])
                for seq, mode in zip((0, 3, 5), modes[:3], strict=True):
                    self.assertEqual(mode["read_addresses"][0], selector)
                    self.assertEqual(mode["stop"], {"pc": 0x5174, "reason": "unsupported opcode",
                                                   "opcode": 0x000a})
                    self.assertEqual(mode["r4"] - 0x4800, calls[seq]["r4"] + 0x18)
                    self.assertEqual(mode["record"]["bytes"], calls[seq]["record"]["bytes"])
                    diff = mode["diff_from_controller"]
                    self.assertEqual([r["controller_word"] >> 24 for r in diff],
                                     [2, 3, 25, 36, 25, 36, 52, 54])
                    for row in diff:
                        self.assertEqual(row["worker_word"] - row["controller_word"], 0x3000)
                        self.assertEqual(row["controller_word"] >> 24, row["worker_word"] >> 24)
                        self.assertIn(row["controller_word"] & 0xffff, (0x9800, 0xa000))
                self.assertEqual([m["record"]["max_index"] for m in modes[:3]], [72, 78, 76])
                self.assertEqual(modes[3]["stop"], {"pc": 0x51d0, "reason": "wfi"})
                self.assertNotIn("record", modes[3])
                for worker, high_index in zip(workers[1:], extra_max, strict=True):
                    base = worker["base"]
                    run, = worker["runs"]
                    offsets = ((0x218, 0x2b4, 0x1d4, 0x134) if not shift else
                               (0x2a0, 0x33c, 0x25c, 0x184))
                    self.assertEqual(run["wait_pcs"], [base + v for v in offsets[:2]])
                    self.assertEqual(run["stop"], {"pc": base + offsets[2],
                                                  "reason": "unsupported opcode", "opcode": 0x000a})
                    self.assertEqual(run["r4"], base + offsets[3])
                    self.assertEqual(run["record"]["max_index"], high_index)
                    self.assertEqual(worker["program_slots"], high_index + 1)

    @unittest.skipUnless(all((SA / f"sa_sabin_{kind}_zima_post.bin").is_file()
                             for kind in ("cp", "mp")), "requires both ZIMA SP bundles")
    def test_two_slot_controls_contain_only_the_shared_prefix(self):
        expected = {"MN_SAZNRSA07": (0x90800, 0x197c0),
                    "ZM_AACE_0x00": (0x90400, 0x35e00),
                    "ZM_AACD_0x00": (0x90400, 0x45e00),
                    "ZM_A3E__0x00": (0x90400, 0x7cf00),
                    "PR_LPCMD0x00": (0x90400, 0x5f200)}
        seen = set()
        for kind in ("cp", "mp"):
            for member, data in sa_members(SA / f"sa_sabin_{kind}_zima_post.bin"):
                if member.name not in expected:
                    continue
                dst, offset = expected[member.name]
                with self.subTest(member=member.name):
                    programs = packed_boot_programs(data)
                    self.assertEqual(list(programs), [dst])
                    image = programs[dst]
                    self.assertEqual((image.offset, image.slots), (offset, 2))
                    self.assertEqual(tuple(zip(*image.planes)), HEADER_SLOTS)
                    seen.add(member.name)
        self.assertEqual(seen, set(expected))

    def test_packed_rows_reject_conflicting_sizes_and_bad_prefix(self):
        blob = bytearray(0x200)
        pack_into("<I", blob, 0x110, 0x50)
        pack_into("<2I", blob, 0x120, 0x40, 0x10)
        pack_into("<4I", blob, 0x140, 0, 0x90400, 0x2e4000, 0x04000200)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            packed_boot_programs(blob)
        pack_into("<I", blob, 0x148, 0x2f4000)
        with self.assertRaisesRegex(ValueError, "prefix"):
            packed_boot_programs(blob)
        pack_into("<I", blob, 0x140, 0x90)
        with self.assertRaisesRegex(ValueError, "outside member"):
            packed_boot_programs(blob)

    def test_control_record_indices_match_three_plane_extents(self):
        cases = (
            ("SA_ROT900x02", 0, 0xc70, 28, 0x35c, 0x4dc),
            ("SA_NV12_0x02", 1, 0x1040, 43, 0xe54, 0xf60),
            ("SA_SHA250x02", 0, 0x1fe0, 33, 0x4bc, 0x564),
            ("SA_SHA250x02", 1, 0x22f8, 49, 0x14e4, 0x15cc),
        )
        for member, bank, offset, count, first, last in cases:
            path = PARTS / f"{member}.bin"
            with self.subTest(member=member, bank=bank):
                if not path.is_file():
                    self.skipTest(f"requires extracted {member}")
                data = path.read_bytes()
                image = boot_programs(data)[PROGRAM_BASE + bank * 0x1000]
                self.assertEqual((image.offset, image.slots), (offset, count))
                self.assertEqual(len(summarize(data, image)["sha256"]), 64)
                joined = join_record(image, data[first:last])
                self.assertEqual({row["index"] for row in joined}, set(range(count)))
                with self.assertRaisesRegex(ValueError, "outside program"):
                    join_record(image, (count << 24).to_bytes(4, "little"))
                if member == "SA_ROT900x02":
                    stage, start, stop, _, payload = rotation(data)
                    self.assertEqual((stage + start, stage + stop, payload), (first, last, data[first:last]))
                elif member == "SA_SHA250x02" and bank == 1:
                    matches = [r for r in joined if r["bit31"]]
                    self.assertEqual([r["index"] for r in matches], [14, 21, 29, 21])
                    self.assertEqual(matches[1]["plane_words"], matches[-1]["plane_words"])

    @unittest.skipUnless(FDN.is_file(), "requires extracted FDN")
    def test_fdn_catalog_sources_are_regular_and_boot_capacity_is_not_global(self):
        data = FDN.read_bytes()
        for block, table, expected in (
                (0x100, 0x9970, [108, 66, 96, 87, 121]),
                (0x1a800, 0x24140, [116, 87, 121, 62])):
            with self.subTest(block=block):
                actual, banks = overlays(data, block)
                self.assertEqual(actual, table)
                initial = boot_programs(data, block)
                self.assertEqual(len(initial), 6)
                for bank, row in enumerate(banks):
                    valid = [image for image in row if image is not None]
                    self.assertEqual([image.slots for image in valid], expected)
                    self.assertEqual(row[0].planes, initial[PROGRAM_BASE + bank * 0x1000].planes)
                    self.assertEqual([i.planes for i in valid],
                                     [i.planes for i in banks[0] if i is not None])
        _, a = overlays(data)
        _, b = overlays(data, 0x1a800)
        self.assertEqual(a[0][4].planes, b[0][2].planes)
        self.assertEqual(tuple(p[:66] for p in b[0][0].planes), a[0][1].planes)
        self.assertEqual(tuple(p[:70] for p in a[0][2].planes),
                         tuple(p[:70] for p in a[0][4].planes))
        with self.assertRaisesRegex(ValueError, "outside program"):
            join_record(a[0][0], data[0x44ec:0x45fc])
        self.assertEqual(max(r["index"] for r in join_record(a[0][4], data[0x44ec:0x45fc])), 120)
        with self.assertRaisesRegex(ValueError, "outside member"):
            program(data, len(data) - 1, 2)

    @unittest.skipUnless(FDN.is_file() and PDN.is_file(), "requires FDN and PDN members")
    def test_pdn_catalog_contains_an_identical_121_slot_program(self):
        fdn, pdn = FDN.read_bytes(), PDN.read_bytes()
        _, a = overlays(fdn)
        table, banks = overlays(pdn)
        self.assertEqual(table, 0xec10)
        self.assertEqual([i.slots for i in banks[0] if i], [121, 122, 105, 114, 62, 39])
        self.assertTrue(all(row[0].planes == a[0][4].planes for row in banks))

    @unittest.skipUnless(all(p.is_file() for p in [CP, FDN, *DICT]), "requires pinned members")
    def test_original_a_controller_requests_all_four_catalog_replacements(self):
        fdn = FDN.read_bytes()
        result = probe(CP.read_bytes(), fdn, dictionaries=[p.read_bytes() for p in DICT],
                       trace_catalog=True)
        run = result["first_workload_fixture"]["caller_continuation"]["program_catalog_continuation"]
        table, banks = overlays(fdn)
        self.assertEqual(run["catalog"]["member_offset"], table)
        self.assertEqual(run["catalog"]["boot_writes"], [
            {"pc": 0x50010130, "address": 0x48168, "size": 4, "value": 0x50019870}])
        expected = [(PROGRAM_BASE + bank * 0x1000, banks[bank][i])
                    for i in range(1, 5) for bank in range(6)]
        self.assertEqual(len(run["program_requests"]), len(expected))
        for request, (destination, image) in zip(run["program_requests"], expected):
            src, dst, w2, w3 = request["words"]
            with self.subTest(slots=image.slots, dst=destination):
                self.assertEqual((src, dst), (0x50019870 + image.offset - table, destination))
                self.assertEqual(descriptor_slots(dst, w2, w3), image.slots)
                self.assertEqual(request["member_offset"], image.offset)
                self.assertEqual(request["source_sha256"], summarize(fdn, image)["sha256"])
        self.assertEqual(run["isolated_calls"], [
            {"record_r4": 0x50000468, "return_pc": 0x50000d46, "program_requests_before_call": 12},
            {"record_r4": 0x50000600, "return_pc": 0x50000c58, "program_requests_before_call": 18},
        ])
        for call, slot_count in zip(run["isolated_calls"], (96, 87)):
            offset = 0x500 + call["record_r4"] - 0x50000000
            image = banks[0][call["program_requests_before_call"] // 6]
            first_span = next(spans(fdn[offset:0xde4]))[1]
            self.assertEqual(image.slots, slot_count)
            self.assertEqual(max(row["index"] for row in join_record(image, first_span)), slot_count - 1)
        self.assertEqual(run["stop"], {"pc": 0x50000b5c, "reason": "unmodeled read",
                                       "address": 0x48100, "size": 4})
        linked = run["isolated_a_to_b_handoff"]
        self.assertEqual((linked["link_table_member_offset"], linked["linked_block"],
                          linked["a_call_return_pc"]), (0x31c00, 0x1a800, 0x50000b7e))
        self.assertEqual(linked["entry"], {"pc": 0x5002a700, "reason": "slice boundary"})
        self.assertEqual(linked["b_catalog"], {"member_offset": 0x24140, "fixture_address": 0x50034040})
        self.assertEqual(linked["external_mapping"], {
            "member_offset": 0x1ac00, "address": 0x50800000, "size": 0x3c00})
        self.assertEqual(linked["external_stop"], {
            "pc": 0x50803352, "reason": "unmodeled read", "address": 0x49c80, "size": 4})


if __name__ == "__main__":
    unittest.main()
