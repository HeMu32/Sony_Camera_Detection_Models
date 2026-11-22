from collections import Counter
from hashlib import sha256
from math import exp
from pathlib import Path
from struct import pack, unpack_from
import tarfile
import unittest

from parsers.sony_sa_post import members
from parsers.sony_sa_program_planes import startup_rows
from probe_a7iv_keypoint_cp_result import BACKUP_MEMBER, published_config
from probe_sa_bed_image import (
    PackedRead, decode_fifth_feature_rows, dictionary_span, locate_second_dictionary, padded_3x3,
    second_dictionary_layout, trace_bed,
)
from probe_sa_fdn_dispatch import pc_literal


ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
SP = ROOT / "extracted/7m4/sabin/sa_sabin_cp_cam_post.bin"
FIRMWARE = ROOT / "firmware/7M4_v602/firmware.tar"
BED_DICT1 = ROOT / "extracted/7m4/sabin/bed_dic1.bin"
BED_DICT2 = ROOT / "extracted/7m4/sabin/bed_dic2.bin"


@unittest.skipUnless(CP.is_file() and SP.is_file(), "requires pinned CP and BED images")
class BedImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp, cls.bundle = CP.read_bytes(), SP.read_bytes()

    @unittest.skipUnless(FIRMWARE.is_file(), "requires the A7 IV firmware archive")
    def test_published_cp_parameters_reach_original_image_and_compute_request(self):
        with tarfile.open(FIRMWARE) as archive:
            config = published_config(self.cp, archive.extractfile(BACKUP_MEMBER).read())
        run = trace_bed(self.cp, self.bundle, config)
        block = bytes.fromhex(run["parameter_hex"])
        self.assertEqual(block[0x32:0x34], bytes((250, 250)))
        self.assertEqual(run["queue_row"][3], 0x44)
        self.assertEqual(run["queue_row"][11:13], (0x40000000, 0x40008080))
        self.assertEqual(run["validator_mask"], 0)
        work, image = run["submissions"]
        self.assertEqual(work["rows"], [[0x20004040, 0x80003724, 0x970000, 0x970000]])
        self.assertEqual(image["rows"], [[0, 0x4c000000, 0, 0],
                                        [0x2002d880, 0x609000, 0xc15f00a0, 0x4057aaaa]])
        self.assertEqual(run["image_read"], PackedRead(107 * 1280 + 192 * 2, 1280, 352, 171))
        self.assertEqual(run["after_image"], {"pc": 0x8c0, "reason": "unsupported opcode", "opcode": 10})
        self.assertEqual(run["compute_arguments"], (0x81c, 0x9003, 0xc0, 0x15800))
        self.assertEqual(run["compute_return"], 0x13f0)
        self.assertEqual(run["compute_record"], {
            "bytes": 0xa4, "sha256": "e045cde60c1a42c7fcbb5a19be6d5338552e7848aa037d87bf0aa16d1b1ad9a4",
        })
        self.assertEqual(run["wrapper_inner_stop"],
                         {"pc": 0x8d6, "reason": "unsupported opcode", "opcode": 9})
        self.assertEqual(run["wrapper_record_pointer"], 0x81c)
        self.assertEqual(dict(run["compute_setup"])[0x40000], 11)
        touched = {i for _, offset, size in run["parameter_reads"] for i in range(offset, offset + size)}
        self.assertTrue({0x00, 0x04, 0x14, 0x1a, 0x20}.issubset(touched))
        self.assertTrue({0x32, 0x33}.isdisjoint(touched))

    def test_original_geometry_validation_precedes_work_or_image_dma(self):
        cases = (((280, 192), (48, 48), 0x600), ((280, 192), (64, 48), 0x600),
                 ((280, 192), (95, 95), 0x600), ((281, 193), (64, 64), 0x80))
        for center, sides, mask in cases:
            with self.subTest(center=center, sides=sides):
                run = trace_bed(self.cp, self.bundle, bytes(0x100), center=center, sides=sides)
                self.assertEqual(run["validator_mask"], mask)
                self.assertFalse(run["submissions"])

    def test_all_five_workers_consume_controller_mailboxes(self):
        partitions = ((0, 11), (11, 11), (22, 11), (33, 11), (44, 10), (54, 10))
        bases = (0x4000, 0x4800, 0x5000, 0x8000, 0x8800)
        for center, crop in (((280, 192), 0x9003), ((288, 202), 0x900b)):
            with self.subTest(center=center):
                run = trace_bed(self.cp, self.bundle, bytes(0x100), center=center, stages=True)
                self.assertEqual(run["controller_partition"], partitions[0])
                self.assertEqual(run["shared_image_entry"]["jobs"], {
                    job: {"record_offset": 0x4cc, "program_offset": offset, "slots": slots}
                    for job, offset, slots in ((0x44, 0x1df0, 37), (0x3e, 0x1d80, 37), (0x43, 0x2c30, 120))
                })
                self.assertEqual(len(run["workers"]), 5)
                for i, (worker, base, partition) in enumerate(zip(run["workers"], bases, partitions[1:])):
                    mailbox = 0x17800 + 0x4c * i
                    self.assertEqual(worker["partition"], partition)
                    self.assertEqual(worker["arguments"], [crop, 0xc0, 0x15800])
                    self.assertEqual(worker["base"], base)
                    self.assertEqual(worker["stop"],
                                     {"pc": base + 0xd8, "reason": "unsupported opcode", "opcode": 10})
                    self.assertEqual(set(worker["mailbox_reads"]), set(range(mailbox, mailbox + 0x4c, 4)))

    def test_completion_picks_directory_block_or_writes_exec_id(self):
        run = trace_bed(self.cp, self.bundle, bytes(0x100), stages=True, exec_id=0x12348657)
        handoff = run["handoff"]
        self.assertEqual(handoff["wait"], {"pc": 0x13fc, "reason": "wfi"})
        self.assertEqual(handoff["mmio_writes"], ((0x48210, 0x3e), (0x48208, 0x80000000)))
        status = handoff["status_paths"]
        self.assertEqual(status[0], status[2])
        self.assertEqual(status[0]["stop"],
                         {"pc": 0xe16, "reason": "unmodeled read", "address": 0x48100, "size": 4})
        self.assertEqual(status[0]["work_writes"],
                         [{"pc": 0xe10, "address": 0xe00040d4, "size": 2, "value": 0x1ff}])
        self.assertEqual(status[1]["stop"], {"pc": 0xe58, "reason": "slice boundary"})
        self.assertEqual(status[1]["work_writes"],
                         [{"pc": 0xe56, "address": 0xe0004090, "size": 4, "value": 0x12348657}])
        self.assertEqual(handoff["directory"], 0x4a900)
        self.assertEqual(handoff["member_offsets"],
                         (0x100, 0x2200, 0x7900, 0xb400, 0x12900, 0x1ae00, 0x21900, 0x31500, 0x3d800))
        self.assertEqual(handoff["branch_target"], 0x50012100)
        self.assertEqual(handoff["boot_boundary"],
                         {"pc": 0x5001218c, "reason": "unmodeled read", "address": 0x48004, "size": 4})
        boot = handoff["boot_fixture"]
        self.assertEqual(boot["waits"], [{"pc": pc, "reason": "wfi"}
                                        for pc in (0x50012244, 0x50012374)])
        self.assertEqual(boot["catalog_fixture_va"], 0x500166f0)
        self.assertEqual(boot["linear_transfer"], (0x4004900, 0x50800000, 0x1d00))
        self.assertEqual(boot["source_member_offset"], 0x2500)
        self.assertEqual(boot["external_stop"],
                         {"pc": 0x50800462, "reason": "unmodeled read", "address": 0x49000, "size": 1})
        fixture = boot["request_fixture"]
        self.assertEqual(fixture["prior_control_write"],
                         {"pc": 0xcc0, "address": 0x49000, "size": 1, "value": 0xf})
        self.assertEqual(fixture["parameter_reads"], ((0x5080049c, 0xc, 0x40004040),
                                                      (0x508004a8, 0x10, 0x400040c0),
                                                      (0x508004b0, 8, 0x40008080)))
        expected = (
            (0x5080070c, 0x50800714, 76, [0x20008080 + 0x75990, 0x4000, 0x4b0000, 0x4b0000]),
            (0x5080075a, 0x50800760, 200, [0x20008080 + 0x75ce4, 0x404c, 0xc70000, 0xc70000]),
            (0x50800536, 0x5080053c, 8192, [0x5800, 0xc0015800, 0x1fff0000, 0x1fff0000]),
        )
        for item, (submit, wait, size, row) in zip(fixture["requests"], expected, strict=True):
            self.assertEqual(item["request"]["submit_pc"], submit)
            self.assertEqual(item["wait"], {"pc": wait, "reason": "wfi"})
            self.assertEqual(item["bytes"], size)
            self.assertEqual(item["request"]["rows"], [row])
        self.assertEqual(fixture["stop"], {"pc": 0x50800e02, "reason": "unmodeled write",
                                           "address": 0x3808, "size": 4, "value": 0})
        self.assertEqual(fixture["scalar_stop"], {"pc": 0x50800e40, "reason": "unmodeled read",
                                                  "address": 0x481cc, "size": 4})
        self.assertEqual({w["address"]: w["value"] for w in fixture["scalar_writes"]},
                         {0x3800: 0, 0x3804: 0x400, 0x3808: 0, 0x380c: 0x50801b00,
                          0x3810: 0x400, 0x3814: 0, 0x3818: 0, 0x381c: 0, 0x3820: 0})
        catalog = fixture["program_fixture"]
        self.assertEqual(catalog["wait"], {"pc": 0x50800ea8, "reason": "wfi"})
        self.assertEqual(catalog["request"]["submit_pc"], 0x50800ea2)
        self.assertEqual(catalog["request"]["rows"], [[0x500173d8, 0x0400f000, 0x83470000, 0x41170200]])
        self.assertEqual((catalog["member_offset"], catalog["slots"]), (0x74d8, 35))
        self.assertEqual(catalog["sha256"], "cf0738ea2bd236880fd8b296670c0c520c47cb9e158bff53f4b877df085a8fdd")
        compute = catalog["compute_fixture"]
        self.assertEqual(compute["wait"], {"pc": 0x50800ee0, "reason": "wfi"})
        self.assertEqual(compute["stop"], {"pc": 0x508018d4, "reason": "unsupported opcode", "opcode": 10})
        self.assertEqual((compute["return"], compute["record"], compute["member_offset"]),
                         (0x50800efe, 0x508002c8, 0x27c8))
        self.assertEqual(compute["arguments"], [0, 8, 0x5000])
        self.assertEqual(compute["record_sha256"], "6d0a3f527a0c4f2ec44ee9afbc31ffa5013348f8958e1adfd6bbb61891accdfa")
        self.assertEqual(compute["slot_indices"],
                         (2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, *range(13, 35), 34, 34, 34))
        self.assertEqual([(w["address"], w["value"]) for w in compute["mailbox_writes"]],
                         [(0x17700, 1), (0x17720, 0x5000), (0x17724, 0x4000),
                          (0x17728, 0x5800), (0x1772c, 0x404c)])
        for worker, (base, wait, opcode, ret, offset, start) in zip(compute["workers"], (
                (0x800, 0x120c, 0x1420, 0x13e4, 0x44d4, 8),
                (0x1800, 0x2200, 0x2414, 0x23d8, 0x5184, 16),
                (0x2800, 0x321c, 0x3430, 0x33f4, 0x5e14, 24)), strict=True):
            self.assertEqual(worker["base"], base)
            self.assertEqual(worker["wait"], {"pc": wait, "reason": "wfi"})
            self.assertEqual(worker["stop"], {"pc": opcode, "reason": "unsupported opcode", "opcode": 10})
            self.assertEqual((worker["return"], worker["record"], worker["member_offset"]),
                             (ret, base + 0x2d4, offset))
            self.assertEqual(worker["arguments"], [start, 8, 0x5000])
            self.assertEqual([(r["address"], r["size"], r["value"]) for r in worker["mailbox_reads"]],
                             [(w["address"], w["size"], w["value"]) for w in compute["mailbox_writes"]])
        mapped = boot["mapped_weight_fixture"]
        self.assertEqual(mapped["table_entry"], (1, 0x20200, 0x90700, 0x40008080))
        self.assertEqual(mapped["parameter_reads"], fixture["parameter_reads"])
        self.assertEqual(mapped["stop"], fixture["stop"])
        self.assertEqual(mapped["scalar_stop"], fixture["scalar_stop"])
        self.assertEqual(mapped["scalar_writes"], fixture["scalar_writes"])
        self.assertEqual(mapped["program_fixture"], catalog)
        for item, (submit, wait, size, row), source in zip(
                mapped["requests"], expected, (0x40020200 + 0x75990, 0x40020200 + 0x75ce4, 0x5800),
                strict=True):
            self.assertEqual(item["request"]["submit_pc"], submit)
            self.assertEqual(item["wait"], {"pc": wait, "reason": "wfi"})
            self.assertEqual(item["bytes"], size)
            self.assertEqual(item["request"]["rows"], [[source, *row[1:]]])

    @unittest.skipUnless(BED_DICT1.is_file(), "requires the published BED dictionary")
    def test_block_one_post_compute_uses_original_dictionary_parameters(self):
        run = trace_bed(self.cp, self.bundle, bytes(0x100), stages=True, dictionary=BED_DICT1.read_bytes())
        result = run["handoff"]["boot_fixture"]["request_fixture"]["post_compute_fixture"]
        self.assertEqual(result["overlay"]["rows"], [[0x500167f0, 0x0400f000, 0x8be70000, 0x43f70200]])
        self.assertEqual((result["program_offset"], result["slots"]), (0x68f0, 127))
        self.assertEqual(result["dictionary_spans"], [(0x49438, 0x7830, 12), (0x49448, 0x7800, 48),
                                                     (0x49478, 0x8000, 48), (0x494a8, 0x8800, 24),
                                                     (0x494c0, 0x9000, 24)])
        self.assertEqual(result["unseeded_stop"],
                         {"pc": 0x50800894, "reason": "unmodeled read", "address": 0x7830, "size": 2})
        self.assertEqual(result["parameter_hex"], "a6fbcb712cfe76b115ff372d")
        self.assertEqual([(r["address"], r["value"]) for r in result["parameter_reads"]],
                         [(0x7830, -1114), (0x7832, 29131)])
        self.assertEqual([(w["address"], w["value"]) for w in result["mmio_writes"][:2]],
                         [(0x40140, 0xfffffba6), (0x40144, 29131)])
        self.assertEqual(result["stop"], {"pc": 0x50800374, "reason": "unsupported opcode", "opcode": 10})
        self.assertEqual((result["record"], result["return"], result["slot_indices"]),
                         (0x508002b0, 0x50800952, (125, 126)))
        following = result["following_fixture"]
        for call, (coefficient, target, value) in zip(following["calls"], (
                (0x7800, 0xac60, 0xfffffba6), (0x7818, 0xca68, 29131)), strict=True):
            self.assertEqual(call["arguments"], (0x50800038, 0x5800, coefficient, target))
            self.assertEqual(call["return"], 0x508009ea)
            self.assertEqual(call["aperture"], ((0x40320, value), (0x40324, 0x5800),
                                               (0x40328, coefficient), (0x4032c, 0x5800),
                                               (0x40330, target)))
        self.assertEqual(following["parameter_offset"], 0x4d7b0)
        self.assertEqual(following["unseeded_stop"], {"pc": 0x50800b42, "reason": "unmodeled read",
                                                     "address": 0x16240, "size": 2})
        self.assertEqual(following["stop"], result["stop"])
        self.assertEqual(following["parameter_writes"][-3:],
                         ((0x40140, 0xfffff286), (0x40144, 0x2f1), (0x40148, 0xffffd799)))
        geometry = following["geometry"]
        self.assertEqual(tuple(geometry[a] for a in (0x40190, 0x40194, 0x40198, 0x4019c,
                                                     0x401a0, 0x401c4, 0x401d0)),
                         (124, 62 * 62 * 2, 24, 2, 60, 30, 30))
        coeff = following["coefficient_fixture"]
        self.assertEqual([(span["offset"], span["destination"], len(span["blocks"])) for span in coeff["spans"]],
                         [(0x4d7c8, 0x16090, 18), (0x4d978, 0x16800, 18),
                          (0x4db28, 0x17000, 18), (0x4dcd8, 0x17800, 18)])
        self.assertEqual(coeff["spans"][0]["blocks"][0],
                         ((-235, -441, -341), (-209, -13, 129), (160, 280, 162)))
        for i, call in enumerate(coeff["calls"]):
            source, output = 0x16090 + 144 * i, 0x5800 + 30 * 30 * 2 * i
            value = (0xfffff286, 0x2f1, 0xffffd799)[i]
            self.assertEqual(call["arguments"], (0x50800184, 0xac60, source, output))
            self.assertEqual(call["aperture"], ((0x40320, value), (0x40324, 0xac60), (0x40328, source),
                                               (0x4032c, 0xac60), (0x40330, output)))
        self.assertEqual(len(coeff["calls"]), 3)
        self.assertEqual(coeff["stop"], {"pc": 0x5080141c, "reason": "wfi"})
        for i, (worker, stop_pc) in enumerate(zip(coeff["workers"], (0x1312, 0x2306, 0x3320), strict=True), 1):
            self.assertEqual(worker["base"], 0x800 + (i - 1) * 0x1000)
            self.assertEqual(worker["unseeded_stop"], {"pc": stop_pc, "reason": "unmodeled read",
                                                       "address": 0x40cb, "size": 1})
            zero, control, left = worker["paths"]
            self.assertEqual([p["controls"] for p in (zero, control, left)], [(0, 0), (0, 13), (-9, 0)])
            self.assertEqual([p["aperture_control"] for p in (zero, control, left)], [0, 13, 0])
            self.assertEqual(len(zero["calls"]), 3)
            for j, call in enumerate(zero["calls"]):
                fw = dict(call["aperture"])
                self.assertEqual((fw[0x40324], fw[0x4032c]), (0xac60, 0xac60))
                self.assertEqual(fw[0x40328], coeff["spans"][i]["destination"] + 144 * j)
                self.assertEqual(fw[0x40330], 0x5800 + 1800 * (3 * i + j))
            plain, changed = (dict(p["calls"][0]["aperture"]) for p in (zero, left))
            self.assertEqual(dict(control["calls"][0]["aperture"]), plain)
            self.assertEqual(changed, plain | {0x40320: plain[0x40320] << 9 & 0xffffffff})

    def test_padded_coefficients_reject_wrong_extent_or_padding(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            padded_3x3(bytes(23))
        with self.assertRaisesRegex(ValueError, "padding"):
            padded_3x3(bytes(7) + bytes((1,)) + bytes(16))

    def test_dictionary_ranges_require_unique_contiguous_bytes(self):
        self.assertEqual(dictionary_span(0x101, 4, ((0x100, b"abc"), (0x103, b"def"))),
                         (b"bcde", [(1, 1, 2), (2, 0, 2)]))
        with self.assertRaisesRegex(ValueError, "no unique source at 0x103"):
            dictionary_span(0x101, 4, ((0x100, b"abc"), (0x104, b"def")))
        with self.assertRaisesRegex(ValueError, "overlapping sources"):
            dictionary_span(0x101, 4, ((0x100, b"abc"), (0x102, b"def")))

    @unittest.skipUnless(BED_DICT1.is_file() and BED_DICT2.is_file(), "requires both published BED dictionaries")
    def test_later_coefficients_uniquely_constrain_second_dictionary_placement(self):
        first, second = BED_DICT1.read_bytes(), BED_DICT2.read_bytes()
        run = trace_bed(self.cp, self.bundle, bytes(0x100), stages=True, exec_id=0x12345678,
                        dictionary=first, dictionary2=second)
        layout = second_dictionary_layout(run, first, second)
        self.assertEqual(layout["base_offset"], 0x70500)
        self.assertEqual(layout["control_bytes"], (-9, 13))
        self.assertEqual([(s["offset"], s["destination"], s["bytes"]) for s in layout["early"]],
                         [(0x5490, 0x4000, 76), (0x57e4, 0x404c, 200)])
        self.assertEqual([(s["offset"], s["destination"], s["bytes"]) for s in layout["later"]],
                         [(0x46f8, 0xafc0, 24), (0x4710, 0xac60, 864), (0x4a70, 0xb000, 864),
                          (0x4dd0, 0xb800, 864), (0x5130, 0xc000, 864)])
        fixture = run["handoff"]["boot_fixture"]["request_fixture"]
        later = fixture["post_compute_fixture"][
            "following_fixture"]["coefficient_fixture"]["next_spans"]
        # Forcing a >=256-byte gap leaves no compatible candidate.
        self.assertEqual(locate_second_dictionary(second, len(first) + 0x100, later[1:], later[:1]), ())
        candidate = fixture["second_dictionary_candidate"]
        self.assertEqual(candidate["layout"], layout)
        self.assertEqual([(r["pc"], r["address"], r["value"]) for r in candidate["config_reads"]],
                         [(0x5080110c, 0x4097, 14), (0x50801114, 0x4090, -11),
                          (0x508013e6, 0x40cb, 13), (0x508013ee, 0x40c4, -9)])
        post = candidate["post_compute"]
        writes = {w["address"]: w["value"] for w in post["mmio_writes"]}
        self.assertEqual((writes[0x40140], writes[0x40144], writes[0x40218]),
                         ((-1114 << 11) & 0xffffffff, 29131 << 11, 14))
        subsequent = post["following_fixture"]
        self.assertEqual(subsequent["aperture_control"], 13)
        self.assertEqual(subsequent["parameter_writes"][-3:],
                         tuple((0x40140 + 4 * i, (v << 9) & 0xffffffff)
                               for i, v in enumerate((-3450, 753, -10343))))
        self.assertEqual(subsequent["geometry"], fixture["post_compute_fixture"]["following_fixture"]["geometry"])
        for worker in subsequent["coefficient_fixture"]["workers"]:
            real = worker["bound_controls"]
            self.assertEqual((real["controls"], real["aperture_control"]), ((-9, 13), 13))
            for plain, bound in zip(worker["paths"][0]["calls"], real["calls"], strict=True):
                before = dict(plain["aperture"])
                self.assertEqual(dict(bound["aperture"]),
                                 before | {0x40320: (before[0x40320] << 9) & 0xffffffff})
        late = subsequent["coefficient_fixture"]["late_parameter_fixture"]
        self.assertEqual(late["unseeded_stop"], {"pc": 0x50800894, "reason": "unmodeled read",
                                               "address": 0xafc0, "size": 2})
        setup = late["setup"]
        self.assertEqual([(r["address"], r["value"]) for r in setup["reads"]],
                         [(0x40ff, 11), (0x40f8, -8), (0xafc0, 758), (0xafc2, -706), (0xafc4, -1244)])
        biases = tuple((v << 8) & 0xffffffff for v in (758, -706, -1244))
        self.assertEqual(setup["parameters"],
                         (*((0x40140 + 4 * i, value) for i, value in enumerate(biases)), (0x40218, 11)))
        for i, (call, bias) in enumerate(zip(late["calls"], biases, strict=True)):
            coeff, target = 0xac60 + 0x120 * i, 0xe400 + 2 * i
            self.assertEqual(call["arguments"], (0x50800038, 0x5800, coeff, target))
            self.assertEqual(call["aperture"], ((0x40320, bias), (0x40324, 0x5800), (0x40328, coeff),
                                               (0x4032c, 0x5800), (0x40330, target)))
        self.assertEqual(late["stop"], {"pc": 0x50800568, "reason": "unmodeled read",
                                       "address": 0x48100, "size": 4})
        handoff = late["handoff"]
        self.assertEqual(handoff["flag_read"], {"pc": 0x5080056a, "reason": "unmodeled read",
                                               "address": 0x48170, "size": 1})
        self.assertEqual(handoff["scan"], {"pc": 0x50800580, "reason": "unmodeled read",
                                          "address": 0x20000000, "size": 4})
        self.assertEqual(handoff["branch"], {"pc": 0x50017800, "reason": "slice boundary"})
        self.assertEqual(handoff["member_offset"], 0x7900)
        boot = handoff["boot"]
        self.assertEqual(boot["waits"], [{"pc": pc, "reason": "wfi"} for pc in (0x50017944, 0x50017a74)])
        self.assertEqual(boot["source_member_offset"], 0x7d00)
        self.assertEqual(boot["linear_transfer"], (0x04005f00, 0x51000000, 0x1200))
        self.assertEqual(boot["external_stop"], {"pc": 0x5100050c, "reason": "wfi"})
        self.assertEqual(handoff["second_wait"], {"pc": 0x51000558, "reason": "wfi"})
        for request, prior, (target, size) in zip(handoff["submissions"], fixture["requests"][:2],
                                                   ((0x4000, 40), (0x4028, 124)), strict=True):
            (source, dest, a, b), = request["rows"]
            (earlier,) = prior["request"]["rows"]
            self.assertEqual(source, earlier[0] + prior["bytes"])
            self.assertEqual(dest, target)
            self.assertEqual(a, b)
            self.assertEqual(a, (size - 1) << 16)
        self.assertEqual(handoff["stop"], {"pc": 0x51000918, "reason": "unmodeled read",
                                          "address": 0x48230, "size": 4})
        third = handoff["compute_fixture"]
        self.assertEqual(third["initial_offsets"], (0x54dc, 0x58ac))
        group_a, group_b = third["groups"]
        for group, (parameter, values, coeff, dest, shift, control) in zip((group_a, group_b), (
                (0x4b9f0, (11721, 9109), 0x8560, 0x5800, 13, 15),
                (0x51160, (27884, -29390, -950), 0x5000, 0xe418, 9, 13)), strict=True):
            self.assertEqual(group["parameter_offset"], parameter)
            self.assertEqual([r["value"] for r in group["reads"]], [control, -shift, *values])
            self.assertEqual(group["aperture"],
                             (*((0x40140 + 4 * i, (v << shift) & 0xffffffff) for i, v in enumerate(values)),
                              (0x40218, control), (0x40328, coeff), (0x40330, dest)))
            self.assertEqual([c["arguments"][0] for c in group["calls"]], [0x510001e8, 0x51000038])
        self.assertEqual(third["stop"], {"pc": 0x51000cf6, "reason": "unmodeled read",
                                        "address": 0x481cc, "size": 4})
        completion = third["completion"]
        paths = completion["paths"]
        self.assertEqual(paths[0], paths[2])
        self.assertEqual(paths[0]["branch"], {"pc": 0x5001b300, "reason": "slice boundary"})
        self.assertEqual(paths[0]["work_writes"],
                         [{"pc": 0x5100039e, "address": 0xe00040d4, "size": 2, "value": 0x3ff}])
        self.assertEqual(paths[1]["work_writes"], [*paths[0]["work_writes"],
                         {"pc": 0x510003ce, "address": 0xe0004090, "size": 4, "value": 0x12345678}])
        self.assertEqual(paths[1]["stop"],
                         {"pc": 0x5100023e, "reason": "unmodeled read", "address": 0x48174, "size": 4})
        self.assertEqual(completion["boot"]["source_member_offset"], 0xb800)
        self.assertEqual(completion["boot"]["linear_transfer"], (0x04006dc0, 0x51800000, 0x2000))
        self.assertEqual([r["rows"] for r in completion["submissions"]],
                         [[[0x2007da84, 0x17e00, 0x330000, 0x330000]],
                          [[0x2007dea8, 0x17e34, 0x8b0000, 0x8b0000]]])
        self.assertEqual(completion["stop"],
                         {"pc": 0x51800ffc, "reason": "unmodeled read", "address": 0x48168, "size": 4})
        fourth = completion["compute_fixture"]
        self.assertEqual(fourth["initial_offsets"], (0x5504, 0x5928))
        self.assertEqual([(x["offset"], x["slots"]) for x in fourth["programs"]],
                         [(0x12420, 47), (0x11910, 118)])
        self.assertEqual(fourth["parameter_offset"], 0x46ea8)
        self.assertEqual(fourth["unseeded_stop"],
                         {"pc": 0x51800b34, "reason": "unmodeled read", "address": 0xdda0, "size": 2})
        self.assertEqual([(r["address"], r["value"]) for r in fourth["reads"][:4]],
                         [(0x17e7e, 11), (0x17e77, -9), (0xdda0, 11474), (0xdda2, -1489)])
        self.assertEqual(fourth["aperture"],
                         ((0x40218, 11), (0x40140, 11474 << 9), (0x40144, (-1489 << 9) & 0xffffffff)))
        self.assertEqual([call["arguments"] for call in fourth["calls"][-2:]],
                         [(0x51800038, 0xb800, 0x4800, 0xa022), (0x51800038, 0xb800, 0x4830, 0xa222)])
        requests = fourth["following_requests"]
        self.assertEqual(len(requests), 24)
        for i in range(12):
            base = 0xa000 + i * 0x200
            self.assertEqual(requests[2 * i]["rows"], [[base, 0xc000a000, 0x1f0000, 0x1f0000],
                                                       [base + 0x1e0, 0xc000a000, 0x1f0000, 0x1f0000]])
            self.assertEqual(requests[2 * i + 1]["rows"], [[base + 0x20, 0xc020a000, 0x10004, 0x1b000d],
                                                           [base + 0x3e, 0xc020a000, 0x10004, 0x1b000d]])
        self.assertEqual(fourth["stop"],
                         {"pc": 0x51801448, "reason": "unmodeled read", "address": 0x481cc, "size": 4})
        final = fourth["completion"]
        self.assertEqual(final["state_one_work_writes"],
                         [{"pc": 0x5180057c, "address": 0xe00040d4, "size": 2, "value": 0x4ff},
                          {"pc": 0x518005b0, "address": 0xe0004090, "size": 4, "value": 0x12345678}])
        self.assertEqual(final["state_one_stop"],
                         {"pc": 0x5180040e, "reason": "unmodeled read", "address": 0x48174, "size": 4})
        self.assertEqual(final["parameter_offset"], 0x49300)
        self.assertEqual(final["unseeded_stop"],
                         {"pc": 0x51800d00, "reason": "unmodeled read", "address": 0xca90, "size": 2})
        self.assertEqual([(r["address"], r["value"]) for r in final["reads"][:4]],
                         [(0x17eab, 14), (0x17ea4, -12), (0xca90, -144), (0xca92, -258)])
        self.assertEqual(final["calls"][0]["aperture"], ((0x40140, (-144 << 12) & 0xffffffff),
                                                        (0x40144, (-258 << 12) & 0xffffffff), (0x40218, 14)))
        self.assertEqual([call["arguments"] for call in final["calls"][1:]],
                         [(0x5180017c, 0xa000, 0x4800, 0xb800), (0x5180017c, 0xa200, 0x4818, 0xb802)])
        self.assertEqual((final["branch"]["pc"], final["next_member_offset"]), (0x50022800, 0x12900))
        fifth = final["fifth_fixture"]
        self.assertEqual(fifth["initial_offsets"], (0x5538, 0x59b4))
        self.assertEqual((fifth["program_offset"], fifth["slots"]), (0x19140, 118))
        self.assertEqual(fifth["parameter_offset"], 0x46c50)
        self.assertEqual(fifth["unseeded_stop"],
                         {"pc": 0x52000c82, "reason": "unmodeled read", "address": 0x92b8, "size": 2})
        self.assertEqual([(r["address"], r["value"]) for r in fifth["reads"][:5]],
                         [(0x17e94, 12), (0x17e8d, -9), (0x92b8, -16978), (0x92ba, -18979), (0x92bc, -15916)])
        self.assertEqual(fifth["control_writes"], ((0x40218, 12), *(
            (0x40140 + i * 4, v << 9 & 0xffffffff) for i, v in enumerate((-16978, -18979, -15916)))))
        self.assertEqual([c["arguments"] for c in fifth["calls"]],
                         [(0x52000038, 0xe400, 0x9210 + 0x30 * i, 0x6cc0 + 0x310 * i) for i in range(3)])
        rows = fifth["mixed_request"]["rows"]
        self.assertEqual(rows[:12], [[0x20004102 + 0x780 * i, 0x80386cc0 + 0x310 * i, 0x370008, 0x370d0d]
                                     for i in range(12)])
        self.assertEqual(rows[12:], [[0x2004ece8, 0x492b8, 0x170000, 0x170000], *(
            [0x2004ef28 + 0x90 * i, dst, 0x8f0000, 0x8f0000]
            for i, dst in enumerate((0x49210, 0x49890, 0x4a090, 0x4a890)))])
        windows = fifth["feature_windows"]
        fixture_bytes = b"".join(bytes((channel, y)) * 28 + b"\xff" * 8
                                 for channel in range(12) for y in range(30))
        for i, window in enumerate(windows):
            self.assertEqual(window, {"source": rows[i][0], "destination_raw": rows[i][1],
                                       "read": PackedRead(i * 0x780, 64, 56, 14)})
            self.assertEqual(window["read"].gather(fixture_bytes),
                             b"".join(bytes((i, y)) * 28 for y in range(14)))
        self.assertEqual(len(windows), 12)
        invalid = [row[:] for row in rows]
        invalid[6][2] ^= 1
        with self.assertRaisesRegex(ValueError, "feature DMA"):
            decode_fifth_feature_rows(dict(fifth["mixed_request"], rows=invalid))
        second_pass = fifth["second_pass"]
        self.assertEqual(second_pass["parameter_offset"], fifth["parameter_offset"])
        self.assertEqual(second_pass["calls"],
                         [(0x52000038, 0x12d80, 0x9180 + 0x30 * i, 0x4800 + 0x310 * i) for i in range(3)])
        self.assertEqual([(r["submit_pc"], len(r["rows"])) for r in second_pass["requests"]],
                         [(0x520015ae, 17), (0x5200176a, 7)])
        self.assertEqual(second_pass["stop"],
                         {"pc": 0x52000df8, "reason": "unmodeled read", "address": 0x15e40, "size": 2})
        later = second_pass["next_compute"]
        self.assertEqual((later["parameter_offset"], later["coefficient_offset"], later["coefficient_blocks"]),
                         (0x47100, 0x47140, 48))
        values = (-8811, 495, -4011, 3503)
        self.assertEqual([r["value"] for r in later["reads"][:4]], list(values))
        shifted = [(v << 9) & 0xffffffff for v in values]
        self.assertEqual(later["setup"], (*((0x40140 + 4 * i, v) for i, v in enumerate(shifted)), (0x40218, 12)))
        for i, (call, value) in enumerate(zip(later["calls"], shifted, strict=True)):
            coef, dest = 0x159c0 + 0x120 * i, 0x4800 + 2 * i
            self.assertEqual(call["arguments"], (0x5200017c, 0xffc0, coef, dest))
            self.assertEqual(call["aperture"], ((0x40320, value), (0x40328, coef), (0x40330, dest)))
        self.assertEqual(later["reload"]["rows"][0], [0x2004f19e, 0x15e40, 0x1d0000, 0x1d0000])
        self.assertEqual(later["stop"], {"pc": 0x52001770, "reason": "wfi"})
        reloaded = later["reloaded_compute"]
        self.assertEqual(reloaded["parameter_offset"], later["parameter_offset"] + 30)
        self.assertEqual([(r["address"], r["value"]) for r in reloaded["reads"][:6]],
                         [(0x17ec1, 12), (0x17eba, -9), (0x15e40, -5595), (0x15e42, 1262),
                          (0x15e44, 3164), (0x15e46, 9239)])
        for i, (call, value) in enumerate(zip(reloaded["calls"], (-5595, 1262, 3164, 9239), strict=True)):
            coef, dest = 0x159c0 + 0x120 * i, 0x481e + 2 * i
            self.assertEqual(call["arguments"], (0x5200017c, 0xffc0, coef, dest))
            self.assertEqual(call["aperture"], ((0x40320, (value << 9) & 0xffffffff),
                                               (0x40328, coef), (0x40330, dest)))
        self.assertEqual([(p["member_offset"], p["slots"]) for p in reloaded["overlays"]],
                         [(0x19c50, 90)] + [(0x1a9e8, 40)] * 3)
        self.assertEqual(reloaded["arguments"], (0x520002e0, 0, 8, 0x10018))
        self.assertEqual(reloaded["unseeded_stop"],
                         {"pc": 0x52001b2c, "reason": "unmodeled read", "address": 0x10018, "size": 2})
        copies = reloaded["copy_fixtures"]
        self.assertEqual([c["seed"] for c in copies], [0x7ff0, 0xfff0])
        for c in copies:
            values = [(c["seed"] + i) & 0xffff for i in range(30)]
            signed = [v - 0x10000 if v >= 0x8000 else v for v in values]
            self.assertEqual([(r["pc"], r["address"], r["size"], r["value"] & 0xffff) for r in c["reads"]], [
                (0x52001b2c, 0x10018 + 8 * i, 2, v)
                for i, v in enumerate(values)])
            self.assertEqual([(w["pc"], w["address"], w["size"], w["value"] & 0xffff) for w in c["writes"]],
                             [(0x52001b30, 0xffd8 + 2 * i, 2, v) for i, v in enumerate(values)])
            self.assertEqual(c["output"], tuple(signed))
            self.assertEqual(c["boundary"], {"pc": 0x52001b3a, "reason": "slice boundary"})
            consumers = c["consumers"]
            first_call, second_call = consumers["calls"]
            self.assertEqual(first_call["request"]["rows"], [[0x2004bc40, 0x10018, 0x027f0000, 0x027f0000]])
            self.assertEqual(second_call["request"]["rows"], [[0x2004bec0, 0x10050, 0x02cf0000, 0x02cf0000]])
            self.assertEqual((first_call["dictionary_offset"], second_call["dictionary_offset"]), (0x43bc0, 0x43e40))
            self.assertEqual(first_call["arguments"], (0x520003a4, 0x10018, 0xffc0, 0xffd8))
            self.assertEqual(second_call["arguments"], (0x52000488, 0xffd8, 0x10050, 0xffc0))
            for call in consumers["calls"]:
                self.assertEqual(call["record_pointer"], call["arguments"][0])
                self.assertEqual(call["dispatch"], {"pc": 0x520023e2, "reason": "unsupported opcode", "opcode": 9})
            self.assertEqual(consumers["stop"],
                             {"pc": 0x52001dc0, "reason": "unsupported opcode", "opcode": 0x3916})
            value_read, exponent_read = consumers["scalar_input_reads"]
            self.assertEqual(value_read, {"pc": 0x52001dbc, "address": 0xffd8, "size": 4,
                                          "value": values[0] | values[1] << 16})
            self.assertEqual((exponent_read["pc"], exponent_read["address"], exponent_read["size"]),
                             (0x52001dbe, 0x17f45, 1))
            sigmoid = consumers["sigmoid_fixture"]
            self.assertEqual(sigmoid["exponents"], (25, 25, -16))
            self.assertEqual(sigmoid["ceiling"], 65535)
            inputs = (-64, -20, -5, -1, 0, 1, 5, 20, 63) * 3 + (-2, 0, 2)
            outputs = (0, 0, 438, 17625, 32768, 47910, 65097, 65535, 65535) * 3 + (7812, 32768, 57723)
            self.assertEqual(sigmoid["input_q25"], tuple(x << 25 for x in inputs))
            self.assertEqual(sigmoid["writes"], [
                {"pc": 0x52001e4e, "address": 0x10050 + 8 * i, "size": 2, "value": v}
                for i, v in enumerate(outputs)])
            self.assertEqual(sigmoid["next_stop"],
                             {"pc": 0x52001e5c, "reason": "unmodeled read", "address": 0x481cc, "size": 4})
            states = sigmoid["state_fixtures"]
            self.assertEqual(states["state_one_stop"],
                             {"pc": 0x5200193a, "reason": "unmodeled read", "address": 0x48230, "size": 4})
            self.assertEqual(states["state_zero_request"]["rows"],
                             [[0x5002a3c0, 0x0400f000, 0x85270000, 0x41b70200]])
            self.assertEqual((states["member_offset"], states["slots"]), (0x1a4c0, 55))
            self.assertEqual(states["next_compute"],
                             {"pc": 0x520023cc, "reason": "unsupported opcode", "opcode": 10})
            self.assertEqual(states["arguments"], (0x520004f0, 0, 8, 0x4800))
            projection = states["following_projection"]
            self.assertEqual(projection["overlay_request"]["rows"],
                             [[0x50029040, 0x0400f000, 0x8b0f0000, 0x43af0200]])
            self.assertEqual((projection["member_offset"], projection["slots"]), (0x19140, 118))
            self.assertEqual(projection["dictionary_offsets"], [0x494d8, 0x494f0, 0x495b0, 0x49670, 0x49730])
            rows = projection["parameter_request"]["rows"]
            self.assertEqual([row[1] for row in rows], [0x10080, 0xffc0, 0x10800, 0x11000, 0x11800])
            self.assertEqual([(r["address"], r["size"], r["value"]) for r in projection["coefficient_reads"]],
                             [(0x10080, 2, 2807), (0x10082, 2, -2659), (0x10084, 2, -99)])
            self.assertEqual(projection["calls"], [
                (0x52000038, 0x4800, 0xffc0, 0x1263e), (0x52000038, 0x4800, 0x10000, 0x12d46),
                (0x52000038, 0x4800, 0x10040, 0x1344e)])
            self.assertEqual(projection["stop"],
                             {"pc": 0x520007d2, "reason": "unmodeled read", "address": 0x48100, "size": 4})
            self.assertEqual(projection["next_branch"], {"pc": 0x5002ad00, "reason": "slice boundary"})
            self.assertEqual(projection["next_member_offset"], 0x1ae00)
            self.assertEqual(projection["startup_request"]["rows"], [[0x5002afa0, 0x18, 0x9f0000, 0x9f0000]])
            self.assertEqual(projection["startup_member_offset"], 0x1b0a0)
            self.assertEqual(projection["startup_stop"], {"pc": 0x5002af84, "reason": "wfi"})
            sixth = projection["sixth_fixture"]
            self.assertEqual(sixth["entry"], {"pc": 0x52800000, "reason": "slice boundary"})
            self.assertEqual(sixth["external_transfer"], (0x0400ac40, 0x52800000, 0x2100))
            self.assertEqual(sixth["source_member_offset"], 0x1b200)
            self.assertEqual([(r["offset"], r["bytes"], r["request"]["rows"][0][1])
                              for r in sixth["parameter_loads"]],
                             [(0x55a0, 76, 0x17e00), (0x5b04, 260, 0x17e4c)])
            self.assertEqual(sixth["catalog_pointer"], 0x500304e0)
            self.assertEqual(sixth["stop"],
                             {"pc": 0x528007ae, "reason": "unmodeled read", "address": 0x48168, "size": 4})
            head = sixth["projection_fixture"]
            self.assertEqual(head["overlay_request"]["rows"],
                             [[0x50030660, 0x0400f000, 0x87370000, 0x42670200]])
            self.assertEqual((head["member_offset"], head["slots"]), (0x20760, 77))
            self.assertEqual(head["dictionary_offsets"],
                             [[0x497f0, *(0x49830 + 288 * i for i in range(6))],
                              [0x497fc, *(0x49ef0 + 288 * i for i in range(6))]])
            self.assertEqual(head["reads"], [{"pc": 0x528008fc, "address": 0xebd0, "size": 2, "value": -16733}])
            self.assertEqual(head["compute"], {"pc": 0x528003b4, "reason": "unsupported opcode", "opcode": 10})
            self.assertEqual(head["arguments"], (0x52800184, 0x18, 0x3c, 0x7d94))
            tiles = head["tiles"]
            self.assertEqual(len(tiles["iterations"]), 5)
            for i, (tile, parameter) in enumerate(zip(tiles["iterations"],
                                                       (-16733, -14812, 13144, -17539, 4826), strict=True)):
                src, dst = (0x4000, 0xa000) if i % 2 else (0x4120, 0xc4c0)
                self.assertEqual(tile["arguments"], (0x52800038, 0x12600, src, dst))
                self.assertEqual(tile["aperture"], ((0x40320, (parameter << 10) & 0xffffffff),
                                                   (0x40324, 0x12600), (0x40328, src),
                                                   (0x4032c, 0x12600), (0x40330, dst)))
                self.assertEqual(tile["request"]["rows"][0],
                                 [0x2000e440 + i * 0x24c0, 0x80000000 | dst, 0x24bf0000, 0x24bf0000])
                self.assertEqual(tile["dictionary_offsets"],
                                 [0x49808 + i * 12, *(0x4a5b0 + (i * 6 + j) * 288 for j in range(6))])
                self.assertTrue(all(bool(row[1] & 0x40000) == (i >= 3) for row in tile["request"]["rows"][1:]))
            self.assertEqual((tiles["member_offset"], tiles["slots"]), (0x20e98, 41))
            self.assertEqual(tiles["merged_request"]["rows"],
                             [[0x2000e440, 0xa000, 0x3fff0000, 0x3fff0000],
                              [0x20012440, 0xe000, 0x3fff0000, 0x3fff0000],
                              [0x20016440, 0x12000, 0x37bf0000, 0x37bf0000]])
            self.assertEqual(tiles["final_compute"], {"pc": 0x52801bb0, "reason": "unsupported opcode", "opcode": 10})
            self.assertEqual(tiles["arguments"], (0x5280019c, 0, 5, 0x157c0))
            self.assertEqual(tiles["stop"],
                             {"pc": 0x5280117c, "reason": "unmodeled read", "address": 0x157c0, "size": 2})
            self.assertEqual(tiles["copy_registers"], (0x157c0, 0xa018, 0xa054))
            self.assertEqual(len(tiles["scalar_fixtures"]), 2)
            for path, expected in zip(tiles["scalar_fixtures"], ((0, 0), (2303, 467)), strict=True):
                values = path["input"]
                self.assertEqual(path["input_reads"],
                                 [{"pc": 0x5280117c, "address": 0x157c0 + 8 * i, "size": 2, "value": v}
                                  for i, v in enumerate(values)])
                self.assertEqual([(r["pc"], r["address"], r["size"], r["value"] & 0xffff)
                                  for r in path["copy_writes"]],
                                 [(0x52801180, 0xa018 + 2 * i, 2, v & 0xffff) for i, v in enumerate(values)])
                self.assertEqual(path["request"]["rows"], [[0x2004c190, 0xa058, 0x027f0000, 0x027f0000]])
                self.assertEqual(path["dictionary_offset"], 0x44110)
                self.assertEqual(path["coefficient_addresses"],
                                 [0xa058 + row * 64 + i * 2 for row in range(2) for i in range(30)])
                products = (sum(x * w for x, w in zip(values, unpack_from("<30h", first, 0x44110 + i * 64)))
                            for i in range(10))
                hidden = tuple(min(32767, (max(0, dot) + 128) >> 8) for dot in products)
                self.assertEqual(hidden[:2], expected)
                self.assertEqual(path["projection_writes"],
                                 [{"pc": 0x528012d6, "address": 0xa000 + 2 * i, "size": 2, "value": v}
                                  for i, v in enumerate(expected)])
                combined = path["combined"]
                self.assertEqual(combined["hidden"], hidden)
                self.assertEqual(combined["second_request"]["rows"], [[0x2004c410, 0xa018, 0x02cf0000, 0x02cf0000]])
                self.assertEqual(combined["second_dictionary_offset"], 0x44390)
                logits = tuple(sum(x * w for x, w in zip(hidden, unpack_from("<10h", first, 0x44390 + i * 24)))
                               for i in range(30))
                self.assertEqual(combined["logits"], logits)
                self.assertEqual([len(job["writes"]) for job in combined["first_workers"]], [2, 2, 2, 1, 1])
                self.assertEqual([len(job["writes"]) for job in combined["second_workers"]], [5] * 5)
                scores = ([36729, 33784, 32497, 36081, 33458, 32411, 33889, 35853, 37802, 34140,
                           34054, 36509, 33994, 37684, 33624, 36577, 31980, 26272, 34143, 33789,
                           35904, 36954, 32987, 32224, 32076, 30995, 36934, 33986, 35481, 32098]
                          if expected == (0, 0) else
                          [27824, 34540, 32766, 31682, 31924, 39832, 31518, 30522, 31153, 31454,
                           32590, 32216, 28864, 31174, 35685, 29060, 34553, 38215, 35470, 33219,
                           29641, 29840, 37201, 32231, 31518, 32506, 30850, 31515, 32784, 35374])
                self.assertEqual(combined["sigmoid_writes"],
                                 [{"pc": 0x528015ac, "address": 0xa000 + 8 * i, "size": 2, "value": v}
                                  for i, v in enumerate(scores)])
                consumer = combined["q16_consumer_fixture"]
                self.assertEqual((consumer["member_offset"], consumer["slots"]), (0x21270, 66))
                launches = consumer["launches"]
                self.assertEqual([job["arguments"] for job in launches],
                                 [(0x52800260, i, 0, 5) for i in range(7)])
                self.assertEqual([job["mailbox"] for job in launches], [
                    (i, 0xa0f0 if i % 2 else 0xbb30, 0xd570 if i % 2 else 0xdc70, 0xa000, 0x17e00, 0x17e4c)
                    for i in range(7)])
                for i, worker in enumerate(launches[0]["workers"], 1):
                    self.assertEqual(worker["arguments"][1:], (0, 5 * i, 5))
                    self.assertEqual(worker["stack_arguments"], (0xbb30, 0xdc70, 0xa000, 0x17e00, 0x17e4c))
                    self.assertEqual([(r["address"], r["value"]) for r in worker["mailbox_reads"]],
                                     list(zip(range(0x17ad8, 0x17af0, 4), launches[0]["mailbox"])))
                geometry = consumer["feature_geometry"]
                self.assertEqual([item["source"] for item in geometry],
                                 [*(0x2000e440 + 224 * i for i in range(7)), 0x2001a060, 0x2001a140])
                feature = b"".join(bytes((row, stripe)) * 112 for row in range(30) for stripe in range(7))
                windows = [item["window"] for item in geometry[:7]]
                for i, window in enumerate(windows):
                    self.assertEqual(window, PackedRead(i * 224, 1568, 224, 30))
                stripes = [window.gather(feature) for window in windows]
                self.assertEqual(b"".join(data[y:y + 224] for y in range(0, 30 * 224, 224) for data in stripes),
                                 feature)
                self.assertEqual([item["window"] for item in geometry[7:]], [None, None])
                self.assertEqual(consumer["stop"],
                                 {"pc": 0x52800564, "reason": "unmodeled read", "address": 0x48100, "size": 4})
                seventh = consumer["seventh_fixture"]
                self.assertEqual(seventh["entry"], {"pc": 0x50031800, "reason": "slice boundary"})
                self.assertEqual(seventh["member_offset"], 0x21900)
                self.assertEqual(seventh["descriptor_request"]["rows"],
                                 [[0x50031aa0, 0x17818, 0x9f0000, 0x9f0000]])
                self.assertEqual((seventh["descriptor_offset"], len(seventh["local_requests"]["rows"])), (0x21ba0, 9))
                self.assertEqual(seventh["external_transfer"], (0x0400c700, 0x53000000, 0x4a00))
                self.assertEqual(seventh["external_member_offset"], 0x21d00)
                self.assertEqual([(r["offset"], r["bytes"], r["request"]["rows"][0][1])
                                  for r in seventh["parameter_loads"]],
                                 [(0x55ec, 232, 0x17b00), (0x5c08, 720, 0x17be8)])
                self.assertEqual((seventh["program_offset"], seventh["slots"]), (0x2fd58, 48))
                previous = [row for batch in consumer["feature_batches"][2:] for row in batch["rows"][:2]]
                self.assertEqual([(row[0], row[2], row[3]) for row in previous],
                                 [(0x2000a0c0 + 88 * 14 * i, 0x3b000b, 0x3b0d0d) for i in range(14)])
                self.assertEqual(seventh["feature_request"]["rows"],
                                 [[0x2000a0c0, 0xfa68, 0x3fff0000, 0x3fff0000],
                                  [0x2000e0c0, 0x13a68, 0x035f0000, 0x035f0000]])
                self.assertEqual(sum((row[2] >> 16) + 1 for row in seventh["feature_request"]["rows"]), 14 * 88 * 14)
                self.assertEqual(seventh["compute"], {"pc": 0x53004044, "reason": "unsupported opcode", "opcode": 10})
                self.assertEqual(seventh["arguments"], (0x5300087c, 0, 7, 0xc0a8))
                projection7 = seventh["projection_fixture"]
                self.assertEqual((projection7["member_offset"], projection7["slots"]), (0x2d700, 118))
                self.assertEqual(projection7["dictionary_offsets"], [0x4caa8, *(0x4cad0 + 288 * i for i in range(6))])
                self.assertEqual(projection7["scalar_parameters"],
                                 [{"pc": 0x5300178c, "address": 0xfb88 + 2 * i, "size": 2, "value": v}
                                  for i, v in enumerate((-5549, 3664, -3084))])
                self.assertEqual(projection7["calls"], [(0x530002c8, 0xfb8e, 0x53000c38, 0x53004324), *(
                    (0x53000184, 0xc0a8, 0x5800 + i * 0x60, 0xb01a + i * 0xd8) for i in range(3))])
                self.assertEqual(len(projection7["copy_requests"]), 18)
                for i, (linear, packed) in enumerate(projection7["copy_requests"]):
                    src = 0xb000 + i * 0xd8
                    self.assertEqual(linear["rows"], [[src, 0xc000b000, 0x110000, 0x110000],
                                                      [src + 0xc0, 0xc000b000, 0x110000, 0x110000]])
                    self.assertEqual(packed["rows"], [[src + at, 0xc018b000, 0x10003, 0xd0006]
                                                      for at in (0x18, 0x28)])
                mixed = projection7["mixed_request"]["rows"]
                self.assertEqual(projection7["mixed_dictionary_offsets"],
                                 [0x4ba90, *(0x4bae8 + i * 672 for i in range(6))])
                self.assertEqual(mixed[7:], [
                    [row[0], 0x60c0a8 + i * 0x540, 0x5f000b, 0x5f0d0d] for i, row in enumerate(previous)])
                late7 = projection7["late_fixture"]
                self.assertEqual(late7["mixed_calls"], [(0x530002c8, 0x160d6, 0x53000c38, 0x53004324), *(
                    (0x53000184, 0xc0a8, 0x5800 + i * 0x60, 0x10a4a + i * 0x200) for i in range(7))])
                self.assertEqual([r["value"] for r in late7["mixed_parameters"]],
                                 [1786, -16654, -13103, -12823, -646, 96, 22934])
                self.assertEqual(len(late7["copy_requests"]), 42)
                for i, (linear, packed) in enumerate(late7["copy_requests"]):
                    src = 0x10a28 + i * 0x200
                    self.assertEqual(linear["rows"], [[src, 0xc0010a28, 0x1f0000, 0x1f0000],
                                                      [src + 0x1e0, 0xc0010a28, 0x1f0000, 0x1f0000]])
                    self.assertEqual(packed["rows"], [[src + at, 0xc0210a28, 0x10004, 0x1b000d]
                                                      for at in (0x20, 0x3e)])
                self.assertEqual(late7["dictionary_offsets"], [0x4d190, *(0x4d1e8 + i * 168 for i in range(6))])
                self.assertEqual([r["value"] for r in late7["late_parameters"]], [-854, 1807, 3315, -612, 1608, 2895, -2002])
                self.assertEqual(late7["late_calls"], [(0x530002c8, 0x18, 0x20, 0x7ad0), *(
                    (0x53000038, 0x10a28 + i * 0x200, 0x5800 + i * 0x18, 0xc0a8 + i * 0x1c0) for i in range(7))])
                self.assertEqual([(o["member_offset"], o["slots"]) for o in late7["overlays"]],
                                 [(0x2e210, 87)] + [(0x30c40, 41)] * 5)
                self.assertEqual(late7["compute"], {"pc": 0x53004044, "reason": "unsupported opcode", "opcode": 10})
                self.assertEqual(late7["arguments"], (0x530002e0, 0, 7, 0x10aa0))
                self.assertEqual(late7["stop"], {"pc": 0x53002842, "reason": "unmodeled read", "address": 0x10aa0, "size": 2})
                self.assertEqual(late7["copy_registers"], (0x10aa0, 0x10a48, 0x10a9c))
                final7 = late7["head_fixture"]
                self.assertEqual(final7["copy_input"], tuple(range(-21, 21)))
                self.assertEqual([(w["pc"], w["address"], w["size"], w["value"] & 0xffff)
                                  for w in final7["copy_writes"]],
                                 [(0x53002846, 0x10a48 + 2 * i, 2, v & 0xffff)
                                  for i, v in enumerate(range(-21, 21))])
                self.assertEqual([(c["offset"], c["size"], c["arguments"]) for c in final7["calls"]],
                                 [(0x44660, 1232, (0x530003a4, 0x10aa0, 0x10a28, 0x10a48)),
                                  (0x44b30, 1344, (0x53000430, 0x10f88, 0x10a48, 0x10a28))])
                self.assertEqual(final7["unseeded_stop"],
                                 {"pc": 0x53002b1c, "reason": "unmodeled read", "address": 0x10f88, "size": 4})
                sigmoid7 = final7["sigmoid_fixture"]
                self.assertEqual(sigmoid7["exponent"], 25)
                self.assertEqual(sigmoid7["input"], tuple(v << 25 for v in (-2, -1, 0, 1, 2, 5, -5)) * 6)
                fp32 = [1039409327, 1049211569, 1056964608, 1060841125, 1063353324, 1065240929, 1004228554] * 6
                self.assertEqual(sigmoid7["fp32_writes"],
                                 [{"pc": 0x53002ba2, "address": 0xa00040c0 + 4 * i, "size": 4, "value": v}
                                  for i, v in enumerate(fp32)])
                self.assertEqual(sigmoid7["writes"],
                                 [{"pc": 0x53002bf4, "address": 0x10a28 + 8 * i, "size": 2, "value": v}
                                  for i, v in enumerate([7812, 17625, 32768, 47910, 57723, 65097, 438] * 6)])
                downstream7 = sigmoid7["consumer_fixture"]
                overlay = downstream7["activation_program"]
                self.assertEqual((overlay["member_offset"], overlay["slots"]), (0x2ea38, 61))
                self.assertEqual(downstream7["consumers"],
                                 [{"arguments": (0x530004a4, i, 0xc0a8, 0x114a8),
                                   "stack_arguments": (0x10a28, 0x17b00, 0x17be8)} for i in range(3)])
                weights7 = downstream7["first_projection"]
                self.assertEqual(weights7["offsets"], [0x4de88, *(0x4dee0 + 672 * i for i in range(6))])
                self.assertEqual([r["value"] for r in weights7["parameters"]],
                                 [-6695, 3770, 16810, -29872, 1273, 4288, 8474])
                self.assertEqual(weights7["calls"], [(0x530002c8, 0x15ab6, 0x53000c38, 0x53004324), *(
                    (0x53000184, 0x114a8, 0x5800 + 0x60 * i, 0xc0ca + 0x200 * i) for i in range(7))])
                self.assertEqual(len(downstream7["copy_requests"]), 42)
                for i, (linear, packed) in enumerate(downstream7["copy_requests"]):
                    src = 0xc0a8 + 0x200 * i
                    self.assertEqual(linear["rows"], [[src + at, 0xc000c0a8, 0x1f0000, 0x1f0000]
                                                      for at in (0, 0x1e0)])
                    self.assertEqual(packed["rows"], [[src + at, 0xc020c0a8, 0x10004, 0x1b000d]
                                                      for at in (0x20, 0x3e)])
                weights7 = downstream7["second_projection"]
                self.assertEqual(weights7["offsets"], [0x4eea0, *(0x4eef8 + 168 * i for i in range(6))])
                self.assertEqual([r["value"] for r in weights7["parameters"]],
                                 [2232, 2906, 1471, 1850, -583, -5991, -404])
                self.assertEqual(weights7["calls"], [(0x530002c8, 0x18, 0x20, 0x7ad0), *(
                    (0x53000038, 0xc0a8 + 0x200 * i, 0x5800 + 0x18 * i, 0x114a8 + 0x1c0 * i) for i in range(7))])
                self.assertEqual([(o["member_offset"], o["slots"]) for o in downstream7["tail_programs"]],
                                 [(0x2eff0, 92)] + [(0x31018, 43)] * 5)
                self.assertEqual(downstream7["arguments"], (0x530005c4, 0, 7, 0xc120))
                self.assertEqual(downstream7["unseeded_stop"],
                                 {"pc": 0x53003434, "reason": "unmodeled read", "address": 0xc120, "size": 2})
                self.assertEqual(downstream7["copy_registers"], (0xc120, 0xc0c8, 0xc11c))
                handoff7 = downstream7["final_fixture"]
                self.assertEqual([(w["pc"], w["address"], w["size"], w["value"] & 0xffff)
                                  for w in handoff7["copy_writes"]],
                                 [(0x53003438, 0xc0c8 + i * 2, 2, v & 0xffff)
                                  for i, v in enumerate(range(-21, 21))])
                self.assertEqual([(c["offset"], c["arguments"]) for c in handoff7["coefficient_loads"]],
                                 [(0x45070, (0x5300068c, 0xc120, 0xc0a8, 0xc0c8)),
                                  (0x45540, (0x53000724, 0xc0c8, 0xc170, 0xc0a8))])
                self.assertEqual(handoff7["unseeded_logits"],
                                 {"pc": 0x530036d6, "reason": "unmodeled read", "address": 0xc0c8, "size": 4})
                self.assertEqual(handoff7["exponents"], (25, 25, -16))
                self.assertEqual(handoff7["input_q25"], sigmoid7["input"])
                self.assertEqual(handoff7["q16_writes"],
                                 [{"pc": 0x53003766, "address": 0xd308 + 8 * i, "size": 2, "value": v}
                                  for i, v in enumerate([7812, 17625, 32768, 47910, 57723, 65097, 438] * 6)])
                self.assertEqual([(o["member_offset"], o["slots"]) for o in handoff7["overlays"]],
                                 [(0x2f890, 51), (0x301d8, 111)])
                self.assertEqual(handoff7["mailbox"], (0xd308, 0x17b00, 0x17be8))
                self.assertEqual(handoff7["consumers"], [(0x53000798, i, 0x114a8, 0xc0a8) for i in range(7)])
                self.assertEqual(handoff7["further_calls"], [(0x53000970, i, 0x16000, 0xc0a8) for i in range(7)])
                self.assertEqual(handoff7["last_offsets"], [0x4d5d8, *(0x4d600 + i * 72 for i in range(6))])
                self.assertEqual(handoff7["last_call"], (0x530009e0, 0, 3, 0x16000))
                self.assertEqual(handoff7["next_entry"], {"pc": 0x50041400, "reason": "slice boundary"})
                self.assertEqual(handoff7["next_member_offset"], 0x31500)
                self.assertEqual(handoff7["descriptor_request"]["rows"],
                                 [[0x500416a0, 0x18, 0x9f0000, 0x9f0000]])
                self.assertEqual(handoff7["descriptor_member_offset"], 0x317a0)
                self.assertEqual(len(handoff7["local_request"]["rows"]), 9)
                self.assertEqual(handoff7["external_transfer"], (0x04010600, 0x53800000, 0x5700))
                self.assertEqual(handoff7["external_member_offset"], 0x31900)
                self.assertEqual(handoff7["eighth_stop"],
                                 {"pc": 0x53800ab4, "reason": "unmodeled write", "address": 0x3b74, "size": 4,
                                  "value": 0})
                eighth = handoff7["eighth_fixture"]
                self.assertEqual((eighth["work_range"], eighth["clear_words"]), ((0x3880, 0x43b0), 716))
                self.assertEqual([(x["offset"], x["bytes"], x["request"]["rows"][0][1])
                                  for x in eighth["parameter_loads"]],
                                 [(0x56d4, 272, 0xa000), (0x5ed8, 1116, 0xa110)])
                self.assertEqual((eighth["overlay_offset"], eighth["slots"]), (0x3bba0, 111))
                self.assertEqual([r["rows"] for r in eighth["initial_copies"]],
                                 [[[src, 0xc0016000, 0x1a3f0000, 0x1a3f0000]]
                                  for src in (0xf000, 0x11000, 0x13000)])
                groups = eighth["projections"]
                self.assertEqual([g["offsets"] for g in groups],
                                 [[0x4f2e8, 0x4f300]] + [[src, *(src + 24 + j * 256 for j in range(6))]
                                                          for src in range(0x4f900, 0x51160, 0x618)])
                self.assertEqual([tuple(r["value"] for r in g["parameters"]) for g in groups],
                                 [(2497, -4532), (-14749, -5584), (-12573, -13173), (-16346, -4069), (6552, -11027)])
                self.assertEqual([g["calls"] for g in groups], [
                    [(0x53800128, 0x16000, coef, dst), (0x53800128, 0x16000, coef + 0x80, dst + 0x70)]
                    for coef, dst in ((0xaa18, 0xcd10), (0x4000, 0xc7d0), (0x4000, 0xc290),
                                      (0x4000, 0xbd50), (0x4000, 0xb810))])
                self.assertEqual([r["rows"] for r in eighth["following_copies"]],
                                 [[[src, 0xc000cd10, 0x53f0000, 0x53f0000]]
                                  for src in range(0x15800, 0x18000, 0x800)])
                self.assertEqual(eighth["stop"],
                                 {"pc": 0x53802aee, "reason": "unmodeled read", "address": 0x48230, "size": 4})
                wide8 = eighth["wide_fixture"]
                self.assertEqual([[r["value"] for r in g["parameters"]] for g in wide8["groups"]],
                                 [[-1283, 5478, 4588, -12661, -17933], [-2019, 1778, -1282, 1510, 4516],
                                  [-32587, 5875, -9859, -573, 2065], [-4102, -5420, 2429, -2638, -9704],
                                  [5053, -10776, -8998, 289, 10670]])
                for i, (g, feature, output) in enumerate(zip(
                        wide8["groups"], (0xcd10, 0xc7d0, 0xc290, 0xbd50, 0xb810),
                        (0xaa00, 0xacd0, 0xafa0, 0xb270, 0xb540), strict=True)):
                    self.assertEqual(g["calls"], [(0x53800038, feature, 0x4000 + j * 0x3c0, output + j * 0x18)
                                                  for j in range(5)])
                    if i < 4:
                        target = (0xc7d0, 0xc290, 0xbd50, 0xb810)[i] | 0xc0000000
                        self.assertEqual([x["rows"] for x in g["copy_requests"]],
                                         [[[src, target, 0x53f0000, 0x53f0000]] for src in range(0x15000, 0x18000, 0x800)])
                last8 = wide8["groups"][-1]
                self.assertEqual(last8["request"]["rows"][3], [0x20078178, 0x10800, 0x12bf0000, 0x12bf0000])
                self.assertEqual(last8["transfers"][3],
                                 {"fragments": [(1, 0x700f8, 1032), (2, 0, 3768)],
                                  "sha256": "298d91bbab79d32b60127e4356fbbb7dc97f84db686f4603c329fd4b9c00b89d"})
                self.assertEqual([t["fragments"][0][0] for t in last8["transfers"][4:]], [2, 2, 2])
                self.assertEqual(wide8["scalar_inputs"], (0xaa00, 0xaa02, 0xaa04))
                self.assertEqual(wide8["stop"],
                                 {"pc": 0x53803632, "reason": "unmodeled read", "address": 0xaa00, "size": 2})
                reduction8 = wide8["reduction_fixture"]
                self.assertEqual((reduction8["config_address"], reduction8["shift_address"]), (0xa0bc, 0xa34e))
                self.assertEqual((reduction8["multiplier"], reduction8["shift"]), (3641, 15))
                self.assertEqual((reduction8["bias"], reduction8["limits"]), ((-16384, 16384), (-32768, 32767)))
                expect_reads = [(pc, 0xaa00 + 24 * i + 8 * (j // 3) + 2 * (j % 3), v)
                                for i, group in enumerate(reduction8["inputs"]) for j, v in enumerate(group)
                                for pc in ((0x53803632, 0x53803634, 0x5380363c)[j % 3],)]
                self.assertEqual([(r["pc"], r["address"], r["value"]) for r in reduction8["reads"]], expect_reads)
                sums = [sum(group) for group in reduction8["inputs"]]
                results = [min(32767, max(-32768, (s * 3641 + (16384 if s >= 0 else -16384)) >> 15))
                           for s in sums]
                self.assertEqual(results, [4, -27, 34, 32767, -32768] * 5)
                self.assertEqual([(w["pc"], w["address"], w["size"], w["value"] & 0xffff)
                                  for w in reduction8["writes"]],
                                 [(0x53803676, 0xb960 + 8 * i, 2, v & 0xffff) for i, v in enumerate(results)])
                self.assertEqual(reduction8["merged_range"], (0xb960, 0xb830, 0xb95c))
                self.assertEqual([(w["pc"], w["address"], w["size"], w["value"] & 0xffff)
                                  for w in reduction8["merge_writes"]],
                                 [(0x53803714, 0xb830 + 2 * i, 2, v & 0xffff) for i, v in enumerate(results)])
                self.assertEqual(reduction8["worker_stop"],
                                 {"pc": 0x53803710, "reason": "unmodeled read", "address": 0xba28, "size": 2})
                combined8 = reduction8["workers_fixture"]
                self.assertEqual(combined8["mailbox"], (0xb960, 0xaa00, 0xa000, 0xa110))
                self.assertEqual(len(combined8["inputs"]), 150)
                combined_values = []
                for group in combined8["inputs"]:
                    total = sum(group)
                    combined_values.append(min(32767, max(-32768, (total * 3641 + (
                        16384 if total >= 0 else -16384)) >> 15)))
                self.assertEqual(combined_values[:25], results)
                for i, job in enumerate(combined8["workers"], 1):
                    start = 25 * i
                    src = 0xaa00 + 24 * start
                    self.assertEqual(job["input_start"], src)
                    rows = combined8["inputs"][start:start + 25]
                    self.assertEqual([(r["address"], r["value"]) for r in job["reads"]],
                                     [(src + 24 * j + 8 * k + 2 * t, row[3 * k + t])
                                      for j, row in enumerate(rows) for k in range(3) for t in (1, 0, 2)])
                    self.assertEqual([(w["address"], w["size"], w["value"] & 0xffff) for w in job["writes"]],
                                     [(0xb960 + 8 * j, 2, combined_values[j] & 0xffff) for j in range(start, start + 25)])
                self.assertEqual([(w["pc"], w["address"], w["size"], w["value"] & 0xffff)
                                  for w in combined8["packed_writes"]],
                                 [(0x53803714, 0xb830 + 2 * i, 2, v & 0xffff) for i, v in enumerate(combined_values)])
                packed8 = combined8["consumer"]
                self.assertEqual((packed8["overlay_offset"], packed8["slots"]), (0x3c608, 84))
                self.assertEqual(packed8["request"]["rows"], [[0x2004db00, 0xb960, 0x11cf0000, 0x11cf0000]])
                self.assertEqual(packed8["dictionary_offset"], 0x45a80)
                self.assertEqual(packed8["mailbox"], (0xb960, 0xb810, 0xb830, 0xa000, 0xa110))
                self.assertEqual(packed8["compute"], {"pc": 0x53804bb8, "reason": "unsupported opcode", "opcode": 10})
                self.assertEqual(packed8["arguments"], (0x5380026c, 0, 3, 0xb960))
                self.assertEqual(packed8["stack_arguments"], (0xb810, 0xb830, 0xa000))
                next8 = packed8["following_projection"]
                self.assertEqual(next8["request"]["rows"], [[0x20008080, 0xba88, 0x12bf0000, 0x12bf0000]])
                self.assertEqual(next8["dictionary_offset"], 0)
                self.assertEqual(next8["mailbox"], (0xb830, 0xba88, 0xb810))
                self.assertEqual(next8["compute"], {"pc": 0x53804bb8, "reason": "unsupported opcode", "opcode": 10})
                self.assertEqual(next8["arguments"], (0x53800344, 0, 25, 0xb830))
                self.assertEqual(next8["unseeded_stop"],
                                 {"pc": 0x53803b1a, "reason": "unmodeled read", "address": 0xb830, "size": 4})
                sigmoid8 = next8["sigmoid_fixture"]
                self.assertEqual(sigmoid8["exponents"], (22, 22, -16))
                logits8 = tuple(x << 22 for x in (-5, -2, -1, 0, 1, 2, 5) * 22)[:150]
                self.assertEqual(sigmoid8["input"], logits8)
                self.assertEqual(sigmoid8["input_reads"],
                                 [{"pc": 0x53803b1a, "address": 0xb830 + 4 * i, "size": 4, "value": v & 0xffffffff}
                                  for i, v in enumerate(logits8)])
                self.assertEqual(sigmoid8["writes"],
                                 [{"pc": 0x53803b0e, "address": 0xc2a0 + 8 * i, "size": 2, "value": v}
                                  for i, v in enumerate(([438, 7812, 17625, 32768, 47910, 57723, 65097] * 22)[:150])])
                self.assertEqual(sigmoid8["end"],
                                 {"pc": 0x53803ed6, "reason": "unmodeled read", "address": 0x481cc, "size": 4})
                post8 = sigmoid8["consumer_fixture"]
                self.assertEqual([(p["member_offset"], p["slots"]) for p in post8["programs"]],
                                 [(0x3cde8, 40), (0x3d1a8, 64)])
                self.assertEqual(post8["arguments"], (0x5380040c, 0, 25, 0xaa00))
                self.assertEqual(post8["stack_arguments"], (0xb810, 0xc2a0, 0xa000, 0xa110))
                self.assertEqual([s["rows"] for s in post8["copies"]],
                                 [[[src, dst, 0x0a8b0000, 0x0a8b0000]] for src, dst in
                                  ((0x16820, 0xc000b810), (0x4000, 0xc0016820), (0x5000, 0xc0016820),
                                   (0x6000, 0xc0016820))])
                self.assertEqual(post8["following_calls"], [(0x538004dc, i, 0, 2) for i in range(12)])
                self.assertEqual(len(post8["batches"]), 14)
                self.assertEqual([b["rows"][0][0] for b in post8["batches"]],
                                 [*(0x20009340 + 18 * i for i in range(11)), 0x200093f6, 0x200093f4, 0x200093f6])
                self.assertEqual([b["rows"][1][0] for b in post8["batches"]],
                                 [*(0x20009408 + 0x5f10 * i for i in range(11)), 0x20045538, 0x20044aa8, 0x20045538])
                self.assertEqual(post8["stop"],
                                 {"pc": 0x538043c6, "reason": "unmodeled read", "address": 0x172b0, "size": 2})
                scalar8 = post8["scalar_heads_fixture"]
                self.assertEqual(scalar8["input"], tuple(range(-50, 50)))
                self.assertEqual(tuple(scalar8[p + "_offset"] for p in ("first", "second", "third")),
                                 (0x433c8, 0x436f0, 0x43888))
                for prefix, count, shift in (("first", 4, 4), ("second", 2, 5), ("third", 2, 2)):
                    offset = scalar8[prefix + "_offset"]
                    bias = unpack_from(f"<{count}h", first, offset)
                    self.assertEqual([r["value"] for r in scalar8[prefix + "_bias_reads"]], list(bias))
                    expected = tuple((v << shift) + sum(x * w for x, w in zip(
                        scalar8["input"], unpack_from("<100h", first, offset + 8 + 200 * i)))
                        for i, v in enumerate(bias))
                    self.assertEqual(scalar8[prefix + "_output"], expected)
                self.assertEqual(scalar8["first_output"], (37721, -219830, -411628, -317016))
                self.assertEqual(scalar8["second_unseeded_stop"],
                                 {"pc": 0x53804526, "reason": "unmodeled read", "address": 0xa004b778, "size": 2})
                self.assertEqual(scalar8["fourth_start"],
                                 {"pc": 0x5380475a, "reason": "unmodeled read", "address": 0xa004baa8, "size": 2})
                writes = {w["address"]: (w["size"], w["value"]) for w in scalar8["work_writes"]}
                for i, (prefix, offset, fixed, floating) in enumerate((
                        ("fourth", 0x43a20, 0xe0004060, 0xe00040b8),
                        ("fifth", 0x43af0, 0xe000408c, 0xe00040b4))):
                    self.assertEqual(scalar8[prefix + "_offset"], offset)
                    bias, = unpack_from("<h", first, offset)
                    self.assertEqual(scalar8[prefix + "_bias"] & 0xffff, bias & 0xffff)
                    expected = sum(x * w for x, w in zip(
                        scalar8["input"], unpack_from("<100h", first, offset + 8)))
                    self.assertEqual(scalar8[prefix + "_sum"], expected & 0xffffffff)
                    self.assertEqual(scalar8[prefix + "_shift"], 0xfffffffb)
                    bits = scalar8["float_words"][i]
                    value, = unpack_from("<f", pack("<I", bits))
                    logit = (expected + (bias << 5)) / (1 << 20)
                    self.assertAlmostEqual(value, 1 / (1 + exp(-logit)), delta=3e-7)
                    self.assertEqual(writes[floating], (4, bits))
                    self.assertEqual(writes[fixed], (4, int(value * 65536)))
                self.assertEqual(writes[0xe00040d4], (2, 0x08ff))
                self.assertEqual(scalar8["handoff"],
                                 {"pc": 0x5004d7ba, "reason": "unmodeled read", "address": 0x481c4, "size": 4})
                self.assertEqual(scalar8["downstream_dma"], [{
                    "pointer_pc": 0x5004d7b0, "pointer": 0x17800, "submit_pc": 0x5004d81c,
                    "rows": [[0x5004d9a0, 0x17818, 0x9f0000, 0x9f0000]],
                }])
                ninth = scalar8["ninth_fixture"]
                self.assertEqual(ninth["code_extent"], (0x3dc00, 0x5300))
                self.assertEqual({r["address"]: r["value"] for r in ninth["work_reads"]},
                                 {a: writes[a][1] for a in range(0xe0004040, 0xe0004058, 4)})
                self.assertEqual([(w["address"], w["value"]) for w in ninth["work_writes"]], [
                    (0xe00040d4, 0x901), (0xe00040d4, 0x902), (0xe00040d6, 5),
                    (0xe00040d4, 0x903), (0xe00040d4, 0x904), (0xe00040d8, 0x22), (0xe00040da, 0x17),
                ])
                self.assertEqual([r["rows"] for r in ninth["input_requests"][1:]], [
                    [[0x14002 + 0x50 * i, 0xc00142d2, 0x400000, 0x400000]] for i in range(9)
                ])
                self.assertEqual(ninth["stop"],
                                 {"pc": 0x54001d88, "reason": "unmodeled read", "address": 0x48168, "size": 4})
                self.assertEqual(ninth["overlays"],
                                 [(0x4a558, 37, 0x04000000 + 0x1000 * i) for i in (1, 2, 3, 4, 5, 0)])
                self.assertEqual(ninth["compute"],
                                 {"pc": 0x54000718, "reason": "unsupported opcode", "opcode": 0xa})
                self.assertEqual(ninth["arguments"], (0x54000674, 0x14002, 0x50, 0x16000))
                self.assertEqual(ninth["return"], 0x54002084)
                self.assertEqual(ninth["record_offset"], 0x3e274)
                self.assertEqual(sha256(bytes.fromhex(ninth["record_hex"])).hexdigest(),
                                 run["shared_image_entry"]["sha256"])
                member = members(self.bundle)[0x44]
                bed = self.bundle[member.offset:member.offset + member.size]
                self.assertEqual(bed[0x4a558:0x4a8d0], bed[0x1df0:0x2168])
                self.assertEqual(tuple(dict(ninth["setup"])[a] for a in range(0x40000, 0x4003c, 4)),
                                 (4, 24, 12, 12, 0x10000000, 0x10000000, 0, 0x800000, 0, 0x800000,
                                  23, 11, 0x1000, 17, 0))
                consumer = ninth["consumer_fixture"]
                self.assertEqual(consumer["overlays"],
                                 [(0x48920, 127, 0x04000000 + 0x1000 * i) for i in range(6)])
                self.assertEqual((consumer["copy_source"], consumer["copy_destination"], consumer["copy_size"]),
                                 (0x16000, 0x17000, 0x480))
                self.assertEqual(consumer["copy_sha256"],
                                 sha256(bytes((i * 17 + 3) & 0xff for i in range(0x480))).hexdigest())
                follow = consumer["next_fixture"]
                self.assertEqual([r["rows"] for r in follow["resource_requests"]], [
                    [[0x2008b6b4, 0x17e04, 0xb30000, 0xb30000]],
                    [[0x2008b768, 0x17c00, 0x2030000, 0x2030000]],
                ])
                resource_base = fixture["post_compute_fixture"]["weight_base_fixture"] + layout["base_offset"]
                for request, offset, size in zip(follow["resource_requests"], (0x13134, 0x131e8), (180, 516)):
                    src, _, flags, _ = request["rows"][0]
                    self.assertEqual((src - resource_base, (flags >> 16) + 1), (offset, size))
                self.assertEqual(0x131e8 + 516, len(second))
                self.assertEqual(follow["broadcast"]["rows"], [
                    [dst, 0xc0017000, 0x47f0000, 0x47f0000] for dst in (0xe800, 0xf000, 0xf800, 0x10000, 0x10800)
                ])
                self.assertEqual(follow["crop"]["rows"], [[0xb118, 0xc0317064, 0x31f0000, 0x271300]])
                self.assertEqual(follow["raster_extent"], (48, 24))
                self.assertEqual(follow["crop_extent"], (40, 20))
                center = bytes(((y * 48 + x) * 17 + 3) & 0xff for y in range(2, 22) for x in range(4, 44))
                self.assertEqual(follow["crop_sha256"], sha256(center).hexdigest())
                self.assertEqual(follow["overlay"], (0x49b20, 37, 0x0400f000))
                self.assertEqual(follow["compute"],
                                 {"pc": 0x5400477c, "reason": "unsupported opcode", "opcode": 0xa})
                self.assertEqual(follow["arguments"], (0x54000458, 0, 2, 0xc9d8))
                self.assertEqual(follow["return"], 0x54002d3a)
                self.assertEqual(follow["tail_offsets"], (0x13134, 0x131e8))
                next_call = follow["following_fixture"]
                self.assertEqual(next_call["overlay"], (0x48920, 127, 0x0400f000))
                self.assertEqual(next_call["offsets"], (0x122ec, *range(0x122fc, 0x12375, 24)))
                self.assertEqual(next_call["resources"]["rows"], [
                    [resource_base + offset, target, flags, flags]
                    for offset, target, flags in zip(
                        next_call["offsets"], (0xcf60, 0x5800, 0x6000, 0x6800, 0xd008, 0xcac8, 0xd800),
                        (0xb0000, *(0x170000,) * 6), strict=True)
                ])
                self.assertEqual(next_call["blocks"][0],
                                 ((4269, -2149, -153), (-1884, -14341, 866), (-95, -215, 13441)))
                self.assertEqual(len(next_call["blocks"]), 6)
                reads = {(r["address"], r["size"]): r["value"] for r in next_call["parameter_reads"]}
                self.assertEqual(reads[0x17c30, 1] & 0xff, 15)
                self.assertEqual(reads[0x17c29, 1] & 0xff, 242)
                self.assertEqual(next_call["compute"],
                                 {"pc": 0x54000718, "reason": "unsupported opcode", "opcode": 0xa})
                self.assertEqual(next_call["arguments"], (0x540002b0, 0x18, 0x30, 0x7b60))
                self.assertEqual(next_call["return"], 0x54001456)

    def test_roll_validator_and_original_fixed_point_tables(self):
        member = members(self.bundle)[0x44]
        bed = self.bundle[member.offset:member.offset + member.size]
        src, dst, flags, _ = next(startup_rows(bed))
        image = bed[src:src + (flags >> 16) + 1]

        def read(address, size):
            return image[address - dst:address - dst + size]

        table_a, table_b = (unpack_from("<25i", image, pc_literal(read, pc, 1) - dst)
                            for pc in (0x12f8, 0x130a))
        self.assertEqual((table_a[12], table_b[12]), (0, 1 << 20))
        for index, roll in enumerate(range(-180, 181, 15)):
            with self.subTest(roll=roll):
                run = trace_bed(self.cp, self.bundle, bytes(0x100), roll=roll)
                self.assertEqual(run["validator_mask"], 0)
                self.assertEqual(run["after_image"]["pc"], 0x8c0)
                setup = dict(run["compute_setup"])
                expected = tuple((8 * v) & 0xffffffff for v in (table_a[index], table_b[index])) * 2
                self.assertEqual(tuple(setup[a] for a in range(0x40018, 0x40028, 4)), expected)
        for roll in (-195, -1, 1, 195):
            with self.subTest(roll=roll):
                run = trace_bed(self.cp, self.bundle, bytes(0x100), roll=roll)
                self.assertEqual(run["validator_mask"], 0x800)
                self.assertFalse(run["submissions"])

    def test_inside_and_clipped_rectangles_keep_both_packed_lanes(self):
        cases = (
            ((288, 202), (96, 96), PackedRead(117 * 1280 + 192 * 2, 1280, 384, 171), {}),
            ((30, 40), (64, 64), PackedRead(0, 1280, 256, 126), {0x1250: 56, 0x11b8: 45}),
            ((620, 450), (64, 64), PackedRead(365 * 1280 + 528 * 2, 1280, 224, 115),
             {0x108e: 66, 0x1124: 56}),
        )
        frame = bytes((i * 83 + 1) & 255 for i in range(1280 * 480))
        for center, sides, expected, counts in cases:
            with self.subTest(center=center, sides=sides):
                run = trace_bed(self.cp, self.bundle, bytes(0x100), center=center, sides=sides)
                self.assertEqual(run["image_read"], expected)
                self.assertEqual(run["after_image"]["pc"], 0x8c0)
                self.assertEqual(Counter(r["submit_pc"] for r in run["submissions"][2:]), counts)
                gathered = expected.gather(frame)
                self.assertEqual(len(gathered), expected.rows * expected.row_bytes)
                for i in range(expected.rows):
                    src, dst = expected.offset + i * expected.pitch, i * expected.row_bytes
                    self.assertEqual(gathered[dst:dst + expected.row_bytes], frame[src:src + expected.row_bytes])
                with self.assertRaisesRegex(ValueError, "truncated"):
                    expected.gather(frame[:expected.offset])

    def test_local_post_dma_crop_matches_independent_clipping(self):
        def pixel(i, j):
            return (53 * i + 67 * j + (i ^ j)) & 255

        width, height = 640, 480
        frame = bytes(pixel(i, j) for j in range(height) for i in range(width))
        for cx in (16, 30, 84, 86, 280, 554, 620):
            for cy in (16, 40, 84, 86, 192, 394, 450):
                with self.subTest(center=(cx, cy)):
                    run = trace_bed(self.cp, self.bundle, bytes(0x100), center=(cx, cy))
                    crop = run["local_image"]
                    self.assertEqual(crop.window_xywh, (cx - 85, cy - 85, 171, 171))
                    read = crop.packed_read
                    offset, pitch, row_bytes = (read.offset // 2, read.pitch // 2, read.row_bytes // 2)
                    raw = b"".join(frame[i:i + row_bytes] for i in range(offset, offset + read.rows * pitch, pitch))
                    expected = bytes(pixel(min(max(i, 0), width - 1), min(max(j, 0), height - 1))
                                     for j in range(cy - 85, cy + 86) for i in range(cx - 85, cx + 86))
                    self.assertEqual(crop.from_local_rows(raw), expected)
                    with self.assertRaisesRegex(ValueError, "exact post-transfer"):
                        crop.from_local_rows(raw[:-1])


if __name__ == "__main__":
    unittest.main()
