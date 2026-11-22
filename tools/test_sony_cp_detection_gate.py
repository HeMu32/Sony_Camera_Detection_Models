"""Replay selected Sony CP decisions with explicit, isolated inputs."""

from pathlib import Path
from struct import pack, unpack
import tarfile
import unittest

from unicorn import UC_HOOK_CODE
from unicorn.arm_const import (UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R1,
                               UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_SP)

from probe_sony_cp_confidence import ConfidenceProbe, GETTER_NAMES, RTTI
from probe_sony_cp_region_scale import BASE, RETURN, CpSlices
from probe_sony_cp_xhd_nms import NmsProbe
from test_sony_cp_region_scale import CP_IMAGES


R5 = CP_IMAGES["A7R V"][0]
OBJECT, INPUT = (RETURN + 0x1000 * i for i in range(1, 3))
STACK = RETURN + 0xe00
R5_VTABLE = 0xc2fb90
R5_SPATIAL_CONFIG = 0x0172424c
R5_SPATIAL_FIELD = 0x11e


def bits(value):
    return unpack("<I", pack("<f", value))[0]


def call(probe, offset, args, extra=()):
    m = probe.machine
    m.mem_write(STACK, pack(f"<{len(extra)}I", *extra))
    for r, v in zip((UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
                     UC_ARM_REG_SP, UC_ARM_REG_LR), (*args, STACK, RETURN | 1)):
        m.reg_write(r, v)
    probe.execute(BASE + offset, RETURN, limit=1200)
    return m.reg_read(UC_ARM_REG_R0)


@unittest.skipUnless(R5.is_file(), "requires the pinned A7R V CP image")
class DetectionGateTest(unittest.TestCase):
    def setUp(self):
        self.probe = CpSlices(R5.read_bytes())
        self.m = self.probe.machine
        for address in (OBJECT, INPUT, R5_SPATIAL_CONFIG, 0x01450290):
            self.probe.map_ram(address)
        getter = self.probe.virtual_methods(R5_VTABLE, RTTI, 0x138 // 4 + 1)[-1]
        self.assertEqual(getter, BASE + 0x1c066d)
        self.m.mem_write(0x01450290, pack("<I", BASE + R5_VTABLE))

        def return_diagnostic(machine, _address, _size, _data):
            machine.reg_write(UC_ARM_REG_PC, machine.reg_read(UC_ARM_REG_LR))

        logger = BASE + 0x1956cc
        self.m.hook_add(UC_HOOK_CODE, return_diagnostic, begin=logger, end=logger)

    def test_spatial_metric_is_intersection_over_smaller_area(self):
        self.m.mem_write(INPUT, pack("<I4H", 0, 20, 20, 20, 20))
        for box, expected in (((0, 0, 100, 100), 1.), ((30, 20, 40, 20), .5),
                              ((31, 20, 40, 20), .45), ((40, 20, 40, 20), 0.)):
            self.m.mem_write(INPUT + 0x20, pack("<I4H", 0, *box))
            got = call(self.probe, 0x1c2cdc, (0, INPUT, INPUT + 0x20, 0))
            self.assertEqual(got, bits(expected))

    def test_spatial_gate_can_clear_flag_independent_of_candidate_scores(self):
        m = self.m
        m.mem_write(INPUT, pack("<f4H", .99, 20, 20, 20, 20))
        m.mem_write(INPUT + 0x40, b"\0")  # Checked category record zero.
        m.mem_write(INPUT + 0x50, b"\1")  # Bypass competing-box branch.
        for threshold, x, score, kept in ((5000, 29, .01, 1), (5000, 30, .99, 1),
                                           (5000, 31, .99, 0), (6000, 30, .99, 0),
                                           (3000, 34, .99, 1), (3000, 35, .99, 0)):
            m.mem_write(R5_SPATIAL_CONFIG + R5_SPATIAL_FIELD, pack("<H", threshold))
            m.mem_write(OBJECT + 0x15c, pack("<f4H", score, x, 20, 40, 20))
            m.mem_write(OBJECT + 0x755, b"\1")
            call(self.probe, 0x6ece30, (OBJECT, INPUT, INPUT + 0x40, INPUT + 0x60),
                 (INPUT + 0x70, INPUT + 0x50))
            self.assertEqual(m.mem_read(OBJECT + 0x755, 1)[0], kept)

    def test_verification_counts_persist_across_calls(self):
        self.m.mem_write(OBJECT + 0x16, b"\2")
        # Diagnostic labels these enum inputs xhd, stpl and verify.
        self.m.mem_write(INPUT, pack("<3I", 1, 0, 0))
        args = (OBJECT, INPUT, INPUT + 4, INPUT + 8)
        observed = []
        for _ in range(7):
            result = call(self.probe, 0x6f5892, args)
            observed.append((result, self.m.mem_read(OBJECT + 0xf, 1)[0]))
        self.assertEqual(observed, [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (2, 5)])
        self.m.mem_write(INPUT, pack("<I", 0))
        self.assertEqual(call(self.probe, 0x6f5892, args), 0)
        self.assertEqual(self.m.mem_read(OBJECT + 0xf, 1), b"\0")


class PublishedConfidenceTest(unittest.TestCase):
    def test_native_backup_and_getters(self):
        cases = {
            "A7R V": ("debug/r5_xhd_fw/firmware.tar", "0110_backup/SYSIPSX-DSLR/GL/CX89200_ALLLANG.bin",
                      (.954, .92699999, .8779)),
            "A1 II": ("debug/a1ii_fw/firmware.tar", "0110_backup/SYSIPSX-DSLR/HC/CX89900_ALLLANG.bin",
                      (.8148, .60399997, .6088)),
        }
        tested = 0
        for camera, (archive, member_name, expected) in cases.items():
            image_path = CP_IMAGES[camera][0]
            if not image_path.is_file() or not Path(archive).is_file():
                continue
            tested += 1
            with self.subTest(camera=camera), tarfile.open(archive, "r:") as tar:
                with tar.extractfile(member_name) as file:
                    member = file.read()
                data = image_path.read_bytes()
                nms, probe = NmsProbe(data), ConfidenceProbe(data)
                addresses = set()
                for i, target in enumerate((1, 2, 4, 5, 6)):
                    address, metadata = nms.load_published_tuning(target, member)
                    self.assertEqual(metadata["configuration_id"], hex(0x60011 + i))
                    address += 0x1d0
                    addresses.add(address)
                    probe.machine.mem_write(address, bytes(nms.machine.mem_read(address, 0x220)))
                self.assertEqual(addresses, probe.records)
                if camera == "A7R V":
                    got = call(probe, 0x1c066c, (0, 0, 0, 0))
                    self.assertEqual(got, bits(.3))
                for getter in GETTER_NAMES[:2]:
                    for sensitivity, face in enumerate((1., .9609, .8906, .539, .164), 1):
                        got = probe.evaluate_loaded(getter, 0, sensitivity)["threshold"]
                        self.assertAlmostEqual(got, face, places=6)
                for target, kind, part, high in ((3, 0, 1, expected[0]), (3, 3, 1, expected[1]),
                                                  (6, 3, 1, expected[2])):
                    for getter, want in zip(GETTER_NAMES[2:], (.5, high)):
                        got = probe.evaluate_loaded(getter, target, 3, xhd_type=kind, part=part)
                        self.assertAlmostEqual(got["threshold"], want, places=6)
        if not tested:
            self.skipTest("requires local published firmware archives and matching CP images")
