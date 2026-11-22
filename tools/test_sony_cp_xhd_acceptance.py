"""Replay the A7R V XHD arbitration gate on synthetic rectangles."""

from pathlib import Path
from struct import pack, unpack
import unittest

from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_SP,
)

from probe_sony_cp_confidence import RTTI
from probe_sony_cp_region_scale import BASE, RETURN, CpSlices


CP = Path(__file__).resolve().parents[1] / "extracted/r5/cp/cpapp-b.bin"
GLOBAL_POST_RESULT = 0x01450290
CONFIG_POINTER = 0x010FD294
VTABLE = 0xC2FB90
GATE = BASE + 0x6ECE30
COVERAGE = BASE + 0x1C2CDC


@unittest.skipUnless(CP.is_file(), "optional pinned A7R V CP image is missing")
class TestXhdAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = CP.read_bytes()

    def setUp(self):
        self.cp = CpSlices(self.image)
        self.m = self.cp.machine
        self.state, self.rect, self.config = (RETURN + i * 0x1000 for i in (1, 2, 3))
        for addr in (self.state, self.rect, self.config, GLOBAL_POST_RESULT):
            self.cp.map_ram(addr)
        methods = self.cp.virtual_methods(VTABLE, RTTI, 0x138 // 4 + 1)
        self.assertEqual(methods[-1], BASE + 0x1C066D)
        # Use the real vtable and getter with an explicit configuration fixture.
        self.m.mem_write(GLOBAL_POST_RESULT, pack("<I", BASE + VTABLE))
        self.m.mem_write(CONFIG_POINTER, pack("<I", self.config))

    def call(self, entry, r0, r1, r2):
        stack = RETURN + 0xE00
        # A nonzero mode skips the subsequent ranking branch, not the gate.
        self.m.mem_write(stack, pack("<II", 0, self.rect + 0x24))
        self.m.mem_write(self.rect + 0x24, b"\x01")
        for reg, value in ((UC_ARM_REG_SP, stack), (UC_ARM_REG_LR, RETURN | 1),
                           (UC_ARM_REG_R0, r0), (UC_ARM_REG_R1, r1), (UC_ARM_REG_R2, r2)):
            self.m.reg_write(reg, value)
        self.cp.execute(entry, RETURN)
        return self.m.reg_read(UC_ARM_REG_R0)

    def check_gate(self, other, expected, threshold, clears):
        first = (100, 100, 100, 100)
        self.m.mem_write(self.rect + 4, pack("<4H", *first))
        self.m.mem_write(self.state + 0x160, pack("<4H", *other))
        self.m.mem_write(self.state + 0x220, b"\x00")
        self.m.mem_write(self.state + 0x755, b"\x01")
        self.m.mem_write(self.config + 0x11E, pack("<H", threshold))
        bits = self.call(COVERAGE, 0, self.rect, self.state + 0x15C)
        self.assertAlmostEqual(unpack("<f", pack("<I", bits))[0], expected)
        self.call(GATE, self.state, self.rect, self.rect + 0x20)
        # The gate only clears the flag on rejection; passing does not set it.
        self.assertEqual(self.m.mem_read(self.state + 0x755, 1), bytes([not clears]))

    def test_geometry_not_model_confidence(self):
        for other, coverage, raw, clears in (
            ((110, 110, 20, 20), 1.0, 9000, False),
            ((125, 100, 100, 100), 0.75, 7500, False),
            ((125, 100, 100, 100), 0.75, 7501, True),
            ((125, 100, 100, 100), 0.75, 0, True),
            ((200, 100, 100, 100), 0.0, 1, True),
        ):
            with self.subTest(other=other, raw=raw):
                self.check_gate(other, coverage, raw, clears)
