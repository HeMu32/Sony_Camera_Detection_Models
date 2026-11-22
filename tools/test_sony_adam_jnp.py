from hashlib import sha256
from pathlib import Path
from struct import pack, pack_into, unpack_from
import unittest

from parsers.nab_extract import extract
from probe_sony_adam_jnp import (
    CONTROL_BYTES, JnpProbe, POSE_BYTES, PoseProjection, original_pose_parameters,
    original_posture_job,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGES = tuple(ROOT / f"extracted/{camera}/adam/adam_app.bin" for camera in ("r5", "1m2"))
CP = (ROOT / "extracted/r5/cp/cpapp-b.bin",
      ROOT / "debug/deepseek_pose_dispatch/fw/a1ii_cp/cpapp-b.bin")
NAB = tuple(ROOT / f"extracted/{camera}/ai_dict/default.nab" for camera in ("r5", "1m2"))
HOST_OUTPUT = tuple(ROOT / f"debug/cpu_raw/{camera}_s01" for camera in ("r5", "m2"))


def fixture(width, height):
    block = bytearray(range(CONTROL_BYTES))
    block[0x20:0x22] = bytes((width - 1, height - 1))
    block[0x1e], block[0x1f], block[0x3f], block[0x47] = 1, 2, 23, 0x90
    pack_into("<7I", block, 0xc8, *(0x200000 + i * 0x1000 for i in range(7)))
    return block


@unittest.skipUnless(all(path.is_file() for path in IMAGES), "requires both extracted Adam images")
class AdamJnpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images = [path.read_bytes() for path in IMAGES]

    def test_same_register_forwarding_for_human_and_r5_animal_grid_fixtures(self):
        offsets = (4, 8, 0x14, *range(0x1c, 0xa8, 4), *range(0xc0, 0xe4, 4))
        for image in self.images:
            jnp = JnpProbe(image)
            for width, height in ((64, 42), (52, 36), (8, 2), (80, 80)):
                with self.subTest(camera=jnp.model, width=width, height=height):
                    block = fixture(width, height)
                    status, writes = jnp.configure(block)
                    self.assertEqual(status, 0)
                    self.assertEqual(sorted(writes),
                                     [(off, unpack_from("<I", block, off)[0]) for off in offsets])

    def test_rejection_does_not_write_registers(self):
        for image in self.images:
            jnp = JnpProbe(image)
            rejects = ((0x47, 0x80), (0x20, 62), (0x21, 0), (0x21, 80), (0x20, 80),
                       (0x3f, 24), (0x3f, 0), (0x1f, 24),
                       *((offset, 1) for offset in range(0xc8, 0xe4, 4)))
            for offset, value in rejects:
                with self.subTest(camera=jnp.model, offset=offset, value=value):
                    block = fixture(64, 42)
                    block[offset] = value
                    self.assertEqual(jnp.configure(block), (1, []))

    def test_register_count_and_timeout_paths(self):
        peaks = [(i * 17) & 31 for i in range(24)]
        for image in self.images:
            jnp = JnpProbe(image)
            with self.subTest(camera=jnp.model):
                self.assertEqual(jnp.counts(peaks, 0xab), ((0, 0), peaks, 0xab))
                self.assertEqual(jnp.counts(peaks, 0xab, busy=True), ((2, 2), [0] * 24, 0))

    @unittest.skipUnless(all(path.is_file() for path in CP), "requires both extracted CP images")
    def test_original_posture_defaults_reach_adam_registers(self):
        for adam, path in zip(self.images, CP):
            jnp = JnpProbe(adam)
            with self.subTest(camera=jnp.model):
                payload, schedule = original_pose_parameters(path.read_bytes(), expected_model=jnp.model)
                self.assertEqual(schedule[1::2], (2, 6, 2))
                self.assertEqual(schedule[2], 0x10400530)
                self.assertEqual(sha256(payload).hexdigest(),
                                 "561914ad7221c6deaf9a0b9c3733146ff1499a1ba5f42ba67176a6705ec4231e")
                registers = jnp.prepare_pose(payload)
                status, writes = jnp.configure(registers)
                self.assertEqual((status, len(writes)), (0, 47))
                values = dict(writes)
                self.assertEqual((1 + (registers[0x20] & 127), 1 + (registers[0x21] & 127)), (64, 42))
                self.assertEqual((1 + (registers[0x3f] & 31), registers[0x3b]), (19, 10))
                self.assertEqual([values[i] for i in (0x28, 0x2c, 0x30, 0x34, 0x40, 0x44)],
                                 [0x19191919, 0x05050500, 0x039901cc, 0x03990399,
                                  0x00660002, 0x902b00c8])
                for i in range(4):
                    updated = bytearray(payload)
                    updated[0x67 + i] += 1
                    status, changed_writes = jnp.configure(jnp.prepare_pose(updated))
                    self.assertEqual(status, 0)
                    self.assertEqual(dict(changed_writes), {**values, 0x28: 0x19191919 + (1 << (8 * i))})

    @unittest.skipUnless(all(path.is_file() for path in (*CP, *NAB)),
                         "requires both extracted CP and NAB images")
    def test_original_slot_one_output_bases_feed_jnp(self):
        for adam, cp, nab in zip(self.images, CP, NAB):
            jnp = JnpProbe(adam)
            with self.subTest(camera=jnp.model):
                job = original_posture_job(cp.read_bytes(), expected_model=jnp.model)
                config = job.postprocess[0] - 0x74
                self.assertEqual(job.procedure_list[:4], (
                    (0, 2, config + 0x74), (1, 2, config + 0x11c),
                    (2, 6, config + 0x530), (0x10, 0, config + 0x61c),
                ))
                boundary = jnp.npu_boundary(job.npu)
                self.assertEqual(boundary["graph"], 1)
                manifest = {row["slot"]: row for row, _ in extract(nab.read_bytes())}
                graph = manifest[boundary["graph"]]
                self.assertEqual(graph["sha256"],
                                 "a5d5ead9478e3b069598483bda95702283d7b0c3d825de02c67250eeea4394b6")
                input_bytes = graph["input_allocation_bytes"]
                output_bytes = graph["output_allocation_bytes"]
                self.assertEqual((input_bytes, output_bytes), (516096, 158592))
                self.assertEqual(boundary["inputs"], ((0x0ff00000, input_bytes),))
                self.assertEqual(boundary["outputs"],
                                 ((0x13100000, output_bytes), (0x13180000, output_bytes)))

                status, writes = jnp.configure(jnp.prepare_pose(job.jnp))
                self.assertEqual(status, 0)
                values = dict(writes)
                first, second = (address for address, _ in boundary["outputs"])
                self.assertEqual((values[0xcc], values[0xc8]), (first, second))
                self.assertEqual((values[0xd0] - first, values[0xdc]), (64 * 42, 64 * 42))

                for src, dst in ((0x3c, 0xc8), (0x40, 0xd4), (0x44, 0xd8),
                                 (0x48, 0xcc), (0x4c, 0xd0), (0x50, 0xdc),
                                 (0x54, None), (0x58, 0xe0), (0x5c, None)):
                    with self.subTest(payload_offset=hex(src)):
                        payload = bytearray(job.jnp)
                        payload[src] ^= 8
                        status, writes = jnp.configure(jnp.prepare_pose(payload))
                        self.assertEqual(status, 0)
                        expected = values if dst is None else {**values, dst: values[dst] ^ 8}
                        self.assertEqual(dict(writes), expected)

    @unittest.skipUnless(all((path / f"Human.{i}.raw").is_file()
                             for path in HOST_OUTPUT for i in range(2)),
                         "requires both saved slot-1 SDK runs")
    def test_saved_slot_one_tensors_are_not_descriptor_sized(self):
        for path in HOST_OUTPUT:
            with self.subTest(run=path.name):
                sizes = tuple((path / f"Human.{i}.raw").stat().st_size for i in range(2))
                self.assertEqual(sizes, (64 * 42 * 40, 64 * 42 * 19))
                self.assertEqual(sum(sizes), 158592)

    def test_cpu_packs_distinct_synthetic_poses_and_handles_empty(self):
        for image in self.images:
            jnp = JnpProbe(image)
            for parts in (2, 24):
                with self.subTest(camera=jnp.model, parts=parts):
                    cases = ((0x12, 0x34, 0x56, 0x78), (1, 2, 3, 4), (127, 112, 94, 57))
                    records, expected = [], []
                    for i, (a, b, c, d) in enumerate(cases):
                        joints = pack(f"<{parts}I", *(0x123 * (k + 1) + i for k in range(parts)))
                        records.append(joints + pack("<3I", a << 19 | b << 6, c << 19 | d << 6,
                                                     ((0x1f - i) << 16) | (0x1234 + i)))
                        expected.append(pack("<BxH4B", 0x1f - i, 0x1234 + i, c, d, a, b)
                                        + joints + bytes((24 - parts) * 4))
                    self.assertEqual(jnp.pack_poses(records, parts=parts, tag=1),
                                     bytes((1, len(cases), 0, 0)) + b"".join(expected))
                    self.assertTrue(all(len(row) == POSE_BYTES for row in expected))
                    self.assertEqual(jnp.pack_poses([], parts=parts, tag=1)[:2], b"\x01\0")
                    self.assertEqual(jnp.pack_poses(records, parts=parts, tag=0)[:2], b"\0\x03")

    @unittest.skipUnless(all(path.is_file() for path in CP), "requires both extracted CP images")
    def test_original_adam_result_through_original_cp_projection(self):
        boxes = ((0, 0, 64, 42), (7, 13, 27, 38), (23, 6, 42, 38))
        records = [pack("<4I", 0x1234, (x2 << 19) | (y2 << 6),
                        (x1 << 19) | (y1 << 6), 0x10234)
                   for x1, y1, x2, y2 in boxes]
        for adam, cp in zip(self.images, CP):
            jnp, projection = JnpProbe(adam), PoseProjection(cp.read_bytes())
            packed = jnp.pack_poses(records, parts=1, tag=3)
            for width, height in ((512, 336), (641, 481), (1919, 1081), (17, 11)):
                with self.subTest(camera=jnp.model, canvas=(width, height)):
                    expected = []
                    for x1, y1, x2, y2 in boxes:
                        x, y = (width * x1) // 64, (height * y1) // 42
                        w, h = (width * (x2 - x1)) // 64, (height * (y2 - y1)) // 42
                        expected.append((x, y, x + w, y + h))
                    self.assertEqual(projection.project(packed, canvas=(width, height)), expected)
                    self.assertEqual(projection.project(b"\x01" + packed[1:], canvas=(width, height)),
                                     expected[:1])
            empty = jnp.pack_poses([], parts=1, tag=3)
            self.assertEqual(projection.project(empty, canvas=(512, 336)), [])
            disabled = jnp.pack_poses(records, parts=1, tag=0)
            self.assertEqual(projection.project(disabled, canvas=(512, 336)), [])
            with self.assertRaisesRegex(ValueError, "complete Adam JNP"):
                projection.project(packed[:4], canvas=(512, 336))


if __name__ == "__main__":
    unittest.main()
