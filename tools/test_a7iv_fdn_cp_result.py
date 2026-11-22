from hashlib import sha256
from pathlib import Path
from random import Random
from struct import pack, pack_into
import tarfile
import unittest

from parsers.fdn_face_frame import merge_pair, project
from probe_a7iv_fdn_cp_result import (probe, probe_dispatch, probe_face_frame, probe_face_pair,
                                      probe_listener, probe_minna_backup,
                                      probe_minna_update, probe_result_pointer,
                                      probe_with_published_minna)
from probe_a7iv_fdn_cp_request import BACKUP_MEMBER


CP = Path(__file__).resolve().parents[1] / "extracted/7m4/cp/cpapp-b.bin"
FIRMWARE = CP.parents[3] / "firmware/7M4_v602/firmware.tar"
UNTOUCHED = b"\xa5" * 0x100


def minna_fixture(records):
    header = bytearray(0x100)
    header[0xc:0x10] = b"BK4\0"
    pack_into("<2I", header, 0x10, 12, len(records))
    groups = bytes(11 * 6) + pack("<HI", len(records), 0)
    items, payload = bytearray(), bytearray()
    for record in records:
        if record is None:
            ref = 0xffffffff
        else:
            ref = len(header) + len(groups) + len(records) * 6 + len(payload)
            payload.extend(pack("<I", len(record)))
            payload.extend(record)
        items.extend(pack("<HI", 0, ref))
    return bytes(header + groups + items + payload)


@unittest.skipUnless(CP.is_file(), "requires the pinned CP image")
class FdnCpResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = CP.read_bytes()

    def test_original_face_frame_writes_rectangle_width_and_height(self):
        cases = (
            ((10, 20, 40, 30), (640, 480), True, 0, (10, 20, 30, 30)),
            ((450, 20, 40, 30), (640, 480), True, 0, (460, 20, 30, 30)),
            ((10, 370, 30, 40), (640, 480), True, 0, (10, 380, 30, 30)),
            ((450, 370, 30, 40), (640, 480), True, 0, (450, 380, 30, 30)),
            ((620, 460, 20, 20), (640, 480), True, 31, (608, 448, 31, 31)),
            ((22, 31, 40, 30), (320, 240), False, 0, (22, 32, 41, 29)),
        )
        for rect, transform, square, minimum, expected in cases:
            with self.subTest(rect=rect, transform=transform):
                params = dict(transform=transform, square=square, minimum=minimum)
                x, y, w, h = probe_face_frame(self.cp, rect, (640, 480), **params)
                self.assertEqual((x, y, w, h), expected)
                self.assertEqual(project(rect, (640, 480), **params), (x, y, x + w, y + h))

    def test_face_frame_integer_projection_matches_original_on_varied_candidates(self):
        rng = Random(0x46434f)
        for i in range(96):
            width, height = rng.randrange(112, 900), rng.randrange(104, 700)
            x, y = rng.randrange(width - 12), rng.randrange(height - 12)
            w, h = rng.randrange(1, width - x - 7), rng.randrange(1, height - y - 7)
            rect, frame = (x, y, w, h), (width, height)
            params = dict(transform=(rng.randrange(width // 2, 2 * width),
                                     rng.randrange(height // 2, 2 * height)),
                          square=bool(i & 1), minimum=rng.randrange(64))
            with self.subTest(rect=rect, frame=frame, **params):
                ox, oy, ow, oh = probe_face_frame(self.cp, rect, frame, **params)
                self.assertEqual(project(rect, frame, **params), (ox, oy, ox + ow, oy + oh))
        with self.assertRaisesRegex(ValueError, "calibration input would reject"):
            project((619, 459, 21, 21), (640, 480), transform=(320, 240))

    def test_stock_same_face_gate_keeps_distinct_boxes_and_fuses_duplicates(self):
        first = (20, 30, 100, 100)
        cases = (
            (first, (120, 30, 100, 100), (100, 100), (3, 3), None),
            (first, (50, 30, 100, 100), (100, 100), (3, 3), None),
            (first, (49, 30, 100, 100), (100, 100), (3, 3), (34, 30, 100, 100)),
            (first, (49, 30, 100, 100), (100, 100), (2, 3), (37, 30, 100, 100)),
            (first, (49, 30, 100, 100), (0, 0), (3, 3), (34, 30, 100, 100)),
            (first, (20, 30, 55, 100), (100, 100), (3, 3), None),
            (first, (20, 30, 56, 100), (100, 300), (3, 3), (20, 30, 67, 100)),
            ((0, 0, 256, 256), (0, 0, 256, 256), (100, 100), (3, 3), None),
        )
        for a, b, weights, sources, expected in cases:
            with self.subTest(a=a, b=b, weights=weights, sources=sources):
                kwargs = {"weights": weights, "sources": sources}
                self.assertEqual(probe_face_pair(self.cp, a, b, **kwargs), (0x8ccd, expected))
                self.assertEqual(merge_pair(a, b, **kwargs), expected)

    def test_assign_va_noncache_resolves_manager_result_pointer(self):
        for offset, descriptor, expected in ((0x40, 0x82000002, 0x82004040),
                                             (0x80, 0x82000002, 0x82004080),
                                             (0x40, 0, 0), (0x40, 0x8200000a, 0)):
            with self.subTest(offset=offset, descriptor=descriptor):
                self.assertEqual(probe_result_pointer(self.cp, 0x502004000, offset, descriptor),
                                 expected)

    def test_one_record_copies_landmark_words_and_manager_metadata(self):
        pairs = (0x80000000, 2, 3, 0x7fffffff, 5, 0xfffffff0)
        result = probe(self.cp, (14, 0xffffffff, *pairs, 0xfedcba98), (-3, 5))
        self.assertEqual(result["left_eye_raw"], pairs[:2])
        self.assertEqual(result["right_eye_raw"], pairs[2:4])
        self.assertEqual(result["nose_root_raw"], pairs[4:])
        self.assertEqual((result["search_type"], result["exec_id"], result["user_data"]),
                         (9, 0x13579bdf, 0x2468ace0))
        self.assertEqual((result["passed"], result["score"], result["other_direction"]), (1, 14, 1))
        raw = bytes.fromhex(result["output_hex"])
        for start, end in ((0, 1), (2, 4), (0xd, 0x14), (0x1c, 0x20),
                           (0x28, 0x2c), (0x39, 0x40)):
            self.assertEqual(raw[start:end], bytes([0xa5]) * (end - start))

    def test_score_requires_signed_strict_greater_than_selected_threshold(self):
        for score, direction, expected in (
                (-4, 0, 0), (-3, 0, 0), (-2, 0, 1), (-2, 1, 0), (5, 1, 0),
                (6, 1, 1), (-1, 0xffffffff, 0), (-0x80000000, 0, 0)):
            with self.subTest(score=score, direction=direction):
                result = probe(self.cp, (score, direction, 1, 2, 3, 4, 5, 6, 7), (-3, 5))
                self.assertEqual((result["passed"], result["score"], result["other_direction"]),
                                 (expected, score, int(direction != 0)))

    def test_driver_selects_a_manager_dispatches_result_and_releases_only_that_slot(self):
        for manager_id, status in ((0, 1), (3, 0), (9, 2), (9, 1)):
            with self.subTest(manager=manager_id, status=status):
                result = probe_dispatch(self.cp, (14, 0, 1, 2, 3, 4, 5, 6, 7), (-3, 5),
                                        manager_id=manager_id, status=status)
                self.assertEqual(result["ready"], int(status == 1))
                self.assertEqual(result["manager_flags"], [int(i != manager_id) for i in range(10)])
                self.assertEqual(result["result"]["left_eye_raw"], (1, 2))
                self.assertEqual((result["result"]["score"], result["result"]["passed"]), (14, 1))

    def test_fc_listener_repackages_landmarks_for_both_result_routes(self):
        points = ((0x80000000, 0x7fffffff), (0x12345678, 0x89abcdef), (37, 41))
        flat = tuple(coordinate for point in points for coordinate in point)
        for route in (0, 1):
            for score, direction, passed in ((14, 1, 1), (-8, 0, 0)):
                with self.subTest(route=route, score=score):
                    cp = probe_dispatch(self.cp, (score, direction, *flat, 0xdeadbeef), (-3, 5),
                                        manager_id=route, status=1, search_type=route)
                    raw = bytes.fromhex(cp["result"]["output_hex"])
                    self.assertEqual(probe_listener(self.cp, raw), {
                        "exec_id": 0x13579bdf, "user_data": 0x2468ace0, "score": score,
                        "passed": passed, "landmarks": points, "other_direction": direction,
                    })

    def test_backup_loader_populates_minna_threshold_words_from_supplied_record(self):
        values = bytearray(range(256))
        for offset, word in ((0xa0, -3), (0xa4, 5), (0xb0, -987)):
            pack_into("<i", values, offset, word)
        others = (b"\x11", b"\x22\x33", b"\x44\x55\x66")
        fixture = minna_fixture((values, *others))
        full = tuple(range(0xb0000, 0xb0004))
        expected = (bytes(values), *(p + UNTOUCHED[len(p):] for p in others))
        self.assertEqual(probe_minna_backup(self.cp, fixture), (4, full, (), expected))
        fixture = minna_fixture((values, None, others[1], others[2]))
        self.assertEqual(probe_minna_backup(self.cp, fixture),
                         (-1, full[:2], (0xb0001,), (bytes(values),) + (UNTOUCHED,) * 3))

    @unittest.skipUnless(FIRMWARE.is_file(), "requires the released A7 IV archive")
    def test_first_released_backup_segment_has_no_minna_record(self):
        with tarfile.open(FIRMWARE, "r:") as tar:
            with tar.extractfile(BACKUP_MEMBER) as member:
                backup = member.read()
        self.assertEqual(probe_minna_backup(self.cp, backup),
                         (-1, (0xb0000,), (0xb0000,), (UNTOUCHED,) * 4))

    @unittest.skipUnless(FIRMWARE.is_file(), "requires the released A7 IV archive")
    def test_second_released_backup_segment_loads_actual_minna_defaults(self):
        with tarfile.open(FIRMWARE, "r:") as tar:
            backup = tar.extractfile(BACKUP_MEMBER).read()
        for score, direction, passed in ((-287995, 0, 0), (-287994, 0, 1),
                                          (-287995, 1, 0), (-287994, 1, 1)):
            with self.subTest(score=score, direction=direction):
                pdn_scores = (590558003 + direction, 590558004 - direction)
                result = probe_with_published_minna(
                    self.cp, backup, (score, direction, 1, 2, 3, 4, 5, 6, 7), pdn_scores=pdn_scores)
                self.assertEqual(result["member_offset"], 0xf3000)
                self.assertEqual(result["loaded_keys"], tuple(range(0xb0000, 0xb0004)))
                self.assertEqual(result["thresholds"], (-287995, -287995, 590558003))
                self.assertEqual(sha256(result["config_blocks"][0]).hexdigest(),
                                 "81588ef4d258ae860b23dc3eafe5e03fb7587e7d63039ac2c7885c3838986343")
                self.assertEqual(result["result"]["passed"], passed)
                self.assertEqual(result["pdn_scores"], pdn_scores)
                self.assertEqual(result["pdn_passes"], (direction, 1 - direction))

    def test_inline_update_copies_only_the_matching_minna_record(self):
        value = bytes(range(0xb4))
        self.assertEqual(probe_minna_update(self.cp, value), value + UNTOUCHED[len(value):])
        self.assertEqual(probe_minna_update(self.cp, value, key=0xb0001), UNTOUCHED)


if __name__ == "__main__":
    unittest.main()
