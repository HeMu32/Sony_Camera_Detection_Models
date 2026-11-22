from pathlib import Path
from struct import pack, unpack
import tarfile
import unittest

from probe_a7iv_fdn_cp_request import BACKUP_MEMBER, WINDOW, probe


CP = Path(__file__).resolve().parents[1] / "extracted/7m4/cp/cpapp-b.bin"
FIRMWARE = CP.parents[3] / "firmware/7M4_v602/firmware.tar"
SP_BUNDLE = CP.parents[1] / "sabin/sa_sabin_cp_cam_post.bin"


def read_factory_backup():
    with tarfile.open(FIRMWARE, "r:") as tar:
        with tar.extractfile(BACKUP_MEMBER) as file:
            return file.read()


@unittest.skipUnless(CP.is_file(), "requires the extracted A7 IV 6.02 CP image")
class FdnCpRequestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = CP.read_bytes()

    def test_original_arm_packs_parameter_fields_and_queue_request(self):
        run = probe(self.cp, manager_id=3, search_type=0x35, usecase=1)
        self.assertEqual(unpack("<4I9H2xI", bytes.fromhex(run["parameter_block_hex"])),
                         (0x40008000, 0x40040000, 0x40046500, 0x40004040,
                          *WINDOW, 0x13579bdf))
        self.assertEqual(bytes.fromhex(run["cp_descriptor_hex"]),
                         pack("<IIQBB6x", 0x80001000, 0, 0x500004000, 3, 1))
        self.assertEqual(bytes.fromhex(run["cp_request_hex"]),
                         pack("<HHIII", 0xf000, 0, 0x30, 0, 0) +
                         pack("<H6BII B3x III", 0xa9a0, 0, 0x40, 1, 0x88, 0, 0,
                              0x70b9eb, 0, 3, 0x40000400, 0x40001000, 0))
        self.assertEqual(run["queue_boundary"],
                         {"pc": "0xb02610", "destination": "0x930101", "length": 48})

    def test_manager_and_usecase_bytes_are_not_search_type(self):
        base = probe(self.cp, manager_id=0, search_type=0, usecase=0)
        changed = probe(self.cp, manager_id=9, search_type=0x35, usecase=1)
        self.assertEqual(changed["manager_search_type_after_build"], 0x35)
        self.assertEqual(changed["parameter_block_hex"], base["parameter_block_hex"])
        a, b = (bytes.fromhex(run["cp_request_hex"]) for run in (base, changed))
        self.assertEqual([i for i, (first, second) in enumerate(zip(a, b)) if first != second],
                         [0x14, 0x20])
        self.assertEqual((b[0x14], b[0x20]), (1, 9))
        search_only = probe(self.cp, manager_id=0, search_type=0xff, usecase=0)
        self.assertEqual(search_only["manager_search_type_after_build"], 0xff)
        self.assertEqual(search_only["cp_request_hex"], base["cp_request_hex"])

    def test_rejects_an_unpinned_cp_image(self):
        with self.assertRaisesRegex(ValueError, "expected pinned"):
            probe(b"not firmware", manager_id=0, search_type=0, usecase=0)

    def test_receiver_uses_real_subrecord_and_stops_at_backup_state_lookup(self):
        run = probe(self.cp, manager_id=3, search_type=0x35, usecase=1, receiver=True)
        self.assertEqual(run["receiver_slices"], {
            "bulk_classifier": {"destination_pc": "0x50b594", "arguments": [0xa9, 0, 0x40]},
            "subrecord_offset": 0x10, "subrecord_length": 0x20, "forwarded_offset": 0x10,
            "subrecord_handler": "0xb03e14", "start_handler": "0x50bbd0",
            "lookup_boundary": {"pc": "0xb02242", "job": 0x40,
                                "request_byte_7": 0, "address": "0x14f8f24"},
            "backup_read": {"pc": "0xac9698", "key": "0x930002",
                            "destination": "0x14f8ee4", "length": 0x100},
        })

    @unittest.skipUnless(FIRMWARE.is_file(), "requires the A7 IV 6.02 tar")
    def test_original_backup_reader_supplies_fdn_job_byte_and_enters_next_predicate(self):
        backup = read_factory_backup()
        run = probe(self.cp, manager_id=3, search_type=0, usecase=1, receiver=True, backup=backup)
        self.assertEqual(run["receiver_slices"]["factory_lookup"], {
            "table_sha256": "d57793b4da4f5eae83e750448e605c9f7ac52e62db3c820b59d2244a6d457626",
            "job_byte": "0x44", "next_predicate_pc": "0xb05ed2",
            "runtime_counter_address": "0x1473968",
        })

    @unittest.skipUnless(FIRMWARE.is_file() and SP_BUNDLE.is_file(),
                         "requires the A7 IV 6.02 tar and extracted camera-post SP bundle")
    def test_cold_scheduler_selects_original_fdn_and_packs_resource_and_preload_requests(self):
        backup, bundle = read_factory_backup(), SP_BUNDLE.read_bytes()
        for usecase, mask in enumerate((0x3, 0x7)):
            with self.subTest(usecase=usecase):
                run = probe(self.cp, manager_id=3, search_type=0, usecase=usecase,
                            backup=backup, bundle=bundle)
                scheduler = run["receiver_slices"]["scheduler_slices"]
                self.assertEqual(scheduler["usecase_mask"], hex(mask))
                self.assertEqual(scheduler["worker_selection"], "0x1000")
                self.assertEqual(scheduler["allocated_request_byte_2"], 1)
                self.assertEqual(scheduler["sp_member"], {
                    "offset": "0xe2d00", "size": 0x31d00, "name": "SA_FDN__SA07",
                    "sha256": "1d146af1db2b188dcaf1bdbf41ec9dee95b6fe6bfc8d76b2295abe3f9a77eead",
                })
                boundary = scheduler["load_boundary"]
                self.assertEqual(boundary["arguments"], [0x4e3, 0, 5000])
                self.assertEqual(boundary["allocation_queue"], "0x61003c")
                self.assertEqual(boundary["allocation_size"], 40)
                self.assertEqual(unpack("<6I", bytes.fromhex(boundary["initialized_prefix_hex"])),
                                 (0x20, 0x4e3, 0, 5000, 0x930101, 0x10000))
                completion = scheduler["resource_completion_fixture"]
                self.assertEqual(unpack("<5I", bytes.fromhex(completion["normalized_reply_hex"])),
                                 (0, 0, 0x11223344, 0x55667788, 0x82000000))
                self.assertEqual(completion["stop_pc"], "0x4e63a4")
                self.assertEqual(completion["descriptor_address"], "0x147397c")
                self.assertEqual(unpack("<4I", bytes.fromhex(completion["descriptor_hex"])),
                                 (1, 1, 0x014739a0, 0xb05ee5))
                self.assertEqual(completion["transfer"], {
                    "size": 0x31d00, "source": "0x330e2d00", "destination": "0x82000000",
                })
                self.assertEqual(completion["entry_lookup_with_fixture_copy"], [
                    {"selector": 0, "status": 0, "returned_entry": "0x82000100"},
                    {"selector": 0xff, "status": 12, "returned_entry": "0x0"},
                ])
                writes = scheduler["sa_argument_write_slice"]
                self.assertEqual(writes["stop_pc"], "0x50cbba")
                self.assertEqual(writes["register_base"], "0xf5440000")
                self.assertEqual(writes["writes"], [
                    {"offset": "0x110", "size": 4, "value": run["sammu_parameter_address"]},
                    {"offset": "0x114", "size": 4, "value": run["sammu_auxiliary_address"]},
                    {"offset": "0x118", "size": 4, "value": "0x0"},
                    {"offset": "0x11c", "size": 4, "value": "0x0"},
                ])
                preload = scheduler["startup_preload"]
                self.assertEqual(preload["destination"], "0x80bd017a")
                self.assertEqual(preload["runtime_count_before_send"], 0)
                self.assertEqual(bytes.fromhex(preload["factory_table_hex"]),
                                 bytes.fromhex("00 b0 00 b1 01 b4 01 b5") + bytes(8))
                self.assertEqual(unpack("<6I", bytes.fromhex(preload["message_hex"])),
                                 (0x41, 1, 10000, 0, 0x88b4, 0))
                self.assertEqual(scheduler["nonzero_counter_fixture"], {
                    "pc": "0x50bca0", "return_code": 0, "ring_address": "0x11ec660",
                    "ring_state": [1, 1, 0],
                    "record_prefix_hex": bytes.fromhex(run["cp_request_hex"])[0x10:].hex(" "),
                })


if __name__ == "__main__":
    unittest.main()
