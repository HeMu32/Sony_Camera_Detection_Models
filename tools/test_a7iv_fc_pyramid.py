from pathlib import Path
from struct import unpack_from
import unittest

from parsers.fdn_image_dma import bind as bind_fdn
from parsers.pdn_image_dma import bind as bind_pdn
from probe_a7iv_fc_pyramid import FRAME_BASE, RC_HANDLE, RC_SOURCE, probe
from probe_a7iv_fdn_cp_request import probe as probe_fdn_cp
from probe_a7iv_pdn_cp_request import probe as probe_pdn
from probe_sa_fdn_parameters import run_fdn_parameters


ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "extracted/7m4/cp/cpapp-b.bin"
FDN = ROOT / "debug/a7iv_parts/SA_FDN__SA07.bin"
PDN = ROOT / "debug/a7iv_parts/SA_PDN__SA07.bin"


@unittest.skipUnless(CP.is_file(), "requires the extracted A7 IV 6.02 CP image")
class FcPyramidTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = CP.read_bytes()
        cls.levels = probe(cls.cp)["frame_records"]

    def test_original_mode_dimensions_and_output_registers(self):
        sizes = [(640, 360), (640, 480), (640, 426), (640, 240), (480, 480),
                 (640, 384), (640, 338)]
        pitches = (640, 320, 192, 128, 64)
        capacities = (640, 320, 160, 80, 40)
        offsets = (0, 0x96000, 0xbb800, 0xc6c00, 0xca800)
        for mode, (width, height) in enumerate(sizes):
            with self.subTest(mode=mode):
                run = probe(self.cp, mode=mode)
                self.assertEqual(run["dimensions"], [width, height])
                self.assertEqual(run["output_records"], tuple(
                    (FRAME_BASE + off, pitch, height >> i, capacity, 1)
                    for i, (off, pitch, capacity) in enumerate(zip(offsets, pitches, capacities))
                ))
                self.assertEqual(run["frame_records"], tuple(
                    (FRAME_BASE + off, width >> i, height >> i, pitch)
                    for i, (off, pitch) in enumerate(zip(offsets, pitches))
                ))
                self.assertEqual(run["mmio"], {
                    "geometry": 0x00a00000 | height,
                    "destination_low_words": tuple(0x20000 + off for off in offsets),
                    "destination_high_control": 0x100,
                    "output_mask": 0x1f,
                })
                self.assertEqual(run["source"], {
                    "device_id": 0xad, "call_site": "0x25dc96", "writer": "0xa4afdc",
                    "control": 0x1802, "start_low": 0, "start_high": 0, "size": 1280 * height,
                })
                msg = bytes.fromhex(run["message_hex"])
                self.assertEqual(unpack_from("<H", msg)[0], 1)
                self.assertEqual(msg[8 + 14:8 + 16], b"\xa5\xa5")
        self.assertEqual(tuple(b - a for a, b in zip(offsets, offsets[1:])),
                         tuple(2 * pitch * (480 >> i) for i, pitch in enumerate(pitches[:4])))

    def test_prc_reconfigure_leaves_control_untouched(self):
        run = probe(self.cp, mode=4, reconfigure=True)
        self.assertEqual(run["source"], {
            "device_id": 0xad, "call_site": "0x25dc62", "writer": "0xa4b112",
            "control": 0, "start_low": 0, "start_high": 0, "size": 1280 * 480,
        })

    def test_original_resize_enables_link_and_skips_explicit_writeback(self):
        for mode, width in ((1, 640), (4, 480)):
            with self.subTest(mode=mode):
                run = probe(self.cp, mode=mode)
                self.assertEqual(run["resize"], {
                    "handle": RC_HANDLE, "callback": 0x00a2009b, "callback_arg": 0,
                    "input_words": (RC_SOURCE, 0), "input_canvas": (1920, 1080),
                    "input_regions": ((0, 0, 1920, 1080),) * 2,
                    "output_canvas": (640, 480), "output_regions": ((0, 0, width, 480),) * 2,
                    "output_kind": 2, "output_selector": run["source"]["device_id"],
                    "output_option": 6 if mode == 4 else 0,
                    "native_config_word": 0x802, "zero_control_config_word": 2,
                    "native_writeback_call_reached": False,
                    "zero_control_writeback_call_reached": True,
                    "config_word_controls": {"selector_1": 0x802, "selector_ae": 0x802,
                                             "selector_100": 0x802, "input_kind_3": 0x820,
                                             "option_1b": 0xb0802},
                })

    @unittest.skipUnless(FDN.is_file(), "requires the pinned FDN member")
    def test_producer_records_reach_fdn_dma_for_valid_geometry(self):
        fdn = FDN.read_bytes()
        for level, (physical, _, _, pitch) in enumerate(self.levels):
            with self.subTest(level=level):
                select = probe_fdn_cp(
                    self.cp, manager_id=3, search_type=0, usecase=1,
                    window=(640, 480, 640, 8, 8, 48, 48, 0, 2),
                    pyramid=self.levels, candidate_extent=48 << level,
                )
                self.assertEqual(select["frame_selection"]["level"], level)
                trace = run_fdn_parameters(
                    fdn, int(select["sammu_parameter_address"], 16),
                    bytes.fromhex(select["parameter_block_hex"]), bytes(0x160), trace_setup=True,
                )
                if level == 4:
                    self.assertNotEqual(trace["validator_mask"], 0)
                    self.assertNotIn("setup_stop", trace)
                    continue
                self.assertEqual(trace["validator_mask"], 0)
                dma = bind_fdn(trace)
                offset = (physical - 0x500000000) // 4
                self.assertEqual(trace["converted_addresses"][0], 0xa0000000 + offset)
                self.assertEqual(dma.source_pitch_bytes, 2 * pitch)
                self.assertEqual(dma.descriptor_words[0], 0x20000000 + offset + dma.source_offset)

    @unittest.skipUnless(PDN.is_file(), "requires the pinned PDN member")
    def test_producer_records_reach_pdn_dma_for_valid_geometry(self):
        pdn = PDN.read_bytes()
        for level, (physical, width, height, pitch) in enumerate(self.levels):
            with self.subTest(level=level):
                run = probe_pdn(self.cp, pdn, pyramid=self.levels, result_level=level,
                                left=(32 << 18, 32 << 18), right=(48 << 18, 32 << 18))
                if level == 4:
                    self.assertNotEqual(run["validator_mask"], 0)
                    self.assertNotIn("image_geometry", run)
                    continue
                self.assertEqual(run["validator_mask"], 0)
                geometry = run["image_geometry"]
                dma = bind_pdn(geometry)
                offset = (physical - 0x500000000) // 4
                self.assertEqual(geometry["frame"], [0xa0000000 + offset, width, height, 2 * pitch])
                self.assertEqual(geometry["image_request"]["rows"][1][0],
                                 0x20000000 + offset + dma.source_offset)


if __name__ == "__main__":
    unittest.main()
