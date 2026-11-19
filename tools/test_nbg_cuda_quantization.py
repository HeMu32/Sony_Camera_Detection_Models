import unittest

import numpy as np
import torch

from nbg_cuda_quantization import round_scaled_u8, tp_requantize_u8


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class QuantizationTest(unittest.TestCase):
    def test_fma_recovers_side_of_tie(self):
        value = torch.tensor([9664, -9664, 3, -3], dtype=torch.float32, device="cuda")
        scale = torch.tensor([0.0018108442891389132] * 2 + [0.5] * 2, device="cuda")
        self.assertEqual(round_scaled_u8(value, scale, 128).tolist(), [145, 111, 130, 126])
        self.assertEqual(round_scaled_u8(value, scale, 128, match_trace=False).tolist(),
                         [146, 110, 130, 126])

    def test_broadcast_and_near_ties_on_nondefault_stream(self):
        rng = np.random.default_rng(1204)
        divisors = rng.integers(1, 1 << 24, size=512).astype(np.float32)
        targets = (rng.integers(0, 256, size=512) + 0.5) / divisors
        scale = targets.astype(np.float32)
        scales = np.stack((np.nextafter(scale, np.float32(0)), scale,
                           np.nextafter(scale, np.float32(np.inf))))[..., None]
        values = np.stack((divisors, -divisors), axis=-1)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value_t = torch.from_numpy(values).cuda()
            scale_t = torch.from_numpy(scales).cuda()
            for relu in (False, True):
                v = np.maximum(values, 0) if relu else values
                expected = np.clip(np.rint(v.astype(np.float64) * scales) + 128, 0, 255)
                result = round_scaled_u8(value_t, scale_t, 128, relu=relu)
                np.testing.assert_array_equal(result.cpu().numpy(), expected)

    def test_tp_add_intermediate_rounding(self):
        centered = torch.tensor([-25, 0, 25], dtype=torch.int32, device="cuda")
        # The 20x20 regression TP command's SE8M15 add rounds to 140.5 / 153.5.
        self.assertEqual(tp_requantize_u8(centered, 147, 0x8519, 17).tolist(), [140, 147, 154])
        self.assertEqual(tp_requantize_u8(centered, 147, 0x8519, 17, match_trace=False).tolist(),
                         [141, 147, 153])


if __name__ == "__main__":
    unittest.main()
