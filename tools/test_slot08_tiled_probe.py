import unittest

import numpy as np

from probe_slot08_tiled import crops, project
from render_anchor_probes import class_scores, suppress


class TiledProbeTest(unittest.TestCase):
    def test_fixed_crops_cover_edges_and_overlap(self):
        self.assertEqual(list(crops(3840, 2560)), [
            (0, 0, 3840, 2560),
            (0, 0, 2560, 1707), (1280, 0, 3840, 1707),
            (0, 853, 2560, 2560), (1280, 853, 3840, 2560),
        ])

    def test_project_undoes_padding_and_scale_before_adding_tile_origin(self):
        geom = {"source_size": [256, 128], "offset": [0, 64], "scale_xy": [.5, .5]}
        crop = (40, 70, 296, 198)
        row = {"xyxy_raster": np.array([-3, 62, 21, 88])}
        np.testing.assert_array_equal(project(row, geom, crop), [40, 70, 82, 118])
        self.assertIsNone(project({"xyxy_raster": np.array([10, 10, 24, 64])}, geom, crop))

    def test_class_score_layout_is_anchor_major(self):
        logits = np.zeros((6 * 5, 2, 2), dtype=np.float32)
        logits[4 * 5 + 1, 0, 1] = 100
        p = class_scores(logits, 5)
        self.assertEqual(p.shape, (6, 2, 2))
        self.assertEqual(p[4, 0, 1], 1.0)
        self.assertAlmostEqual(p[5, 0, 1], 0.2)

    def test_merge_removes_tile_duplicate_but_keeps_disjoint_subject(self):
        rows = [dict(xy=np.array(box), classification_softmax=p) for box, p in (
            ([0, 0, 20, 20], .9), ([1, 1, 21, 21], .96), ([40, 4, 55, 20], .93)
        )]
        self.assertEqual([r["classification_softmax"] for r in suppress(rows, "xy")], [.96, .93])


if __name__ == "__main__":
    unittest.main()
